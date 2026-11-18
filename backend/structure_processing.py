"""Parse protein structures and build validated 20 x 20 interface features."""

from __future__ import annotations

import io
import math
import ast
import csv
import gzip
import json
import os
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from Bio.Align import PairwiseAligner
from typing import BinaryIO, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, NeighborSearch
from Bio.PDB.Residue import Residue
from Bio.SeqUtils import seq1

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AA_ORDER)}
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
SABDAB_API = "https://sabdab.opig.stats.ox.ac.uk/api"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_heavy(atom) -> bool:
    """Ignore hydrogen/deuterium atoms and keep one selected alternate location."""
    element = (atom.element or atom.get_name()[0]).strip().upper()
    return element not in {"H", "D"}


def _residue_letter(residue: Residue) -> str | None:
    """Return a standard one-letter amino-acid code, or None for other residues."""
    try:
        letter = seq1(residue.resname, custom_map={}).upper()
    except Exception:
        return None
    return letter if letter in AA_TO_INDEX else None


def parse_structure(source: str | Path | BinaryIO, filename: str | None = None):
    """Parse PDB or mmCIF input; file-like objects need a filename hint."""
    name = filename or str(source)
    suffix = Path(name).suffix.lower()
    if hasattr(source, "read"):
        contents = source.read()
        if isinstance(contents, bytes):
            try:
                contents = contents.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(f"Structure file is not valid UTF-8 text: {exc}") from exc
        handle = io.StringIO(contents)
    else:
        handle = str(source)
    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    elif suffix in {".pdb", ".ent"}:
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported structure extension: {suffix or '(missing)'}")
    try:
        structure = parser.get_structure("sample", handle)
    except Exception as exc:
        raise ValueError(f"Structure could not be parsed: {exc}") from exc
    models = list(structure.get_models())
    if not models:
        raise ValueError("Structure contains no coordinate model")
    return models[0]

# from Bio.PDB import PDBIO
# model = parse_structure(r"C:\Users\yadavalli charishma\Downloads\splits\splits_final\pdb_00002r0z_H_L.cif")

# io = PDBIO()
# io.set_structure(model)
# io.save("parsed_structure.pdb")

def _chain_residues(model, chain_ids: Sequence[str], residue_ranges: Mapping[str, tuple] | None = None) -> tuple[list[Residue], list]:
    """Collect standard amino-acid residues and their heavy atoms for chosen chains."""
    requested = [str(chain_id).strip() for chain_id in chain_ids]
    if not requested or "" in requested:
        raise ValueError("Chain identifiers are missing")
    found = [chain.id for chain in model.get_chains()]
    resolved = {}
    for requested_id in requested:
        if requested_id in found:
            resolved[requested_id] = requested_id
            continue
        matches = [actual_id for actual_id in found if actual_id.casefold() == requested_id.casefold()]
        if len(matches) == 1:
            resolved[requested_id] = matches[0]
        elif len(matches) > 1:
            raise ValueError(f"Chain ID {requested_id!r} has an ambiguous case-insensitive match: {matches}")
        else:
            raise ValueError(f"Required chain(s) not present: {requested_id}")
    wanted = set(resolved.values())
    actual_ranges = {
        resolved.get(str(chain_id).strip(), str(chain_id).strip()): residue_range
        for chain_id, residue_range in (residue_ranges or {}).items()
    }

    residues: list[Residue] = []
    atoms = []
    for chain in model.get_chains():
        if chain.id not in wanted:
            continue
        for residue in chain.get_residues():
            if actual_ranges and chain.id in actual_ranges:
                start, end = actual_ranges[chain.id]
                residue_number = residue.id[1]
                insertion_code = residue.id[2].strip()
                if not (start <= (residue_number, insertion_code) <= end):
                    continue
            if _residue_letter(residue) is None:
                continue
            residue_atoms = [atom for atom in residue.get_atoms() if _is_heavy(atom)]
            if not residue_atoms:
                continue
            residues.append(residue)
            atoms.extend(residue_atoms)
    if not residues:
        raise ValueError(f"Selected chain(s) {sorted(wanted)} contain no usable standard amino-acid residues")
    if not atoms or not np.isfinite(np.asarray([atom.coord for atom in atoms])).all():
        raise ValueError("Selected chains have missing or non-finite atom coordinates")
    return residues, atoms


def _residue_key(residue: Residue) -> tuple[str, tuple]:
    return residue.get_parent().id, residue.id


@lru_cache(maxsize=1)
def _sabdab_imgt_records() -> list[dict]:
    """Read only the antibody IMGT metadata needed for CDR counts from the supplied ZIP."""
    archive = Path(os.getenv("SPLITS_FINAL_ZIP", str(Path(__file__).resolve().parents[1] / "data" / "raw" / "splits_final.zip")))
    if not archive.is_file():
        return []
    member = "splits_final/abag_split.csv"
    records = []
    with zipfile.ZipFile(archive) as zipped:
        with zipped.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
            for row in csv.DictReader(text):
                if row.get("VH_numbering_list") or row.get("VL_numbering_list"):
                    records.append(row)
    return records


def _identifier_core(value: str) -> str:
    """Normalize PDB IDs and dataset instance names for metadata matching."""
    stem = Path(str(value)).stem.lower()
    if stem.startswith("pdb_"):
        accession = stem[4:].split("_", 1)[0]
        return accession[-4:]
    return stem[:4]


def _parse_imgt_numbering(raw: str) -> list[tuple[str, int, str]]:
    """Convert a SAbDab residue/IMGT list into (amino acid, position, insertion) entries."""
    if not raw:
        return []
    try:
        values = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    parsed = []
    for item in values if isinstance(values, (list, tuple)) else []:
        try:
            amino_acid, number = item
            position, insertion = number
            letter = str(amino_acid).upper()
            position = int(position)
            if letter in AA_TO_INDEX:
                parsed.append((letter, position, str(insertion or "").strip()))
        except (TypeError, ValueError):
            continue
    return parsed


def _sabdab_request_json(path: str) -> dict:
    """Fetch one record from the public SAbDab2 API."""
    request = Request(
        f"{SABDAB_API}{path}",
        headers={"Accept": "application/json", "User-Agent": "AntibodyAntigenClassifier/1.0"},
    )
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def _segment_numbering_for_local_mapper(segment: Mapping) -> str:
    """Convert SAbDab2 numbering into the local CSV numbering format."""
    entries = []
    for item in segment.get("numbering") or []:
        try:
            position, amino_acid = item
            entries.append([amino_acid, position])
        except (TypeError, ValueError):
            continue
    return repr(entries)


def _download_sabdab_fallback(structure_id: str) -> tuple[list[dict], Path | None, str | None]:
    """Fetch CDR numbering and an IMGT-numbered mmCIF for a PDB ID missing locally."""
    raw_id = Path(str(structure_id)).stem.lower()
    valid_pdb_name = bool(
        re.fullmatch(r"[a-z0-9]{4}", raw_id)
        or re.fullmatch(r"pdb_[a-z0-9]{8}(?:_.*)?", raw_id)
    )
    if not valid_pdb_name:
        return [], None, "SAbDab lookup needs a PDB ID or a PDB-named structure file."
    pdb_core = _identifier_core(structure_id)
    if len(pdb_core) != 4 or not pdb_core.isalnum():
        return [], None, "SAbDab lookup needs a valid four-character PDB ID."

    extended_id = f"pdb_0000{pdb_core}"
    try:
        pdb_entry = _sabdab_request_json(f"/pdb/{extended_id}")
        rows = []
        for instance in pdb_entry.get("antibody_instances") or []:
            instance_id = str(instance.get("id", ""))
            if not instance_id:
                continue
            details = _sabdab_request_json(f"/antibody-instance/{quote(instance_id, safe='')}")
            heavy = details.get("heavy_like_segment") or {}
            light = details.get("light_like_segment") or {}
            heavy_chain = heavy.get("sabdab_auth_asym_id")
            light_chain = light.get("sabdab_auth_asym_id")
            if not heavy_chain and not light_chain:
                continue
            row = {"INSTANCE": instance_id, "PDB_ID": extended_id}
            if heavy_chain:
                row.update({
                    "Hchain": heavy_chain,
                    "VH_numbering_list": _segment_numbering_for_local_mapper(heavy),
                    "CDRH1": heavy.get("cdr1", ""),
                    "CDRH2": heavy.get("cdr2", ""),
                    "CDRH3": heavy.get("cdr3", ""),
                })
            if light_chain:
                row.update({
                    "Lchain": light_chain,
                    "VL_numbering_list": _segment_numbering_for_local_mapper(light),
                    "CDRL1": light.get("cdr1", ""),
                    "CDRL2": light.get("cdr2", ""),
                    "CDRL3": light.get("cdr3", ""),
                })
            rows.append(row)
    except HTTPError as exc:
        if exc.code == 404:
            return [], None, None
        return [], None, f"SAbDab returned HTTP {exc.code}."
    except (URLError, TimeoutError) as exc:
        return [], None, f"Could not reach SAbDab: {exc}"
    except Exception as exc:
        return [], None, f"SAbDab lookup failed: {exc}"

    if not rows:
        return [], None, None

    output_path = PROJECT_ROOT / "data" / "processed" / f"{pdb_core}_sabdab_imgt.cif"
    if output_path.is_file():
        return rows, output_path, None
    try:
        request = Request(
            f"{SABDAB_API}/download/structures",
            data=json.dumps({"structures": [{"pdb_id": extended_id, "full_structure": True}]}).encode("utf-8"),
            headers={
                "Accept": "chemical/x-mmcif",
                "Content-Type": "application/json",
                "User-Agent": "AntibodyAntigenClassifier/1.0",
            },
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            structure_bytes = response.read(100 * 1024 * 1024 + 1)
            if len(structure_bytes) > 100 * 1024 * 1024:
                raise ValueError("SAbDab structure download exceeds the 100 MB limit.")
            if response.headers.get("Content-Encoding", "").lower() == "gzip" or structure_bytes.startswith(b"\x1f\x8b"):
                structure_bytes = gzip.decompress(structure_bytes)
        parse_structure(io.BytesIO(structure_bytes), filename=output_path.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".cif.part")
        temporary_path.write_bytes(structure_bytes)
        temporary_path.replace(output_path)
        return rows, output_path, None
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return rows, None, f"CDR annotations were found in SAbDab, but its mmCIF download failed: {exc}"


def _align_residues_to_imgt(residues: list[Residue], numbered: list[tuple[str, int, str]]) -> dict[int, tuple[int, str]]:
    """Align observed coordinates to SAbDab's precomputed IMGT sequence positions."""
    if not residues or not numbered:
        return {}
    structure_sequence = "".join(_residue_letter(residue) for residue in residues)
    metadata_sequence = "".join(letter for letter, _position, _insertion in numbered)
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(structure_sequence, metadata_sequence)[0]
    mapping = {}
    for (structure_start, structure_end), (meta_start, meta_end) in zip(*alignment.aligned):
        for structure_index, metadata_index in zip(range(structure_start, structure_end), range(meta_start, meta_end)):
            mapping[structure_index] = numbered[metadata_index][1:]
    return mapping


def analyze_cdr_regions(
    model,
    partner_a_chains: Sequence[str],
    partner_b_chains: Sequence[str],
    cutoff_angstrom: float,
    partner_a_residue_ranges: Mapping[str, tuple] | None = None,
    partner_b_residue_ranges: Mapping[str, tuple] | None = None,
    structure_id: str = "",
) -> dict:
    """Map supplied SAbDab IMGT numbering onto observed heavy/light-chain residues."""
    try:
        records = _sabdab_imgt_records()
        exact = [row for row in records if str(row.get("INSTANCE", "")).lower() == Path(structure_id).stem.lower()]
        matches = exact or [row for row in records if _identifier_core(row.get("PDB_ID", "")) == _identifier_core(structure_id)]
        sabdab_structure_path = None
        lookup_error = None
        used_sabdab_api = False
        if not matches:
            matches, sabdab_structure_path, lookup_error = _download_sabdab_fallback(structure_id)
            used_sabdab_api = bool(matches)
            if not matches:
                message = lookup_error or "No antibody record for this PDB ID was found in the local dataset or SAbDab."
                return {"status": "not_detected", "numbering_scheme": "IMGT", "message": message, "antibody_domains": []}

        residues_a, atoms_a = _chain_residues(model, partner_a_chains, partner_a_residue_ranges)
        residues_b, atoms_b = _chain_residues(model, partner_b_chains, partner_b_residue_ranges)
        interface_a, interface_b, _ = _interface_residues(atoms_a, atoms_b, cutoff_angstrom)
        interface_keys = {_residue_key(residue) for residue in interface_a | interface_b}
        antibody_domains = []
        mapping_warnings = []
        chains = {chain.id: chain for chain in model.get_chains()}
        all_partner_chains = list(dict.fromkeys([*partner_a_chains, *partner_b_chains]))
        for row in matches:
            for chain_type, chain_column, numbering_column, sequence_columns in (
                ("heavy", "Hchain", "VH_numbering_list", ("CDRH1", "CDRH2", "CDRH3")),
                ("light", "Lchain", "VL_numbering_list", ("CDRL1", "CDRL2", "CDRL3")),
            ):
                metadata_chain_id = str(row.get(chain_column, "")).strip()
                chain_id = next((actual for actual in chains if actual.casefold() == metadata_chain_id.casefold()), None)
                if not chain_id:
                    continue
                residues = [residue for residue in chains[chain_id].get_residues()
                            if _residue_letter(residue) and any(_is_heavy(atom) for atom in residue.get_atoms())]
                numbered = _parse_imgt_numbering(row.get(numbering_column, ""))
                mapping = _align_residues_to_imgt(residues, numbered)
                if not mapping:
                    mapping_warnings.append(f"Chain {chain_id}: supplied IMGT numbering could not be aligned to observed residues.")
                    continue
                prefix = "H" if chain_type == "heavy" else "L"
                cdr_regions = []
                for region_index, sequence_column in enumerate(sequence_columns, start=1):
                    region_name = f"CDR-{prefix}{region_index}"
                    expected_sequence = str(row.get(sequence_column, "")).strip().upper()
                    mapped_residues = []
                    for residue_index, residue in enumerate(residues):
                        if residue_index not in mapping:
                            continue
                        position, insertion = mapping[residue_index]
                        if not (27 <= position <= 38 or 56 <= position <= 65 or 105 <= position <= 117):
                            continue
                        expected_region = 1 if 27 <= position <= 38 else 2 if 56 <= position <= 65 else 3
                        if expected_region != region_index:
                            continue
                        residue_id = residue.id
                        mapped_residues.append({
                            "region": region_name,
                            "imgt_position": f"{prefix}{position}{insertion}",
                            "amino_acid": _residue_letter(residue),
                            "pdb_residue_number": int(residue_id[1]),
                            "pdb_insertion_code": str(residue_id[2]).strip(),
                            "pdb_residue_name": residue.resname,
                            "is_interface_residue": _residue_key(residue) in interface_keys,
                        })
                    cdr_regions.append({
                        "name": region_name,
                        "sequence": "".join(item["amino_acid"] for item in mapped_residues),
                        "residue_count": len(mapped_residues),
                        "interface_residue_count": sum(item["is_interface_residue"] for item in mapped_residues),
                        "residues": mapped_residues,
                        "reference_sequence": expected_sequence,
                    })
                partner = "Partner A" if chain_id in partner_a_chains else "Partner B" if chain_id in partner_b_chains else "Other chain"
                antibody_domains.append({"chain_id": chain_id, "partner": partner, "chain_type": chain_type,
                                         "domain_number": 1, "cdr_regions": cdr_regions})
            if antibody_domains:
                break

        if antibody_domains:
            status = "available"
            if used_sabdab_api:
                message = "CDR residues were mapped from SAbDab2 IMGT annotations. Counts include residues observed in the analyzed structure."
                if sabdab_structure_path:
                    message += " The IMGT-numbered mmCIF is ready to download."
                elif lookup_error:
                    message += f" {lookup_error}"
            else:
                message = "CDR residues were mapped using the supplied SAbDab IMGT numbering metadata. Counts include residues observed in the structure."
        else:
            status = "not_detected"
            message = "No matching heavy/light chain with usable IMGT numbering was present in this structure."
        result = {
            "status": status,
            "numbering_scheme": "IMGT",
            "message": message,
            "antibody_domains": antibody_domains,
            "mapping_warnings": mapping_warnings,
        }
        if sabdab_structure_path and sabdab_structure_path.is_file():
            result["structure_download_url"] = f"/api/sabdab-structure/{_identifier_core(structure_id)}"
        return result
    except Exception as exc:
        return {
            "status": "error",
            "numbering_scheme": "IMGT",
            "message": f"CDR analysis failed: {exc}",
            "antibody_domains": [],
        }


def _interface_residues(atoms_a: Sequence, atoms_b: Sequence, cutoff: float) -> tuple[set, set, int]:
    """Find residue pairs with any heavy-atom contact at the configured cutoff."""
    all_atoms = list(atoms_a) + list(atoms_b)
    pairs = NeighborSearch(all_atoms).search_all(cutoff, level="R")
    residues_a = {_residue_key(atom.get_parent()) for atom in atoms_a}
    residues_b = {_residue_key(atom.get_parent()) for atom in atoms_b}
    interface_a, interface_b = set(), set()
    contact_count = 0
    for first, second in pairs:
        first_key, second_key = _residue_key(first), _residue_key(second)
        if first_key in residues_a and second_key in residues_b:
            interface_a.add(first)
            interface_b.add(second)
            contact_count += 1
        elif second_key in residues_a and first_key in residues_b:
            interface_a.add(second)
            interface_b.add(first)
            contact_count += 1
    return interface_a, interface_b, contact_count


def find_interacting_chain_pairs(model, cutoff_angstrom: float = 5.0) -> list[dict]:
    """Find pairs of protein chains with at least one cross-chain heavy-atom contact."""
    if cutoff_angstrom <= 0 or not math.isfinite(cutoff_angstrom):
        raise ValueError("Distance threshold must be a finite positive number")

    protein_chains = []
    for chain in model.get_chains():
        try:
            _residues, atoms = _chain_residues(model, [chain.id])
        except ValueError:
            continue
        protein_chains.append((chain.id, atoms))

    pairs = []
    for index, (chain_a, atoms_a) in enumerate(protein_chains):
        for chain_b, atoms_b in protein_chains[index + 1:]:
            _interface_a, _interface_b, contact_count = _interface_residues(
                atoms_a, atoms_b, cutoff_angstrom
            )
            if contact_count:
                pairs.append({
                    "chain_a": chain_a,
                    "chain_b": chain_b,
                    "cross_partner_residue_contacts": contact_count,
                })
    return pairs


def _intrachain_counts(residues: Iterable[Residue], cutoff: float) -> tuple[np.ndarray, int, int]:
    """Count side-chain heavy-atom contacts between distinct interface residues.

    Residue pairs are unordered. Same-amino-acid pairs are counted as observed but
    excluded from the matrix because the required diagonal is zero. Counts in the
    matrix therefore normalize over represented, off-diagonal amino-acid pairs.
    """
    selected = list(residues)
    counts = np.zeros((20, 20), dtype=np.float64)
    sidechain_atoms = [
        atom for residue in selected for atom in residue.get_atoms()
        if _is_heavy(atom) and atom.get_name().strip().upper() not in BACKBONE_ATOMS
    ]
    if len(sidechain_atoms) < 2:
        return counts, 0, 0
    residue_pairs = NeighborSearch(sidechain_atoms).search_all(cutoff, level="R")
    observed = 0
    homotypic = 0
    seen: set[tuple[tuple[str, tuple], tuple[str, tuple]]] = set()
    selected_keys = {_residue_key(residue) for residue in selected}
    for first, second in residue_pairs:
        if first is second:
            continue
        first_key, second_key = _residue_key(first), _residue_key(second)
        if first_key not in selected_keys or second_key not in selected_keys:
            continue
        pair_key = tuple(sorted((first_key, second_key), key=repr))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        first_index = AA_TO_INDEX[_residue_letter(first)]
        second_index = AA_TO_INDEX[_residue_letter(second)]
        observed += 1
        if first_index == second_index:
            homotypic += 1
            continue
        low, high = sorted((first_index, second_index))
        counts[low, high] += 1
    return counts, observed, homotypic


def validate_matrix(matrix: np.ndarray) -> None:
    """Reject malformed, non-finite, negative, or incorrectly normalized features."""
    if matrix.shape != (20, 20):
        raise ValueError(f"Matrix shape must be (20, 20), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Matrix contains non-finite values")
    if np.any(matrix < 0):
        raise ValueError("Matrix contains negative values")
    if not np.allclose(np.diag(matrix), 0.0):
        raise ValueError("Matrix diagonal must be zero")
    if not math.isclose(float(np.tril(matrix, -1).sum()), 1.0, abs_tol=1e-6):
        raise ValueError("Antibody/general-partner lower triangle must sum to 1")
    if not math.isclose(float(np.triu(matrix, 1).sum()), 1.0, abs_tol=1e-6):
        raise ValueError("Antigen/general-partner upper triangle must sum to 1")


def build_interaction_matrix(
    source: str | Path | BinaryIO,
    partner_a_chains: Sequence[str],
    partner_b_chains: Sequence[str],
    *,
    cutoff_angstrom: float = 5.0,
    filename: str | None = None,
    sample_id: str = "unknown",
    partner_a_residue_ranges: Mapping[str, tuple] | None = None,
    partner_b_residue_ranges: Mapping[str, tuple] | None = None,
) -> tuple[np.ndarray, dict]:
    """Validate a structure and create partner-specific normalized contact features.

    First, interface residues are defined by cross-partner heavy-atom contacts.
    Then, side-chain contacts between distinct interface residues are counted
    within each selected partner entity. Partner A is stored in the lower
    triangle (row amino acid > column amino acid); partner B is stored in the
    upper triangle. Diagonal cells remain zero. Each triangle is normalized
    independently to sum to one.
    """
    if cutoff_angstrom <= 0 or not math.isfinite(cutoff_angstrom):
        raise ValueError("Distance threshold must be a finite positive number")
    model = source if hasattr(source, "get_chains") else parse_structure(source, filename=filename)
    available_chains = [chain.id for chain in model.get_chains()]

    def resolve_ids(chain_ids: Sequence[str]) -> list[str]:
        actual_ids = []
        for chain_id in chain_ids:
            requested_id = str(chain_id).strip()
            if requested_id in available_chains:
                actual_id = requested_id
            else:
                matches = [value for value in available_chains if value.casefold() == requested_id.casefold()]
                actual_id = matches[0] if len(matches) == 1 else requested_id
            if actual_id not in actual_ids:
                actual_ids.append(actual_id)
        return actual_ids

    actual_a_chains = resolve_ids(partner_a_chains)
    actual_b_chains = resolve_ids(partner_b_chains)
    actual_a_ranges = {
        next((value for value in available_chains if value.casefold() == key.casefold()), key): value
        for key, value in (partner_a_residue_ranges or {}).items()
    }
    actual_b_ranges = {
        next((value for value in available_chains if value.casefold() == key.casefold()), key): value
        for key, value in (partner_b_residue_ranges or {}).items()
    }
    residues_a, atoms_a = _chain_residues(model, actual_a_chains, actual_a_ranges)
    residues_b, atoms_b = _chain_residues(model, actual_b_chains, actual_b_ranges)
    if {_residue_key(r) for r in residues_a} & {_residue_key(r) for r in residues_b}:
        raise ValueError("Selected partner residue ranges overlap; interaction roles are ambiguous")
    interface_a, interface_b, cross_contacts = _interface_residues(atoms_a, atoms_b, cutoff_angstrom)
    if cross_contacts == 0:
        raise ValueError(f"No cross-partner heavy-atom contacts within {cutoff_angstrom:g} Å")
    if len(interface_a) < 2 or len(interface_b) < 2:
        raise ValueError("Fewer than two interface residues on one or both partners")

    counts_a, observed_a, homotypic_a = _intrachain_counts(interface_a, cutoff_angstrom)
    counts_b, observed_b, homotypic_b = _intrachain_counts(interface_b, cutoff_angstrom)
    total_a, total_b = counts_a.sum(), counts_b.sum()
    if total_a <= 0 or total_b <= 0:
        raise ValueError("One or both partners have no representable off-diagonal interface residue contacts")

    matrix = np.zeros((20, 20), dtype=np.float64)
    matrix[np.tril_indices(20, -1)] = (counts_a.T / total_a)[np.tril_indices(20, -1)]
    matrix[np.triu_indices(20, 1)] = (counts_b / total_b)[np.triu_indices(20, 1)]
    validate_matrix(matrix)
    metadata = {
        "sample_id": sample_id,
        "cutoff_angstrom": cutoff_angstrom,
        "amino_acid_order": AA_ORDER,
        "partner_a_chains": actual_a_chains,
        "partner_b_chains": actual_b_chains,
        "partner_a_residue_ranges": {k: list(v) for k, v in actual_a_ranges.items()},
        "partner_b_residue_ranges": {k: list(v) for k, v in actual_b_ranges.items()},
        "partner_a_interface_residues": len(interface_a),
        "partner_b_interface_residues": len(interface_b),
        "cross_partner_residue_contacts": cross_contacts,
        "partner_a_observed_sidechain_contacts": observed_a,
        "partner_b_observed_sidechain_contacts": observed_b,
        "partner_a_homotypic_contacts_excluded_by_zero_diagonal": homotypic_a,
        "partner_b_homotypic_contacts_excluded_by_zero_diagonal": homotypic_b,
        "normalization": "Each partner's off-diagonal amino-acid contact counts divided by that partner's represented off-diagonal total; each triangle sums to 1.",
        "feature_definition": "Cross-partner contacts select interface residues; features count side-chain heavy-atom intrapartner contacts among those interface residues.",
    }
    return matrix.astype(np.float32), metadata