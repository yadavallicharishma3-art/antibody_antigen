"""Parse protein structures and build validated 20 x 20 interface features."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping, Sequence

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, NeighborSearch
from Bio.PDB.Residue import Residue
from Bio.SeqUtils import seq1

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AA_ORDER)}
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


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
