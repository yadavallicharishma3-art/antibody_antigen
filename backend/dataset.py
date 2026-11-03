"""Load the supplied archives, validate every record, and build data splits."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .structure_processing import AA_ORDER, build_interaction_matrix, parse_structure
except ImportError:  # Also support running this file directly during development.
    from structure_processing import AA_ORDER, build_interaction_matrix, parse_structure

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PPI_LIST = Path(r"C:\Users\yadavalli charishma\ABAG\pdb_4960list.txt")
DEFAULT_PPI_ZIP = Path(r"C:\Users\yadavalli charishma\ABAG\ppi_structures.zip")
DEFAULT_ABAG_ZIP = Path(r"C:\Users\yadavalli charishma\Downloads\splits\splits_final.zip")
OUTPUT_DIR = ROOT / "data" / "processed"
CLASS_NAMES = {0: "General Protein-Protein Interaction", 1: "Antibody-Antigen Interaction"}
_WORKER_PPI_ARCHIVE = None
_WORKER_ABAG_ARCHIVE = None
_WORKER_PPI_INDEX = None
_WORKER_ABAG_INDEX = None
_WORKER_CUTOFF = 5.0


def _chain_list(value: str | None) -> list[str]:
    """Split a metadata chain list while rejecting placeholder values."""
    if not value:
        return []
    value = value.strip()
    if value in {"/", "-", "NA", "NaN", "nan", "None"}:
        return []
    return [part.strip() for part in re.split(r"[/,;\s]+", value) if part.strip()]


def _parse_range_record(tokens: list[str]) -> tuple[str, dict, dict]:
    """Read chain IDs from the PPI list, including optional insertion codes.

    The normal record contains eight tokens. A few records include insertion
    codes after residue numbers, so token positions are parsed by numeric fields
    instead of assuming a fixed column count.
    """
    if len(tokens) < 8:
        raise ValueError(f"Expected at least 8 whitespace-separated fields, got {len(tokens)}")
    structure_id = tokens[1].lower()
    # Most rows use the plain eight-column format. For the five rows with
    # insertion codes, a bounded backtracking parser distinguishes chain IDs
    # from optional insertion-code fields without guessing a fixed position.
    def parse_two(cursor: int, chosen: list[dict]) -> list[dict] | None:
        if len(chosen) == 2:
            return chosen if cursor == len(tokens) else None
        if cursor >= len(tokens):
            return None
        chain_id = tokens[cursor]
        start = cursor + 1
        if start >= len(tokens) or not re.fullmatch(r"-?\d+", tokens[start]):
            return None
        start_options = [(int(tokens[start]), "", start + 1)]
        if start + 1 < len(tokens) and not re.fullmatch(r"-?\d+", tokens[start + 1]):
            start_options.append((int(tokens[start]), tokens[start + 1], start + 2))
        candidates = []
        for start_num, start_ins, end_index in start_options:
            if end_index >= len(tokens) or not re.fullmatch(r"-?\d+", tokens[end_index]):
                continue
            end_num = int(tokens[end_index])
            end_options = [("", end_index + 1)]
            if end_index + 1 < len(tokens) and not re.fullmatch(r"-?\d+", tokens[end_index + 1]):
                end_options.append((tokens[end_index + 1], end_index + 2))
            for end_ins, next_cursor in end_options:
                if start_num <= end_num:
                    result = parse_two(next_cursor, chosen + [{"chain": chain_id, "range": ((start_num, start_ins), (end_num, end_ins))}])
                    if result:
                        candidates.append(result)
        return candidates[0] if len(candidates) == 1 else None

    parsed = parse_two(2, [])
    if not parsed:
        raise ValueError(f"Could not unambiguously parse partner chains and residue ranges: {' '.join(tokens)}")
    return structure_id, parsed[0], parsed[1]


def _structure_index(archive: zipfile.ZipFile) -> dict[str, list[str]]:
    """Index PDB/mmCIF entries by their basename without extracting files."""
    index: dict[str, list[str]] = defaultdict(list)
    for name in archive.namelist():
        if name.lower().endswith((".cif", ".mmcif", ".pdb", ".ent")):
            index[Path(name).stem.lower()].append(name)
    return index


def _load_positive_records(archive: zipfile.ZipFile, index: dict[str, list[str]]) -> Iterator[dict]:
    """Yield antibody-antigen records from the supplied AbAg split table."""
    with archive.open("splits_final/abag_split.csv") as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        required = {"INSTANCE", "PDB_ID", "Hchain", "Lchain", "agchains", "ab_ag_split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"abag_split.csv is missing required columns: {sorted(missing)}")
        for row in reader:
            instance = (row.get("INSTANCE") or "").strip()
            yield {
                "sample_id": instance,
                "label": 1,
                "source": "splits_final.zip:abag_split.csv",
                "structure_id": (row.get("PDB_ID") or instance.split("_")[1] if "_" in instance else "").lower(),
                "structure_key": instance.lower(),
                "group_id": (row.get("PDB_ID") or instance).lower(),
                "partner_a_chains": [x for x in [row.get("Hchain", "").strip(), row.get("Lchain", "").strip()] if x],
                "partner_b_chains": _chain_list(row.get("agchains")),
                "partner_a_ranges": {},
                "partner_b_ranges": {},
                "source_split": row.get("ab_ag_split", "").strip().lower(),
            }


def _load_negative_records(list_path: Path) -> Iterator[dict]:
    """Yield general PPI records; their label follows the supplied control-list provenance."""
    with list_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            tokens = line.split()
            try:
                structure_id, first, second = _parse_range_record(tokens)
                range_a, range_b = {first["chain"]: first["range"]}, {second["chain"]: second["range"]}
                record = {
                    "sample_id": f"ppi_{tokens[0]}_{structure_id}",
                    "label": 0,
                    "source": "pdb_4960list.txt (general PPI control list)",
                    "structure_id": structure_id,
                    "structure_key": structure_id,
                    "group_id": structure_id,
                    "partner_a_chains": [first["chain"]],
                    "partner_b_chains": [second["chain"]],
                    "partner_a_ranges": range_a,
                    "partner_b_ranges": range_b,
                    "source_split": "",
                    "source_line": line_number,
                }
            except Exception as exc:
                record = {
                    "sample_id": f"ppi_line_{line_number}",
                    "label": 0,
                    "source": "pdb_4960list.txt (general PPI control list)",
                    "structure_id": tokens[1].lower() if len(tokens) > 1 else "",
                    "structure_key": tokens[1].lower() if len(tokens) > 1 else "",
                    "group_id": tokens[1].lower() if len(tokens) > 1 else f"line_{line_number}",
                    "partner_a_chains": [],
                    "partner_b_chains": [],
                    "partner_a_ranges": {},
                    "partner_b_ranges": {},
                    "source_split": "",
                    "source_line": line_number,
                    "pre_error": str(exc),
                }
            yield record


def inspect_inputs(ppi_list: Path, ppi_zip: Path, abag_zip: Path) -> dict:
    """Inspect record counts and structure matches before processing features."""
    with zipfile.ZipFile(ppi_zip) as ppi_archive, zipfile.ZipFile(abag_zip) as abag_archive:
        ppi_index, abag_index = _structure_index(ppi_archive), _structure_index(abag_archive)
        negatives = list(_load_negative_records(ppi_list))
        positive_count = 0
        source_splits = Counter()
        positive_missing = []
        positive_multiple = []
        with abag_archive.open("splits_final/abag_split.csv") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            for row in reader:
                positive_count += 1
                key = (row.get("INSTANCE") or "").strip().lower()
                source_splits[(row.get("ab_ag_split") or "").lower()] += 1
                matches = abag_index.get(key, [])
                if not matches:
                    positive_missing.append(key)
                elif len(matches) > 1:
                    positive_multiple.append(key)
        negative_missing = [x["structure_id"] for x in negatives if not ppi_index.get(x["structure_key"])]
        negative_multiple = [x["structure_id"] for x in negatives if len(ppi_index.get(x["structure_key"], [])) > 1]
        return {
            "positive_records": positive_count,
            "positive_label": CLASS_NAMES[1],
            "positive_source_splits": dict(source_splits),
            "positive_structure_files": sum(1 for x in abag_index.values() for n in x if n.lower().endswith((".cif", ".mmcif", ".pdb", ".ent"))),
            "positive_unmatched_records": len(positive_missing),
            "positive_ambiguous_matches": len(positive_multiple),
            "positive_unmatched_examples": positive_missing[:20],
            "negative_records": len(negatives),
            "negative_label": CLASS_NAMES[0],
            "negative_structure_files": sum(1 for x in ppi_index.values() for n in x),
            "negative_unmatched_records": len(negative_missing),
            "negative_ambiguous_matches": len(negative_multiple),
            "negative_unmatched_examples": negative_missing[:20],
            "negative_metadata_parse_errors": sum(bool(x.get("pre_error")) for x in negatives),
            "label_basis": "Positive records come from abag_split.csv. Negative records come from the supplied pdb_4960list.txt control list, as identified by its provenance and the accompanying study description.",
            "note": "The source split column is preserved as metadata; final splits are regenerated by structure group to prevent the observed PDB_ID train/test overlap.",
        }


def _assign_group_splits(labels: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Create grouped stratified train/validation/test labels with no group overlap."""
    if len(set(labels.tolist())) != 2:
        raise ValueError("Both classes must be present after structure validation to create splits")
    group_counts = {label: len(set(groups[labels == label].tolist())) for label in set(labels.tolist())}
    folds = min(5, min(group_counts.values()))
    if folds < 3:
        raise ValueError(f"At least 3 independent structure groups per class are needed for grouped splitting; found {group_counts}")
    first = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    split_labels = np.full(len(labels), "train", dtype="U10")
    train_val_indices, test_indices = next(first.split(np.zeros(len(labels)), labels, groups))
    split_labels[test_indices] = "test"
    remaining_labels, remaining_groups = labels[train_val_indices], groups[train_val_indices]
    inner_group_counts = {label: len(set(remaining_groups[remaining_labels == label].tolist())) for label in set(labels.tolist())}
    inner_folds = min(5, min(inner_group_counts.values()))
    if inner_folds < 2:
        raise ValueError(f"Not enough training structure groups for a validation split: {inner_group_counts}")
    second = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=seed + 1)
    _, validation_local = next(second.split(np.zeros(len(train_val_indices)), remaining_labels, remaining_groups))
    split_labels[train_val_indices[validation_local]] = "validation"
    for name in ("train", "validation", "test"):
        selected = split_labels == name
        if len(np.unique(labels[selected])) < 2:
            raise ValueError(f"The grouped {name} split does not contain both classes")
    for first_name, second_name in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(groups[split_labels == first_name]) & set(groups[split_labels == second_name]):
            raise ValueError(f"Structure-group leakage detected between {first_name} and {second_name}")
    return split_labels


def _source_fingerprint(paths: list[Path], cutoff_angstrom: float, seed: int, batch_size: int) -> str:
    """Return a checkpoint key that changes when inputs or processing settings change."""
    source_state = []
    for path in paths:
        stat = path.stat()
        source_state.append({"path": str(path.resolve()), "size": stat.st_size, "modified_ns": stat.st_mtime_ns})
    payload = json.dumps({"sources": source_state, "cutoff": cutoff_angstrom, "seed": seed,
                          "batch_size": batch_size, "schema": 2}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _process_one_record(record: dict, ppi_archive, abag_archive, ppi_index: dict, abag_index: dict,
                        cutoff_angstrom: float) -> tuple[dict, np.ndarray | None]:
    """Validate one record and return its audit row plus a matrix only if accepted."""
    status, reason, matrix, feature_meta = "rejected", "", None, {}
    archive = abag_archive if record["label"] == 1 else ppi_archive
    index = abag_index if record["label"] == 1 else ppi_index
    matches = index.get(record["structure_key"], [])
    matched = len(matches) == 1
    parsed = False
    if record.get("pre_error"):
        reason = record["pre_error"]
    elif not matches:
        reason = f"No structure file matched identifier {record['structure_key']}"
    elif len(matches) > 1:
        reason = f"Ambiguous structure identifier maps to {len(matches)} files"
    elif not record["partner_a_chains"] or not record["partner_b_chains"]:
        reason = "Required partner chain identifiers are missing or ambiguous"
    else:
        try:
            member = matches[0]
            with archive.open(member) as stream:
                model = parse_structure(stream, filename=member)
                parsed = True
                matrix, feature_meta = build_interaction_matrix(
                    model, record["partner_a_chains"], record["partner_b_chains"],
                    cutoff_angstrom=cutoff_angstrom, filename=member, sample_id=record["sample_id"],
                    partner_a_residue_ranges=record.get("partner_a_ranges"),
                    partner_b_residue_ranges=record.get("partner_b_ranges"),
                )
            status = "accepted"
        except Exception as exc:
            reason = str(exc)
    row = {
        "sample_id": record["sample_id"], "structure_id": record["structure_id"],
        "structure_file": matches[0] if matched else "", "structure_matched": matched,
        "structure_parsed": parsed, "source": record["source"], "label": record["label"],
        "class_name": CLASS_NAMES[record["label"]],
        "partner_a_chains": ",".join(record["partner_a_chains"]),
        "partner_b_chains": ",".join(record["partner_b_chains"]),
        "partner_a_residue_ranges": json.dumps(record.get("partner_a_ranges", {})),
        "partner_b_residue_ranges": json.dumps(record.get("partner_b_ranges", {})),
        "source_split": record["source_split"], "validation_status": status,
        "failure_reason": reason, "feature_metadata": json.dumps(feature_meta, ensure_ascii=False),
        "dataset_split": "",
    }
    return row, matrix


def _initialize_worker(ppi_zip: str, abag_zip: str, cutoff_angstrom: float) -> None:
    """Open read-only archives once in each worker process."""
    global _WORKER_PPI_ARCHIVE, _WORKER_ABAG_ARCHIVE, _WORKER_PPI_INDEX, _WORKER_ABAG_INDEX, _WORKER_CUTOFF
    _WORKER_PPI_ARCHIVE = zipfile.ZipFile(ppi_zip)
    _WORKER_ABAG_ARCHIVE = zipfile.ZipFile(abag_zip)
    _WORKER_PPI_INDEX = _structure_index(_WORKER_PPI_ARCHIVE)
    _WORKER_ABAG_INDEX = _structure_index(_WORKER_ABAG_ARCHIVE)
    _WORKER_CUTOFF = cutoff_angstrom


def _process_record_in_worker(record: dict) -> tuple[dict, np.ndarray | None]:
    return _process_one_record(record, _WORKER_PPI_ARCHIVE, _WORKER_ABAG_ARCHIVE,
                               _WORKER_PPI_INDEX, _WORKER_ABAG_INDEX, _WORKER_CUTOFF)


def process_dataset(
    ppi_list: Path = DEFAULT_PPI_LIST,
    ppi_zip: Path = DEFAULT_PPI_ZIP,
    abag_zip: Path = DEFAULT_ABAG_ZIP,
    output_dir: Path = OUTPUT_DIR,
    cutoff_angstrom: float = 5.0,
    seed: int = 42,
    batch_size: int = 100,
    fresh: bool = False,
    workers: int = 4,
) -> dict:
    """Process every source record using resumable, atomic batch checkpoints."""
    if batch_size < 1:
        raise ValueError("Checkpoint batch size must be at least 1")
    if workers < 1:
        raise ValueError("Worker count must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_key = _source_fingerprint([ppi_list, ppi_zip, abag_zip], cutoff_angstrom, seed, batch_size)
    checkpoint_dir = checkpoint_root / checkpoint_key
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_info = {
        "fingerprint": checkpoint_key, "sources": [str(p.resolve()) for p in (ppi_list, ppi_zip, abag_zip)],
        "cutoff_angstrom": cutoff_angstrom, "seed": seed, "batch_size": batch_size,
        "note": "Each completed batch is stored atomically and can be reused after an interruption.",
    }
    (checkpoint_dir / "checkpoint_info.json").write_text(json.dumps(checkpoint_info, indent=2), encoding="utf-8")
    chunk_arrays: list[np.ndarray] = []
    chunk_labels: list[np.ndarray] = []
    chunk_groups: list[np.ndarray] = []
    chunk_ids: list[np.ndarray] = []
    manifest: list[dict] = []
    with zipfile.ZipFile(ppi_zip) as ppi_archive, zipfile.ZipFile(abag_zip) as abag_archive:
        ppi_index, abag_index = _structure_index(ppi_archive), _structure_index(abag_archive)
        records = list(_load_positive_records(abag_archive, abag_index)) + list(_load_negative_records(ppi_list))
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker,
                                 initargs=(str(ppi_zip), str(abag_zip), cutoff_angstrom)) as pool:
            for start in range(0, len(records), batch_size):
                end = min(start + batch_size, len(records))
                checkpoint_path = checkpoint_dir / f"batch_{start:06d}_{end:06d}.npz"
                cached = checkpoint_path.exists() and not fresh
                if cached:
                    try:
                        cached_data = np.load(checkpoint_path, allow_pickle=False)
                        rows = [json.loads(value) for value in cached_data["manifest_json"].tolist()]
                        # Recheck batches rejected for missing IDs because the
                        # structure validator now permits unique case-only matches.
                        if any(str(row.get("failure_reason", "")).startswith("Required chain(s) not present:") for row in rows):
                            cached_data.close()
                            cached_data = None
                            cached = False
                        else:
                            accepted_rows = [row for row in rows if row.get("validation_status") == "accepted"]
                            cached_x = cached_data["X"]
                            expected_ids = [row["sample_id"] for row in accepted_rows]
                            expected_labels = [row["label"] for row in accepted_rows]
                            if (cached_x.shape != (len(accepted_rows), 20, 20)
                                    or cached_data["sample_ids"].tolist() != expected_ids
                                    or cached_data["y"].tolist() != expected_labels):
                                print(f"Checkpoint {checkpoint_path.name} does not match its accepted manifest rows; recomputing.", flush=True)
                                cached_data.close()
                                cached_data = None
                                cached = False
                    except Exception as exc:
                        print(f"Checkpoint {checkpoint_path.name} is unreadable and will be recomputed: {exc}", flush=True)
                        try:
                            cached_data.close()
                        except Exception:
                            pass
                        cached = False
                if cached:
                    try:
                        chunk_X = cached_data["X"]
                        chunk_y = cached_data["y"]
                        chunk_group = cached_data["groups"]
                        chunk_id = cached_data["sample_ids"]
                    except Exception as exc:
                        print(f"Checkpoint {checkpoint_path.name} is unreadable and will be recomputed: {exc}", flush=True)
                        cached_data.close()
                        cached_data = None
                        cached = False
                if not cached:
                    # Materialize the map once so manifest rows and matrices
                    # are both collected from the same worker results.
                    batch_results = list(pool.map(_process_record_in_worker, records[start:end], chunksize=1))
                    rows = [row for row, _ in batch_results]
                    accepted = [(row, matrix) for row, matrix in batch_results if matrix is not None]
                    accepted_status_count = sum(row.get("validation_status") == "accepted" for row in rows)
                    if accepted_status_count != len(accepted):
                        examples = [(row.get("sample_id"), row.get("validation_status"), matrix is not None)
                                    for row, matrix in batch_results if row.get("validation_status") == "accepted"][:5]
                        raise RuntimeError(
                            f"Worker results disagree in batch {start}:{end}: "
                            f"{accepted_status_count} accepted rows but {len(accepted)} matrices; examples={examples}"
                        )
                    chunk_X = np.stack([matrix for _, matrix in accepted]).astype(np.float32) if accepted else np.empty((0, 20, 20), dtype=np.float32)
                    chunk_y = np.asarray([row["label"] for row, _ in accepted], dtype=np.int64)
                    chunk_group = np.asarray([row["structure_id"].lower() for row, _ in accepted], dtype=str)
                    chunk_id = np.asarray([row["sample_id"] for row, _ in accepted], dtype=str)
                    temporary = checkpoint_path.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, X=chunk_X, y=chunk_y, groups=chunk_group, sample_ids=chunk_id,
                                            manifest_json=np.asarray([json.dumps(row, ensure_ascii=False) for row in rows], dtype=str))
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, checkpoint_path)
                    cached_data = None
                chunk_arrays.append(np.asarray(chunk_X))
                chunk_labels.append(np.asarray(chunk_y))
                chunk_groups.append(np.asarray(chunk_group))
                chunk_ids.append(np.asarray(chunk_id))
                manifest.extend(rows)
                if cached_data is not None:
                    cached_data.close()
                accepted_so_far = sum(array.shape[0] for array in chunk_arrays)
                print(f"Checkpointed {end}/{len(records)} records; accepted so far={accepted_so_far}; reused={cached}; workers={workers}", flush=True)

    matrix_array = np.concatenate(chunk_arrays, axis=0) if chunk_arrays else np.empty((0, 20, 20), dtype=np.float32)
    labels_array = np.concatenate(chunk_labels) if chunk_labels else np.asarray([], dtype=np.int64)
    groups_array = np.concatenate(chunk_groups) if chunk_groups else np.asarray([], dtype=str)
    sample_ids = np.concatenate(chunk_ids).tolist() if chunk_ids else []
    matched_count = Counter(row["label"] for row in manifest if row["structure_matched"])
    parsed_count = sum(bool(row["structure_parsed"]) for row in manifest)
    repeated_structure_records = Counter(row["structure_id"] for row in manifest if row["validation_status"] == "accepted")
    split_limitation = ""
    if len(labels_array):
        try:
            split_array = _assign_group_splits(labels_array, groups_array, seed)
        except ValueError as exc:
            split_array = np.full(len(labels_array), "unassigned", dtype="U12")
            split_limitation = str(exc)
    else:
        split_array = np.asarray([], dtype="U12")
        split_limitation = "No structures passed validation, so train/validation/test splitting is unavailable."
    split_names = split_array.astype(str).tolist()
    accepted_by_id = {sample_id: split for sample_id, split in zip(sample_ids, split_names)}
    for row in manifest:
        row["dataset_split"] = accepted_by_id.get(row["sample_id"], "")

    np.savez_compressed(
        output_dir / "dataset_matrices.npz", X=matrix_array, y=labels_array,
        groups=groups_array, sample_ids=np.asarray(sample_ids), splits=np.asarray(split_names),
    )
    manifest_path = output_dir / "manifest.csv"
    # Ensure dataset_split is present for both accepted and rejected records.
    for row in manifest:
        row.setdefault("dataset_split", "")
    fieldnames = list(manifest[0].keys())
    pd.DataFrame(manifest, columns=fieldnames).to_csv(manifest_path, index=False)

    status_counts = Counter(row["validation_status"] for row in manifest)
    label_counts = Counter(row["label"] for row in manifest)
    rejection_reasons = Counter(row["failure_reason"] for row in manifest if row["validation_status"] == "rejected")
    record_id_counts = Counter(row["sample_id"] for row in manifest)
    accepted_structure_counts = Counter(row["structure_id"] for row in manifest if row["validation_status"] == "accepted")
    split_report = {}
    for split_name in ("train", "validation", "test"):
        mask = split_array == split_name
        split_report[split_name] = {
            "total": int(mask.sum()),
            "negative": int(np.sum(labels_array[mask] == 0)),
            "positive": int(np.sum(labels_array[mask] == 1)),
        }
    summary = {
        "total_records": len(manifest),
        "matched_to_structure_file": int(sum(matched_count.values())),
        "matched_positive": int(matched_count[1]),
        "matched_negative": int(matched_count[0]),
        "original_negative_records": int(label_counts[0]),
        "original_positive_records": int(label_counts[1]),
        "structures_parsed_and_accepted": int(status_counts["accepted"]),
        "structures_successfully_parsed": int(parsed_count),
        "structures_rejected": int(status_counts["rejected"]),
        "valid_matrices_generated": int(status_counts["accepted"]),
        "rejection_reasons": dict(rejection_reasons.most_common()),
        "repeated_structure_ids_in_accepted_records": sum(count > 1 for count in repeated_structure_records.values()),
        "duplicate_record_ids": sum(count - 1 for count in record_id_counts.values() if count > 1),
        "extra_accepted_records_sharing_a_structure_id": sum(count - 1 for count in accepted_structure_counts.values() if count > 1),
        "accepted_samples_per_split": split_report,
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "cutoff_angstrom": cutoff_angstrom,
        "seed": seed,
        "split_method": "StratifiedGroupKFold on structure/PDB identifier groups; no group crosses train, validation, or test.",
        "source_split_handling": "Source split values are recorded, but final splits are regenerated because 111 PDB_ID values appear in both supplied train and test partitions.",
        "split_limitation": split_limitation or None,
        "feature_definition": "Cross-partner heavy-atom contacts define interface residues; side-chain heavy-atom contacts among interface residues are counted within each partner. Partner B occupies upper triangle; partner A occupies lower triangle; diagonal is zero; each triangle is normalized independently.",
        "normalization_note": "Same-amino-acid contact pairs are tracked but omitted because the requested diagonal is zero; each triangle denominator includes only represented off-diagonal pairs.",
        "manifest_path": str(manifest_path),
        "matrices_path": str(output_dir / "dataset_matrices.npz"),
    }
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", action="store_true", help="Inspect counts and matches without processing structures")
    parser.add_argument("--ppi-list", type=Path, default=Path(os.getenv("PPI_LIST_PATH", str(DEFAULT_PPI_LIST))))
    parser.add_argument("--ppi-zip", type=Path, default=Path(os.getenv("PPI_STRUCTURES_ZIP", str(DEFAULT_PPI_ZIP))))
    parser.add_argument("--abag-zip", type=Path, default=Path(os.getenv("SPLITS_FINAL_ZIP", str(DEFAULT_ABAG_ZIP))))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=100, help="Records committed per resumable checkpoint")
    parser.add_argument("--workers", type=int, default=4, help="Parallel structure-validation processes")
    parser.add_argument("--fresh", action="store_true", help="Recompute all batches instead of reusing checkpoints")
    args = parser.parse_args()
    if args.inspect:
        print(json.dumps(inspect_inputs(args.ppi_list, args.ppi_zip, args.abag_zip), indent=2))
    else:
        print(json.dumps(process_dataset(args.ppi_list, args.ppi_zip, args.abag_zip, args.output_dir,
                                         args.cutoff, args.seed, args.batch_size, args.fresh, args.workers), indent=2))


if __name__ == "__main__":
    main()
