"""Data loading, representation validation, leakage auditing, and diversity analysis."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


@dataclass
class DatasetBundle:
    """Encapsulates loaded tensors, labels, and metadata for a split."""
    X: np.ndarray  # Shape: (N, 20, 21, 1)
    y: np.ndarray  # Shape: (N,)
    sample_ids: List[str]
    pdb_ids: List[str]
    metadata: List[Dict[str, Any]]
    split_name: str

    def __len__(self) -> int:
        return len(self.y)

    @property
    def positive_count(self) -> int:
        return int(np.sum(self.y == 1))

    @property
    def negative_count(self) -> int:
        return int(np.sum(self.y == 0))

    @property
    def class_ratio(self) -> Dict[int, float]:
        total = len(self.y)
        if total == 0:
            return {0: 0.0, 1: 0.0}
        return {
            0: float(self.negative_count / total),
            1: float(self.positive_count / total),
        }

    @property
    def unique_pdbs(self) -> Set[str]:
        return set(self.pdb_ids)


def load_dataset_split(
    csv_path: Path,
    base_dir: Optional[Path] = None,
    split_name: Optional[str] = None,
) -> DatasetBundle:
    """Load a dataset split from CSV and verify all representations on disk.

    Args:
        csv_path: Path to train.csv, validation.csv, or test.csv.
        base_dir: Base directory for resolving relative representation paths.
        split_name: Optional name tag ('train', 'validation', 'test').

    Returns:
        DatasetBundle containing X (N, 20, 21, 1), y (N,), and metadata.
    """
    base = base_dir or BASE_DIR
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        return DatasetBundle(
            X=np.empty((0, 20, 21, 1), dtype=np.float32),
            y=np.empty((0,), dtype=np.int32),
            sample_ids=[],
            pdb_ids=[],
            metadata=[],
            split_name=split_name or csv_path.stem,
        )

    tensors = []
    labels = []
    sample_ids = []
    pdb_ids = []
    metadata = []

    for _, row in df.iterrows():
        sample_id = str(row["sample_id"]).strip()
        pdb_id = str(row["pdb_id"]).strip().upper()
        rel_rep_path = str(row["representation_path"]).strip()
        label_val = int(row["label"])

        # Resolve path
        full_rep_path = Path(rel_rep_path)
        if not full_rep_path.is_absolute():
            full_rep_path = base / rel_rep_path

        if not full_rep_path.exists():
            raise FileNotFoundError(
                f"Representation file missing for sample {sample_id}: {full_rep_path}"
            )

        tensor_2d = np.load(full_rep_path)

        # Validate representation invariants
        if tensor_2d.shape != (20, 21):
            raise ValueError(
                f"Invalid tensor shape for {sample_id}: got {tensor_2d.shape}, expected (20, 21)"
            )

        if np.isnan(tensor_2d).any() or np.isinf(tensor_2d).any():
            raise ValueError(
                f"Representation for {sample_id} contains NaN or Inf values"
            )

        t_min = float(np.min(tensor_2d))
        t_max = float(np.max(tensor_2d))
        if t_min < 0.0 or t_max > 1.0001:
            raise ValueError(
                f"Representation values for {sample_id} outside [0, 1] range: [{t_min}, {t_max}]"
            )

        # Reshape to (20, 21, 1)
        tensor_3d = np.expand_dims(tensor_2d.astype(np.float32), axis=-1)

        tensors.append(tensor_3d)
        labels.append(label_val)
        sample_ids.append(sample_id)
        pdb_ids.append(pdb_id)
        metadata.append(row.to_dict())

    X_arr = np.array(tensors, dtype=np.float32)
    y_arr = np.array(labels, dtype=np.int32)

    return DatasetBundle(
        X=X_arr,
        y=y_arr,
        sample_ids=sample_ids,
        pdb_ids=pdb_ids,
        metadata=metadata,
        split_name=split_name or csv_path.stem,
    )


def audit_splits_leakage(
    train_bundle: DatasetBundle,
    val_bundle: DatasetBundle,
    test_bundle: DatasetBundle,
) -> Dict[str, Any]:
    """Verify zero structure-level or cluster-level leakage across splits.

    Target:
        train ∩ val = 0
        train ∩ test = 0
        val ∩ test = 0
    """
    def _get_set(bundle: DatasetBundle, key: str) -> Set[str]:
        return {str(m.get(key, "")).strip() for m in bundle.metadata if m.get(key) not in [None, "", "nan"]}

    pdbs_train = set(train_bundle.pdb_ids)
    pdbs_val = set(val_bundle.pdb_ids)
    pdbs_test = set(test_bundle.pdb_ids)

    ab_train = _get_set(train_bundle, "ab_cluster")
    ab_val = _get_set(val_bundle, "ab_cluster")
    ab_test = _get_set(test_bundle, "ab_cluster")

    ag_train = _get_set(train_bundle, "ag_cluster")
    ag_val = _get_set(val_bundle, "ag_cluster")
    ag_test = _get_set(test_bundle, "ag_cluster")

    abag_train = _get_set(train_bundle, "ab_ag_cluster")
    abag_val = _get_set(val_bundle, "ab_ag_cluster")
    abag_test = _get_set(test_bundle, "ab_ag_cluster")

    pdb_train_val = pdbs_train & pdbs_val
    pdb_train_test = pdbs_train & pdbs_test
    pdb_val_test = pdbs_val & pdbs_test

    abag_train_val = abag_train & abag_val
    abag_train_test = abag_train & abag_test
    abag_val_test = abag_val & abag_test

    leakage_detected = any([
        len(pdb_train_val) > 0,
        len(pdb_train_test) > 0,
        len(pdb_val_test) > 0,
        len(abag_train_val) > 0,
        len(abag_train_test) > 0,
        len(abag_val_test) > 0,
    ])

    return {
        "pdb_overlap_train_val": sorted(list(pdb_train_val)),
        "pdb_overlap_train_test": sorted(list(pdb_train_test)),
        "pdb_overlap_val_test": sorted(list(pdb_val_test)),
        "ab_ag_cluster_overlap_train_val": sorted(list(abag_train_val)),
        "ab_ag_cluster_overlap_train_test": sorted(list(abag_train_test)),
        "ab_ag_cluster_overlap_val_test": sorted(list(abag_val_test)),
        "ab_cluster_overlap_train_test": sorted(list(ab_train & ab_test)),
        "ag_cluster_overlap_train_test": sorted(list(ag_train & ag_test)),
        "leakage_detected": leakage_detected,
        "is_leakage_free": not leakage_detected,
    }


def compute_representation_diversity(
    bundle: DatasetBundle,
) -> Dict[str, Any]:
    """Calculate diagnostics to confirm representations have real diversity and are not collapsed.

    Computes:
    - total samples
    - count of unique representation hashes
    - non-zero cell statistics per sample
    - matrix value statistics
    - mean pairwise Frobenius distance between sample representations
    """
    if len(bundle) == 0:
        return {"total_samples": 0}

    hashes = set()
    nonzero_counts = []
    flat_values = []

    for i in range(len(bundle)):
        mat = bundle.X[i, :, :, 0]
        h = hashlib.sha256(mat.tobytes()).hexdigest()
        hashes.add(h)
        nonzero_counts.append(int(np.count_nonzero(mat)))
        flat_values.append(mat.flatten())

    flat_all = np.concatenate(flat_values)

    # Compute pairwise Frobenius distance across sample pairs
    n = len(bundle)
    pairwise_distances = []
    for i in range(min(n, 25)):
        for j in range(i + 1, min(n, 25)):
            diff = bundle.X[i, :, :, 0] - bundle.X[j, :, :, 0]
            dist = float(np.linalg.norm(diff))
            pairwise_distances.append(dist)

    mean_dist = float(np.mean(pairwise_distances)) if pairwise_distances else 0.0
    min_dist = float(np.min(pairwise_distances)) if pairwise_distances else 0.0

    return {
        "split": bundle.split_name,
        "total_samples": len(bundle),
        "unique_representation_hashes": len(hashes),
        "has_duplicate_representations": len(hashes) < len(bundle),
        "nonzero_cells_min": int(np.min(nonzero_counts)),
        "nonzero_cells_max": int(np.max(nonzero_counts)),
        "nonzero_cells_mean": float(np.mean(nonzero_counts)),
        "nonzero_cells_median": float(np.median(nonzero_counts)),
        "nonzero_cells_std": float(np.std(nonzero_counts)),
        "matrix_val_min": float(np.min(flat_all)),
        "matrix_val_max": float(np.max(flat_all)),
        "matrix_val_mean": float(np.mean(flat_all)),
        "matrix_val_std": float(np.std(flat_all)),
        "mean_pairwise_frobenius_distance": mean_dist,
        "min_pairwise_frobenius_distance": min_dist,
        "is_representation_collapsed": bool(len(hashes) <= 1 or mean_dist < 1e-4),
    }


def load_all_splits(
    processed_dir: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> Tuple[DatasetBundle, DatasetBundle, DatasetBundle, Dict[str, Any]]:
    """Load train, validation, and test splits and audit leakage."""
    p_dir = processed_dir or PROCESSED_DIR
    base = base_dir or BASE_DIR

    train_path = p_dir / "train.csv"
    val_path = p_dir / "validation.csv"
    test_path = p_dir / "test.csv"

    train_bundle = load_dataset_split(train_path, base_dir=base, split_name="train")
    val_bundle = load_dataset_split(val_path, base_dir=base, split_name="validation")
    test_bundle = load_dataset_split(test_path, base_dir=base, split_name="test")

    leakage_audit = audit_splits_leakage(train_bundle, val_bundle, test_bundle)
    return train_bundle, val_bundle, test_bundle, leakage_audit


def generate_mismatched_negatives(
    bundle: DatasetBundle,
    seed: int = 42,
) -> DatasetBundle:
    """Generate author-compatible non-cognate mismatched Ab-Ag pairs strictly within the split.

    Following Zhang et al. 2024 Cell 16 in part1.ipynb:
    Pairs antibody from complex i with antigen from complex j (i != j).
    Strictly stays within the split to prevent cross-split structural leakage.
    Assigns label=0 to mismatched pairs.
    """
    n = len(bundle)
    if n < 2:
        return bundle

    rng = np.random.RandomState(seed)
    mismatched_tensors = []
    mismatched_sample_ids = []
    mismatched_pdb_ids = []
    mismatched_metadata = []

    # In 20x21 representation:
    # rows 0-19: row r has (r + 1) antibody elements (flipped upper triangle)
    # and (20 - r) antigen elements (upper triangle)
    # Total per row: (r + 1) + (20 - r) = 21
    # To swap antigens while keeping antibodies:
    # For each row r, columns 0..r are antibody, columns (r+1)..20 are antigen
    used_pairs = set()
    for i in range(n):
        attempts = 0
        j = i
        while (j == i or (i, j) in used_pairs) and attempts < 100:
            j = int(rng.randint(0, n))
            attempts += 1
        used_pairs.add((i, j))

        mat_ab = bundle.X[i, :, :, 0]
        mat_ag = bundle.X[j, :, :, 0]

        # Combine: for each row r:
        # ab columns: 0 to (r + 1)
        # ag columns: (r + 1) to 21
        mismatched_mat = np.zeros((20, 21), dtype=np.float32)
        for r in range(20):
            ab_len = r + 1
            mismatched_mat[r, :ab_len] = mat_ab[r, :ab_len]
            mismatched_mat[r, ab_len:] = mat_ag[r, ab_len:]

        mismatched_tensors.append(np.expand_dims(mismatched_mat, axis=-1))
        mismatched_sample_ids.append(f"mismatch_{bundle.sample_ids[i]}_x_{bundle.sample_ids[j]}")
        mismatched_pdb_ids.append(f"{bundle.pdb_ids[i]}_{bundle.pdb_ids[j]}")
        mismatched_metadata.append({
            "sample_id": f"mismatch_{bundle.sample_ids[i]}_x_{bundle.sample_ids[j]}",
            "ab_source": bundle.sample_ids[i],
            "ag_source": bundle.sample_ids[j],
            "label": 0,
            "is_mismatched": True,
        })

    # Combine cognate (positives) + mismatched (negatives)
    all_X = np.concatenate([bundle.X, np.array(mismatched_tensors, dtype=np.float32)], axis=0)
    all_y = np.concatenate([bundle.y, np.zeros(len(mismatched_tensors), dtype=np.int32)], axis=0)
    all_samples = bundle.sample_ids + mismatched_sample_ids
    all_pdbs = bundle.pdb_ids + mismatched_pdb_ids
    all_meta = bundle.metadata + mismatched_metadata

    # Deterministic shuffle
    perm = rng.permutation(len(all_y))
    return DatasetBundle(
        X=all_X[perm],
        y=all_y[perm],
        sample_ids=[all_samples[p] for p in perm],
        pdb_ids=[all_pdbs[p] for p in perm],
        metadata=[all_meta[p] for p in perm],
        split_name=f"{bundle.split_name}_balanced_mismatch",
    )
