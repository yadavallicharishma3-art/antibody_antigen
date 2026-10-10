"""Structural interaction matrix generation, per-complex normalization, and serialization."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .constants import STANDARD_AA_3TO1
from .models import ComplexData
from .contacts import (
    find_intermolecular_contacts,
    find_intramolecular_contacts,
    IntermolecularContactResult,
    IntramolecularContactResult,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_MATRICES_DIR = BASE_DIR / "data" / "processed" / "matrices"
PROCESSED_MATRICES_DIR.mkdir(parents=True, exist_ok=True)

# Alphabetical list of 3-letter codes matching author's part1.ipynb and prompt
AMINO_3_CODES = sorted(list(STANDARD_AA_3TO1.keys()))
AMINO_1_CODES = [STANDARD_AA_3TO1[c] for c in AMINO_3_CODES]
AA_INDEX_MAP = {aa3: idx for idx, aa3 in enumerate(AMINO_3_CODES)}


@dataclass
class MatrixStatistics:
    """Summary statistics and cryptographic fingerprint of a representation."""
    shape: Tuple[int, ...]
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    sum_val: float
    nonzero_count: int
    sha256: str


@dataclass
class RepresentationResult:
    """Complete result of Phase 3 feature matrix generation."""
    pdb_id: str
    intermolecular: IntermolecularContactResult
    antigen_intramolecular: IntramolecularContactResult
    antibody_intramolecular: IntramolecularContactResult

    raw_ag_triangle: np.ndarray        # (20, 20) with NaNs below diagonal
    raw_ab_triangle: np.ndarray        # (20, 20) with NaNs below diagonal
    normalized_ag_triangle: np.ndarray # (20, 20) min-max normalized
    normalized_ab_triangle: np.ndarray # (20, 20) min-max normalized

    matrix_20x20: np.ndarray           # (20, 20) float32 canonical matrix
    tensor_20x21: np.ndarray           # (20, 21) float32 author CNN tensor

    stats_20x20: MatrixStatistics
    stats_20x21: MatrixStatistics


def build_upper_triangle(pair_counts: Dict[Tuple[str, str], int]) -> Tuple[np.ndarray, np.ndarray]:
    """Construct (20, 20) upper triangle chart and apply per-complex min-max normalization.

    Matches author's part1.ipynb cell 3 & 7:
        chart -= nanmin(chart)
        chart /= (nanmax(chart) - nanmin(chart))

    Returns:
        Tuple of (raw_chart, normalized_chart) with NaNs below diagonal.
    """
    raw_chart = np.empty((20, 20), dtype=np.float32)
    raw_chart.fill(np.nan)

    for r in range(20):
        for c in range(r, 20):
            fst = AMINO_3_CODES[r]
            snd = AMINO_3_CODES[c]
            comb = (fst, snd)
            raw_chart[r, c] = pair_counts.get(comb, 0)

    nan_min = float(np.nanmin(raw_chart))
    nan_max = float(np.nanmax(raw_chart))

    norm_chart = raw_chart.copy()
    if nan_max > nan_min:
        norm_chart -= nan_min
        norm_chart /= (nan_max - nan_min)
    else:
        # Zero-range edge case: all populated values are identical (e.g. all 0)
        norm_chart.fill(0.0)
        for r in range(20):
            for c in range(r):
                norm_chart[r, c] = np.nan

    return raw_chart, norm_chart


def compute_matrix_statistics(matrix: np.ndarray) -> MatrixStatistics:
    """Calculate deterministic summary statistics and SHA256 hash."""
    clean_arr = np.nan_to_num(matrix, nan=0.0).astype(np.float32)
    # Round to 6 decimals for hash stability across platforms
    hash_bytes = clean_arr.round(6).tobytes()
    sha = hashlib.sha256(hash_bytes).hexdigest()

    return MatrixStatistics(
        shape=tuple(matrix.shape),
        min_val=float(clean_arr.min()),
        max_val=float(clean_arr.max()),
        mean_val=float(clean_arr.mean()),
        std_val=float(clean_arr.std()),
        sum_val=float(clean_arr.sum()),
        nonzero_count=int((clean_arr > 0).sum()),
        sha256=sha,
    )


def build_interaction_representation(
    complex_data: ComplexData,
    cutoff: float = 5.0
) -> RepresentationResult:
    """Generate the complete structural interaction representation for a complex.

    Workflow:
    1. Intermolecular contacts (5 Å cutoff, sidechain heavy atoms).
    2. Interface residues (Antibody vs Antigen).
    3. Filter antibody CDR interface residues.
    4. Intramolecular contacts among antigen interface residues.
    5. Intramolecular contacts among antibody CDR interface residues.
    6. Amino-acid pair frequency counting.
    7. Per-complex min-max normalization.
    8. Matrix assembly:
       - matrix_20x20: Antigen in upper triangle, Antibody in lower triangle.
       - tensor_20x21: Concatenated flipped antibody + antigen rows.
    """
    # 1-3: Intermolecular contacts & interface residues
    inter_result = find_intermolecular_contacts(complex_data, cutoff=cutoff)

    ag_interface_residues = list(inter_result.antigen_interface_residues.values())
    ab_cdr_interface_residues = list(inter_result.cdr_interface_residues.values())

    # 4: Antigen intramolecular contacts
    ag_intra = find_intramolecular_contacts(ag_interface_residues, cutoff=cutoff)

    # 5: Antibody CDR intramolecular contacts
    ab_intra = find_intramolecular_contacts(ab_cdr_interface_residues, cutoff=cutoff)

    # 6-7: Triangular charts and per-complex normalization
    raw_ag, norm_ag = build_upper_triangle(ag_intra.pair_frequencies)
    raw_ab, norm_ab = build_upper_triangle(ab_intra.pair_frequencies)

    # 8a: Author's (20, 21) tensor
    flipped_ab = np.flip(norm_ab)
    cat_chart = np.concatenate([flipped_ab, norm_ag], axis=1)
    tensor_20x21 = np.array([row[~np.isnan(row)] for row in cat_chart], dtype=np.float32)

    # 8b: Canonical (20, 20) matrix
    matrix_20x20 = np.zeros((20, 20), dtype=np.float32)
    for r in range(20):
        for c in range(r, 20):
            matrix_20x20[r, c] = norm_ag[r, c] if not np.isnan(norm_ag[r, c]) else 0.0
            matrix_20x20[c, r] = norm_ab[r, c] if not np.isnan(norm_ab[r, c]) else 0.0

    stats_20x20 = compute_matrix_statistics(matrix_20x20)
    stats_20x21 = compute_matrix_statistics(tensor_20x21)

    return RepresentationResult(
        pdb_id=complex_data.pdb_id,
        intermolecular=inter_result,
        antigen_intramolecular=ag_intra,
        antibody_intramolecular=ab_intra,
        raw_ag_triangle=raw_ag,
        raw_ab_triangle=raw_ab,
        normalized_ag_triangle=norm_ag,
        normalized_ab_triangle=norm_ab,
        matrix_20x20=matrix_20x20,
        tensor_20x21=tensor_20x21,
        stats_20x20=stats_20x20,
        stats_20x21=stats_20x21,
    )


def export_representation(
    rep_result: RepresentationResult,
    output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """Export matrix arrays (.npy and .csv) to the processed directory for inspection.

    Returns:
        Dict of file paths created.
    """
    dest = output_dir if output_dir else PROCESSED_MATRICES_DIR
    dest.mkdir(parents=True, exist_ok=True)

    pdb = rep_result.pdb_id.upper()
    paths = {}

    # 1. Export standard author CNN input tensor ({pdb}.npy matching prompt specification)
    path_default_npy = dest / f"{pdb}.npy"
    np.save(path_default_npy, rep_result.tensor_20x21)
    paths["default_npy"] = path_default_npy

    path_default_csv = dest / f"{pdb}.csv"
    np.savetxt(path_default_csv, rep_result.tensor_20x21, delimiter=",", fmt="%.6f")
    paths["default_csv"] = path_default_csv

    # 2. Export 20x20 canonical matrix
    path_20x20_npy = dest / f"{pdb}_matrix_20x20.npy"
    np.save(path_20x20_npy, rep_result.matrix_20x20)
    paths["matrix_20x20_npy"] = path_20x20_npy

    path_20x20_csv = dest / f"{pdb}_matrix_20x20.csv"
    np.savetxt(path_20x20_csv, rep_result.matrix_20x20, delimiter=",", fmt="%.6f",
               header=",".join(AMINO_1_CODES), comments="")
    paths["matrix_20x20_csv"] = path_20x20_csv

    # 3. Export 20x21 author tensor
    path_20x21_npy = dest / f"{pdb}_tensor_20x21.npy"
    np.save(path_20x21_npy, rep_result.tensor_20x21)
    paths["tensor_20x21_npy"] = path_20x21_npy

    path_20x21_csv = dest / f"{pdb}_tensor_20x21.csv"
    np.savetxt(path_20x21_csv, rep_result.tensor_20x21, delimiter=",", fmt="%.6f")
    paths["tensor_20x21_csv"] = path_20x21_csv

    return paths
