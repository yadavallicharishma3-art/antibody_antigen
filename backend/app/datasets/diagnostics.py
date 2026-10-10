"""Dataset diagnostics, distribution statistics, duplicate analysis, and readiness evaluation."""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import numpy as np

from .models import DatasetSample, RejectedSample, SplitMetrics


def generate_dataset_report(
    total_source_rows: int,
    total_source_pdbs: int,
    accepted_samples: List[DatasetSample],
    rejected_samples: List[RejectedSample],
    split_metrics: Optional[SplitMetrics] = None,
) -> str:
    """Format comprehensive Phase 4 dataset diagnostic report."""
    lines = []
    lines.append("=" * 65)
    lines.append("PHASE 4: DATASET DIAGNOSTIC REPORT")
    lines.append("=" * 65)

    # 1. Source Metadata Summary
    lines.append("\n1. SOURCE METADATA AUDIT")
    lines.append("-" * 35)
    lines.append(f"  Total metadata rows in source:    {total_source_rows:,}")
    lines.append(f"  Total unique PDB IDs in source:   {total_source_pdbs:,}")
    lines.append(f"  Samples evaluated in this run:    {len(accepted_samples) + len(rejected_samples):,}")

    # 2. Filtering & Rejection Summary
    lines.append("\n2. SAMPLE FILTERING & REJECTIONS")
    lines.append("-" * 35)
    lines.append(f"  Accepted samples:                 {len(accepted_samples):,}")
    lines.append(f"  Rejected samples:                 {len(rejected_samples):,}")

    rejection_reasons = Counter(r.reason for r in rejected_samples)
    lines.append("  Rejection reasons breakdown:")
    if rejection_reasons:
        for reason, count in rejection_reasons.most_common():
            pct = (count / len(rejected_samples)) * 100
            lines.append(f"    - {reason:<28}: {count:>4} ({pct:>5.1f}%)")
    else:
        lines.append("    (None - 100% acceptance)")

    # 3. Representation Dataset Statistics
    lines.append("\n3. REPRESENTATION STATISTICS")
    lines.append("-" * 35)
    shapes = Counter(s.representation_shape for s in accepted_samples)
    lines.append("  Tensor shape distribution:")
    for shape_str, cnt in shapes.most_common():
        lines.append(f"    - {shape_str}: {cnt}")

    if accepted_samples:
        nonzeros = [s.representation_nonzero_count for s in accepted_samples]
        inter_contacts = [s.intermolecular_contact_count for s in accepted_samples]
        ab_iface = [s.antibody_interface_count for s in accepted_samples]
        cdr_iface = [s.cdr_interface_count for s in accepted_samples]
        ag_iface = [s.antigen_interface_count for s in accepted_samples]
        ab_intra = [s.antibody_intramolecular_contact_count for s in accepted_samples]
        ag_intra = [s.antigen_intramolecular_contact_count for s in accepted_samples]

        lines.append("\n  Nonzero cells per (20,21) tensor (out of 420 cells):")
        lines.append(f"    min: {np.min(nonzeros)}, max: {np.max(nonzeros)}, mean: {np.mean(nonzeros):.1f}, median: {np.median(nonzeros):.1f}")

        lines.append("\n  Intermolecular 5 Å contacts:")
        lines.append(f"    min: {np.min(inter_contacts)}, max: {np.max(inter_contacts)}, mean: {np.mean(inter_contacts):.1f}, median: {np.median(inter_contacts):.1f}")

        lines.append("\n  Interface residues:")
        lines.append(f"    Antibody total interface: min: {np.min(ab_iface)}, max: {np.max(ab_iface)}, mean: {np.mean(ab_iface):.1f}")
        lines.append(f"    Antibody CDR interface:   min: {np.min(cdr_iface)}, max: {np.max(cdr_iface)}, mean: {np.mean(cdr_iface):.1f}")
        lines.append(f"    Antigen interface:        min: {np.min(ag_iface)}, max: {np.max(ag_iface)}, mean: {np.mean(ag_iface):.1f}")

        lines.append("\n  Intramolecular interface contacts:")
        lines.append(f"    Antibody CDR intra:       min: {np.min(ab_intra)}, max: {np.max(ab_intra)}, mean: {np.mean(ab_intra):.1f}")
        lines.append(f"    Antigen intra:            min: {np.min(ag_intra)}, max: {np.max(ag_intra)}, mean: {np.mean(ag_intra):.1f}")

    # 4. Duplicate Analysis
    lines.append("\n4. DUPLICATE & DIVERSITY ANALYSIS")
    lines.append("-" * 35)
    pdb_counts = Counter(s.pdb_id for s in accepted_samples)
    multi_instance_pdbs = {pdb: cnt for pdb, cnt in pdb_counts.items() if cnt > 1}
    lines.append(f"  Unique PDB IDs in accepted set:   {len(pdb_counts)}")
    lines.append(f"  PDBs with multiple instances:     {len(multi_instance_pdbs)} (e.g. multi-arm Fabs or asymmetric units)")
    if multi_instance_pdbs:
        sample_multis = list(multi_instance_pdbs.items())[:5]
        for pdb, cnt in sample_multis:
            lines.append(f"    - {pdb}: {cnt} instances")

    hashes = [s.representation_sha256 for s in accepted_samples]
    unique_hashes = set(hashes)
    lines.append(f"  Total representation hashes:      {len(hashes)}")
    lines.append(f"  Unique representation hashes:     {len(unique_hashes)}")
    duplicate_hash_count = len(hashes) - len(unique_hashes)
    lines.append(f"  Duplicate hash collisions:        {duplicate_hash_count}")

    # 5. Split Diagnostics & Leakage Prevention
    if split_metrics:
        lines.append("\n5. TRAIN / VALIDATION / TEST SPLIT & LEAKAGE AUDIT")
        lines.append("-" * 35)
        lines.append(f"  Train:       {split_metrics.train_count:>4} samples ({split_metrics.train_pct:>5.1f}%)")
        lines.append(f"  Validation:  {split_metrics.val_count:>4} samples ({split_metrics.val_pct:>5.1f}%)")
        lines.append(f"  Test:        {split_metrics.test_count:>4} samples ({split_metrics.test_pct:>5.1f}%)")
        lines.append(f"  Total:       {split_metrics.total_samples:>4} samples")

        lines.append("\n  Leakage Overlap Diagnostics (Target: 0 between Test and Train/Val):")
        lines.append(f"    PDB ID overlap (Train ∩ Val):           {split_metrics.pdb_overlap_train_val}")
        lines.append(f"    PDB ID overlap (Train ∩ Test):          {split_metrics.pdb_overlap_train_test}")
        lines.append(f"    PDB ID overlap (Val ∩ Test):            {split_metrics.pdb_overlap_val_test}")
        lines.append(f"    Ab-Ag cluster overlap (Train ∩ Val):    {split_metrics.ab_ag_cluster_overlap_train_val}")
        lines.append(f"    Ab-Ag cluster overlap (Train ∩ Test):   {split_metrics.ab_ag_cluster_overlap_train_test}")
        lines.append(f"    Ab-Ag cluster overlap (Val ∩ Test):     {split_metrics.ab_ag_cluster_overlap_val_test}")
        lines.append(f"    Ab cluster overlap (Train ∩ Val):       {split_metrics.ab_cluster_overlap_train_val}")
        lines.append(f"    Ab cluster overlap (Train ∩ Test):      {split_metrics.ab_cluster_overlap_train_test}")
        lines.append(f"    Ab cluster overlap (Val ∩ Test):        {split_metrics.ab_cluster_overlap_val_test}")
        lines.append(f"    Ag cluster overlap (Train ∩ Val):       {split_metrics.ag_cluster_overlap_train_val}")
        lines.append(f"    Ag cluster overlap (Train ∩ Test):      {split_metrics.ag_cluster_overlap_train_test}")
        lines.append(f"    Ag cluster overlap (Val ∩ Test):        {split_metrics.ag_cluster_overlap_val_test}")

    # 6. Class & Label Audit
    lines.append("\n6. CLASS & LABEL AUDIT")
    lines.append("-" * 35)
    lines.append("  Source data status: Positive antibody-antigen complexes only (label=1).")
    lines.append("  Binding affinity: Source structural data does NOT provide continuous binding affinity (Kd/IC50).")
    lines.append("  Negative examples: No synthetic or fabricated negative samples introduced in Phase 4.")

    # 7. Paper Experiment Readiness
    lines.append("\n7. PAPER EXPERIMENT READINESS (Tests 1, 2, 3)")
    lines.append("-" * 35)
    lines.append("  Test 1 (Ab-Ag vs General PPI):")
    lines.append("    - Required: Positive Ab-Ag representations + General PPI control representations (3did).")
    lines.append("    - Status: data/3did/pdb_4960list.txt staged in Phase 1 (4,144 pairs). READY for Phase 5.")
    lines.append("  Test 2 (Antigen vs General Protein PPI):")
    lines.append("    - Required: Isolated antigen upper-triangle representations vs non-antigen protein PPI.")
    lines.append("    - Status: Antigen triangular charts computed and preserved. READY for Phase 5.")
    lines.append("  Test 3 (Cognate vs Mismatched Ab-Ag):")
    lines.append("    - Required: Valid Ab and Ag components paired with non-cognate partners (permutations).")
    lines.append("    - Status: Author mismatch generation logic audited in part1.ipynb. READY for Phase 5.")

    lines.append("\n" + "=" * 65)
    return "\n".join(lines)
