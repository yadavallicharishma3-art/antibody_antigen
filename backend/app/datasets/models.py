"""Data models for dataset construction, manifest records, and rejection tracking."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class DatasetSample:
    """Represents a successfully processed, accepted antibody-antigen complex."""
    sample_id: str
    pdb_id: str
    structure_path: str
    antibody_heavy_chain: str
    antibody_light_chain: str
    antigen_chains: str
    sabdab_id: str
    ab_cluster: str
    ag_cluster: str
    ab_ag_cluster: str
    label: int  # 1 for positive cognate Ab-Ag complex
    representation_path: str
    representation_shape: str
    intermolecular_contact_count: int
    antibody_interface_count: int
    cdr_interface_count: int
    antigen_interface_count: int
    antibody_intramolecular_contact_count: int
    antigen_intramolecular_contact_count: int
    representation_nonzero_count: int
    representation_sha256: str
    processing_status: str = "ACCEPTED"
    split: str = "unassigned"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RejectedSample:
    """Represents a sample rejected by the structural filtering pipeline."""
    sample_id: str
    pdb_id: str
    reason: str
    detailed_error: str
    stage_failed: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SplitMetrics:
    """Diagnostic metrics for dataset splits and cluster overlaps."""
    total_samples: int
    train_count: int
    val_count: int
    test_count: int
    train_pct: float
    val_pct: float
    test_pct: float

    # Overlap diagnostics
    pdb_overlap_train_val: int
    pdb_overlap_train_test: int
    pdb_overlap_val_test: int

    ab_cluster_overlap_train_val: int
    ab_cluster_overlap_train_test: int
    ab_cluster_overlap_val_test: int

    ag_cluster_overlap_train_val: int
    ag_cluster_overlap_train_test: int
    ag_cluster_overlap_val_test: int

    ab_ag_cluster_overlap_train_val: int
    ab_ag_cluster_overlap_train_test: int
    ab_ag_cluster_overlap_val_test: int
