"""Phase 4 Test Suite: Dataset Construction, Validation, and Leakage-Safe Splitting.

Validates:
Test A — Manifest schema is valid.
Test B — Every accepted sample has an existing representation file.
Test C — Every representation has shape (20, 21).
Test D — No accepted sample has NaN or Inf values in the final tensor.
Test E — Representation hashes are deterministic.
Test F — Rejected samples have explicit categorized reasons.
Test G — PDB IDs do not unexpectedly overlap between train/validation and test.
Test H — Cluster leakage checks verify zero Ab-Ag cluster overlap with test.
Test I — Split generation is deterministic across runs with fixed seed.
Test J — Unseen structure (1A3R) processes through the same production pipeline.
Test K — Rebuilding dataset produces identical manifests and hashes.
Test L — No benchmark-specific hardcoded processing branches exist in source.
"""

import csv
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from backend.app.datasets import (
    DatasetSample,
    RejectedSample,
    DatasetBuilder,
    ClusterAwareSplitter,
)
from backend.app.structure import (
    analyze_biological_entities,
    build_interaction_representation,
    compute_matrix_statistics,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
REJECTED_PATH = PROCESSED_DIR / "rejected_samples.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "validation.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

VALID_REASONS = {
    "missing_pdb_id",
    "missing_structure",
    "structure_parse_error",
    "missing_antibody_metadata",
    "missing_antigen_metadata",
    "cdr_mapping_failure",
    "no_interface_contacts",
    "no_valid_representation",
    "unsupported_structure",
}


# -----------------------------------------------------------------------------
# Test A — Manifest schema is valid
# -----------------------------------------------------------------------------
def test_a_manifest_schema_is_valid():
    """Verify that dataset_manifest.csv and rejected_samples.csv exist and have valid headers."""
    assert MANIFEST_PATH.exists(), f"Missing {MANIFEST_PATH}"
    assert REJECTED_PATH.exists(), f"Missing {REJECTED_PATH}"

    expected_manifest_cols = {
        "sample_id", "pdb_id", "structure_path", "antibody_heavy_chain",
        "antibody_light_chain", "antigen_chains", "sabdab_id", "ab_cluster",
        "ag_cluster", "ab_ag_cluster", "label", "representation_path",
        "representation_shape", "intermolecular_contact_count",
        "antibody_interface_count", "cdr_interface_count", "antigen_interface_count",
        "antibody_intramolecular_contact_count", "antigen_intramolecular_contact_count",
        "representation_nonzero_count", "representation_sha256",
        "processing_status", "split",
    }

    manifest_df = pd.read_csv(MANIFEST_PATH)
    assert not manifest_df.empty, "dataset_manifest.csv is empty"
    for col in expected_manifest_cols:
        assert col in manifest_df.columns, f"Missing column '{col}' in manifest"

    expected_rejected_cols = {"sample_id", "pdb_id", "reason", "detailed_error", "stage_failed"}
    rejected_df = pd.read_csv(REJECTED_PATH)
    assert not rejected_df.empty, "rejected_samples.csv is empty"
    for col in expected_rejected_cols:
        assert col in rejected_df.columns, f"Missing column '{col}' in rejected samples"


# -----------------------------------------------------------------------------
# Test B — Every accepted sample has an existing representation
# -----------------------------------------------------------------------------
def test_b_every_accepted_sample_has_representation():
    """Verify that every representation_path in the manifest exists on disk."""
    df = pd.read_csv(MANIFEST_PATH)
    for _, row in df.iterrows():
        rel_path = row["representation_path"]
        full_path = BASE_DIR / rel_path
        assert full_path.exists(), f"Representation file not found: {full_path}"
        assert full_path.stat().st_size > 0, f"Representation file is empty: {full_path}"


# -----------------------------------------------------------------------------
# Test C — Every representation has shape (20, 21)
# -----------------------------------------------------------------------------
def test_c_representation_shapes():
    """Verify that every accepted tensor has shape (20, 21)."""
    df = pd.read_csv(MANIFEST_PATH)
    for _, row in df.iterrows():
        full_path = BASE_DIR / row["representation_path"]
        arr = np.load(full_path)
        assert arr.shape == (20, 21), f"Unexpected shape {arr.shape} for {row['sample_id']}"


# -----------------------------------------------------------------------------
# Test D — No NaN or Inf values in tensors
# -----------------------------------------------------------------------------
def test_d_no_nans_or_infs():
    """Verify that all saved tensors contain strictly valid finite numbers in [0.0, 1.0]."""
    df = pd.read_csv(MANIFEST_PATH)
    for _, row in df.iterrows():
        full_path = BASE_DIR / row["representation_path"]
        arr = np.load(full_path)
        assert not np.isnan(arr).any(), f"NaN detected in {row['sample_id']}"
        assert not np.isinf(arr).any(), f"Inf detected in {row['sample_id']}"
        assert arr.min() >= 0.0, f"Value < 0.0 in {row['sample_id']}"
        assert arr.max() <= 1.0, f"Value > 1.0 in {row['sample_id']}"


# -----------------------------------------------------------------------------
# Test E — Representation hashes are deterministic
# -----------------------------------------------------------------------------
def test_e_representation_hashes_are_deterministic():
    """Re-compute hashes for sample complexes and verify exact match with manifest."""
    df = pd.read_csv(MANIFEST_PATH).head(3)
    for _, row in df.iterrows():
        clean_pdb = row["pdb_id"]
        full_path = BASE_DIR / row["representation_path"]
        loaded_arr = np.load(full_path)
        computed_stats = compute_matrix_statistics(loaded_arr)
        assert computed_stats.sha256 == row["representation_sha256"]


# -----------------------------------------------------------------------------
# Test F — Rejected samples have explicit reasons
# -----------------------------------------------------------------------------
def test_f_rejected_samples_have_explicit_reasons():
    """Verify all rejected samples have a recognized reason, detailed error, and stage."""
    df = pd.read_csv(REJECTED_PATH)
    for _, row in df.iterrows():
        reason = str(row["reason"]).strip()
        assert reason in VALID_REASONS, f"Unknown rejection reason '{reason}'"
        assert str(row["detailed_error"]).strip() != ""
        assert str(row["stage_failed"]).strip() != ""


# -----------------------------------------------------------------------------
# Test G — PDB IDs do not unexpectedly overlap between splits
# -----------------------------------------------------------------------------
def test_g_pdb_ids_do_not_overlap_unexpectedly():
    """Verify that test PDB IDs do not overlap with train or validation PDB IDs."""
    assert TRAIN_PATH.exists() and VAL_PATH.exists() and TEST_PATH.exists()
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_pdbs = set(train_df["pdb_id"].dropna().unique())
    val_pdbs = set(val_df["pdb_id"].dropna().unique())
    test_pdbs = set(test_df["pdb_id"].dropna().unique())

    assert len(train_pdbs & test_pdbs) == 0, f"PDB overlap between train and test: {train_pdbs & test_pdbs}"
    assert len(val_pdbs & test_pdbs) == 0, f"PDB overlap between val and test: {val_pdbs & test_pdbs}"
    assert len(train_pdbs & val_pdbs) == 0, f"PDB overlap between train and val: {train_pdbs & val_pdbs}"


# -----------------------------------------------------------------------------
# Test H — Cluster leakage checks verify zero overlap with test
# -----------------------------------------------------------------------------
def test_h_cluster_leakage_checks():
    """Verify that ab_ag_cluster overlap between test and train/val is zero."""
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    def _get_clusters(df, col):
        return {str(c) for c in df[col].dropna() if str(c).strip() != ""}

    train_abag = _get_clusters(train_df, "ab_ag_cluster")
    val_abag = _get_clusters(val_df, "ab_ag_cluster")
    test_abag = _get_clusters(test_df, "ab_ag_cluster")

    assert len(train_abag & test_abag) == 0, "Ab-Ag cluster overlap between train and test"
    assert len(val_abag & test_abag) == 0, "Ab-Ag cluster overlap between val and test"
    assert len(train_abag & val_abag) == 0, "Ab-Ag cluster overlap between train and val"


# -----------------------------------------------------------------------------
# Test I — Split generation is deterministic
# -----------------------------------------------------------------------------
def test_i_split_generation_is_deterministic():
    """Verify that ClusterAwareSplitter produces identical split assignments across runs."""
    manifest_df = pd.read_csv(MANIFEST_PATH)
    samples = [DatasetSample(**row.to_dict()) for _, row in manifest_df.iterrows()]

    splitter = ClusterAwareSplitter(seed=42)
    train1, val1, test1, _ = splitter.split_samples(samples)

    # Re-instantiate samples
    samples2 = [DatasetSample(**row.to_dict()) for _, row in manifest_df.iterrows()]
    train2, val2, test2, _ = splitter.split_samples(samples2)

    assert [s.sample_id for s in train1] == [s.sample_id for s in train2]
    assert [s.sample_id for s in val1] == [s.sample_id for s in val2]
    assert [s.sample_id for s in test1] == [s.sample_id for s in test2]


# -----------------------------------------------------------------------------
# Test J — Unseen structure (1A3R) processes through the same pipeline
# -----------------------------------------------------------------------------
def test_j_unseen_structure_pipeline():
    """Verify that unseen structure 1A3R processes through production pipeline successfully."""
    struct_path = BASE_DIR / "data" / "structures" / "1A3R.cif"
    assert struct_path.exists()

    comp = analyze_biological_entities(struct_path)
    assert comp.antibody.heavy_chain_id == "H"
    assert comp.antibody.light_chain_id == "L"
    assert "P" in comp.antigen_chain_ids

    # All 6 CDRs mapped
    for cdr_name in ["H1", "H2", "H3", "L1", "L2", "L3"]:
        assert cdr_name in comp.antibody.cdrs
        assert comp.antibody.cdrs[cdr_name].mapping_status == "MATCH"

    rep = build_interaction_representation(comp)
    assert rep.tensor_20x21.shape == (20, 21)
    assert rep.intermolecular.contact_count > 0
    assert rep.stats_20x21.nonzero_count > 0
    assert rep.stats_20x21.sha256 != ""


# -----------------------------------------------------------------------------
# Test K — Rebuilding dataset produces identical manifests and hashes
# -----------------------------------------------------------------------------
def test_k_rebuilding_dataset_produces_same_manifests(tmp_path):
    """Verify that rebuilding on the same subset produces identical manifests and hashes."""
    builder = DatasetBuilder(output_dir=tmp_path)
    # Test on a deterministic 5-sample candidate set
    candidates = {"1EJO", "1NBZ", "1KC5", "1DQJ", "1A3R"}
    accepted1, _ = builder.build_dataset(candidate_pdbs=candidates, cached_only=True)
    accepted2, _ = builder.build_dataset(candidate_pdbs=candidates, cached_only=True)

    assert len(accepted1) == len(accepted2)
    hashes1 = {s.sample_id: s.representation_sha256 for s in accepted1}
    hashes2 = {s.sample_id: s.representation_sha256 for s in accepted2}
    assert hashes1 == hashes2


# -----------------------------------------------------------------------------
# Test L — No benchmark-specific hardcoded processing branches exist
# -----------------------------------------------------------------------------
def test_l_no_benchmark_specific_hardcoded_processing():
    """Inspect source code of datasets and structure modules for hardcoded benchmark branches."""
    import backend.app.datasets.builder as builder_mod
    import backend.app.structure.contacts as contacts_mod
    import backend.app.structure.representation as rep_mod

    modules_to_check = [builder_mod, contacts_mod, rep_mod]
    benchmark_ids = ["1EJO", "1NBZ", "1KC5", "1DQJ"]

    for mod in modules_to_check:
        src = inspect.getsource(mod)
        for b_id in benchmark_ids:
            # Check for conditional logic branching on benchmark IDs
            assert f'== "{b_id}"' not in src, f"Hardcoded branch on {b_id} in {mod.__name__}"
            assert f"== '{b_id}'" not in src, f"Hardcoded branch on {b_id} in {mod.__name__}"
            assert f'in ["{b_id}"' not in src, f"Hardcoded list with {b_id} in {mod.__name__}"
