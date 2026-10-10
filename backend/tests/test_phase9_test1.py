"""Phase 9 Test Suite: Test 1 Scientific Evaluation — Antibody-Antigen vs General PPI.

Comprehensive validation covering:
Test 1  — Dataset class definitions (Class 1: antibody-antigen, Class 0: general PPI).
Test 2  — Positive and negative class validity and balance (74 samples: 37 Pos, 37 Neg).
Test 3  — Dataset count integrity across splits (50 Train, 12 Val, 12 Test).
Test 4  — Zero duplicate sample IDs across the entire dataset.
Test 5  — Representation shape is strictly (20, 21, 1) for both classes.
Test 6  — Standard 20 amino-acid ordering and orientation preserved.
Test 7  — General PPI representations do NOT contain or fabricate CDR loop data.
Test 8  — Predicted probabilities satisfy 0.0 <= p <= 1.0.
Test 9  — Confusion matrix components sum to partition sample count.
Test 10 — Test 1 artifacts are properly serialized in model/test1/.
Test 11 — Dataset construction is deterministic given fixed seed.
Test 12 — Phase 3 representation hash regression (1EJO, 1NBZ, 1KC5, 1DQJ, 1A14).
Test 13 — Real general PPI structure (1YZ5) computes valid (20, 21) tensor with 0 NaNs.
Test 14 — Zero sample leakage across train, val, and test partitions.
Test 15 — Exact Phase 5 CNN architecture equivalence and spatial dimension progression.
Test 16 — PDB-level train/validation/test disjointness (zero PDB overlap).
Test 17 — Negative rejection-rule validity (Zhang et al. Cell 4 divide-by-zero verification).
"""

import json
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from backend.app.ml import (
    build_antibody_antigen_cnn,
    compute_general_ppi_representation,
    build_test1_dataset,
    run_test1_evaluation,
    save_test1_artifacts,
    DEFAULT_TEST1_DIR,
    DEFAULT_MODEL_PATH,
    load_inference_model,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEST1_DIR = BASE_DIR / "model" / "test1"
STRUCTURES_DIR = BASE_DIR / "data" / "structures"


@pytest.fixture(scope="module")
def test1_data():
    """Build Test 1 dataset splits once for the test module."""
    tr, va, te, summary = build_test1_dataset(seed=42)
    return tr, va, te, summary


# -----------------------------------------------------------------------------
# Test 1 — Dataset class definitions
# -----------------------------------------------------------------------------
def test_1_dataset_class_definitions(test1_data):
    """Verify Class 1 is antibody-antigen and Class 0 is general PPI."""
    tr, va, te, summary = test1_data
    for bundle in (tr, va, te):
        for meta, label in zip(bundle.metadata, bundle.y):
            if label == 1:
                assert meta["class_type"] == "antibody_antigen"
            elif label == 0:
                assert meta["class_type"] == "general_ppi"
            else:
                pytest.fail(f"Invalid label {label}")


# -----------------------------------------------------------------------------
# Test 2 — Positive and negative class validity and balance
# -----------------------------------------------------------------------------
def test_2_class_balance_and_validity(test1_data):
    """Verify that dataset contains equal counts of positive and negative samples."""
    tr, va, te, summary = test1_data
    total_pos = tr.positive_count + va.positive_count + te.positive_count
    total_neg = tr.negative_count + va.negative_count + te.negative_count

    assert total_pos == 37, f"Expected 37 positive samples, got {total_pos}"
    assert total_neg == 37, f"Expected 37 negative samples, got {total_neg}"
    assert summary["total_samples"] == 74


# -----------------------------------------------------------------------------
# Test 3 — Dataset count integrity across splits
# -----------------------------------------------------------------------------
def test_3_split_count_integrity(test1_data):
    """Verify that train, val, and test splits sum to total samples."""
    tr, va, te, summary = test1_data
    assert len(tr) + len(va) + len(te) == 74
    assert len(tr) == summary["train_samples"]
    assert len(va) == summary["val_samples"]
    assert len(te) == summary["test_samples"]
    assert summary["train_samples"] == 50
    assert summary["val_samples"] == 12
    assert summary["test_samples"] == 12


# -----------------------------------------------------------------------------
# Test 4 — No duplicate sample IDs
# -----------------------------------------------------------------------------
def test_4_no_duplicate_sample_ids(test1_data):
    """Verify that all sample IDs are completely unique."""
    tr, va, te, _ = test1_data
    all_sids = tr.sample_ids + va.sample_ids + te.sample_ids
    assert len(all_sids) == len(set(all_sids)), f"Duplicate sample IDs detected: {len(all_sids)} vs {len(set(all_sids))}"


# -----------------------------------------------------------------------------
# Test 5 — Representation shape is strictly (20, 21, 1)
# -----------------------------------------------------------------------------
def test_5_representation_shapes(test1_data):
    """Verify tensor shapes match (N, 20, 21, 1) for all splits."""
    tr, va, te, _ = test1_data
    for b in (tr, va, te):
        assert b.X.ndim == 4
        assert b.X.shape[1:] == (20, 21, 1)
        assert not np.isnan(b.X).any()
        assert not np.isinf(b.X).any()
        assert np.all(b.X >= 0.0)
        assert np.all(b.X <= 1.0001)


# -----------------------------------------------------------------------------
# Test 6 — Standard amino acid ordering
# -----------------------------------------------------------------------------
def test_6_amino_acid_ordering():
    """Verify that the standard 20 amino acid list is preserved."""
    from backend.app.structure.representation import AMINO_3_CODES, AMINO_1_CODES
    assert len(AMINO_3_CODES) == 20
    assert len(AMINO_1_CODES) == 20
    assert AMINO_3_CODES == sorted(AMINO_3_CODES)
    assert AMINO_3_CODES[0] == "ALA"
    assert AMINO_3_CODES[-1] == "VAL"


# -----------------------------------------------------------------------------
# Test 7 — No fake CDR data for General PPI
# -----------------------------------------------------------------------------
def test_7_no_fake_cdr_for_general_ppi(test1_data):
    """Verify that General PPI metadata does not invent or claim CDR data."""
    tr, va, te, _ = test1_data
    for b in (tr, va, te):
        for meta in b.metadata:
            if meta.get("class_type") == "general_ppi":
                assert "cdr_interface_residues" not in meta
                assert "cdrh3_cluster" not in meta
                assert "cdrl1_residues" not in meta
                assert meta.get("is_general_ppi") is True


# -----------------------------------------------------------------------------
# Test 8 — Probability range
# -----------------------------------------------------------------------------
def test_8_probability_range(test1_data):
    """Verify that CNN inference outputs valid probabilities in [0, 1]."""
    tr, _, _, _ = test1_data
    model = build_antibody_antigen_cnn()
    probs = model.predict(tr.X[:6], verbose=0).flatten()

    assert len(probs) == 6
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    del model
    tf.keras.backend.clear_session()


# -----------------------------------------------------------------------------
# Test 9 — Confusion matrix consistency
# -----------------------------------------------------------------------------
def test_9_confusion_matrix_consistency(test1_data):
    """Verify confusion matrix arithmetic on dummy predictions."""
    from backend.app.ml.cross_validation import compute_metrics_from_predictions
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.8, 0.4, 0.2, 0.7, 0.9, 0.3])
    m = compute_metrics_from_predictions(y_true, y_prob, threshold=0.5)

    cm = m["confusion_matrix"]
    total = cm["true_positive"] + cm["false_positive"] + cm["true_negative"] + cm["false_negative"]
    assert total == len(y_true)
    assert cm["true_positive"] == 2
    assert cm["false_negative"] == 1
    assert cm["true_negative"] == 2
    assert cm["false_positive"] == 1


# -----------------------------------------------------------------------------
# Test 10 — Test 1 artifact serialization
# -----------------------------------------------------------------------------
def test_10_artifact_serialization():
    """Verify that Test 1 artifacts are created and adhere to schema in model/test1/."""
    summary_path = TEST1_DIR / "dataset_summary.json"
    eval_path = TEST1_DIR / "evaluation.json"
    preds_path = TEST1_DIR / "predictions.json"
    preds_csv = TEST1_DIR / "predictions.csv"
    report_path = TEST1_DIR / "test1_report.txt"

    assert summary_path.exists(), f"Missing {summary_path}"
    assert eval_path.exists(), f"Missing {eval_path}"
    assert preds_path.exists(), f"Missing {preds_path}"
    assert preds_csv.exists(), f"Missing {preds_csv}"
    assert report_path.exists(), f"Missing {report_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        ds = json.load(f)
    assert ds["total_samples"] == 74
    assert ds["positive_count"] == 37
    assert ds["negative_count"] == 37
    assert ds["pdb_level_disjoint"] is True
    assert ds["train_val_pdb_overlap"] == 0
    assert ds["train_test_pdb_overlap"] == 0
    assert ds["val_test_pdb_overlap"] == 0

    with open(eval_path, "r", encoding="utf-8") as f:
        ev = json.load(f)
    assert "train" in ev
    assert "validation" in ev
    assert "test" in ev
    assert ev["test"]["sample_count"] == 12


# -----------------------------------------------------------------------------
# Test 11 — Deterministic dataset construction
# -----------------------------------------------------------------------------
def test_11_deterministic_dataset_construction():
    """Verify that build_test1_dataset produces identical splits given seed 42."""
    tr1, va1, te1, _ = build_test1_dataset(seed=42)
    tr2, va2, te2, _ = build_test1_dataset(seed=42)

    assert tr1.sample_ids == tr2.sample_ids
    assert va1.sample_ids == va2.sample_ids
    assert te1.sample_ids == te2.sample_ids
    assert tr1.pdb_ids == tr2.pdb_ids
    assert te1.pdb_ids == te2.pdb_ids
    np.testing.assert_array_equal(tr1.y, tr2.y)
    np.testing.assert_array_equal(te1.y, te2.y)


# -----------------------------------------------------------------------------
# Test 12 — Phase 3 representation regression
# -----------------------------------------------------------------------------
def test_12_phase3_representation_regression():
    """Verify that approved Phase 3 representation hashes remain exact."""
    import hashlib
    from backend.app.structure import analyze_biological_entities, build_interaction_representation, resolve_structure_path

    expected_hashes = {
        "1EJO": "8d505d81e339f64194dbd02f1f96d39e979d62ff55102261eea4b75baef235f4",
        "1NBZ": "eabd0045cffb2f7988d61d9d28103d870e9f9938917b1cc6c74e18904f9ba91a",
        "1KC5": "f1f3d2218b685463ab1c29b92cb4b4d9732382a2546025dd791f7b3714d955d9",
        "1DQJ": "43a1a031a9ca13711ca47ead0a2b40b0828bbedd4fc4d8c9a0b62205ff65ec79",
        "1A14": "6f3764142075b19764e25603a28914df26cc5b5d5914a76b79436907b8002a51",
    }

    for pdb, exp_sha in expected_hashes.items():
        cif = resolve_structure_path(pdb, allow_download=False)
        comp = analyze_biological_entities(cif)
        rep = build_interaction_representation(comp)
        act_sha = rep.stats_20x21.sha256
        assert act_sha == exp_sha, f"Hash mismatch for {pdb}: {act_sha} != {exp_sha}"


# -----------------------------------------------------------------------------
# Test 13 — Real General PPI structure execution (1YZ5)
# -----------------------------------------------------------------------------
def test_13_real_general_ppi_structure_execution():
    """Verify that real general PPI structure 1YZ5 computes cleanly without error."""
    cif_file = STRUCTURES_DIR / "1YZ5.cif"
    assert cif_file.exists(), "1YZ5.cif not found in data/structures"

    tensor, meta = compute_general_ppi_representation(cif_file, "A", "B", pdb_id="1YZ5")
    assert tensor.shape == (20, 21)
    assert not np.isnan(tensor).any()
    assert not np.isinf(tensor).any()
    assert np.count_nonzero(tensor) > 0
    assert meta["p1_interface_residues"] > 0
    assert meta["p2_interface_residues"] > 0
    assert meta["p1_intra_contacts"] > 0
    assert meta["p2_intra_contacts"] > 0
    assert meta["is_general_ppi"] is True


# -----------------------------------------------------------------------------
# Test 14 — Zero sample leakage across partitions
# -----------------------------------------------------------------------------
def test_14_zero_sample_leakage(test1_data):
    """Verify strict partition independence across train, val, and test."""
    tr, va, te, _ = test1_data
    s_tr = set(tr.sample_ids)
    s_va = set(va.sample_ids)
    s_te = set(te.sample_ids)

    assert len(s_tr & s_va) == 0, "Train-Val sample leakage!"
    assert len(s_tr & s_te) == 0, "Train-Test sample leakage!"
    assert len(s_va & s_te) == 0, "Val-Test sample leakage!"


# -----------------------------------------------------------------------------
# Test 15 — Exact Phase 5 CNN architecture equivalence and spatial progression
# -----------------------------------------------------------------------------
def test_15_exact_cnn_architecture_equivalence():
    """Verify that Test 1 CNN architecture exactly matches Phase 5 production model."""
    test1_cnn = build_antibody_antigen_cnn(input_shape=(20, 21, 1), name="test1_model")
    assert test1_cnn.input_shape == (None, 20, 21, 1)
    assert test1_cnn.output_shape == (None, 1)
    assert test1_cnn.count_params() == 199681

    # Verify layer configuration
    layer_types = [layer.__class__.__name__ for layer in test1_cnn.layers]
    expected_types = [
        "Conv2D", "MaxPooling2D",
        "Conv2D", "MaxPooling2D",
        "Conv2D", "MaxPooling2D",
        "Flatten",
        "Dense",
        "Dense",
        "Dense",
    ]
    assert layer_types == expected_types

    # Verify padding is 'same' on all conv layers
    conv_layers = [l for l in test1_cnn.layers if isinstance(l, tf.keras.layers.Conv2D)]
    assert len(conv_layers) == 3
    for conv in conv_layers:
        assert conv.padding == "same"
        assert conv.kernel_size == (3, 3)
        assert conv.filters == 32

    # Verify spatial progression
    dummy = tf.zeros((1, 20, 21, 1))
    x = dummy
    shapes = [x.shape]
    for layer in test1_cnn.layers:
        x = layer(x)
        shapes.append(x.shape)

    # (None, 20, 21, 1) -> Conv -> (1, 20, 21, 32) -> Pool -> (1, 10, 11, 32)
    # -> Conv -> (1, 10, 11, 32) -> Pool -> (1, 5, 6, 32)
    # -> Conv -> (1, 5, 6, 32) -> Pool -> (1, 3, 3, 32)
    # -> Flatten -> (1, 288) -> Dense -> (1, 512) -> Dense -> (1, 64) -> Dense -> (1, 1)
    assert tuple(shapes[2]) == (1, 10, 11, 32)
    assert tuple(shapes[4]) == (1, 5, 6, 32)
    assert tuple(shapes[6]) == (1, 3, 3, 32)
    assert tuple(shapes[7]) == (1, 288)
    assert tuple(shapes[8]) == (1, 512)
    assert tuple(shapes[9]) == (1, 64)
    assert tuple(shapes[10]) == (1, 1)

    del test1_cnn
    tf.keras.backend.clear_session()


# -----------------------------------------------------------------------------
# Test 16 — PDB-level train/validation/test disjointness
# -----------------------------------------------------------------------------
def test_16_pdb_level_disjointness(test1_data):
    """Verify strict PDB-level partition independence (0 PDB overlap)."""
    tr, va, te, summary = test1_data

    tr_pdbs = set(tr.pdb_ids)
    va_pdbs = set(va.pdb_ids)
    te_pdbs = set(te.pdb_ids)

    # Disjointness across all 3 partitions
    assert len(tr_pdbs & va_pdbs) == 0, f"PDB overlap Train-Val: {tr_pdbs & va_pdbs}"
    assert len(tr_pdbs & te_pdbs) == 0, f"PDB overlap Train-Test: {tr_pdbs & te_pdbs}"
    assert len(va_pdbs & te_pdbs) == 0, f"PDB overlap Val-Test: {va_pdbs & te_pdbs}"

    # Positive and negative PDB partition checks
    tr_pos = {p for p, y in zip(tr.pdb_ids, tr.y) if y == 1}
    va_pos = {p for p, y in zip(va.pdb_ids, va.y) if y == 1}
    te_pos = {p for p, y in zip(te.pdb_ids, te.y) if y == 1}

    tr_neg = {p for p, y in zip(tr.pdb_ids, tr.y) if y == 0}
    va_neg = {p for p, y in zip(va.pdb_ids, va.y) if y == 0}
    te_neg = {p for p, y in zip(te.pdb_ids, te.y) if y == 0}

    assert len(tr_pos & va_pos) == 0, "Positive PDB Train-Val overlap!"
    assert len(tr_pos & te_pos) == 0, "Positive PDB Train-Test overlap!"
    assert len(va_pos & te_pos) == 0, "Positive PDB Val-Test overlap!"

    assert len(tr_neg & va_neg) == 0, "Negative PDB Train-Val overlap!"
    assert len(tr_neg & te_neg) == 0, "Negative PDB Train-Test overlap!"
    assert len(va_neg & te_neg) == 0, "Negative PDB Val-Test overlap!"

    # Cross-class contamination check
    all_pos_pdbs = tr_pos | va_pos | te_pos
    all_neg_pdbs = tr_neg | va_neg | te_neg
    assert len(all_pos_pdbs & all_neg_pdbs) == 0, "Cross-class PDB contamination!"
    assert len(all_pos_pdbs) == 31
    assert len(all_neg_pdbs) == 37


# -----------------------------------------------------------------------------
# Test 17 — Negative rejection-rule validity (Zhang et al. Cell 4 divide-by-zero)
# -----------------------------------------------------------------------------
def test_17_negative_rejection_rule_validity():
    """Verify that 1IB1 and 1HT2 correctly trigger the divide-by-zero normalization rejection."""
    cif_1ib1 = STRUCTURES_DIR / "1IB1.cif"
    if cif_1ib1.exists():
        with pytest.raises(ValueError) as excinfo:
            compute_general_ppi_representation(cif_1ib1, "D", "E", pdb_id="1IB1")
        assert "Zero intramolecular contacts" in str(excinfo.value)
        assert "divide-by-zero normalization failure" in str(excinfo.value)

    cif_1ht2 = STRUCTURES_DIR / "1HT2.cif"
    if cif_1ht2.exists():
        with pytest.raises(ValueError) as excinfo:
            compute_general_ppi_representation(cif_1ht2, "H", "J", pdb_id="1HT2")
        assert "Zero intramolecular contacts" in str(excinfo.value)
        assert "divide-by-zero normalization failure" in str(excinfo.value)
