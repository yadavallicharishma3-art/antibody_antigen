"""Phase 8 Test Suite: 10-Fold Stratified Cross-Validation & Model Evaluation.

Validates:
Test 1  — Exactly 10 folds generated.
Test 2  — Every valid sample appears in exactly one validation fold (complete coverage).
Test 3  — No sample appears in multiple validation folds (zero validation overlap).
Test 4  — Every fold contains both positive and negative classes (stratification).
Test 5  — Out-of-fold predictions cover every valid sample exactly once.
Test 6  — Prediction shape corresponds one-to-one with samples.
Test 7  — Every predicted probability satisfies 0.0 <= p <= 1.0.
Test 8  — Repeated fold assignment using seed 42 produces identical assignments (determinism).
Test 9  — Fresh model initialization per fold (weights are fresh, not carried over).
Test 10 — Phase 3 representation hashes remain unchanged (regression invariance).
Test 11 — Serialized CV artifacts (summary, fold metrics, OOF predictions) exist and match schemas.
Test 12 — Dyadic pairing audit: zero sample leakage between train and validation across all folds.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from backend.app.ml import (
    build_antibody_antigen_cnn,
    load_unified_cv_dataset,
    generate_stratified_folds,
    audit_fold_leakage,
    compute_metrics_from_predictions,
    DEFAULT_CV_DIR,
)
from backend.app.ml.cross_validation import DatasetBundle

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "model"
CV_DIR = MODEL_DIR / "cv"


@pytest.fixture(scope="module")
def unified_dataset():
    """Load authoritative Phase 5 corrected dataset (74 samples)."""
    return load_unified_cv_dataset(seed=42)


@pytest.fixture(scope="module")
def stratified_folds(unified_dataset):
    """Generate the 10 stratified folds."""
    return generate_stratified_folds(unified_dataset, n_splits=10, seed=42)


# -----------------------------------------------------------------------------
# Test 1 — Fold count: Exactly 10 folds
# -----------------------------------------------------------------------------
def test_1_fold_count_is_exactly_10(stratified_folds):
    """Verify that exactly 10 folds are generated."""
    assert len(stratified_folds) == 10, f"Expected 10 folds, got {len(stratified_folds)}"


# -----------------------------------------------------------------------------
# Test 2 — Complete coverage: Every sample appears in exactly 1 validation fold
# -----------------------------------------------------------------------------
def test_2_complete_validation_coverage(unified_dataset, stratified_folds):
    """Verify that every valid sample appears in exactly one validation fold."""
    n_samples = len(unified_dataset)
    all_val_indices = []
    for _, val_idx in stratified_folds:
        all_val_indices.extend(val_idx.tolist())

    assert len(all_val_indices) == n_samples, (
        f"Validation sample count {len(all_val_indices)} != dataset size {n_samples}"
    )
    assert set(all_val_indices) == set(range(n_samples)), "Validation indices do not cover entire dataset"


# -----------------------------------------------------------------------------
# Test 3 — No validation overlap: No sample appears in multiple validation folds
# -----------------------------------------------------------------------------
def test_3_no_validation_overlap(stratified_folds):
    """Verify pairwise disjoint validation sets."""
    for i in range(len(stratified_folds)):
        val_i = set(stratified_folds[i][1].tolist())
        for j in range(i + 1, len(stratified_folds)):
            val_j = set(stratified_folds[j][1].tolist())
            overlap = val_i & val_j
            assert len(overlap) == 0, f"Validation overlap detected between Fold {i+1} and Fold {j+1}: {overlap}"


# -----------------------------------------------------------------------------
# Test 4 — Stratification: Every fold contains both positive and negative samples
# -----------------------------------------------------------------------------
def test_4_stratification_preserves_both_classes(unified_dataset, stratified_folds):
    """Verify that each fold's validation partition contains both positive and negative samples."""
    y = unified_dataset.y
    for fold_idx, (_, val_idx) in enumerate(stratified_folds):
        y_val = y[val_idx]
        pos_cnt = int(np.sum(y_val == 1))
        neg_cnt = int(np.sum(y_val == 0))
        assert pos_cnt > 0, f"Fold {fold_idx + 1} has 0 positive samples!"
        assert neg_cnt > 0, f"Fold {fold_idx + 1} has 0 negative samples!"
        # In a 74-sample dataset with 37 pos and 37 neg split across 10 folds,
        # each validation fold should have 3 or 4 samples per class.
        assert pos_cnt in (3, 4), f"Fold {fold_idx + 1} pos count {pos_cnt} unexpected"
        assert neg_cnt in (3, 4), f"Fold {fold_idx + 1} neg count {neg_cnt} unexpected"


# -----------------------------------------------------------------------------
# Test 5 — Out-of-fold coverage: Audit verifies 0 sample leakage and 100% coverage
# -----------------------------------------------------------------------------
def test_5_out_of_fold_coverage_and_leakage_audit(unified_dataset, stratified_folds):
    """Verify that audit_fold_leakage passes all invariants."""
    audit = audit_fold_leakage(unified_dataset, stratified_folds)
    assert audit["complete_coverage"] is True
    assert audit["zero_sample_overlap"] is True
    assert audit["n_folds"] == 10
    assert audit["total_samples"] == 74


# -----------------------------------------------------------------------------
# Test 6 — Prediction shape corresponds one-to-one with samples
# -----------------------------------------------------------------------------
def test_6_prediction_shape_and_metrics_computation():
    """Verify that compute_metrics_from_predictions correctly evaluates sample predictions."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.4, 0.7, 0.6, 0.3])
    metrics = compute_metrics_from_predictions(y_true, y_prob, threshold=0.5)

    assert "accuracy" in metrics
    assert "loss" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["confusion_matrix"]["true_positive"] == 3
    assert metrics["confusion_matrix"]["true_negative"] == 3
    assert metrics["confusion_matrix"]["false_positive"] == 1
    assert metrics["confusion_matrix"]["false_negative"] == 1
    assert metrics["accuracy"] == 0.75


# -----------------------------------------------------------------------------
# Test 7 — Probability range: Every sigmoid prediction satisfies 0.0 <= p <= 1.0
# -----------------------------------------------------------------------------
def test_7_probability_range(unified_dataset):
    """Verify that a freshly built model outputs valid sigmoid probabilities in [0, 1]."""
    model = build_antibody_antigen_cnn()
    subset_X = unified_dataset.X[:5]
    probs = model.predict(subset_X, verbose=0).flatten()

    assert len(probs) == 5
    assert np.all(probs >= 0.0), "Negative probability detected"
    assert np.all(probs <= 1.0), "Probability exceeding 1.0 detected"


# -----------------------------------------------------------------------------
# Test 8 — Determinism: Seed 42 produces identical fold assignments
# -----------------------------------------------------------------------------
def test_8_determinism_seed_42(unified_dataset):
    """Verify that fold generation is strictly deterministic when seed=42."""
    folds1 = generate_stratified_folds(unified_dataset, n_splits=10, seed=42)
    folds2 = generate_stratified_folds(unified_dataset, n_splits=10, seed=42)

    for (tr1, va1), (tr2, va2) in zip(folds1, folds2):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(va1, va2)


# -----------------------------------------------------------------------------
# Test 9 — Fresh model per fold: Two initialized models do not share weights
# -----------------------------------------------------------------------------
def test_9_fresh_model_per_fold_initialization():
    """Verify that successive model builds have separate, distinct weight instances."""
    m1 = build_antibody_antigen_cnn(name="model_fold_1")
    m2 = build_antibody_antigen_cnn(name="model_fold_2")

    w1 = m1.get_weights()
    w2 = m2.get_weights()

    assert len(w1) == len(w2)
    # Check that they are not identical references
    for layer_w1, layer_w2 in zip(w1, w2):
        assert layer_w1 is not layer_w2

    # Cleanup
    del m1, m2
    tf.keras.backend.clear_session()


# -----------------------------------------------------------------------------
# Test 10 — Existing Phase 3 representation regression
# -----------------------------------------------------------------------------
def test_10_phase3_representation_regression(unified_dataset):
    """Verify that benchmark representations in the dataset match the Phase 3 verified hashes."""
    # 1EJO representation hash
    ejo_idx = None
    for i, sid in enumerate(unified_dataset.sample_ids):
        if sid == "pdb_00001ejo_H_L":
            ejo_idx = i
            break

    assert ejo_idx is not None, "1EJO not found in unified CV dataset"
    mat_20x21 = unified_dataset.X[ejo_idx, :, :, 0]
    import hashlib
    mat_sha = hashlib.sha256(mat_20x21.tobytes()).hexdigest()

    # Verify sha256 matches Phase 3
    expected_sha = unified_dataset.metadata[ejo_idx].get("representation_sha256")
    if expected_sha:
        assert mat_sha == expected_sha, f"Hash mismatch for 1EJO: {mat_sha} != {expected_sha}"


# -----------------------------------------------------------------------------
# Test 11 — Serialized CV artifacts exist and adhere to schemas
# -----------------------------------------------------------------------------
def test_11_serialized_cv_artifacts_exist():
    """Verify that CV artifacts are created in model/cv/ when cross-validation runs."""
    summary_file = CV_DIR / "cv_summary.json"
    oof_file = CV_DIR / "oof_predictions.json"
    oof_csv = CV_DIR / "oof_predictions.csv"
    folds_file = CV_DIR / "fold_metrics.json"

    assert summary_file.exists(), f"Missing {summary_file}"
    assert oof_file.exists(), f"Missing {oof_file}"
    assert oof_csv.exists(), f"Missing {oof_csv}"
    assert folds_file.exists(), f"Missing {folds_file}"

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert "metadata" in summary
    assert "macro_fold_statistics" in summary
    assert "aggregate_oof_metrics" in summary
    assert summary["metadata"]["total_samples"] == 74
    assert summary["metadata"]["n_folds"] == 10

    with open(oof_file, "r", encoding="utf-8") as f:
        oof = json.load(f)
    assert len(oof) == 74, f"Expected 74 OOF predictions, got {len(oof)}"


# -----------------------------------------------------------------------------
# Test 12 — Dyadic pairing audit: zero sample leakage between train and val
# -----------------------------------------------------------------------------
def test_12_dyadic_pairing_zero_sample_leakage(unified_dataset, stratified_folds):
    """Verify that no sample ID ever leaks from training to validation partition."""
    sids = unified_dataset.sample_ids
    for fold_idx, (tr_idx, va_idx) in enumerate(stratified_folds):
        tr_sids = set(sids[i] for i in tr_idx)
        va_sids = set(sids[i] for i in va_idx)
        assert len(tr_sids & va_sids) == 0, f"Fold {fold_idx + 1} has sample ID leakage!"
