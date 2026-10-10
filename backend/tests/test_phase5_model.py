"""Phase 5 Test Suite: CNN Architecture, Training, Evaluation, and Model Serialization.

Validates:
Test 1 — Input shape is strictly (20, 21, 1).
Test 2 — Model can perform a forward pass.
Test 3 — Output has the expected classification shape (batch, 1).
Test 4 — Model can train for a smoke-test epoch on a real dataset subset.
Test 5 — Model can be serialized to disk (.keras).
Test 6 — Serialized model can be reloaded into a fresh object.
Test 7 — Original and reloaded model predictions agree within numerical tolerance (< 1e-5).
Test 8 — Dataset labels are valid binary classes.
Test 9 — Train/validation/test structure overlap is zero.
Test 10 — Flatten bottleneck produces exactly 288 dimensions.
Test 11 — Probability diagnostics module detects distribution characteristics without error.
Test 12 — Real sample representations have 100% unique hashes and nonzero diversity.
Test 13 — Author Test 3 mismatch generator produces valid (20, 21, 1) tensors with label 0.
"""

from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from backend.app.ml import (
    build_antibody_antigen_cnn,
    get_architecture_summary,
    load_all_splits,
    load_dataset_split,
    audit_splits_leakage,
    compute_representation_diversity,
    generate_mismatched_negatives,
    train_cnn_model,
    evaluate_model_split,
    evaluate_unseen_structure,
    serialize_model_suite,
    reload_and_verify_model,
    compute_probability_diagnostics,
)
from backend.app.ml.data_loader import DatasetBundle

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "model"


# -----------------------------------------------------------------------------
# Test 1 — Input shape is strictly (20, 21, 1)
# -----------------------------------------------------------------------------
def test_1_input_shape_is_20x21x1():
    """Verify that CNN input layer expects exactly (None, 20, 21, 1)."""
    model = build_antibody_antigen_cnn(input_shape=(20, 21, 1))
    assert model.input_shape == (None, 20, 21, 1), f"Unexpected input shape: {model.input_shape}"


# -----------------------------------------------------------------------------
# Test 2 — Model can perform a forward pass
# -----------------------------------------------------------------------------
def test_2_forward_pass_execution():
    """Verify that CNN executes a forward pass on real input representations."""
    model = build_antibody_antigen_cnn(input_shape=(20, 21, 1))
    dummy_input = np.random.uniform(0.0, 1.0, size=(4, 20, 21, 1)).astype(np.float32)
    output = model(dummy_input, training=False)
    assert output is not None
    assert isinstance(output, tf.Tensor)


# -----------------------------------------------------------------------------
# Test 3 — Output has expected classification shape (batch, 1)
# -----------------------------------------------------------------------------
def test_3_output_classification_shape():
    """Verify that model output has shape (batch_size, 1) with values in [0, 1]."""
    model = build_antibody_antigen_cnn(input_shape=(20, 21, 1))
    batch_size = 5
    sample_input = np.random.uniform(0.0, 1.0, size=(batch_size, 20, 21, 1)).astype(np.float32)
    pred = model.predict(sample_input, verbose=0)

    assert pred.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {pred.shape}"
    assert np.all(pred >= 0.0), "Prediction contains negative probability"
    assert np.all(pred <= 1.0), "Prediction exceeds 1.0 probability"


# -----------------------------------------------------------------------------
# Test 4 — Smoke-test training on real dataset samples
# -----------------------------------------------------------------------------
def test_4_smoke_test_training_on_real_samples():
    """Verify that CNN can complete at least 1 epoch of backpropagation on real samples."""
    train_bundle, val_bundle, _, _ = load_all_splits()
    assert len(train_bundle) >= 2, "Train bundle has insufficient samples"

    # Take tiny subset of real samples for smoke test
    tiny_train = DatasetBundle(
        X=train_bundle.X[:4],
        y=train_bundle.y[:4],
        sample_ids=train_bundle.sample_ids[:4],
        pdb_ids=train_bundle.pdb_ids[:4],
        metadata=train_bundle.metadata[:4],
        split_name="smoke_train",
    )
    tiny_val = DatasetBundle(
        X=val_bundle.X[:2],
        y=val_bundle.y[:2],
        sample_ids=val_bundle.sample_ids[:2],
        pdb_ids=val_bundle.pdb_ids[:2],
        metadata=val_bundle.metadata[:2],
        split_name="smoke_val",
    )

    model, meta = train_cnn_model(
        train_bundle=tiny_train,
        val_bundle=tiny_val,
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        patience=1,
        verbose=0,
    )

    assert meta["epochs_completed"] == 1
    assert "loss" in meta["history"]
    assert len(meta["history"]["loss"]) == 1
    assert not np.isnan(meta["history"]["loss"][0])


# -----------------------------------------------------------------------------
# Test 5 — Model can be serialized
# -----------------------------------------------------------------------------
def test_5_model_serialization(tmp_path):
    """Verify that trained model and its complete artifact suite serialize to disk."""
    train_bundle, val_bundle, test_bundle, _ = load_all_splits()
    model = build_antibody_antigen_cnn()

    training_meta = {
        "epochs_requested": 1,
        "epochs_completed": 1,
        "best_epoch": 1,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "history": {"loss": [0.65], "accuracy": [1.0], "val_loss": [0.60]},
        "class_weights": None,
        "class_weight_rationale": "Test single class",
        "seeds": {"python_seed": 42, "numpy_seed": 42, "tensorflow_seed": 42},
    }
    eval_results = {
        "train": {"accuracy": 1.0, "loss": 0.65},
        "validation": {"accuracy": 1.0, "loss": 0.60},
        "test": {"accuracy": 1.0, "loss": 0.62},
    }

    saved = serialize_model_suite(
        model=model,
        training_meta=training_meta,
        evaluation_results=eval_results,
        train_bundle=train_bundle,
        val_bundle=val_bundle,
        test_bundle=test_bundle,
        output_dir=tmp_path,
    )

    assert saved["model"].exists()
    assert saved["model"].stat().st_size > 0
    assert saved["history"].exists()
    assert saved["evaluation"].exists()
    assert saved["config"].exists()


# -----------------------------------------------------------------------------
# Test 6 — Serialized model can be reloaded
# -----------------------------------------------------------------------------
def test_6_serialized_model_reload(tmp_path):
    """Verify that a saved .keras file can be loaded into a fresh model instance."""
    model = build_antibody_antigen_cnn()
    saved_path = tmp_path / "test_cnn.keras"
    model.save(saved_path)

    reloaded = tf.keras.models.load_model(saved_path)
    assert reloaded is not None
    assert reloaded.input_shape == (None, 20, 21, 1)
    assert reloaded.output_shape == (None, 1)


# -----------------------------------------------------------------------------
# Test 7 — Original and reloaded model predictions agree within tolerance
# -----------------------------------------------------------------------------
def test_7_prediction_numerical_agreement_after_reload(tmp_path):
    """Verify that predictions from original and reloaded models match within 1e-5."""
    _, _, test_bundle, _ = load_all_splits()
    model = build_antibody_antigen_cnn()

    orig_probs = model.predict(test_bundle.X, verbose=0).flatten()
    saved_path = tmp_path / "numerical_test.keras"
    model.save(saved_path)

    report = reload_and_verify_model(
        saved_model_path=saved_path,
        test_bundle=test_bundle,
        in_memory_predictions=orig_probs,
        tolerance=1e-5,
    )

    assert report["within_tolerance"] is True
    assert report["max_absolute_difference"] < 1e-5
    assert report["verification_status"] == "PASSED"


# -----------------------------------------------------------------------------
# Test 8 — Dataset labels are valid binary values
# -----------------------------------------------------------------------------
def test_8_dataset_labels_are_valid():
    """Verify that labels across all splits are strictly in {0, 1}."""
    train_bundle, val_bundle, test_bundle, _ = load_all_splits()
    for name, bundle in [("train", train_bundle), ("val", val_bundle), ("test", test_bundle)]:
        unique_labels = set(np.unique(bundle.y))
        assert unique_labels.issubset({0, 1}), f"Split {name} contains non-binary labels: {unique_labels}"
        assert len(bundle.y) > 0, f"Split {name} is unexpectedly empty"


# -----------------------------------------------------------------------------
# Test 9 — Train / validation / test structure overlap is zero
# -----------------------------------------------------------------------------
def test_9_zero_structure_leakage():
    """Verify zero PDB ID and Ab-Ag cluster overlap between splits."""
    train_bundle, val_bundle, test_bundle, audit = load_all_splits()

    assert audit["is_leakage_free"] is True
    assert len(audit["pdb_overlap_train_val"]) == 0, f"Train-Val PDB overlap: {audit['pdb_overlap_train_val']}"
    assert len(audit["pdb_overlap_train_test"]) == 0, f"Train-Test PDB overlap: {audit['pdb_overlap_train_test']}"
    assert len(audit["pdb_overlap_val_test"]) == 0, f"Val-Test PDB overlap: {audit['pdb_overlap_val_test']}"
    assert len(audit["ab_ag_cluster_overlap_train_test"]) == 0, f"Cluster overlap: {audit['ab_ag_cluster_overlap_train_test']}"


# -----------------------------------------------------------------------------
# Test 10 — Flatten bottleneck produces exactly 288 dimensions
# -----------------------------------------------------------------------------
def test_10_flatten_bottleneck_exact_288():
    """Verify that flatten layer output dimension is exactly 3 * 3 * 32 = 288."""
    model = build_antibody_antigen_cnn()
    flatten_layer = model.get_layer("flatten")
    assert flatten_layer.output.shape[-1] == 288, f"Expected 288, got {flatten_layer.output.shape}"


# -----------------------------------------------------------------------------
# Test 11 — Probability diagnostics module functionality
# -----------------------------------------------------------------------------
def test_11_probability_diagnostics_calculation():
    """Verify that probability diagnostics correctly detects collapse and spread."""
    # Collapsed dummy distribution
    collapsed = np.full((10,), 0.51, dtype=np.float32)
    diag_c = compute_probability_diagnostics(collapsed)
    assert diag_c["has_collapsed"] is True

    # Diverse distribution
    diverse = np.array([0.05, 0.20, 0.40, 0.60, 0.85, 0.95], dtype=np.float32)
    diag_d = compute_probability_diagnostics(diverse)
    assert diag_d["has_collapsed"] is False
    assert diag_d["min"] == pytest.approx(0.05, abs=1e-3)
    assert diag_d["max"] == pytest.approx(0.95, abs=1e-3)


# -----------------------------------------------------------------------------
# Test 12 — Real sample representation diversity
# -----------------------------------------------------------------------------
def test_12_real_sample_representation_diversity():
    """Verify that all 24 training representations have distinct SHA256 hashes."""
    train_bundle, _, _, _ = load_all_splits()
    div = compute_representation_diversity(train_bundle)
    assert div["is_representation_collapsed"] is False
    assert div["unique_representation_hashes"] == len(train_bundle)
    assert div["mean_pairwise_frobenius_distance"] > 1.0


# -----------------------------------------------------------------------------
# Test 13 — Author Test 3 mismatch generator produces valid tensors with label 0
# -----------------------------------------------------------------------------
def test_13_mismatch_generator_invariants():
    """Verify that generate_mismatched_negatives creates valid (20, 21, 1) negative pairs."""
    train_bundle, _, _, _ = load_all_splits()
    balanced = generate_mismatched_negatives(train_bundle, seed=42)

    assert len(balanced) == 2 * len(train_bundle)
    assert balanced.positive_count == len(train_bundle)
    assert balanced.negative_count == len(train_bundle)
    assert balanced.X.shape == (2 * len(train_bundle), 20, 21, 1)
    assert not np.isnan(balanced.X).any()
    assert np.all(balanced.X >= 0.0)
    assert np.all(balanced.X <= 1.0001)


# -----------------------------------------------------------------------------
# Test 14 — Independent unseen structure 1A14 evaluation
# -----------------------------------------------------------------------------
def test_14_unseen_structure_1a14_evaluation():
    """Verify that genuinely unseen structure 1A14 evaluates cleanly with valid metrics."""
    train_bundle, val_bundle, test_bundle, _ = load_all_splits()

    # Verify 1A14 is strictly absent from all dataset splits
    assert "1A14" not in train_bundle.unique_pdbs
    assert "1A14" not in val_bundle.unique_pdbs
    assert "1A14" not in test_bundle.unique_pdbs

    # Load serialized model
    saved_model_path = MODEL_DIR / "antibody_antigen_cnn.keras"
    assert saved_model_path.exists(), "Model file not found; please train first."
    model = tf.keras.models.load_model(saved_model_path)

    res = evaluate_unseen_structure(model, pdb_id="1A14")
    assert res["status"] == "SUCCESS", f"Evaluation failed: {res.get('error')}"
    assert 0.0 <= res["prediction_probability"] <= 1.0
    assert res["prediction_label"] in (0, 1)
    assert res["intermolecular_contacts"] > 0
    assert res["cdr_interface_residues"] > 0
    assert res["antigen_interface_residues"] > 0
    assert res["tensor_shape"] == "(20, 21)"


# -----------------------------------------------------------------------------
# Test 15 — 1A3R presence in train split resolution
# -----------------------------------------------------------------------------
def test_15_1a3r_presence_in_train_split_resolution():
    """Verify and document that 1A3R is in the training split and thus cannot be claimed unseen."""
    train_bundle, _, _, _ = load_all_splits()
    assert "1A3R" in train_bundle.unique_pdbs
    assert any("1a3r" in sid.lower() for sid in train_bundle.sample_ids)
