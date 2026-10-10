"""Phase 6 Test Suite: Real End-to-End ML Inference Pipeline & FastAPI Endpoints.

Validates:
Test 1 — Serialized CNN model loads cleanly from disk.
Test 2 — Model input shape is (None, 20, 21, 1) and output is (None, 1).
Test 3 — Known benchmark structure 1EJO executes full pipeline dynamically.
Test 4 — Benchmark structure 1KC5 executes full pipeline independently.
Test 5 — Genuinely unseen structure 1A14 executes full pipeline cleanly.
Test 6 — Invalid PDB ID formats raise InvalidPdbIdError / HTTP 400.
Test 7 — Missing or corrupted model path raises ModelLoadError cleanly.
Test 8 — Representation tensor consistency (20x20 canonical, 20x21 CNN, 1x20x21x1 batch).
Test 9 — Numerical determinism: identical structure produces identical predictions.
Test 10 — Critical Phase 3 representation regression across 4 benchmark structures.
Test 11 — Model reload verification: fresh disk reload yields identical prediction.
Test 12 — Scientific sanity check: 1EJO, 1KC5, and 1A14 produce distinct structural representations.
Test 13 — FastAPI GET /api/health endpoint returns model readiness.
Test 14 — FastAPI POST /api/predict returns valid payload with scientific interpretation.
Test 15 — FastAPI POST /api/predict handles invalid PDB IDs and non-existent structures.
"""

from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.ml.inference import (
    load_inference_model,
    predict_interaction,
    run_pipeline_for_pdb,
    validate_pdb_id,
    validate_inference_input,
    ModelLoadError,
    InvalidPdbIdError,
    StructureNotFoundError,
    DEFAULT_MODEL_PATH,
)
from backend.app.structure import (
    analyze_biological_entities,
    build_interaction_representation,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "model" / "antibody_antigen_cnn.keras"

client = TestClient(app)


# -----------------------------------------------------------------------------
# Test 1 — Serialized CNN model loads cleanly from disk
# -----------------------------------------------------------------------------
def test_1_model_loads_successfully():
    """Verify that the production serialized model exists and loads cleanly."""
    assert MODEL_PATH.exists(), f"Model artifact missing at {MODEL_PATH}"
    model = load_inference_model(MODEL_PATH, force_reload=True)
    assert model is not None
    assert hasattr(model, "predict")


# -----------------------------------------------------------------------------
# Test 2 — Model input shape is (None, 20, 21, 1) and output is (None, 1)
# -----------------------------------------------------------------------------
def test_2_model_input_and_output_shapes():
    """Verify strict compatibility with the 20x21 CNN architecture."""
    model = load_inference_model(MODEL_PATH)
    assert model.input_shape == (None, 20, 21, 1), f"Unexpected input shape: {model.input_shape}"
    assert model.output_shape == (None, 1), f"Unexpected output shape: {model.output_shape}"


# -----------------------------------------------------------------------------
# Test 3 — Known benchmark structure 1EJO executes full pipeline dynamically
# -----------------------------------------------------------------------------
def test_3_known_benchmark_1ejo_pipeline():
    """Verify complete dynamic pipeline execution on benchmark 1EJO."""
    res = run_pipeline_for_pdb("1EJO", allow_download=False)

    assert res["success"] is True
    assert res["pdb_id"] == "1EJO"
    assert "H" in res["structure"]["antibody_heavy_chains"]
    assert "L" in res["structure"]["antibody_light_chains"]
    assert "P" in res["structure"]["antigen_chains"]

    # Contact counts dynamically extracted
    assert res["contacts"]["intermolecular"] > 50
    assert res["contacts"]["antibody_interface_residues"] > 5
    assert res["contacts"]["cdr_interface_residues"] > 5
    assert res["contacts"]["antigen_interface_residues"] > 5

    # Representation shapes
    assert res["representation"]["canonical_shape"] == [20, 20]
    assert res["representation"]["cnn_shape"] == [20, 21, 1]
    assert res["representation"]["nonzero_count"] > 10

    # Prediction
    prob = res["prediction"]["probability"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    assert res["prediction"]["predicted_class"] in (0, 1)
    assert res["prediction"]["threshold"] == 0.5
    assert "cognate" in res["prediction"]["class_label"] or "mismatched" in res["prediction"]["class_label"]
    assert "Test-3" in res["prediction"]["task_description"]


# -----------------------------------------------------------------------------
# Test 4 — Benchmark structure 1KC5 executes full pipeline independently
# -----------------------------------------------------------------------------
def test_4_benchmark_1kc5_pipeline():
    """Verify independent dynamic pipeline execution on benchmark 1KC5."""
    res = run_pipeline_for_pdb("1KC5", allow_download=False)

    assert res["success"] is True
    assert res["pdb_id"] == "1KC5"
    assert res["contacts"]["intermolecular"] == 120
    assert res["contacts"]["cdr_interface_residues"] == 10
    assert res["representation"]["nonzero_count"] == 10

    prob = res["prediction"]["probability"]
    assert 0.0 <= prob <= 1.0
    assert res["prediction"]["predicted_class"] in (0, 1)


# -----------------------------------------------------------------------------
# Test 5 — Genuinely unseen structure 1A14 executes full pipeline cleanly
# -----------------------------------------------------------------------------
def test_5_unseen_structure_1a14_pipeline():
    """Verify inference on 1A14 (held-out complex absent from train/val/test)."""
    res = run_pipeline_for_pdb("1A14", allow_download=False)

    assert res["success"] is True
    assert res["pdb_id"] == "1A14"
    assert res["contacts"]["intermolecular"] == 184
    assert res["contacts"]["cdr_interface_residues"] == 15
    assert res["representation"]["nonzero_count"] == 21

    prob = res["prediction"]["probability"]
    assert 0.0 <= prob <= 1.0
    assert res["prediction"]["predicted_class"] in (0, 1)


# -----------------------------------------------------------------------------
# Test 6 — Invalid PDB ID formats raise InvalidPdbIdError
# -----------------------------------------------------------------------------
def test_6_invalid_pdb_id_format_errors():
    """Verify rejection of malformed or invalid PDB codes."""
    for invalid in ["INVALID", "toolong123", "12", ""]:
        with pytest.raises(InvalidPdbIdError):
            validate_pdb_id(invalid)


# -----------------------------------------------------------------------------
# Test 7 — Missing or corrupted model path raises ModelLoadError cleanly
# -----------------------------------------------------------------------------
def test_7_missing_model_path_error():
    """Verify ModelLoadError is raised when model file does not exist."""
    non_existent = BASE_DIR / "model" / "non_existent_model.keras"
    with pytest.raises(ModelLoadError) as exc_info:
        load_inference_model(non_existent, force_reload=True)
    assert "not found" in str(exc_info.value).lower()


# -----------------------------------------------------------------------------
# Test 8 — Representation tensor consistency
# -----------------------------------------------------------------------------
def test_8_tensor_consistency_and_normalization():
    """Verify that validated inference input is strictly (1, 20, 21, 1) and in [0, 1]."""
    comp = analyze_biological_entities("1EJO")
    rep = build_interaction_representation(comp)

    # Validate 2D to 4D transformation
    tensor_4d = validate_inference_input(rep.tensor_20x21)
    assert tensor_4d.shape == (1, 20, 21, 1)
    assert tensor_4d.dtype == np.float32
    assert not np.isnan(tensor_4d).any()
    assert not np.isinf(tensor_4d).any()
    assert float(np.min(tensor_4d)) >= 0.0
    assert float(np.max(tensor_4d)) <= 1.0001


# -----------------------------------------------------------------------------
# Test 9 — Numerical determinism
# -----------------------------------------------------------------------------
def test_9_numerical_determinism():
    """Running inference twice on the same structure yields identical outputs."""
    res1 = run_pipeline_for_pdb("1EJO", allow_download=False)
    res2 = run_pipeline_for_pdb("1EJO", allow_download=False)

    assert res1["prediction"]["probability"] == pytest.approx(res2["prediction"]["probability"], abs=1e-6)
    assert res1["prediction"]["predicted_class"] == res2["prediction"]["predicted_class"]
    assert res1["representation"]["sha256"] == res2["representation"]["sha256"]


# -----------------------------------------------------------------------------
# Test 10 — Critical Phase 3 representation regression across 4 benchmark structures
# -----------------------------------------------------------------------------
def test_10_phase3_representation_regression():
    """Verify Phase 6 representation generation exactly reproduces Phase 3 SHA256 hashes."""
    expected_stats = {
        "1EJO": {"contacts": 233, "cdr_iface": 17, "ag_iface": 11, "nonzero": 29, "prefix": "8d505d81e339f641"},
        "1NBZ": {"contacts": 278, "cdr_iface": 17, "ag_iface": 16, "nonzero": 21, "prefix": "eabd0045cffb2f79"},
        "1KC5": {"contacts": 120, "cdr_iface": 10, "ag_iface": 6, "nonzero": 10, "prefix": "f1f3d2218b685463"},
        "1DQJ": {"contacts": 315, "cdr_iface": 16, "ag_iface": 16, "nonzero": 23, "prefix": "43a1a031a9ca1371"},
    }

    for pdb_id, exp in expected_stats.items():
        res = run_pipeline_for_pdb(pdb_id, allow_download=False)
        assert res["contacts"]["intermolecular"] == exp["contacts"]
        assert res["contacts"]["cdr_interface_residues"] == exp["cdr_iface"]
        assert res["contacts"]["antigen_interface_residues"] == exp["ag_iface"]
        assert res["representation"]["nonzero_count"] == exp["nonzero"]
        assert res["representation"]["sha256"].startswith(exp["prefix"])


# -----------------------------------------------------------------------------
# Test 11 — Model reload verification
# -----------------------------------------------------------------------------
def test_11_model_reload_agreement():
    """Predictions from cached model and freshly reloaded disk model agree within 1e-6."""
    model_cached = load_inference_model(force_reload=False)
    comp = analyze_biological_entities("1EJO")
    rep = build_interaction_representation(comp)

    pred1 = predict_interaction(rep.tensor_20x21, model=model_cached)

    model_fresh = load_inference_model(force_reload=True)
    pred2 = predict_interaction(rep.tensor_20x21, model=model_fresh)

    assert pred1["probability"] == pytest.approx(pred2["probability"], abs=1e-6)
    assert pred1["predicted_class"] == pred2["predicted_class"]


# -----------------------------------------------------------------------------
# Test 12 — Scientific sanity check: 1EJO, 1KC5, 1A14 produce distinct representations
# -----------------------------------------------------------------------------
def test_12_scientific_sanity_diverse_representations():
    """Verify that distinct structures produce distinct representations and hashes."""
    structures = ["1EJO", "1KC5", "1A14"]
    hashes = set()
    nonzero_counts = set()

    for pdb in structures:
        res = run_pipeline_for_pdb(pdb, allow_download=False)
        hashes.add(res["representation"]["sha256"])
        nonzero_counts.add(res["representation"]["nonzero_count"])

    assert len(hashes) == len(structures), f"Duplicate hashes detected across structures: {hashes}"
    assert len(nonzero_counts) == len(structures), f"Identical nonzero counts detected: {nonzero_counts}"


# -----------------------------------------------------------------------------
# Test 13 — FastAPI GET /api/health endpoint
# -----------------------------------------------------------------------------
def test_13_api_health_endpoint():
    """Verify /api/health reports online status and model readiness."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["architecture"]["input_shape"] == [None, 20, 21, 1]
    assert data["architecture"]["output_shape"] == [None, 1]


# -----------------------------------------------------------------------------
# Test 14 — FastAPI POST /api/predict with 1EJO
# -----------------------------------------------------------------------------
def test_14_api_predict_benchmark_1ejo():
    """Verify /api/predict returns full validated JSON for benchmark 1EJO."""
    response = client.post("/api/predict", json={"pdb_id": "1EJO"})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["pdb_id"] == "1EJO"
    assert "prediction" in data
    assert "structure" in data
    assert "contacts" in data
    assert "representation" in data
    assert 0.0 <= data["prediction"]["probability"] <= 1.0
    assert data["contacts"]["intermolecular"] == 233


# -----------------------------------------------------------------------------
# Test 15 — FastAPI POST /api/predict handles errors cleanly
# -----------------------------------------------------------------------------
def test_15_api_predict_error_handling():
    """Verify /api/predict returns HTTP 400 for malformed PDB IDs and 404 for missing structures."""
    # Malformed PDB ID -> HTTP 400
    resp_invalid = client.post("/api/predict", json={"pdb_id": "INVALID"})
    assert resp_invalid.status_code == 400

    # Non-existent PDB ID (with download failure) -> HTTP 404
    resp_notfound = client.post("/api/predict", json={"pdb_id": "0ZZZ"})
    assert resp_notfound.status_code == 404
