"""Phase 10 Test Suite: Final System Integration, Production Pipeline & Scientific Audit.

Validates the complete end-to-end production architecture:
Test 1  — Production model file integrity, input/output shapes, 199,681 parameters.
Test 2  — 1EJO live end-to-end processing through run_pipeline_for_pdb().
Test 3  — 1A14 live end-to-end processing through run_pipeline_for_pdb().
Test 4  — INVALID PDB ID error handling (raises InvalidPdbIdError).
Test 5  — Complete Phase 3 representation hash regression (all 5 complexes exact).
Test 6  — FastAPI GET /api/health endpoint returns 200 and model_loaded: True.
Test 7  — FastAPI POST /api/predict with 1EJO returns 200 and valid schema.
Test 8  — FastAPI POST /api/predict with 1A14 returns 200 and valid schema.
Test 9  — FastAPI POST /api/predict with INVALID returns HTTP 400 with detail.
Test 10 — Inference determinism across repeated executions on 1EJO and 1A14.
Test 11 — Function contract audit (run_pipeline_for_pdb takes str, predict_interaction takes ndarray).
Test 12 — Artifact segregation (model/cv/ and model/test1/ remain preserved).
Test 13 — Frontend stack integrity (Plain HTML5, CSS, Vanilla JS, zero React/Vite/JSX).
"""

from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.ml import (
    load_inference_model,
    run_pipeline_for_pdb,
    predict_interaction,
    validate_inference_input,
    InvalidPdbIdError,
    DEFAULT_MODEL_PATH,
)
from backend.app.structure import (
    analyze_biological_entities,
    build_interaction_representation,
    resolve_structure_path,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "model"
FRONTEND_DIR = BASE_DIR / "frontend"


# -----------------------------------------------------------------------------
# Test 1 — Production model integrity
# -----------------------------------------------------------------------------
def test_1_production_model_integrity():
    """Verify that model/antibody_antigen_cnn.keras exists with exact shape and parameters."""
    assert DEFAULT_MODEL_PATH.exists(), f"Production model missing at {DEFAULT_MODEL_PATH}"
    model = load_inference_model()
    assert model.input_shape == (None, 20, 21, 1), f"Unexpected input shape: {model.input_shape}"
    assert model.output_shape == (None, 1), f"Unexpected output shape: {model.output_shape}"
    assert model.count_params() == 199681, f"Parameter count mismatch: {model.count_params()}"


# -----------------------------------------------------------------------------
# Test 2 — 1EJO live end-to-end processing
# -----------------------------------------------------------------------------
def test_2_end_to_end_1ejo():
    """Verify live structural pipeline and CNN prediction on benchmark 1EJO."""
    res = run_pipeline_for_pdb("1EJO", allow_download=False)

    assert res["success"] is True
    assert res["pdb_id"] == "1EJO"

    # Structural facts
    struct = res["structure"]
    assert "H" in struct["antibody_heavy_chains"]
    assert "L" in struct["antibody_light_chains"]
    assert "P" in struct["antigen_chains"]

    # Contacts
    contacts = res["contacts"]
    assert contacts["intermolecular"] == 233
    assert contacts["antibody_interface_residues"] == 22
    assert contacts["cdr_interface_residues"] == 17
    assert contacts["antigen_interface_residues"] == 11

    # Representation & hash
    rep = res["representation"]
    assert rep["canonical_shape"] == [20, 20]
    assert rep["cnn_shape"] == [20, 21, 1]
    assert rep["sha256"] == "8d505d81e339f64194dbd02f1f96d39e979d62ff55102261eea4b75baef235f4"

    # Prediction
    pred = res["prediction"]
    assert 0.0 <= pred["probability"] <= 1.0
    assert pred["predicted_class"] in (0, 1)


# -----------------------------------------------------------------------------
# Test 3 — 1A14 live end-to-end processing
# -----------------------------------------------------------------------------
def test_3_end_to_end_1a14():
    """Verify live structural pipeline and CNN prediction on unseen structure 1A14."""
    res = run_pipeline_for_pdb("1A14", allow_download=False)

    assert res["success"] is True
    assert res["pdb_id"] == "1A14"

    rep = res["representation"]
    assert rep["sha256"] == "6f3764142075b19764e25603a28914df26cc5b5d5914a76b79436907b8002a51"

    pred = res["prediction"]
    assert 0.0 <= pred["probability"] <= 1.0
    assert pred["predicted_class"] in (0, 1)


# -----------------------------------------------------------------------------
# Test 4 — INVALID PDB ID error handling
# -----------------------------------------------------------------------------
def test_4_invalid_pdb_error_handling():
    """Verify that malformed or non-4-character PDB inputs raise InvalidPdbIdError cleanly."""
    for bad in ["INVALID", "", "1E", "1EJOX", "12", "----"]:
        with pytest.raises(InvalidPdbIdError):
            run_pipeline_for_pdb(bad)


# -----------------------------------------------------------------------------
# Test 5 — Complete Phase 3 representation hash regression
# -----------------------------------------------------------------------------
def test_5_complete_phase3_hash_regression():
    """Verify exact preservation of all 5 approved Phase 3 representation hashes."""
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
# Test 6 — FastAPI GET /api/health endpoint
# -----------------------------------------------------------------------------
def test_6_api_health_endpoint():
    """Verify GET /api/health returns HTTP 200, status ok, and model_loaded: True."""
    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["architecture"]["trainable_parameters"] == 199681


# -----------------------------------------------------------------------------
# Test 7 — FastAPI POST /api/predict with 1EJO
# -----------------------------------------------------------------------------
def test_7_api_predict_1ejo():
    """Verify POST /api/predict with 1EJO returns HTTP 200 with complete response."""
    client = TestClient(app)
    response = client.post("/api/predict", json={"pdb_id": "1EJO"})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["pdb_id"] == "1EJO"
    assert data["representation"]["sha256"] == "8d505d81e339f64194dbd02f1f96d39e979d62ff55102261eea4b75baef235f4"
    assert len(data["representation"]["matrix_20x20"]) == 20
    assert len(data["representation"]["matrix_20x20"][0]) == 20
    assert "probability" in data["prediction"]


# -----------------------------------------------------------------------------
# Test 8 — FastAPI POST /api/predict with 1A14
# -----------------------------------------------------------------------------
def test_8_api_predict_1a14():
    """Verify POST /api/predict with 1A14 returns HTTP 200."""
    client = TestClient(app)
    response = client.post("/api/predict", json={"pdb_id": "1A14"})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["pdb_id"] == "1A14"
    assert data["representation"]["sha256"] == "6f3764142075b19764e25603a28914df26cc5b5d5914a76b79436907b8002a51"


# -----------------------------------------------------------------------------
# Test 9 — FastAPI POST /api/predict with INVALID returns HTTP 400
# -----------------------------------------------------------------------------
def test_9_api_predict_invalid_pdb():
    """Verify POST /api/predict with INVALID returns HTTP 400."""
    client = TestClient(app)
    response = client.post("/api/predict", json={"pdb_id": "INVALID"})

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert data["detail"]["error_type"] == "InvalidPdbId"


# -----------------------------------------------------------------------------
# Test 10 — Inference determinism across repeated executions
# -----------------------------------------------------------------------------
def test_10_inference_determinism():
    """Verify that repeated inference produces numerically identical results."""
    for pdb in ["1EJO", "1A14"]:
        r1 = run_pipeline_for_pdb(pdb, allow_download=False)
        r2 = run_pipeline_for_pdb(pdb, allow_download=False)

        diff = abs(r1["prediction"]["probability"] - r2["prediction"]["probability"])
        assert diff < 1e-6, f"Nondeterministic inference for {pdb}: diff = {diff}"
        assert r1["representation"]["sha256"] == r2["representation"]["sha256"]


# -----------------------------------------------------------------------------
# Test 11 — Function contract audit
# -----------------------------------------------------------------------------
def test_11_function_contract_audit():
    """Verify that abstraction layers correctly enforce input type contracts."""
    # run_pipeline_for_pdb accepts a string
    res = run_pipeline_for_pdb("1EJO", allow_download=False)
    assert isinstance(res, dict)

    # predict_interaction accepts a numeric 20x21 numpy array
    dummy_tensor = np.zeros((20, 21), dtype=np.float32)
    pred_res = predict_interaction(dummy_tensor)
    assert "probability" in pred_res

    # validate_inference_input rejects non-numeric or malformed tensor
    with pytest.raises(Exception):
        validate_inference_input("1EJO")  # String cannot be passed as tensor


# -----------------------------------------------------------------------------
# Test 12 — Artifact segregation
# -----------------------------------------------------------------------------
def test_12_artifact_segregation():
    """Verify that production, CV, and Test 1 artifacts are kept segregated."""
    assert DEFAULT_MODEL_PATH.exists()
    assert (MODEL_DIR / "cv" / "cv_summary.json").exists()
    assert (MODEL_DIR / "cv" / "oof_predictions.json").exists()
    assert (MODEL_DIR / "test1" / "dataset_summary.json").exists()
    assert (MODEL_DIR / "test1" / "evaluation.json").exists()
    assert (MODEL_DIR / "test1" / "test1_report.txt").exists()


# -----------------------------------------------------------------------------
# Test 13 — Frontend stack integrity
# -----------------------------------------------------------------------------
def test_13_frontend_stack_integrity():
    """Verify frontend consists only of Plain HTML5, Plain CSS, and Vanilla JavaScript."""
    index_html = FRONTEND_DIR / "index.html"
    styles_css = FRONTEND_DIR / "styles.css"
    app_js = FRONTEND_DIR / "app.js"

    assert index_html.exists()
    assert styles_css.exists()
    assert app_js.exists()

    with open(index_html, "r", encoding="utf-8") as f:
        html = f.read()

    # Confirm NO frontend frameworks
    assert "react" not in html.lower()
    assert "vite" not in html.lower()
    assert "vue" not in html.lower()
    assert "angular" not in html.lower()
    assert "svelte" not in html.lower()
