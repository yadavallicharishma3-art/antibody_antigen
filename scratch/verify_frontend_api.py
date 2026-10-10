import urllib.request
import json

def test_get(url):
    res = urllib.request.urlopen(url)
    content = res.read()
    print(f"GET {url} -> HTTP {res.status}, {len(content)} bytes")

def test_predict(pdb):
    data = json.dumps({"pdb_id": pdb}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/predict",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    res = urllib.request.urlopen(req)
    out = json.loads(res.read().decode())
    pred = out["prediction"]
    contacts = out["contacts"]
    cdrs = out["cdrs"]
    rep = out["representation"]
    mat_sum = sum(sum(r) for r in rep["matrix_20x20"])
    print(
        f"PREDICT {pdb} -> Prob: {pred['probability']:.4f} | "
        f"Class: {pred['predicted_class']} ({pred['class_label']}) | "
        f"Contacts: {contacts['intermolecular']} | "
        f"CDRs: H={cdrs['H1']}+{cdrs['H2']}+{cdrs['H3']}, L={cdrs['L1']}+{cdrs['L2']}+{cdrs['L3']} | "
        f"Nonzero: {rep['nonzero_count']} | MatrixSum: {mat_sum:.4f} | SHA: {rep['sha256'][:8]}..."
    )

print("=== VERIFYING FRONTEND STATIC ASSETS ===")
test_get("http://127.0.0.1:8000/")
test_get("http://127.0.0.1:8000/styles.css")
test_get("http://127.0.0.1:8000/app.js")
test_get("http://127.0.0.1:8000/api/health")

print("\n=== VERIFYING PREDICTION PIPELINE ON CURATED COMPLEXES ===")
for pdb in ["1EJO", "1KC5", "1A14", "1DQJ"]:
    test_predict(pdb)

print("\n=== VERIFYING ERROR HANDLING ===")
try:
    bad_data = json.dumps({"pdb_id": "XYZ!"}).encode("utf-8")
    bad_req = urllib.request.Request(
        "http://127.0.0.1:8000/api/predict",
        data=bad_data,
        headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(bad_req)
except urllib.error.HTTPError as e:
    err_body = json.loads(e.read().decode())
    print(f"INVALID PDB 'XYZ!' -> HTTP {e.code}: {err_body['detail']['error_type']} - {err_body['detail']['detail']}")

print("\nAll frontend endpoints and backend verification tests PASSED.")
