"""Verify live server endpoints and static assets."""

import requests
import json

base = "http://127.0.0.1:8000"

print("=== 1. Static HTML Root ===")
r_root = requests.get(f"{base}/")
print(f"Status: {r_root.status_code}, Length: {len(r_root.text)}, Has Title: {'Antibody' in r_root.text}")
assert r_root.status_code == 200
assert "index.html" in r_root.text or "Antibody" in r_root.text

print("\n=== 2. Static CSS ===")
r_css = requests.get(f"{base}/styles.css")
print(f"Status: {r_css.status_code}, Length: {len(r_css.text)}, Has CSS rules: {':root' in r_css.text}")
assert r_css.status_code == 200
assert ":root" in r_css.text

print("\n=== 3. Static JS ===")
r_js = requests.get(f"{base}/app.js")
print(f"Status: {r_js.status_code}, Length: {len(r_js.text)}, Has JS functions: {'checkBackendHealth' in r_js.text}")
assert r_js.status_code == 200
assert "checkBackendHealth" in r_js.text

print("\n=== 4. API Health Endpoint ===")
r_health = requests.get(f"{base}/api/health")
print(f"Status: {r_health.status_code}, Body: {r_health.json()}")
assert r_health.status_code == 200
assert r_health.json()["model_loaded"] is True

print("\n=== 5. API Predict (1EJO) ===")
r_pred1 = requests.post(f"{base}/api/predict", json={"pdb_id": "1EJO"})
print(f"Status: {r_pred1.status_code}")
p1 = r_pred1.json()
print(f"1EJO: prob={p1['prediction']['probability']:.4f}, class={p1['prediction']['predicted_class']}, contacts={p1['contacts']['intermolecular']}, CDR_iface={p1['contacts']['cdr_interface_residues']}, has_matrix={len(p1['representation']['matrix_20x20']) == 20}")
assert r_pred1.status_code == 200
assert p1["prediction"]["predicted_class"] == 1

print("\n=== 6. API Predict (1A14 - Unseen) ===")
r_pred2 = requests.post(f"{base}/api/predict", json={"pdb_id": "1A14"})
print(f"Status: {r_pred2.status_code}")
p2 = r_pred2.json()
print(f"1A14: prob={p2['prediction']['probability']:.4f}, class={p2['prediction']['predicted_class']}, contacts={p2['contacts']['intermolecular']}, CDR_iface={p2['contacts']['cdr_interface_residues']}, has_matrix={len(p2['representation']['matrix_20x20']) == 20}")
assert r_pred2.status_code == 200
assert p2["prediction"]["predicted_class"] == 0

print("\n=== 7. API Predict (INVALID PDB) ===")
r_pred3 = requests.post(f"{base}/api/predict", json={"pdb_id": "INVALID"})
print(f"Status: {r_pred3.status_code}, Error: {r_pred3.json()}")
assert r_pred3.status_code == 400

print("\n>>> ALL LIVE FRONTEND & BACKEND INTEGRATION CHECKS PASSED SUCCESSFULLY! <<<")
