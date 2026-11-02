"""FastAPI web API for structure validation, prediction, and project reports."""

from __future__ import annotations

import io
import csv
import os
import re
import json
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from ml_pipeline import MODEL_PATH, predict_matrix
from structure_processing import (
    AA_ORDER,
    build_interaction_matrix,
    find_interacting_chain_pairs,
    parse_structure,
)

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
PROCESSED = ROOT / "data" / "processed"
MAX_STRUCTURE_BYTES = 100 * 1024 * 1024
PDB_ID_PATTERN = re.compile(r"(?:[A-Za-z0-9]{4}|pdb_[A-Za-z0-9]{8})\Z")

app = FastAPI(title="Antibody–Antigen Interaction Classification")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


def _download_structure(pdb_id: str) -> bytes:
    """Download a PDBx/mmCIF file from the official RCSB file service."""
    if pdb_id.lower().startswith("pdb_"):
        normalized_id = "pdb_" + pdb_id[4:].upper()
    else:
        normalized_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{normalized_id}.cif"
    request = Request(url, headers={"User-Agent": "AntibodyAntigenClassifier/1.0"})
    with urlopen(request, timeout=30) as response:
        contents = response.read(MAX_STRUCTURE_BYTES + 1)
    if len(contents) > MAX_STRUCTURE_BYTES:
        raise ValueError("The structure file exceeds the 100 MB limit.")
    return contents


def _pdb_core(value: str) -> str:
    """Normalize standard and extended PDB names to their four-character ID."""
    match = re.match(r"^(pdb_[A-Za-z0-9]{8}|[A-Za-z0-9]{4})", value.strip())
    if not match:
        return value.strip().lower()
    identifier = match.group(1).lower()
    return identifier[-4:] if identifier.startswith("pdb_") else identifier


@lru_cache(maxsize=1)
def _known_partner_assignments() -> dict[str, list[dict]]:
    """Use supplied dataset chain annotations when a PDB ID is already known."""
    manifest_path = PROCESSED / "manifest.csv"
    if not manifest_path.exists():
        return {}
    assignments: dict[str, list[dict]] = {}
    seen = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("validation_status") != "accepted":
                continue
            chain_a = [value for value in row.get("partner_a_chains", "").split(",") if value]
            chain_b = [value for value in row.get("partner_b_chains", "").split(",") if value]
            if not chain_a or not chain_b:
                continue
            key = _pdb_core(row.get("structure_id", ""))
            ranges_a = {
                chain: (tuple(bounds[0]), tuple(bounds[1]))
                for chain, bounds in json.loads(row.get("partner_a_residue_ranges") or "{}").items()
            }
            ranges_b = {
                chain: (tuple(bounds[0]), tuple(bounds[1]))
                for chain, bounds in json.loads(row.get("partner_b_residue_ranges") or "{}").items()
            }
            identity = (key, tuple(chain_a), tuple(chain_b), json.dumps(ranges_a, sort_keys=True), json.dumps(ranges_b, sort_keys=True))
            if identity in seen:
                continue
            seen.add(identity)
            assignments.setdefault(key, []).append({
                "chain_a": chain_a,
                "chain_b": chain_b,
                "ranges_a": ranges_a,
                "ranges_b": ranges_b,
            })
    return assignments


@app.get("/")
def index():
    """Serve the project frontend."""
    return FileResponse(FRONTEND / "index.html")


@app.post("/api/predict")
async def predict(
    pdb_id: str = Form(default=""),
    structure: UploadFile | None = File(default=None),
):
    """Fetch or read a structure, detect contacting chains, and classify it."""
    pdb_id = pdb_id.strip()
    has_upload = structure is not None and bool(structure.filename)
    if pdb_id and has_upload:
        return JSONResponse(
            {"error": "Enter a PDB ID or upload a structure file, not both."},
            status_code=400,
        )
    if not pdb_id and not has_upload:
        return JSONResponse({"error": "Enter a PDB ID or choose a PDB/mmCIF file."}, status_code=400)

    if pdb_id:
        if not PDB_ID_PATTERN.fullmatch(pdb_id):
            return JSONResponse(
                {"error": "Enter a valid 4-character PDB ID or an extended PDB ID."},
                status_code=400,
            )
        filename = f"{pdb_id}.cif"
        try:
            contents = await run_in_threadpool(_download_structure, pdb_id)
        except HTTPError as exc:
            if exc.code == 404:
                return JSONResponse({"error": f"PDB ID {pdb_id} was not found in RCSB."}, status_code=404)
            return JSONResponse({"error": f"RCSB could not return this structure (HTTP {exc.code})."}, status_code=502)
        except URLError as exc:
            return JSONResponse({"error": f"Could not reach RCSB: {exc.reason}"}, status_code=502)
        except TimeoutError:
            return JSONResponse({"error": "RCSB structure download timed out. Try again."}, status_code=504)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=413)
        except Exception as exc:
            return JSONResponse({"error": f"Could not retrieve the structure: {exc}"}, status_code=502)
    else:
        filename = structure.filename
        if Path(filename).suffix.lower() not in {".pdb", ".ent", ".cif", ".mmcif"}:
            return JSONResponse(
                {"error": "Choose a PDB or mmCIF structure file."},
                status_code=400,
            )
        contents = await structure.read(MAX_STRUCTURE_BYTES + 1)
        await structure.close()
        if len(contents) > MAX_STRUCTURE_BYTES:
            return JSONResponse({"error": "Structure file exceeds the 100 MB limit."}, status_code=413)
        pdb_id = Path(filename).stem

    try:
        cutoff = float(os.getenv("CONTACT_CUTOFF_ANGSTROM", "5.0"))
        model = parse_structure(io.BytesIO(contents), filename=filename)
        chain_pairs = find_interacting_chain_pairs(model, cutoff)
    except Exception as exc:
        return JSONResponse(
            {"error": str(exc), "validation_status": "rejected"},
            status_code=422,
        )

    if not chain_pairs:
        return JSONResponse(
            {"error": "No contacting protein-chain pairs were found in this structure.", "validation_status": "rejected"},
            status_code=422,
        )

    known_specs = _known_partner_assignments().get(_pdb_core(pdb_id), [])
    automatic_specs = [
        {"chain_a": [pair["chain_a"]], "chain_b": [pair["chain_b"]], "ranges_a": {}, "ranges_b": {}}
        for pair in chain_pairs
    ]

    def score_specs(specs: list[dict]) -> list[dict]:
        scored = []
        for spec in specs:
            chains_a, chains_b = spec["chain_a"], spec["chain_b"]
            ranges_a, ranges_b = spec.get("ranges_a", {}), spec.get("ranges_b", {})
            try:
                matrix_ab, metadata_ab = build_interaction_matrix(
                    model, chains_a, chains_b, cutoff_angstrom=cutoff,
                    filename=filename, sample_id=pdb_id,
                    partner_a_residue_ranges=ranges_a,
                    partner_b_residue_ranges=ranges_b,
                )
                matrix_ba, _metadata_ba = build_interaction_matrix(
                    model, chains_b, chains_a, cutoff_angstrom=cutoff,
                    filename=filename, sample_id=pdb_id,
                    partner_a_residue_ranges=ranges_b,
                    partner_b_residue_ranges=ranges_a,
                )
            except Exception:
                continue
            try:
                prediction_ab = predict_matrix(matrix_ab, MODEL_PATH)
                prediction_ba = predict_matrix(matrix_ba, MODEL_PATH)
            except FileNotFoundError as exc:
                raise RuntimeError(str(exc)) from exc
            probability = (
                prediction_ab["probability_antibody_antigen"]
                + prediction_ba["probability_antibody_antigen"]
            ) / 2
            scored.append({
                "probability": probability,
                "matrix": matrix_ab,
                "metadata": metadata_ab,
                "chain_a": chains_a,
                "chain_b": chains_b,
                "cross_partner_residue_contacts": metadata_ab["cross_partner_residue_contacts"],
            })
        return scored

    try:
        candidates = score_specs(known_specs) if known_specs else []
        if not candidates:
            candidates = score_specs(automatic_specs)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)

    if not candidates:
        return JSONResponse(
            {"error": "Contacting chains were found, but no pair passed feature validation.", "validation_status": "rejected"},
            status_code=422,
        )

    best = max(candidates, key=lambda candidate: candidate["probability"])
    probability = float(best["probability"])
    label = int(probability >= 0.5)
    class_name = "Antibody-Antigen Interaction" if label else "General Protein-Protein Interaction"
    prediction = {
        "label": label,
        "class_name": class_name,
        "confidence": max(probability, 1 - probability),
        "probability_antibody_antigen": probability,
        "interpretation": "This is a model prediction, not experimental confirmation of a biological interaction.",
        "partner_a_chains": best["chain_a"],
        "partner_b_chains": best["chain_b"],
    }

    return {
        "pdb_id": pdb_id,
        "validation_status": "accepted",
        "message": "Structure passed validation and a prediction was generated.",
        "matrix": best["matrix"].tolist(),
        "amino_acid_order": AA_ORDER,
        "feature_metadata": best["metadata"],
        "prediction": prediction,
        "candidate_interactions": len(candidates),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "5000")))
