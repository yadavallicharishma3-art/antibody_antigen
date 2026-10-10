"""FastAPI Application entrypoint for Antibody-Antigen Interaction Analyzer V2.

Exposes:
- GET  /api/health  Model status and architecture check
- POST /api/predict Full structural preprocessing + CNN inference
- Static Frontend   Mounted at / and /app for interactive research tool UI
"""

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.ml import load_inference_model

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model artifact into memory upon service startup."""
    try:
        load_inference_model()
    except Exception:
        # Allow app to start even if model is not yet trained,
        # so /api/health can report unready status clearly.
        pass
    yield


app = FastAPI(
    title="Antibody-Antigen Interaction Analyzer V2",
    description=(
        "Production ML inference backend reproducing Zhang et al. 2024 "
        "structural interaction analysis. Predicts cognate vs mismatched antibody-antigen pairing "
        "from 5 A intermolecular contact graphs and intramolecular amino-acid pair frequencies."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes first
app.include_router(router)

# Mount static frontend application
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
