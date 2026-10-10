"""FastAPI API routes for antibody-antigen interaction analysis and model health checks."""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any

from backend.app.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    ErrorResponse,
)
from backend.app.ml import (
    run_pipeline_for_pdb,
    load_inference_model,
    InvalidPdbIdError,
    StructureNotFoundError,
    InvalidComplexError,
    NoValidContactsError,
    ModelLoadError,
    DEFAULT_MODEL_PATH,
)

router = APIRouter(prefix="/api", tags=["Inference"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and model readiness check",
    description="Reports the service status and whether the serialized CNN model is loaded.",
)
def health_check() -> HealthResponse:
    """Verify backend service and model readiness."""
    model_loaded = False
    arch_info = None

    try:
        model = load_inference_model()
        model_loaded = True
        arch_info = {
            "input_shape": list(model.input_shape),
            "output_shape": list(model.output_shape),
            "trainable_parameters": model.count_params(),
        }
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_path=str(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else None,
        architecture=arch_info,
        version="2.0.0",
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid PDB identifier format"},
        404: {"model": ErrorResponse, "description": "Structure file not found"},
        422: {"model": ErrorResponse, "description": "Structure is not a valid antibody-antigen complex or lacks contacts"},
        500: {"model": ErrorResponse, "description": "Model loading or server-side failure"},
    },
    summary="End-to-end antibody-antigen interaction inference",
    description=(
        "Executes the full structural pipeline (structure loading, antibody/antigen chain identification, "
        "IMGT CDR mapping, 5 A contact extraction, 20x21 representation) and computes cognate-pair "
        "classification probability using the trained CNN (Zhang et al. 2024 Test-3 paradigm)."
    ),
)
def predict_interaction_endpoint(request: PredictRequest) -> PredictResponse:
    """Run full structural analysis and ML inference on a given PDB code."""
    pdb_id = request.pdb_id

    try:
        result = run_pipeline_for_pdb(pdb_id, allow_download=True)
        return PredictResponse(**result)

    except InvalidPdbIdError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "success": False,
                "pdb_id": pdb_id,
                "error_type": "InvalidPdbId",
                "detail": str(exc),
            },
        )
    except StructureNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "success": False,
                "pdb_id": pdb_id,
                "error_type": "StructureNotFound",
                "detail": str(exc),
            },
        )
    except (InvalidComplexError, NoValidContactsError) as exc:
        err_type = "InvalidComplex" if isinstance(exc, InvalidComplexError) else "NoValidContacts"
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "success": False,
                "pdb_id": pdb_id,
                "error_type": err_type,
                "detail": str(exc),
            },
        )
    except ModelLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "pdb_id": pdb_id,
                "error_type": "ModelLoadError",
                "detail": str(exc),
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "pdb_id": pdb_id,
                "error_type": "InferencePipelineError",
                "detail": f"Unexpected error during structural inference: {exc}",
            },
        )
