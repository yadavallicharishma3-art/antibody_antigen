"""Pydantic schemas for data interchange and API contracts."""

from .analysis import (
    MatrixStats,
    ResidueItem,
    CDRDetails,
    InterfaceSummary,
    PredictionOutput,
    StructureAnalysisResult,
)
from .inference import (
    PredictRequest,
    PredictionDetail,
    StructureDetail,
    CDRDetailCounts,
    ContactDetail,
    RepresentationDetail,
    PredictResponse,
    HealthResponse,
    ErrorResponse,
)

__all__ = [
    "MatrixStats",
    "ResidueItem",
    "CDRDetails",
    "InterfaceSummary",
    "PredictionOutput",
    "StructureAnalysisResult",
    "PredictRequest",
    "PredictionDetail",
    "StructureDetail",
    "CDRDetailCounts",
    "ContactDetail",
    "RepresentationDetail",
    "PredictResponse",
    "HealthResponse",
    "ErrorResponse",
]
