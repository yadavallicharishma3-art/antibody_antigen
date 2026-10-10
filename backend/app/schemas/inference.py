"""Pydantic schemas for the Phase 6 inference API and scientific diagnostics."""

import re
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Request payload for antibody-antigen interaction inference."""
    pdb_id: str = Field(
        ...,
        description="4-character PDB identifier (e.g. '1EJO', '1KC5', '1A14')",
        examples=["1EJO"],
    )


class PredictionDetail(BaseModel):
    """Detailed model prediction output."""
    probability: float = Field(..., description="Cognate-pair classification probability from CNN sigmoid activation")
    predicted_class: int = Field(..., description="Binary classification (1 = cognate native pair, 0 = non-cognate mismatched pair)")
    threshold: float = Field(0.5, description="Classification decision threshold")
    class_label: str = Field(..., description="Scientific label ('cognate_pair' or 'mismatched_pair')")
    interpretation: str = Field(..., description="Scientific interpretation note")
    task_description: str = Field(
        default="Zhang et al. 2024 Test-3 cognate-vs-mismatched pairing classification. Not an experimental binding affinity estimate.",
        description="Explicit scientific scope of the classification task",
    )


class StructureDetail(BaseModel):
    """Identified macromolecular chains and biological entities."""
    antibody_heavy_chains: List[str] = Field(default_factory=list, description="Identified antibody heavy chain IDs")
    antibody_light_chains: List[str] = Field(default_factory=list, description="Identified antibody light chain IDs")
    antigen_chains: List[str] = Field(default_factory=list, description="Identified protein antigen chain IDs")


class CDRDetailCounts(BaseModel):
    """Residue counts across mapped CDR loops."""
    H1: int = Field(0, description="Residue count in CDR-H1")
    H2: int = Field(0, description="Residue count in CDR-H2")
    H3: int = Field(0, description="Residue count in CDR-H3")
    L1: int = Field(0, description="Residue count in CDR-L1")
    L2: int = Field(0, description="Residue count in CDR-L2")
    L3: int = Field(0, description="Residue count in CDR-L3")


class ContactDetail(BaseModel):
    """Counts of structural contacts and interface residues."""
    intermolecular: int = Field(..., description="Count of intermolecular atom contacts <= 5.0 A (side-chain heavy atoms only)")
    antibody_interface_residues: int = Field(..., description="Unique antibody residues at the interface")
    cdr_interface_residues: int = Field(..., description="Subset of antibody interface residues located within CDR loops")
    antigen_interface_residues: int = Field(..., description="Unique antigen residues at the interface")


class RepresentationDetail(BaseModel):
    """Details of the generated 2D structural matrix and CNN input tensor."""
    canonical_shape: List[int] = Field(default=[20, 20], description="Shape of canonical upper-triangle matrix [20, 20]")
    cnn_shape: List[int] = Field(default=[20, 21, 1], description="Shape of author-compatible CNN input tensor [20, 21, 1]")
    nonzero_count: int = Field(..., description="Number of non-zero entries in the 20x21 tensor")
    sha256: str = Field(..., description="SHA256 hash of the 20x21 tensor for reproducibility auditing")
    matrix_20x20: Optional[List[List[float]]] = Field(None, description="20x20 normalized contact frequency matrix")


class PredictResponse(BaseModel):
    """Complete scientific response returned by the /api/predict endpoint."""
    success: bool = Field(True, description="Whether the inference pipeline executed successfully")
    pdb_id: str = Field(..., description="Normalized 4-character PDB ID")
    prediction: PredictionDetail
    structure: StructureDetail
    cdrs: CDRDetailCounts
    contacts: ContactDetail
    representation: RepresentationDetail


class HealthResponse(BaseModel):
    """Response payload for the /api/health endpoint."""
    status: str = Field("ok", description="Service health status")
    model_loaded: bool = Field(..., description="Whether the trained CNN model artifact is loaded and verified")
    model_path: Optional[str] = Field(None, description="Path to the serialized model artifact")
    architecture: Optional[Dict[str, Any]] = Field(None, description="Summary of verified CNN input/output shapes")
    version: str = Field("2.0.0", description="API version")


class ErrorResponse(BaseModel):
    """Standardized error payload returned on pipeline failures."""
    success: bool = Field(False, description="Failure indicator")
    pdb_id: Optional[str] = Field(None, description="PDB ID if available")
    error_type: str = Field(..., description="Classification of the error (e.g. InvalidPdbId, StructureNotFound)")
    detail: str = Field(..., description="Detailed explanation of the failure")
