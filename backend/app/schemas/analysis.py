"""Data models and schemas for structural analysis, feature representations, and CNN inference."""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class MatrixStats(BaseModel):
    """Statistical summary of 20x20 contact matrix for scientific auditing."""
    shape: List[int] = Field(..., description="Shape of the matrix, e.g. [20, 20]")
    min_val: float = Field(..., description="Minimum value in normalized matrix")
    max_val: float = Field(..., description="Maximum value in normalized matrix")
    mean_val: float = Field(..., description="Mean value across all matrix elements")
    std_val: float = Field(..., description="Standard deviation across all matrix elements")
    sum_val: float = Field(..., description="Sum of all matrix elements")
    nonzero_count: int = Field(..., description="Number of non-zero cells")
    sha256: str = Field(..., description="SHA256 hash of rounded float representation")


class ResidueItem(BaseModel):
    """Structural residue identifier."""
    chain_id: str
    res_seq: int
    res_name: str
    insertion_code: Optional[str] = " "


class CDRDetails(BaseModel):
    """Complementarity-determining region annotation."""
    cdr_name: str = Field(..., description="L1, L2, L3, H1, H2, H3")
    chain_id: str
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    residues: List[ResidueItem] = []


class InterfaceSummary(BaseModel):
    """Summary of intermolecular contacts and interface residue definitions."""
    antibody_residues: List[ResidueItem] = []
    antigen_residues: List[ResidueItem] = []
    intermolecular_contact_count: int = 0
    antibody_intramolecular_contact_count: int = 0
    antigen_intramolecular_contact_count: int = 0


class PredictionOutput(BaseModel):
    """Detailed model prediction output exposing raw and calibrated tensors."""
    experiment_id: str = Field(..., description="test1, test2, or test3")
    predicted_class: int = Field(..., description="Binary predicted class index (0 or 1)")
    class_label: str = Field(..., description="Scientific label of predicted class")
    raw_output: float = Field(..., description="Raw output logit/activation before rounding")
    probability: float = Field(..., description="Estimated probability of class 1")
    input_shape: List[int] = Field(..., description="Tensor shape fed to the CNN")


class StructureAnalysisResult(BaseModel):
    """Complete scientific response containing intermediate structural features and prediction."""
    pdb_id: str
    structure_source: str = Field(..., description="Local cache, AbDb, SAbDab, or RCSB PDB")
    antibody_chains: List[str] = []
    antigen_chains: List[str] = []
    cdrs: Dict[str, CDRDetails] = {}
    interface: InterfaceSummary
    matrix: List[List[float]] = Field(..., description="20x20 normalized contact frequency matrix")
    matrix_stats: MatrixStats
    prediction: Optional[PredictionOutput] = None
