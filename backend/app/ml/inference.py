"""End-to-end ML inference engine for antibody-antigen interaction analysis.

Provides:
- Singleton model loader for serialized CNN artifacts.
- Validation of PDB IDs, input tensors, and macromolecular complexes.
- Full structural pipeline integration:
    PDB ID -> Structure parsing -> Entity/CDR identification -> 5 A contact extraction
    -> 20x20 canonical matrix -> 20x21 CNN tensor -> CNN prediction.
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import tensorflow as tf

from backend.app.structure import (
    analyze_biological_entities,
    build_interaction_representation,
    resolve_structure_path,
    StructureLoadError,
    ComplexData,
    RepresentationResult,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "model" / "antibody_antigen_cnn.keras"

# Global cached model singleton
_CACHED_MODEL: Optional[tf.keras.Model] = None
_CACHED_MODEL_PATH: Optional[Path] = None


class InferenceException(Exception):
    """Base exception for inference errors."""
    pass


class ModelLoadError(InferenceException):
    """Raised when the serialized CNN model cannot be loaded or is incompatible."""
    pass


class InvalidPdbIdError(InferenceException):
    """Raised when an invalid PDB identifier format is supplied."""
    pass


class StructureNotFoundError(InferenceException):
    """Raised when the requested structure cannot be found or downloaded."""
    pass


class InvalidComplexError(InferenceException):
    """Raised when the structure is not a valid antibody-antigen complex."""
    pass


class NoValidContactsError(InferenceException):
    """Raised when no intermolecular contacts <= 5.0 A exist between antibody and antigen."""
    pass


def validate_pdb_id(raw_pdb: str) -> str:
    """Validate and normalize a 4-character PDB code.

    Args:
        raw_pdb: User-supplied identifier (e.g. '1ejo', ' 1KC5 ').

    Returns:
        Clean 4-character uppercase PDB ID.

    Raises:
        InvalidPdbIdError: If the ID does not conform to the 4-alphanumeric format.
    """
    if not raw_pdb or not isinstance(raw_pdb, str):
        raise InvalidPdbIdError("PDB ID must be a non-empty string.")

    clean = raw_pdb.strip().upper()
    clean = clean.replace("PDB_", "").replace(".CIF", "").replace(".PDB", "")

    if not re.match(r"^[0-9A-Z]{4}$", clean):
        raise InvalidPdbIdError(
            f"Invalid PDB ID '{raw_pdb}'. Expected exactly 4 alphanumeric characters (e.g. '1EJO')."
        )
    return clean


def load_inference_model(
    model_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
) -> tf.keras.Model:
    """Load and verify the trained antibody-antigen CNN model artifact.

    Uses a singleton cache so the model is loaded into memory only once per process.

    Args:
        model_path: Optional custom path to .keras model file.
        force_reload: If True, bypasses cache and reloads from disk.

    Returns:
        Loaded, verified tf.keras.Model.

    Raises:
        ModelLoadError: If file is missing or architecture is incompatible.
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH

    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not force_reload and _CACHED_MODEL is not None and _CACHED_MODEL_PATH == target_path:
        return _CACHED_MODEL

    if not target_path.exists():
        raise ModelLoadError(
            f"Model artifact not found at '{target_path}'. Please train the model first."
        )

    try:
        model = tf.keras.models.load_model(target_path)
    except Exception as exc:
        raise ModelLoadError(f"Failed to deserialize Keras model at '{target_path}': {exc}") from exc

    # Architecture validation
    if model.input_shape != (None, 20, 21, 1):
        raise ModelLoadError(
            f"Incompatible model input shape: expected (None, 20, 21, 1), got {model.input_shape}"
        )

    if model.output_shape != (None, 1):
        raise ModelLoadError(
            f"Incompatible model output shape: expected (None, 1), got {model.output_shape}"
        )

    _CACHED_MODEL = model
    _CACHED_MODEL_PATH = target_path
    return _CACHED_MODEL


def validate_inference_input(tensor: np.ndarray) -> np.ndarray:
    """Validate and format a 20x21 input representation for CNN inference.

    Args:
        tensor: 2D or 3D numpy array.

    Returns:
        4D batch tensor of shape (1, 20, 21, 1) and float32 dtype.

    Raises:
        InferenceException: If shape or values are invalid.
    """
    if not isinstance(tensor, np.ndarray):
        tensor = np.array(tensor, dtype=np.float32)

    if np.isnan(tensor).any() or np.isinf(tensor).any():
        raise InferenceException("Input tensor contains NaN or Inf values.")

    if tensor.shape == (20, 21):
        formatted = np.expand_dims(tensor.astype(np.float32), axis=(0, -1))
    elif tensor.shape == (20, 21, 1):
        formatted = np.expand_dims(tensor.astype(np.float32), axis=0)
    elif tensor.shape == (1, 20, 21, 1):
        formatted = tensor.astype(np.float32)
    else:
        raise InferenceException(
            f"Invalid representation shape {tensor.shape}; expected (20, 21) or (1, 20, 21, 1)."
        )

    # Check normalized value range
    min_v, max_v = float(np.min(formatted)), float(np.max(formatted))
    if min_v < -1e-5 or max_v > 1.0001:
        raise InferenceException(
            f"Tensor values out of normalized range [0.0, 1.0]: min={min_v:.4f}, max={max_v:.4f}"
        )

    return formatted


def predict_interaction(
    tensor_20x21: np.ndarray,
    model: Optional[tf.keras.Model] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Execute model inference on an author-compatible 20x21 representation.

    Args:
        tensor_20x21: 20x21 normalized contact representation.
        model: Optional pre-loaded model (loads singleton if None).
        threshold: Classification threshold (default 0.5).

    Returns:
        Dictionary containing probability, predicted_class, class_label, interpretation.
    """
    active_model = model or load_inference_model()
    batch_input = validate_inference_input(tensor_20x21)

    raw_prediction = active_model.predict(batch_input, verbose=0)
    probability = float(raw_prediction[0, 0])
    predicted_class = int(probability >= threshold)

    class_label = "cognate_pair" if predicted_class == 1 else "mismatched_pair"
    if predicted_class == 1:
        interpretation = (
            "Predicted cognate antibody-antigen complex. Intramolecular contact distribution "
            "is consistent with native crystal complex pairing (Zhang et al. 2024 Test-3 paradigm)."
        )
    else:
        interpretation = (
            "Predicted non-cognate / mismatched antibody-antigen pairing. Intramolecular contact "
            "distribution deviates from native pairing."
        )

    return {
        "probability": probability,
        "predicted_class": predicted_class,
        "threshold": threshold,
        "class_label": class_label,
        "interpretation": interpretation,
        "task_description": (
            "Zhang et al. 2024 Test-3 cognate-vs-mismatched pairing classification. "
            "Not an experimentally calibrated binding affinity estimate."
        ),
    }


def run_pipeline_for_pdb(
    pdb_id: str,
    allow_download: bool = True,
    model: Optional[tf.keras.Model] = None,
) -> Dict[str, Any]:
    """Execute the full end-to-end structural processing and CNN inference pipeline.

    PDB ID
    -> PDB/mmCIF structure resolution
    -> Antibody and antigen identification
    -> IMGT CDR mapping
    -> 5 A side-chain heavy atom contact extraction
    -> Interface residue identification
    -> 20x20 and 20x21 representation generation
    -> CNN model inference

    Args:
        pdb_id: 4-character PDB code.
        allow_download: Whether to allow fetching from RCSB PDB if not cached.
        model: Optional preloaded model.

    Returns:
        Structured response dictionary matching PredictResponse schema.

    Raises:
        InvalidPdbIdError: If PDB ID format is invalid.
        StructureNotFoundError: If structure file cannot be found/downloaded.
        InvalidComplexError: If structure lacks antibody or antigen chains.
        NoValidContactsError: If no intermolecular contacts <= 5.0 A exist.
        ModelLoadError: If CNN model artifact is missing or invalid.
    """
    clean_pdb = validate_pdb_id(pdb_id)

    # 1. Structure resolution
    try:
        struct_path = resolve_structure_path(clean_pdb, allow_download=allow_download)
    except StructureLoadError as exc:
        raise StructureNotFoundError(
            f"Structure '{clean_pdb}' could not be located or retrieved: {exc}"
        ) from exc

    # 2. Biological entity identification & CDR mapping (Phase 2)
    try:
        complex_data: ComplexData = analyze_biological_entities(struct_path)
    except Exception as exc:
        raise InvalidComplexError(
            f"Failed to analyze biological entities for '{clean_pdb}': {exc}"
        ) from exc

    # Verify antibody chains
    heavy_id = complex_data.antibody.heavy_chain_id
    light_id = complex_data.antibody.light_chain_id
    if not heavy_id and not light_id:
        raise InvalidComplexError(
            f"Structure '{clean_pdb}' does not contain recognized antibody heavy or light chains."
        )

    # Verify antigen chains
    if not complex_data.antigen_chain_ids:
        raise InvalidComplexError(
            f"Structure '{clean_pdb}' does not contain recognized protein antigen chains."
        )

    # 3. 5 A contact extraction & representation generation (Phase 3)
    try:
        rep: RepresentationResult = build_interaction_representation(complex_data, cutoff=5.0)
    except Exception as exc:
        raise InferenceException(
            f"Failed to extract contacts or build representation for '{clean_pdb}': {exc}"
        ) from exc

    if rep.intermolecular.contact_count == 0:
        raise NoValidContactsError(
            f"Structure '{clean_pdb}' has 0 intermolecular contacts <= 5.0 A between antibody and antigen."
        )

    # 4. CNN Model Inference (Phase 5 / Phase 6)
    pred_result = predict_interaction(rep.tensor_20x21, model=model)

    # 5. Assemble response structure
    heavy_chains = [heavy_id] if heavy_id else []
    light_chains = [light_id] if light_id else []
    antigen_chains = sorted(list(complex_data.antigen_chain_ids))

    cdr_counts = {
        "H1": len(complex_data.antibody.cdrs.get("H1").residues) if "H1" in complex_data.antibody.cdrs else 0,
        "H2": len(complex_data.antibody.cdrs.get("H2").residues) if "H2" in complex_data.antibody.cdrs else 0,
        "H3": len(complex_data.antibody.cdrs.get("H3").residues) if "H3" in complex_data.antibody.cdrs else 0,
        "L1": len(complex_data.antibody.cdrs.get("L1").residues) if "L1" in complex_data.antibody.cdrs else 0,
        "L2": len(complex_data.antibody.cdrs.get("L2").residues) if "L2" in complex_data.antibody.cdrs else 0,
        "L3": len(complex_data.antibody.cdrs.get("L3").residues) if "L3" in complex_data.antibody.cdrs else 0,
    }

    return {
        "success": True,
        "pdb_id": clean_pdb,
        "prediction": pred_result,
        "structure": {
            "antibody_heavy_chains": heavy_chains,
            "antibody_light_chains": light_chains,
            "antigen_chains": antigen_chains,
        },
        "cdrs": cdr_counts,
        "contacts": {
            "intermolecular": rep.intermolecular.contact_count,
            "antibody_interface_residues": rep.intermolecular.antibody_interface_count,
            "cdr_interface_residues": rep.intermolecular.cdr_interface_count,
            "antigen_interface_residues": rep.intermolecular.antigen_interface_count,
        },
        "representation": {
            "canonical_shape": list(rep.matrix_20x20.shape),
            "cnn_shape": [20, 21, 1],
            "nonzero_count": rep.stats_20x21.nonzero_count,
            "sha256": rep.stats_20x21.sha256,
            "matrix_20x20": rep.matrix_20x20.tolist(),
        },
    }
