"""Model serialization, reproducibility configuration export, and reload verification."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import tensorflow as tf

from .cnn import get_architecture_summary
from .data_loader import DatasetBundle

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "model"

AMINO_ACIDS_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
]


def serialize_model_suite(
    model: tf.keras.Model,
    training_meta: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    train_bundle: DatasetBundle,
    val_bundle: DatasetBundle,
    test_bundle: DatasetBundle,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Serialize the trained Keras model, training history, evaluation metrics, and config."""
    out_dir = output_dir or DEFAULT_MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "antibody_antigen_cnn.keras"
    history_path = out_dir / "training_history.json"
    evaluation_path = out_dir / "evaluation.json"
    config_path = out_dir / "model_config.json"

    # 1. Save Keras model in native format
    model.save(model_path)

    # 2. Save training history
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_meta.get("history", {}), f, indent=2)

    # 3. Save evaluation metrics (clean NaN values for standard JSON)
    def _clean_for_json(obj: Any) -> Any:
        if isinstance(obj, float):
            if np.isnan(obj):
                return None
            if np.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return _clean_for_json(obj.tolist())
        if isinstance(obj, dict):
            return {k: _clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean_for_json(x) for x in obj]
        return obj

    clean_eval = _clean_for_json(evaluation_results)
    with open(evaluation_path, "w", encoding="utf-8") as f:
        json.dump(clean_eval, f, indent=2)

    # 4. Save comprehensive reproducibility config
    arch_summary = get_architecture_summary(model)
    config = {
        "project": "Antibody-Antigen Interaction Analyzer V2",
        "phase": 5,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "paper_reference": "Zhang et al., 2024 (BioSystems 243:105264)",
        "representation_version": "Phase3_Author_Compatible_20x21",
        "input_shape": [20, 21, 1],
        "amino_acid_ordering": AMINO_ACIDS_ORDER,
        "normalization": "Per-complex min-max independent normalization ([0, 1] range)",
        "contact_definition": "5.0 A heavy atom distance cutoff between side chains",
        "cdr_definition": "IMGT numbering from SAbDab metadata",
        "model_architecture": arch_summary,
        "optimizer": {
            "name": model.optimizer.__class__.__name__,
            "learning_rate": training_meta.get("learning_rate", 1e-4),
        },
        "loss": "binary_crossentropy",
        "training_configuration": {
            "epochs_requested": training_meta.get("epochs_requested"),
            "epochs_completed": training_meta.get("epochs_completed"),
            "best_epoch": training_meta.get("best_epoch"),
            "batch_size": training_meta.get("batch_size"),
            "early_stopping_monitor": training_meta.get("early_stopping_monitor"),
            "early_stopping_patience": training_meta.get("early_stopping_patience"),
            "class_weights": training_meta.get("class_weights"),
            "class_weight_rationale": training_meta.get("class_weight_rationale"),
            "seeds": training_meta.get("seeds"),
        },
        "dataset_split_statistics": {
            "train_samples": len(train_bundle),
            "train_unique_pdbs": len(train_bundle.unique_pdbs),
            "train_positives": train_bundle.positive_count,
            "train_negatives": train_bundle.negative_count,
            "val_samples": len(val_bundle),
            "val_unique_pdbs": len(val_bundle.unique_pdbs),
            "val_positives": val_bundle.positive_count,
            "val_negatives": val_bundle.negative_count,
            "test_samples": len(test_bundle),
            "test_unique_pdbs": len(test_bundle.unique_pdbs),
            "test_positives": test_bundle.positive_count,
            "test_negatives": test_bundle.negative_count,
        },
        "saved_artifacts": {
            "model_path": str(model_path.resolve().relative_to(DEFAULT_MODEL_DIR.parent.resolve())) if model_path.resolve().is_relative_to(DEFAULT_MODEL_DIR.parent.resolve()) else str(model_path),
            "history_path": str(history_path.resolve().relative_to(DEFAULT_MODEL_DIR.parent.resolve())) if history_path.resolve().is_relative_to(DEFAULT_MODEL_DIR.parent.resolve()) else str(history_path),
            "evaluation_path": str(evaluation_path.resolve().relative_to(DEFAULT_MODEL_DIR.parent.resolve())) if evaluation_path.resolve().is_relative_to(DEFAULT_MODEL_DIR.parent.resolve()) else str(evaluation_path),
            "config_path": str(config_path.resolve().relative_to(DEFAULT_MODEL_DIR.parent.resolve())) if config_path.resolve().is_relative_to(DEFAULT_MODEL_DIR.parent.resolve()) else str(config_path),
        },
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return {
        "model": model_path,
        "history": history_path,
        "evaluation": evaluation_path,
        "config": config_path,
    }


def reload_and_verify_model(
    saved_model_path: Path,
    test_bundle: DatasetBundle,
    in_memory_predictions: Optional[np.ndarray] = None,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """Reload serialized model into fresh object and verify prediction equality within tolerance."""
    if not saved_model_path.exists():
        raise FileNotFoundError(f"Serialized model not found at {saved_model_path}")

    # Load fresh model instance
    reloaded_model = tf.keras.models.load_model(saved_model_path)

    # Compute predictions
    reloaded_probs = reloaded_model.predict(test_bundle.X, verbose=0).flatten().astype(float)

    if in_memory_predictions is not None:
        orig_probs = in_memory_predictions.flatten().astype(float)
        diffs = np.abs(orig_probs - reloaded_probs)
        max_diff = float(np.max(diffs))
        mean_diff = float(np.mean(diffs))
        within_tolerance = bool(max_diff <= tolerance)
    else:
        max_diff = 0.0
        mean_diff = 0.0
        within_tolerance = True

    return {
        "saved_model_path": str(saved_model_path),
        "test_sample_count": len(test_bundle),
        "reloaded_input_shape": list(reloaded_model.input_shape),
        "reloaded_output_shape": list(reloaded_model.output_shape),
        "max_absolute_difference": max_diff,
        "mean_absolute_difference": mean_diff,
        "tolerance": tolerance,
        "within_tolerance": within_tolerance,
        "verification_status": "PASSED" if within_tolerance else "FAILED",
        "sample_reloaded_probabilities": [float(p) for p in reloaded_probs[:5]],
    }
