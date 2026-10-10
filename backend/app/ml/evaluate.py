"""Model evaluation, metrics computation, probability diagnostics, and collapse detection."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf

from .data_loader import DatasetBundle


def compute_probability_diagnostics(
    probabilities: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate statistical diagnostics on model predicted probabilities.

    Specifically inspects whether predictions collapse into a narrow constant (e.g. ~51%)
    as observed in flawed prior implementations.
    """
    if len(probabilities) == 0:
        return {
            "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0,
            "unique_count": 0, "fraction_near_05": 0.0, "has_collapsed": False,
        }

    probs = probabilities.astype(float)
    p_min = float(np.min(probs))
    p_max = float(np.max(probs))
    p_mean = float(np.mean(probs))
    p_median = float(np.median(probs))
    p_std = float(np.std(probs))

    # Rounded unique values
    rounded = np.round(probs, decimals=4)
    unique_count = int(len(np.unique(rounded)))

    # Fraction of predictions within narrow range [0.45, 0.55]
    fraction_near_05 = float(np.mean((probs >= 0.45) & (probs <= 0.55)))

    # Collapse detection criteria:
    # 1. Standard deviation < 0.005, or
    # 2. Max - Min < 0.01, or
    # 3. > 90% of predictions within [0.49, 0.53] with unique_count <= 2
    has_collapsed = bool(
        p_std < 0.005 or
        (p_max - p_min) < 0.01 or
        (float(np.mean((probs >= 0.49) & (probs <= 0.53))) > 0.90 and unique_count <= 2)
    )

    diagnostics: Dict[str, Any] = {
        "min": p_min,
        "max": p_max,
        "mean": p_mean,
        "median": p_median,
        "std": p_std,
        "unique_count": unique_count,
        "fraction_near_05": fraction_near_05,
        "has_collapsed": has_collapsed,
        "collapse_assessment": (
            "WARNING: Probability collapse detected (predictions concentrated narrowly around constant)"
            if has_collapsed
            else "NORMAL: Predicted probabilities show responsive variation across inputs"
        ),
    }

    # Class-wise statistics if labels provided and both classes present
    if labels is not None and len(labels) == len(probs):
        unique_classes = set(np.unique(labels))
        diagnostics["class_breakdown"] = {}
        for c in [0, 1]:
            if c in unique_classes:
                c_mask = (labels == c)
                c_probs = probs[c_mask]
                diagnostics["class_breakdown"][f"class_{c}"] = {
                    "count": int(np.sum(c_mask)),
                    "min": float(np.min(c_probs)) if len(c_probs) else 0.0,
                    "max": float(np.max(c_probs)) if len(c_probs) else 0.0,
                    "mean": float(np.mean(c_probs)) if len(c_probs) else 0.0,
                    "std": float(np.std(c_probs)) if len(c_probs) else 0.0,
                }

    return diagnostics


def evaluate_model_split(
    model: tf.keras.Model,
    bundle: DatasetBundle,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate a trained CNN model on a single dataset bundle.

    Computes loss, accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
    and probability diagnostics.
    """
    if len(bundle) == 0:
        return {"split": bundle.split_name, "sample_count": 0}

    # Forward pass predictions
    y_prob = model.predict(bundle.X, verbose=0).flatten().astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    y_true = bundle.y.astype(int)

    # Compute binary cross entropy loss
    eps = 1e-7
    clipped_probs = np.clip(y_prob, eps, 1.0 - eps)
    bce = -np.mean(y_true * np.log(clipped_probs) + (1 - y_true) * np.log(1.0 - clipped_probs))
    loss_val = float(bce)

    # Accuracy
    accuracy = float(np.mean(y_pred == y_true))

    # Confusion matrix components
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if tp > 0 else 0.0)
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if tp > 0 else 0.0)
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # ROC-AUC calculation (handling single-class splits safely)
    unique_labels = np.unique(y_true)
    auc_note = None
    if len(unique_labels) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception as exc:
            roc_auc = float("nan")
            auc_note = f"ROC-AUC computation failed: {exc}"
    else:
        roc_auc = float("nan")
        auc_note = f"ROC-AUC undefined when only one class is present in split (class={unique_labels[0]})"

    # Probability diagnostics
    prob_diagnostics = compute_probability_diagnostics(y_prob, y_true)

    # Sample-level predictions
    sample_predictions = []
    for i in range(len(bundle)):
        sample_predictions.append({
            "sample_id": bundle.sample_ids[i],
            "pdb_id": bundle.pdb_ids[i],
            "true_label": int(y_true[i]),
            "pred_probability": float(y_prob[i]),
            "pred_label": int(y_pred[i]),
            "correct": bool(y_pred[i] == y_true[i]),
        })

    return {
        "split": bundle.split_name,
        "sample_count": len(bundle),
        "positive_count": int(np.sum(y_true == 1)),
        "negative_count": int(np.sum(y_true == 0)),
        "loss": loss_val,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "auc_note": auc_note,
        "threshold": threshold,
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "probability_diagnostics": prob_diagnostics,
        "sample_predictions": sample_predictions,
    }


def evaluate_all_splits(
    model: tf.keras.Model,
    train_bundle: DatasetBundle,
    val_bundle: DatasetBundle,
    test_bundle: DatasetBundle,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate trained model across train, validation, and test splits."""
    train_metrics = evaluate_model_split(model, train_bundle, threshold=threshold)
    val_metrics = evaluate_model_split(model, val_bundle, threshold=threshold)
    test_metrics = evaluate_model_split(model, test_bundle, threshold=threshold)

    return {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }


def evaluate_unseen_structure(
    model: tf.keras.Model,
    pdb_id: str = "1A14",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate trained model on an independent unseen structure."""
    from backend.app.structure import (
        analyze_biological_entities,
        build_interaction_representation,
        resolve_structure_path,
    )
    clean_pdb = pdb_id.strip().upper()
    try:
        struct_path = resolve_structure_path(clean_pdb, allow_download=False)
        comp = analyze_biological_entities(struct_path)
        rep = build_interaction_representation(comp)
        tensor_input = np.expand_dims(rep.tensor_20x21.astype(np.float32), axis=(0, -1))

        pred_prob = float(model.predict(tensor_input, verbose=0)[0, 0])
        pred_label = int(pred_prob >= threshold)

        return {
            "pdb_id": clean_pdb,
            "status": "SUCCESS",
            "prediction_probability": pred_prob,
            "prediction_label": pred_label,
            "threshold": threshold,
            "intermolecular_contacts": rep.intermolecular.contact_count,
            "cdr_interface_residues": rep.intermolecular.cdr_interface_count,
            "antigen_interface_residues": rep.intermolecular.antigen_interface_count,
            "tensor_shape": str(rep.tensor_20x21.shape),
            "representation_sha256": rep.stats_20x21.sha256,
        }
    except Exception as exc:
        return {
            "pdb_id": clean_pdb,
            "status": "FAILED",
            "error": str(exc),
        }
