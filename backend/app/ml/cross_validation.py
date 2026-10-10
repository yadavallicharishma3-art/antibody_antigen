"""10-Fold Stratified Cross-Validation workflow for antibody-antigen interaction CNN.

Implements the Zhang et al. 2024 Test-3-style cross-validation evaluation:
- Real Phase 5 corrected dataset (37 positive, 37 negative, 74 total samples, 31 unique PDBs).
- Stratified 10-fold cross-validation preserving class distribution.
- Strict isolation: fresh CNN initialization per fold, no weight sharing or cross-fold leakage.
- Comprehensive fold-level, macro-averaged, and aggregate out-of-fold (OOF) metrics.
- Probability diagnostics and collapse monitoring.
- Export of metrics, predictions, and assignments to model/cv/.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from .cnn import build_antibody_antigen_cnn
from .data_loader import (
    DatasetBundle,
    load_all_splits,
    generate_mismatched_negatives,
)
from .evaluate import (
    compute_probability_diagnostics,
)
from .train import (
    set_reproducibility_seeds,
    compute_training_class_weights,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CV_DIR = BASE_DIR / "model" / "cv"


def load_unified_cv_dataset(
    base_dir: Optional[Path] = None,
    seed: int = 42,
) -> DatasetBundle:
    """Load the authoritative Phase 5 corrected dataset (74 samples: 37 pos, 37 neg).

    Reconstructs the balanced cognate and non-cognate pairs from Phase 4 splits
    using the exact within-split mismatch generation from Phase 5.
    """
    base = base_dir or BASE_DIR
    train_bundle, val_bundle, test_bundle, _ = load_all_splits(base_dir=base)

    # Generate author-compatible within-split mismatched negatives
    train_b = generate_mismatched_negatives(train_bundle, seed=seed)
    val_b = generate_mismatched_negatives(val_bundle, seed=seed)
    test_b = generate_mismatched_negatives(test_bundle, seed=seed)

    all_X = np.concatenate([train_b.X, val_b.X, test_b.X], axis=0)
    all_y = np.concatenate([train_b.y, val_b.y, test_b.y], axis=0)
    all_samples = train_b.sample_ids + val_b.sample_ids + test_b.sample_ids
    all_pdbs = train_b.pdb_ids + val_b.pdb_ids + test_b.pdb_ids
    all_meta = train_b.metadata + val_b.metadata + test_b.metadata

    return DatasetBundle(
        X=all_X,
        y=all_y,
        sample_ids=all_samples,
        pdb_ids=all_pdbs,
        metadata=all_meta,
        split_name="unified_cv_74",
    )


def generate_stratified_folds(
    bundle: DatasetBundle,
    n_splits: int = 10,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition the dataset into n_splits stratified folds with fixed seed."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(bundle.X, bundle.y))


def audit_fold_leakage(
    bundle: DatasetBundle,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """Audit cross-validation folds for data leakage, coverage, and class balance.

    Invariants checked:
    - 0 sample overlap between training and validation sets for every fold.
    - Exactly 100% out-of-fold sample coverage (every sample in exactly 1 validation fold).
    - No sample present in multiple validation folds.
    - Class counts per fold.
    - Dyadic PDB structure overlap audit.
    """
    n_total = len(bundle)
    all_val_indices: List[int] = []
    fold_audits = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())

        # 1. Sample overlap between train and val
        sample_overlap = train_set & val_set
        assert len(sample_overlap) == 0, f"Sample overlap in fold {fold_idx + 1}: {sample_overlap}"

        # 2. Check sample IDs
        train_sids = {bundle.sample_ids[i] for i in train_idx}
        val_sids = {bundle.sample_ids[i] for i in val_idx}
        sid_overlap = train_sids & val_sids
        assert len(sid_overlap) == 0, f"Sample ID overlap in fold {fold_idx + 1}: {sid_overlap}"

        all_val_indices.extend(val_idx.tolist())

        # Class counts
        y_val = bundle.y[val_idx]
        y_tr = bundle.y[train_idx]
        pos_val = int(np.sum(y_val == 1))
        neg_val = int(np.sum(y_val == 0))
        pos_tr = int(np.sum(y_tr == 1))
        neg_tr = int(np.sum(y_tr == 0))

        # PDB overlap audit
        pos_pdbs_val = {bundle.pdb_ids[i] for i in val_idx if bundle.y[i] == 1}
        pos_pdbs_tr = {bundle.pdb_ids[i] for i in train_idx if bundle.y[i] == 1}
        pos_pdb_overlap = sorted(list(pos_pdbs_val & pos_pdbs_tr))

        fold_audits.append({
            "fold": fold_idx + 1,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "train_pos": pos_tr,
            "train_neg": neg_tr,
            "val_pos": pos_val,
            "val_neg": neg_val,
            "val_sample_ids": sorted(list(val_sids)),
            "pos_pdb_overlap": pos_pdb_overlap,
            "has_sample_overlap": False,
        })

    # Coverage checks
    unique_val = set(all_val_indices)
    assert len(unique_val) == n_total, f"Incomplete validation coverage: {len(unique_val)} != {n_total}"
    assert len(all_val_indices) == n_total, f"Validation indices contain duplicates: {len(all_val_indices)} != {n_total}"

    return {
        "total_samples": n_total,
        "n_folds": len(folds),
        "complete_coverage": True,
        "zero_sample_overlap": True,
        "fold_audits": fold_audits,
    }


def compute_metrics_from_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute binary classification metrics with zero fabrication."""
    y_pred = (y_prob >= threshold).astype(int)

    # Binary cross-entropy loss
    eps = 1e-7
    clipped = np.clip(y_prob, eps, 1.0 - eps)
    bce = -np.mean(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped))
    loss_val = float(bce)

    # Accuracy
    acc = float(np.mean(y_pred == y_true))

    # Confusion matrix
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # ROC-AUC
    unique_labels = np.unique(y_true)
    auc_note = None
    if len(unique_labels) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception as exc:
            auc = float("nan")
            auc_note = str(exc)
    else:
        auc = float("nan")
        auc_note = f"Undefined with single class {unique_labels[0]}"

    return {
        "loss": loss_val,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "auc_note": auc_note,
        "threshold": threshold,
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "matrix": [[tn, fp], [fn, tp]],
        },
    }


def train_and_evaluate_single_fold(
    fold_idx: int,
    train_bundle: DatasetBundle,
    val_bundle: DatasetBundle,
    epochs: int = 60,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    patience: int = 15,
    seed: int = 42,
    monitor: str = "val_loss",
    dropout_rate: float = 0.0,
    threshold: float = 0.5,
    verbose: int = 0,
) -> Dict[str, Any]:
    """Train a fresh CNN instance on one fold and evaluate on held-out validation data.

    Invariants enforced:
    - Fresh model initialized from scratch with distinct weights.
    - Trained exclusively on fold's training bundle.
    - Early stopping monitored on validation loss with restore_best_weights=True.
    - Model discarded after fold evaluation.
    """
    fold_seed = seed + fold_idx * 100
    set_reproducibility_seeds(fold_seed)

    # Compute class weights strictly on fold training data
    class_weights, _ = compute_training_class_weights(train_bundle.y)

    # 1. Fresh model instance
    model = build_antibody_antigen_cnn(
        input_shape=(20, 21, 1),
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        name=f"cnn_fold_{fold_idx + 1}",
    )

    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=verbose,
        )
    ]

    # 2. Train on fold training set
    history = model.fit(
        train_bundle.X,
        train_bundle.y,
        validation_data=(val_bundle.X, val_bundle.y),
        epochs=epochs,
        batch_size=min(batch_size, len(train_bundle)),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=verbose,
    )

    epochs_completed = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history[monitor])) + 1 if monitor in history.history else epochs_completed

    # 3. Predict on held-out fold
    val_probs = model.predict(val_bundle.X, verbose=0).flatten().astype(float)
    val_metrics = compute_metrics_from_predictions(val_bundle.y, val_probs, threshold=threshold)

    # Also compute training metrics for overfit assessment
    tr_probs = model.predict(train_bundle.X, verbose=0).flatten().astype(float)
    tr_metrics = compute_metrics_from_predictions(train_bundle.y, tr_probs, threshold=threshold)

    # Probability diagnostics
    val_prob_diag = compute_probability_diagnostics(val_probs, val_bundle.y)
    tr_prob_diag = compute_probability_diagnostics(tr_probs, train_bundle.y)

    # Sample predictions for OOF collection
    sample_preds = []
    for i in range(len(val_bundle)):
        p = float(val_probs[i])
        lbl = int(p >= threshold)
        y_t = int(val_bundle.y[i])
        sample_preds.append({
            "sample_id": val_bundle.sample_ids[i],
            "pdb_id": val_bundle.pdb_ids[i],
            "true_label": y_t,
            "pred_probability": p,
            "pred_label": lbl,
            "correct": bool(lbl == y_t),
            "fold": fold_idx + 1,
        })

    # Discard model explicitly
    del model
    tf.keras.backend.clear_session()

    return {
        "fold": fold_idx + 1,
        "seed": fold_seed,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "train_samples": len(train_bundle),
        "val_samples": len(val_bundle),
        "validation_metrics": val_metrics,
        "training_metrics": tr_metrics,
        "validation_probability_diagnostics": val_prob_diag,
        "training_probability_diagnostics": tr_prob_diag,
        "sample_predictions": sample_preds,
    }


def run_stratified_cross_validation(
    bundle: Optional[DatasetBundle] = None,
    n_splits: int = 10,
    epochs: int = 60,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    patience: int = 15,
    seed: int = 42,
    threshold: float = 0.5,
    dropout_rate: float = 0.0,
    verbose: int = 0,
) -> Dict[str, Any]:
    """Execute complete 10-fold stratified cross-validation on the antibody-antigen CNN.

    Returns:
        Structured dictionary containing:
        - metadata & configuration
        - leakage audit
        - fold-by-fold metrics
        - macro/mean summary across folds (mean ± std, min, max)
        - aggregate out-of-fold metrics and confusion matrix
        - complete out-of-fold predictions
        - probability diagnostics
    """
    dataset = bundle or load_unified_cv_dataset(seed=seed)
    folds = generate_stratified_folds(dataset, n_splits=n_splits, seed=seed)
    leakage_audit = audit_fold_leakage(dataset, folds)

    fold_results = []
    all_oof_predictions: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_bundle = DatasetBundle(
            X=dataset.X[train_idx],
            y=dataset.y[train_idx],
            sample_ids=[dataset.sample_ids[i] for i in train_idx],
            pdb_ids=[dataset.pdb_ids[i] for i in train_idx],
            metadata=[dataset.metadata[i] for i in train_idx],
            split_name=f"cv_train_fold_{fold_idx + 1}",
        )
        val_bundle = DatasetBundle(
            X=dataset.X[val_idx],
            y=dataset.y[val_idx],
            sample_ids=[dataset.sample_ids[i] for i in val_idx],
            pdb_ids=[dataset.pdb_ids[i] for i in val_idx],
            metadata=[dataset.metadata[i] for i in val_idx],
            split_name=f"cv_val_fold_{fold_idx + 1}",
        )

        fold_res = train_and_evaluate_single_fold(
            fold_idx=fold_idx,
            train_bundle=train_bundle,
            val_bundle=val_bundle,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            seed=seed,
            dropout_rate=dropout_rate,
            threshold=threshold,
            verbose=verbose,
        )
        fold_results.append(fold_res)
        all_oof_predictions.extend(fold_res["sample_predictions"])

    # Verify OOF prediction coverage
    assert len(all_oof_predictions) == len(dataset), (
        f"Mismatch in OOF predictions count: {len(all_oof_predictions)} vs {len(dataset)}"
    )

    # 1. Macro-averaged / fold statistics
    metric_keys = ["accuracy", "loss", "roc_auc", "precision", "recall", "f1"]
    fold_stats = {}
    for k in metric_keys:
        vals = [f["validation_metrics"][k] for f in fold_results if not np.isnan(f["validation_metrics"][k])]
        if vals:
            fold_stats[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "folds_computed": len(vals),
            }
        else:
            fold_stats[k] = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "folds_computed": 0}

    # 2. Aggregate OOF metrics (all 74 held-out predictions evaluated together)
    oof_y_true = np.array([p["true_label"] for p in all_oof_predictions], dtype=int)
    oof_y_prob = np.array([p["pred_probability"] for p in all_oof_predictions], dtype=float)

    aggregate_oof_metrics = compute_metrics_from_predictions(oof_y_true, oof_y_prob, threshold=threshold)
    aggregate_oof_prob_diag = compute_probability_diagnostics(oof_y_prob, oof_y_true)

    return {
        "metadata": {
            "task": "Zhang et al. 2024 Test-3-style cognate-vs-mismatched pairing classification",
            "evaluation_method": "10-Fold Stratified Cross-Validation",
            "total_samples": len(dataset),
            "positive_samples": dataset.positive_count,
            "negative_samples": dataset.negative_count,
            "unique_pdbs": len(dataset.unique_pdbs),
            "n_folds": n_splits,
            "random_seed": seed,
            "training_configuration": {
                "optimizer": "Adam",
                "learning_rate": learning_rate,
                "epochs_max": epochs,
                "batch_size": batch_size,
                "early_stopping_patience": patience,
                "early_stopping_monitor": "val_loss",
                "decision_threshold": threshold,
                "dropout_rate": dropout_rate,
            },
        },
        "leakage_audit": leakage_audit,
        "fold_results": fold_results,
        "macro_fold_statistics": fold_stats,
        "aggregate_oof_metrics": aggregate_oof_metrics,
        "aggregate_oof_probability_diagnostics": aggregate_oof_prob_diag,
        "oof_predictions": all_oof_predictions,
    }


def save_cross_validation_artifacts(
    cv_results: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Serialize cross-validation metrics, predictions, and summaries to disk."""
    out_dir = output_dir or DEFAULT_CV_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_file = out_dir / "cv_summary.json"
    folds_file = out_dir / "fold_metrics.json"
    oof_json_file = out_dir / "oof_predictions.json"
    oof_csv_file = out_dir / "oof_predictions.csv"
    assignments_file = out_dir / "fold_assignments.json"
    report_file = out_dir / "cv_report.txt"

    # 1. Summary JSON
    summary_data = {
        "metadata": cv_results["metadata"],
        "macro_fold_statistics": cv_results["macro_fold_statistics"],
        "aggregate_oof_metrics": cv_results["aggregate_oof_metrics"],
        "aggregate_oof_probability_diagnostics": cv_results["aggregate_oof_probability_diagnostics"],
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 2. Fold metrics JSON
    with open(folds_file, "w", encoding="utf-8") as f:
        json.dump(cv_results["fold_results"], f, indent=2)

    # 3. OOF predictions JSON & CSV
    with open(oof_json_file, "w", encoding="utf-8") as f:
        json.dump(cv_results["oof_predictions"], f, indent=2)

    df_oof = pd.DataFrame(cv_results["oof_predictions"])
    df_oof.to_csv(oof_csv_file, index=False)

    # 4. Fold assignments
    assignments = []
    for f in cv_results["leakage_audit"]["fold_audits"]:
        assignments.append({
            "fold": f["fold"],
            "train_size": f["train_size"],
            "val_size": f["val_size"],
            "val_pos": f["val_pos"],
            "val_neg": f["val_neg"],
            "val_sample_ids": f["val_sample_ids"],
            "pos_pdb_overlap": f["pos_pdb_overlap"],
        })
    with open(assignments_file, "w", encoding="utf-8") as f:
        json.dump(assignments, f, indent=2)

    # 5. Human-readable text report
    s = cv_results["macro_fold_statistics"]
    agg = cv_results["aggregate_oof_metrics"]
    diag = cv_results["aggregate_oof_probability_diagnostics"]
    cm = agg["confusion_matrix"]

    lines = [
        "=" * 72,
        "PHASE 8: 10-FOLD STRATIFIED CROSS-VALIDATION EVALUATION REPORT",
        "=" * 72,
        f"Task: {cv_results['metadata']['task']}",
        f"Evaluation: {cv_results['metadata']['evaluation_method']} (Seed: {cv_results['metadata']['random_seed']})",
        f"Dataset: {cv_results['metadata']['total_samples']} samples ({cv_results['metadata']['positive_samples']} Pos, {cv_results['metadata']['negative_samples']} Neg, {cv_results['metadata']['unique_pdbs']} unique PDBs)",
        "",
        "-" * 72,
        "1. MACRO-AVERAGED FOLD METRICS (Mean ± Std, [Min, Max]):",
        "-" * 72,
        f"Accuracy:  {s['accuracy']['mean'] * 100:6.2f}% ± {s['accuracy']['std'] * 100:5.2f}%   [{s['accuracy']['min'] * 100:5.2f}%, {s['accuracy']['max'] * 100:5.2f}%]",
        f"ROC-AUC:   {s['roc_auc']['mean']:7.4f}  ± {s['roc_auc']['std']:6.4f}   [{s['roc_auc']['min']:6.4f}, {s['roc_auc']['max']:6.4f}]",
        f"Loss:      {s['loss']['mean']:7.4f}  ± {s['loss']['std']:6.4f}   [{s['loss']['min']:6.4f}, {s['loss']['max']:6.4f}]",
        f"Precision: {s['precision']['mean']:7.4f}  ± {s['precision']['std']:6.4f}   [{s['precision']['min']:6.4f}, {s['precision']['max']:6.4f}]",
        f"Recall:    {s['recall']['mean']:7.4f}  ± {s['recall']['std']:6.4f}   [{s['recall']['min']:6.4f}, {s['recall']['max']:6.4f}]",
        f"F1-Score:  {s['f1']['mean']:7.4f}  ± {s['f1']['std']:6.4f}   [{s['f1']['min']:6.4f}, {s['f1']['max']:6.4f}]",
        "",
        "-" * 72,
        "2. AGGREGATE OUT-OF-FOLD (OOF) EVALUATION (All 74 samples):",
        "-" * 72,
        f"OOF Accuracy:   {agg['accuracy'] * 100:.2f}% ({cm['true_positive'] + cm['true_negative']}/{len(df_oof)})",
        f"OOF ROC-AUC:    {agg['roc_auc']:.4f}",
        f"OOF Precision:  {agg['precision']:.4f}",
        f"OOF Recall:     {agg['recall']:.4f}",
        f"OOF F1-Score:   {agg['f1']:.4f}",
        f"OOF Loss (BCE): {agg['loss']:.4f}",
        f"Confusion Matrix: TP={cm['true_positive']}, FP={cm['false_positive']}, TN={cm['true_negative']}, FN={cm['false_negative']}",
        "",
        "-" * 72,
        "3. PROBABILITY DISTRIBUTION & COLLAPSE DIAGNOSTICS:",
        "-" * 72,
        f"Min Probability:     {diag['min']:.4f}",
        f"Max Probability:     {diag['max']:.4f}",
        f"Mean Probability:    {diag['mean']:.4f}",
        f"Median Probability:  {diag['median']:.4f}",
        f"Standard Deviation:  {diag['std']:.4f}",
        f"Unique Values (4d):  {diag['unique_count']}",
        f"Fraction in [0.45, 0.55]: {diag['fraction_near_05'] * 100:.1f}%",
        f"Assessment:          {diag['collapse_assessment']}",
        "",
        "-" * 72,
        "4. FOLD-BY-FOLD DETAILED RESULTS:",
        "-" * 72,
    ]

    for fr in cv_results["fold_results"]:
        vm = fr["validation_metrics"]
        vcm = vm["confusion_matrix"]
        pd_ = fr["validation_probability_diagnostics"]
        lines.append(
            f"Fold {fr['fold']:2d} | Val N={fr['val_samples']:2d} | Acc={vm['accuracy']*100:5.1f}% | "
            f"AUC={vm['roc_auc']:6.4f} | Loss={vm['loss']:6.4f} | Prec={vm['precision']:6.4f} | Rec={vm['recall']:6.4f} | F1={vm['f1']:6.4f} | "
            f"TP={vcm['true_positive']} FP={vcm['false_positive']} TN={vcm['true_negative']} FN={vcm['false_negative']} | "
            f"Prob=[{pd_['min']:.3f}, {pd_['max']:.3f}] Epochs={fr['epochs_completed']}"
        )

    lines.extend([
        "",
        "=" * 72,
        "SCIENTIFIC INTERPRETATION & METHODOLOGICAL LIMITATIONS:",
        "=" * 72,
        "1. This evaluation measures 10-fold cross-validation on the Test-3-style cognate-vs-mismatched classification task.",
        "2. The dataset is small (74 samples across 31 unique PDBs) relative to the 199,681 CNN parameter capacity.",
        "3. Negative pairs are synthetic non-cognate mismatches (decoys), NOT experimentally verified non-binders.",
        "4. Model outputs reflect structural interaction pairing plausibility, NOT physical binding affinities or experimental KD.",
        "=" * 72,
    ])

    report_text = "\n".join(lines)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    return {
        "summary": summary_file,
        "fold_metrics": folds_file,
        "oof_json": oof_json_file,
        "oof_csv": oof_csv_file,
        "assignments": assignments_file,
        "report": report_file,
    }
