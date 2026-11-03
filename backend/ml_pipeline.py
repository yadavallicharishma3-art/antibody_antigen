"""Train, cross-validate, evaluate, and load the specified TensorFlow CNN."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "dataset_matrices.npz"
MODEL_PATH = ROOT / "models" / "cnn_20x20_three_conv.keras"
OUTPUT_DIR = ROOT / "data" / "processed"
CLASS_NAMES = {0: "General Protein-Protein Interaction", 1: "Antibody-Antigen Interaction"}
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
ARCHITECTURE_VERSION = "20x20x1-conv32-pool-conv32-pool-conv32-pool-flatten-dense512-dense64-sigmoid-v1"


def _tensorflow():
    """Import TensorFlow only when training or prediction is requested."""
    try:
        import tensorflow as tf 
    except Exception as exc:
        raise RuntimeError("TensorFlow is required for CNN training and prediction. Install requirements.txt first.") from exc
    return tf 


def create_model():
    """Build the requested 20x20 CNN with three convolution and pooling stages."""
    tf = _tensorflow()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(20, 20, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predicted = (probabilities >= 0.5).astype(int)
    auc = None
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, probabilities))
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_score": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
        "test_samples": int(len(y_true)),
    }


def _class_weights(labels: np.ndarray) -> dict[int, float]:
    counts = np.bincount(labels.astype(int), minlength=2)
    total = int(counts.sum())
    return {i: total / (2 * int(count)) for i, count in enumerate(counts) if count > 0}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _save_training_class_mean_heatmap(features: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    """Plot class-average matrices using training samples only, avoiding test-set peeking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = []
    sample_counts = []
    for label in (0, 1):
        class_matrices = features[labels == label]
        if len(class_matrices) == 0:
            raise ValueError(f"Cannot plot class {label}: no training matrices were found")
        means.append(class_matrices.mean(axis=0))
        sample_counts.append(len(class_matrices))

    color_limit = max(float(matrix.max()) for matrix in means)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    image = None
    for axis, matrix, label, count in zip(axes, means, (0, 1), sample_counts):
        image = axis.imshow(matrix, cmap="magma", vmin=0, vmax=color_limit, interpolation="nearest")
        axis.set_title(f"{CLASS_NAMES[label]}\nTraining samples: {count}")
        axis.set_xticks(range(20), AA_ORDER, fontsize=8)
        axis.set_yticks(range(20), AA_ORDER, fontsize=8)
        axis.set_xlabel("Amino acid")
        axis.set_ylabel("Amino acid")
    figure.suptitle("Mean Interaction Matrices by Class — Training Split Only")
    figure.colorbar(image, ax=axes, label="Mean normalized contact frequency", shrink=0.82)
    figure.text(0.5, -0.01, "Partner A: lower triangle · Partner B: upper triangle · diagonal: zero",
                ha="center", fontsize=9)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "training_class_mean_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int = 10,
    seed: int = 42,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 12,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Train a fresh CNN for each group-aware stratified fold on training data only."""
    tf = _tensorflow()
    group_counts = [len(set(groups[y == label].tolist())) for label in (0, 1)]
    actual_folds = min(folds, *group_counts)
    if actual_folds < 2:
        raise ValueError(f"Cross-validation needs at least two independent groups per class; counts are {group_counts}")
    fingerprint_input = np.ascontiguousarray(X).tobytes() + y.tobytes() + groups.astype(str).tobytes()
    fingerprint_input += f"{ARCHITECTURE_VERSION}:{actual_folds}:{seed}:{epochs}:{batch_size}:{patience}".encode("ascii")
    fingerprint = hashlib.sha256(fingerprint_input).hexdigest()[:16]
    checkpoint_path = output_dir / f"cross_validation_checkpoint_{fingerprint}.json"
    backup_root = output_dir / "cross_validation_backups" / fingerprint
    saved_folds = []
    if checkpoint_path.exists():
        saved_folds = json.loads(checkpoint_path.read_text(encoding="utf-8")).get("fold_results", [])
    completed = {int(item["fold"]) for item in saved_folds}
    splitter = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
    fold_results = list(saved_folds)
    print(f"Cross-validation: {actual_folds} folds; starting with fresh CNN weights for each fold.", flush=True)
    for fold_number, (train_indices, validation_indices) in enumerate(splitter.split(X, y, groups), start=1):
        if fold_number in completed:
            print(f"Fold {fold_number}/{actual_folds}: already completed; reusing its saved metrics.", flush=True)
            continue
        print(f"Starting fold {fold_number}/{actual_folds} "
              f"(train={len(train_indices)}, validation={len(validation_indices)}).", flush=True)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed + fold_number)
        model = create_model()
        model.fit(
            X[train_indices], y[train_indices],
            validation_data=(X[validation_indices], y[validation_indices]),
            epochs=epochs, batch_size=batch_size, verbose=0,
            class_weight=_class_weights(y[train_indices]),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
                tf.keras.callbacks.BackupAndRestore(backup_dir=str(backup_root / f"fold_{fold_number}")),
            ],
        )
        probabilities = model.predict(X[validation_indices], verbose=0).reshape(-1)
        result = _metrics(y[validation_indices], probabilities)
        result["fold"] = fold_number
        result["train_samples"] = int(len(train_indices))
        result["validation_samples"] = int(len(validation_indices))
        result["train_groups"] = int(len(set(groups[train_indices].tolist())))
        result["validation_groups"] = int(len(set(groups[validation_indices].tolist())))
        fold_results.append(result)
        fold_results.sort(key=lambda item: int(item["fold"]))
        _write_json(checkpoint_path, {"architecture": ARCHITECTURE_VERSION, "fingerprint": fingerprint,
                                      "folds": actual_folds, "fold_results": fold_results})
        auc = "n/a" if result["roc_auc"] is None else f"{result['roc_auc']:.4f}"
        print(f"Completed fold {fold_number}/{actual_folds}: "
              f"accuracy={result['accuracy']:.4f}, precision={result['precision']:.4f}, "
              f"recall={result['recall']:.4f}, F1={result['f1_score']:.4f}, ROC-AUC={auc}.", flush=True)
        del model

    metric_names = ("accuracy", "precision", "recall", "f1_score", "roc_auc")
    aggregate = {}
    for metric in metric_names:
        values = [item[metric] for item in fold_results if item[metric] is not None]
        aggregate[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0} if values else None
    return {"architecture": ARCHITECTURE_VERSION, "folds": actual_folds, "seed": seed,
            "fold_results": fold_results, "mean_and_std": aggregate,
            "method": "StratifiedGroupKFold; a fresh CNN is initialized for each fold. The separate test set is not passed to this function."}


def _save_training_artifacts(history: dict, result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.csv"
    if not history_path.exists():
        keys = sorted({key for values in history.values() for key in values})
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", *keys])
            writer.writeheader()
            for index in range(max((len(values) for values in history.values()), default=0)):
                row = {"epoch": index + 1}
                row.update({key: values[index] for key, values in history.items() if index < len(values)})
                writer.writerow(row)
    plot_history = {}
    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if rows:
        for key in rows[0]:
            if key != "epoch":
                plot_history[key] = [float(row[key]) for row in rows if row.get(key, "")]
    else:
        plot_history = history
    _write_json(output_dir / "evaluation_results.json", result)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric, filename, title in (("accuracy", "training_accuracy.png", "Training and validation accuracy"),
                                    ("loss", "training_loss.png", "Training and validation loss")):
        plt.figure(figsize=(7, 4))
        for key in (metric, f"val_{metric}"):
            if key in plot_history:
                plt.plot(plot_history[key], label=key.replace("_", " ").title())
        plt.title(title); plt.xlabel("Epoch"); plt.ylabel(metric.title()); plt.legend(); plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160); plt.close()

    matrix = np.asarray(result["confusion_matrix"])
    plt.figure(figsize=(5, 4)); plt.imshow(matrix, cmap="Blues"); plt.colorbar()
    plt.xticks([0, 1], ["General PPI", "Antibody-Antigen"], rotation=15)
    plt.yticks([0, 1], ["General PPI", "Antibody-Antigen"])
    plt.xlabel("Predicted class"); plt.ylabel("True class"); plt.title("Test confusion matrix")
    for row in range(2):
        for column in range(2):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.tight_layout(); plt.savefig(output_dir / "confusion_matrix.png", dpi=160); plt.close()

    roc_auc = result.get("roc_auc")
    if roc_auc is not None and "test_labels" in result:
        from sklearn.metrics import RocCurveDisplay
        RocCurveDisplay.from_predictions(np.asarray(result["test_labels"]), np.asarray(result["test_probabilities"]))
        plt.title(f"Test ROC curve (AUC = {roc_auc:.3f})"); plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=160); plt.close()


def train_pipeline(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    output_dir: Path = OUTPUT_DIR,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 12,
    seed: int = 42,
    folds: int = 10,
) -> dict:
    """Run CV on training samples, train with validation data, then test once."""
    tf = _tensorflow()
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}. Run backend/dataset.py first.")
    data = np.load(data_path, allow_pickle=False)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    groups = np.asarray(data["groups"], dtype=str)
    splits = np.asarray(data["splits"], dtype=str)
    if X.ndim != 3 or X.shape[1:] != (20, 20):
        raise ValueError(f"Expected matrices shaped (N, 20, 20), got {X.shape}")
    if not np.isfinite(X).all() or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Dataset contains invalid features or labels")
    X = X[..., np.newaxis]
    train = splits == "train"; validation = splits == "validation"; test = splits == "test"
    if not all(mask.any() for mask in (train, validation, test)):
        raise ValueError("Train, validation, and test splits must each contain samples")
    for name, mask in (("train", train), ("validation", validation), ("test", test)):
        if len(np.unique(y[mask])) < 2:
            raise ValueError(f"{name} split does not contain both classes; grouped split needs review")
        print(f"{name}: {int(mask.sum())} samples; negative={int(np.sum(y[mask] == 0))}; positive={int(np.sum(y[mask] == 1))}")
    if set(groups[train]) & set(groups[validation]) or set(groups[train]) & set(groups[test]) or set(groups[validation]) & set(groups[test]):
        raise ValueError("Data leakage detected: a structure group occurs in multiple splits")

    _save_training_class_mean_heatmap(X[train, :, :, 0], y[train], output_dir)

    cv_result = run_cross_validation(X[train], y[train], groups[train], folds=folds, seed=seed, epochs=epochs,
                                     batch_size=batch_size, patience=patience, output_dir=output_dir)
    _write_json(output_dir / "cross_validation_results.json", cv_result)

    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    model = create_model()
    model.summary()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    training_fingerprint_input = np.ascontiguousarray(X[train]).tobytes() + y[train].tobytes() + y[validation].tobytes()
    training_fingerprint_input += f"{ARCHITECTURE_VERSION}:{seed}:{epochs}:{batch_size}:{patience}".encode("ascii")
    training_fingerprint = hashlib.sha256(training_fingerprint_input).hexdigest()[:16]
    backup_dir = output_dir / "training_backups" / training_fingerprint
    completion_marker = output_dir / f"final_training_complete_{training_fingerprint}.json"
    resuming = backup_dir.exists()
    training_complete = completion_marker.exists() and model_path.exists()
    if model_path.exists() and not resuming and not training_complete:
        model_path.unlink()
    history_path = output_dir / "training_history.csv"
    if not resuming and not training_complete and history_path.exists():
        history_path.unlink()
    best_state_path = output_dir / f"best_validation_loss_{training_fingerprint}.json"
    if not resuming and not training_complete and best_state_path.exists():
        best_state_path.unlink()
    best_threshold = None
    if resuming and best_state_path.exists():
        best_threshold = json.loads(best_state_path.read_text(encoding="utf-8")).get("best_val_loss")

    def record_best_validation_loss(_epoch: int, logs: dict | None = None) -> None:
        value = (logs or {}).get("val_loss")
        if value is None:
            return
        current = json.loads(best_state_path.read_text(encoding="utf-8")).get("best_val_loss") if best_state_path.exists() else None
        if current is None or float(value) < float(current):
            _write_json(best_state_path, {"best_val_loss": float(value)})

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True, mode="min",
                                           initial_value_threshold=best_threshold, verbose=0),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0),
        tf.keras.callbacks.LambdaCallback(on_epoch_end=record_best_validation_loss),
        tf.keras.callbacks.LambdaCallback(on_train_end=lambda _logs: _write_json(completion_marker, {"training_complete": True})),
        tf.keras.callbacks.CSVLogger(str(history_path), append=resuming),
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(backup_dir)),
    ]
    if training_complete:
        print("Final model training is already checkpointed; reusing the saved model.", flush=True)
        history_dict = {}
    else:
        history = model.fit(
            X[train], y[train], validation_data=(X[validation], y[validation]),
            epochs=epochs, batch_size=batch_size, verbose=1,
            class_weight=_class_weights(y[train]),
            callbacks=callbacks,
        )
        history_dict = {key: [float(value) for value in values] for key, values in history.history.items()}
    selected_model = tf.keras.models.load_model(model_path)
    probabilities = selected_model.predict(X[test], verbose=0).reshape(-1)
    result = _metrics(y[test], probabilities)
    result.update({
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "architecture": ARCHITECTURE_VERSION,
        "model_path": str(model_path),
        "selected_by": "minimum validation loss with early stopping",
        "split_counts": {name: {"total": int(mask.sum()), "negative": int(np.sum(y[mask] == 0)), "positive": int(np.sum(y[mask] == 1))}
                         for name, mask in (("train", train), ("validation", validation), ("test", test))},
        "test_labels": y[test].astype(int).tolist(),
        "test_probabilities": probabilities.astype(float).tolist(),
        "metrics_explanation": {
            "accuracy": "Fraction of test predictions that are correct.",
            "precision": "Of predicted antibody-antigen cases, the fraction that are truly antibody-antigen.",
            "recall": "Of true antibody-antigen cases, the fraction found by the model.",
            "f1_score": "Harmonic mean of precision and recall.",
            "roc_auc": "Ability to rank positive cases above negative cases across thresholds; absent when only one class is present.",
            "confusion_matrix": "Rows are true classes and columns are predicted classes, ordered negative then positive.",
        },
    })
    _save_training_artifacts(history_dict, result, output_dir)
    return result


@lru_cache(maxsize=2)
def _load_prediction_model(model_path: str):
    """Load a trained model once for repeated structure-pair predictions."""
    return _tensorflow().keras.models.load_model(Path(model_path))


def predict_matrix(matrix: np.ndarray, model_path: Path = MODEL_PATH) -> dict:
    """Predict one validated matrix; confidence is model probability, not proof."""
    from structure_processing import validate_matrix
    validate_matrix(np.asarray(matrix))
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}. Train the model first.")
    model = _load_prediction_model(str(model_path.resolve()))
    probability = float(model.predict(np.asarray(matrix, dtype=np.float32).reshape(1, 20, 20, 1), verbose=0)[0][0])
    label = int(probability >= 0.5)
    return {"label": label, "class_name": CLASS_NAMES[label], "probability_antibody_antigen": probability,
            "confidence": max(probability, 1.0 - probability),
            "interpretation": "This is a model prediction, not experimental confirmation of a biological interaction."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_pipeline(args.data, args.model, args.output_dir, epochs=args.epochs,
                            batch_size=args.batch_size, folds=args.folds, seed=args.seed)
    print(json.dumps({key: value for key, value in result.items() if key not in {"test_labels", "test_probabilities"}}, indent=2))


if __name__ == "__main__":
    main()
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def _save_training_class_mean_heatmap(
    features: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot average interaction matrices for each class using training data only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_means = []
    sample_counts = []

    for label in (0, 1):
        class_matrices = features[labels == label]
        if len(class_matrices) == 0:
            raise ValueError(f"No training matrices found for class {label}.")

        class_means.append(class_matrices.mean(axis=0))
        sample_counts.append(len(class_matrices))

    color_limit = max(float(matrix.max()) for matrix in class_means)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    for axis, matrix, label, count in zip(
        axes, class_means, (0, 1), sample_counts
    ):
        image = axis.imshow(
            matrix,
            cmap="magma",
            vmin=0,
            vmax=color_limit,
            interpolation="nearest",
        )
        axis.set_title(f"{CLASS_NAMES[label]}\nTraining samples: {count}")
        axis.set_xticks(range(20), AA_ORDER, fontsize=8)
        axis.set_yticks(range(20), AA_ORDER, fontsize=8)
        axis.set_xlabel("Amino acid")
        axis.set_ylabel("Amino acid")

    figure.suptitle("Mean Interaction Matrices by Class — Training Split Only")
    figure.colorbar(
        image,
        ax=axes,
        label="Mean normalized contact frequency",
        shrink=0.82,
    )
    figure.text(
        0.5,
        -0.01,
        "Partner A: lower triangle · Partner B: upper triangle · diagonal: zero",
        ha="center",
        fontsize=9,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_dir / "training_class_mean_heatmap.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)