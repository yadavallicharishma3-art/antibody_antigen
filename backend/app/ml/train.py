"""Deterministic training orchestration with validation-monitored early stopping."""

import os
import random
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from .cnn import build_antibody_antigen_cnn
from .data_loader import DatasetBundle


def set_reproducibility_seeds(seed: int = 42) -> Dict[str, int]:
    """Set explicit random seeds across all environments for deterministic behavior."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return {
        "python_seed": seed,
        "numpy_seed": seed,
        "tensorflow_seed": seed,
    }


def compute_training_class_weights(y_train: np.ndarray) -> Tuple[Optional[Dict[int, float]], str]:
    """Calculate class weights strictly from training data labels.

    Never uses validation or test labels.
    If only one class is present, returns None with an explanatory rationale.
    """
    unique_classes, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))

    if len(class_counts) <= 1:
        rationale = (
            f"Single-class training data detected (classes present: {list(class_counts.keys())}). "
            "Class weighting is mathematically inapplicable; training proceeds without artificial sample weighting."
        )
        return None, rationale

    total = len(y_train)
    weights = {}
    for c, cnt in class_counts.items():
        weights[int(c)] = float(total / (len(class_counts) * cnt))

    rationale = (
        f"Balanced class weights computed from training split: {weights} "
        f"(class 0: {class_counts.get(0, 0)}, class 1: {class_counts.get(1, 0)})."
    )
    return weights, rationale


def train_cnn_model(
    train_bundle: DatasetBundle,
    val_bundle: DatasetBundle,
    epochs: int = 60,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    patience: int = 15,
    seed: int = 42,
    monitor: str = "val_loss",
    dropout_rate: float = 0.0,
    verbose: int = 1,
) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """Train the CNN on the training split, validated solely against the validation split.

    Strict leakage constraints:
    - Test set is NEVER touched during training.
    - Early stopping restores best weights strictly based on validation performance.
    """
    seed_info = set_reproducibility_seeds(seed)

    # Inspect class balance
    class_weights, weight_rationale = compute_training_class_weights(train_bundle.y)

    model = build_antibody_antigen_cnn(
        input_shape=(20, 21, 1),
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
    )

    callbacks = [
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=verbose,
        )
    ]

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

    # Format training history
    history_dict = {}
    for k, v in history.history.items():
        history_dict[k] = [float(x) for x in v]

    total_epochs_run = len(history_dict.get("loss", []))
    best_epoch = int(np.argmin(history_dict[monitor])) + 1 if monitor in history_dict else total_epochs_run

    training_meta = {
        "seeds": seed_info,
        "epochs_requested": epochs,
        "epochs_completed": total_epochs_run,
        "best_epoch": best_epoch,
        "early_stopping_monitor": monitor,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "class_weights": class_weights,
        "class_weight_rationale": weight_rationale,
        "train_samples": len(train_bundle),
        "val_samples": len(val_bundle),
        "history": history_dict,
    }

    return model, training_meta
