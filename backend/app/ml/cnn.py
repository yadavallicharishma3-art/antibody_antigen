"""Convolutional Neural Network architecture for antibody-antigen interaction analysis.

Implements the Zhang et al., 2024 architecture:
- 3 Convolutional stages with 32 filters, 3x3 kernels, ReLU activation, same padding
- Max pooling (2x2) with same padding after each conv layer
- Dimensionality progression: (20, 21, 1) -> (10, 11, 32) -> (5, 6, 32) -> (3, 3, 32)
- Flatten bottleneck: exactly 3 * 3 * 32 = 288 dimensions
- Fully connected: Dense(512, relu) -> Dense(64, relu) -> Dense(1, sigmoid)
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_antibody_antigen_cnn(
    input_shape: Tuple[int, int, int] = (20, 21, 1),
    learning_rate: float = 1e-4,
    dropout_rate: float = 0.0,
    l2_reg: float = 0.0,
    name: str = "antibody_antigen_cnn",
) -> tf.keras.Model:
    """Build the author-compatible CNN model.

    Args:
        input_shape: Spatial dimensions of the input tensor, default (20, 21, 1).
        learning_rate: Learning rate for the Adam optimizer.
        dropout_rate: Optional dropout rate (default 0.0, author tested 0.25 commented out).
        l2_reg: Optional L2 kernel regularizer weight.
        name: Name for the Keras model.

    Returns:
        Compiled tf.keras.Model.
    """
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    model = models.Sequential(name=name)
    model.add(layers.Input(shape=input_shape, name="input_tensor"))

    # ---------------------------------------------------------
    # Stage 1: Conv2D(32, 3x3) -> (20, 21, 32) -> Pool -> (10, 11, 32)
    # ---------------------------------------------------------
    model.add(
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=reg,
            name="conv1",
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="pool1"))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="drop1"))

    # ---------------------------------------------------------
    # Stage 2: Conv2D(32, 3x3) -> (10, 11, 32) -> Pool -> (5, 6, 32)
    # ---------------------------------------------------------
    model.add(
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=reg,
            name="conv2",
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="pool2"))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="drop2"))

    # ---------------------------------------------------------
    # Stage 3: Conv2D(32, 3x3) -> (5, 6, 32) -> Pool -> (3, 3, 32)
    # ---------------------------------------------------------
    model.add(
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_regularizer=reg,
            name="conv3",
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="pool3"))

    # ---------------------------------------------------------
    # Flatten -> exactly 3 * 3 * 32 = 288 dimensions
    # ---------------------------------------------------------
    model.add(layers.Flatten(name="flatten"))

    # ---------------------------------------------------------
    # Dense classification stages
    # ---------------------------------------------------------
    model.add(
        layers.Dense(
            units=512,
            activation="relu",
            kernel_regularizer=reg,
            name="dense_512",
        )
    )
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="drop3"))

    model.add(
        layers.Dense(
            units=64,
            activation="relu",
            kernel_regularizer=reg,
            name="dense_64",
        )
    )

    model.add(
        layers.Dense(
            units=1,
            activation="sigmoid",
            name="classification_output",
        )
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )

    return model


def get_architecture_summary(model: tf.keras.Model) -> dict:
    """Extract structured architecture metadata for model_config.json."""
    layer_info = []
    for layer in model.layers:
        info = {
            "name": layer.name,
            "class": layer.__class__.__name__,
            "output_shape": [int(x) if x is not None else None for x in layer.output.shape],
            "param_count": layer.count_params(),
        }
        if hasattr(layer, "filters"):
            info["filters"] = layer.filters
        if hasattr(layer, "kernel_size"):
            info["kernel_size"] = list(layer.kernel_size)
        if hasattr(layer, "activation") and hasattr(layer.activation, "__name__"):
            info["activation"] = layer.activation.__name__
        layer_info.append(info)

    return {
        "model_name": model.name,
        "input_shape": list(model.input_shape),
        "total_parameters": model.count_params(),
        "trainable_parameters": sum(tf.keras.backend.count_params(w) for w in model.trainable_weights),
        "non_trainable_parameters": sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights),
        "layers": layer_info,
        "flatten_dim": 288,
    }
