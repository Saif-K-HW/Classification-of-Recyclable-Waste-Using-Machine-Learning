"""
Model Definitions
Builds baseline and MobileNetV2 models used for training and fine-tuning
"""

from __future__ import annotations

import tensorflow as tf

from config import BASE_LR, FINE_TUNE_LR, IMAGE_SIZE


def _compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    # Use a shared compile setup so all models train under the same objective
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_baseline_cnn(num_classes: int) -> tf.keras.Model:
    # Build a lightweight CNN baseline for comparison against transfer learning
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="baseline_cnn",
    )
    return _compile_model(model, learning_rate=BASE_LR)


def build_mobilenet_frozen(num_classes: int, model_name: str = "mobilenet_frozen") -> tf.keras.Model:
    # Build a transfer-learning model with ImageNet features frozen
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return _compile_model(model, learning_rate=BASE_LR)


def enable_mobilenet_fine_tuning(
    model: tf.keras.Model,
    unfreeze_last_n_layers: int = 20,
    learning_rate: float = FINE_TUNE_LR,
) -> tf.keras.Model:
    # Find the MobileNet backbone inside the wrapped model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find MobileNetV2 base model inside the provided model.")

    # Unfreeze only the deepest layers to adapt while keeping stable low-level features
    base_model.trainable = True

    for layer in base_model.layers[:-unfreeze_last_n_layers]:
        layer.trainable = False

    return _compile_model(model, learning_rate=learning_rate)
