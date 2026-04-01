"""
Model Definitions
Builds baseline and transfer-learning models used for training and fine-tuning
"""

from __future__ import annotations

import tensorflow as tf

import config
from config import BASE_LR, IMAGE_SIZE


@tf.keras.utils.register_keras_serializable(package="custom")
class SparseCategoricalCrossentropyWithLabelSmoothing(tf.keras.losses.Loss):
    # SparseCategoricalCrossentropy in this TF version has no label_smoothing arg, so we handle it here.
    def __init__(
        self,
        label_smoothing: float = 0.0,
        name: str = "sparse_categorical_crossentropy_with_label_smoothing",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes, dtype=y_pred.dtype)
        return tf.keras.losses.categorical_crossentropy(
            y_true_one_hot,
            y_pred,
            label_smoothing=self.label_smoothing,
        )

    def get_config(self) -> dict:
        config_dict = super().get_config()
        config_dict.update({"label_smoothing": self.label_smoothing})
        return config_dict


@tf.keras.utils.register_keras_serializable(package="custom")
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    # Focal loss helps focus optimization on harder examples.
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        label_smoothing: float = 0.0,
        name: str = "sparse_categorical_focal_loss",
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.gamma = float(gamma)
        self.alpha = float(alpha) if alpha is not None else None
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes, dtype=y_pred.dtype)

        if self.label_smoothing > 0:
            smoothing_value = self.label_smoothing / tf.cast(num_classes, y_pred.dtype)
            y_true_one_hot = (1.0 - self.label_smoothing) * y_true_one_hot + smoothing_value

        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        loss = -tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)

        if self.alpha is not None:
            loss = loss * tf.cast(self.alpha, y_pred.dtype)

        return loss

    def get_config(self) -> dict:
        config_dict = super().get_config()
        config_dict.update(
            {
                "gamma": self.gamma,
                "alpha": self.alpha,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config_dict


@tf.keras.utils.register_keras_serializable(package="custom")
class BackbonePreprocess(tf.keras.layers.Layer):
    # Avoid Lambda layers so models reload cleanly across Keras safe deserialization rules.
    def __init__(self, backbone: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.backbone = backbone.lower()
        if self.backbone not in {"mobilenetv2", "efficientnetb0", "resnet50", "densenet121"}:
            raise ValueError(f"Unsupported backbone for preprocessing: {backbone}")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        image_255 = inputs * 255.0
        if self.backbone == "mobilenetv2":
            return tf.keras.applications.mobilenet_v2.preprocess_input(image_255)
        if self.backbone == "efficientnetb0":
            return tf.keras.applications.efficientnet.preprocess_input(image_255)
        if self.backbone == "resnet50":
            return tf.keras.applications.resnet50.preprocess_input(image_255)
        return tf.keras.applications.densenet.preprocess_input(image_255)

    def get_config(self) -> dict:
        config_dict = super().get_config()
        config_dict.update({"backbone": self.backbone})
        return config_dict


def get_loss(loss_name: str = "cross_entropy", label_smoothing: float = 0.0) -> tf.keras.losses.Loss:
    # Keep one loss factory so experiments can switch objectives safely.
    if loss_name == "cross_entropy":
        if label_smoothing > 0:
            return SparseCategoricalCrossentropyWithLabelSmoothing(label_smoothing=label_smoothing)
        return tf.keras.losses.SparseCategoricalCrossentropy()

    if loss_name == "focal":
        return SparseCategoricalFocalLoss(gamma=2.0, alpha=None, label_smoothing=label_smoothing)

    raise ValueError(f"Unsupported loss: {loss_name}")


def _compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    loss_name: str = "cross_entropy",
    label_smoothing: float = 0.0,
) -> tf.keras.Model:
    # Use a shared compile setup so all models train under the same objective
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=get_loss(loss_name=loss_name, label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )
    return model


def build_baseline_cnn(
    num_classes: int,
    learning_rate: float = BASE_LR,
    loss_name: str = "cross_entropy",
    label_smoothing: float = 0.0,
) -> tf.keras.Model:
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
    return _compile_model(
        model,
        learning_rate=learning_rate,
        loss_name=loss_name,
        label_smoothing=label_smoothing,
    )


def _get_backbone_model(backbone: str) -> tf.keras.Model:
    # Use pretrained ImageNet backbones for fair architecture comparisons.
    backbone_key = backbone.lower()

    if backbone_key == "mobilenetv2":
        return tf.keras.applications.MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
    if backbone_key == "efficientnetb0":
        return tf.keras.applications.EfficientNetB0(
            input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
        )
    if backbone_key == "resnet50":
        return tf.keras.applications.ResNet50(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
    if backbone_key == "densenet121":
        return tf.keras.applications.DenseNet121(
            input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
        )

    raise ValueError(f"Unsupported backbone: {backbone}")


def _apply_backbone_preprocessing(inputs: tf.Tensor, backbone: str) -> tf.Tensor:
    # Loader emits [0,1], so each backbone gets a small model-side range adapter.
    backbone_key = backbone.lower()
    return BackbonePreprocess(backbone=backbone_key, name=f"{backbone_key}_preprocess")(inputs)


def build_transfer_model(
    num_classes: int,
    backbone: str = "resnet50",
    model_name: str | None = None,
    learning_rate: float = BASE_LR,
    loss_name: str = "cross_entropy",
    label_smoothing: float = 0.0,
) -> tf.keras.Model:
    # Share one classification head so differences mostly come from the backbone.
    backbone_key = backbone.lower()
    raw_base_model = _get_backbone_model(backbone_key)
    # Wrapping with an explicit name keeps fine-tuning lookup stable after save/load.
    base_model = tf.keras.Model(
        inputs=raw_base_model.input,
        outputs=raw_base_model.output,
        name=f"{backbone_key}_base",
    )
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))
    x = _apply_backbone_preprocessing(inputs, backbone=backbone_key)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    resolved_name = model_name or f"{backbone_key}_frozen"
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=resolved_name)
    return _compile_model(
        model,
        learning_rate=learning_rate,
        loss_name=loss_name,
        label_smoothing=label_smoothing,
    )


def build_mobilenet_frozen(num_classes: int, model_name: str = "mobilenet_frozen") -> tf.keras.Model:
    # Keep backward compatibility for existing imports.
    return build_transfer_model(num_classes=num_classes, backbone="mobilenetv2", model_name=model_name)


def enable_backbone_fine_tuning(
    model: tf.keras.Model,
    backbone: str,
    unfreeze_last_n_layers: int | None = None,
    learning_rate: float | None = None,
    loss_name: str = "cross_entropy",
    label_smoothing: float = 0.0,
) -> tf.keras.Model:
    # Unfreeze only deep layers to adapt without fully discarding pretrained features.
    backbone_key = backbone.lower()
    target_name = f"{backbone_key}_base"
    candidate_tokens = [target_name, backbone_key]

    base_model = None
    for layer in model.layers:
        if not isinstance(layer, tf.keras.Model):
            continue
        layer_name = layer.name.lower()
        if any(token in layer_name for token in candidate_tokens):
            base_model = layer
            break

    if base_model is None:
        nested_models = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
        if len(nested_models) == 1:
            # Older checkpoints may not preserve our preferred name token.
            base_model = nested_models[0]

    if base_model is None:
        nested_names = [layer.name for layer in model.layers if isinstance(layer, tf.keras.Model)]
        raise ValueError(
            f"Could not find backbone model for '{backbone_key}'. "
            f"Looked for tokens {candidate_tokens}. Nested models: {nested_names}"
        )

    base_model.trainable = True

    unfreeze_count = int(unfreeze_last_n_layers or config.FINETUNE_UNFREEZE_LAYERS)
    if unfreeze_count < 1:
        unfreeze_count = 1
    unfreeze_count = min(unfreeze_count, len(base_model.layers))

    for layer in base_model.layers[:-unfreeze_count]:
        layer.trainable = False

    resolved_lr = learning_rate if learning_rate is not None else config.FINETUNE_LR
    return _compile_model(
        model,
        learning_rate=resolved_lr,
        loss_name=loss_name,
        label_smoothing=label_smoothing,
    )


def enable_mobilenet_fine_tuning(
    model: tf.keras.Model,
    unfreeze_last_n_layers: int = 20,
    learning_rate: float | None = None,
) -> tf.keras.Model:
    # Keep backward compatibility for existing MobileNet fine-tuning calls.
    return enable_backbone_fine_tuning(
        model=model,
        backbone="mobilenetv2",
        unfreeze_last_n_layers=unfreeze_last_n_layers,
        learning_rate=learning_rate,
    )
