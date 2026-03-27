"""
Model Training
Trains baseline and transfer-learning models and logs experiment artifacts
"""

from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import (
    EPOCHS_FINE,
    EPOCHS_FROZEN,
    LOGS_DIR,
    METRICS_GLOBAL_DIR,
    MODEL_NAMES,
    MODELS_DIR,
    get_model_plots_dir,
    ensure_directories,
    set_global_seed,
)
from data_loader import load_datasets, load_split_dataframe
from model import build_baseline_cnn, build_mobilenet_frozen, enable_mobilenet_fine_tuning


def _build_callbacks(model_name: str) -> Tuple[List[tf.keras.callbacks.Callback], str]:
    # Configure callbacks for early stopping, checkpointing, and LR scheduling
    checkpoint_path = str(MODELS_DIR / f"{model_name}_best.keras")

    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]
    return callbacks, checkpoint_path


def _save_history_and_plots(model_name: str, history: tf.keras.callbacks.History) -> pd.DataFrame:
    # Save raw training history and learning curves for each model
    model_plots_dir = get_model_plots_dir(model_name)

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(LOGS_DIR / f"{model_name}_history.csv", index=False)

    if {"accuracy", "val_accuracy"}.issubset(history_df.columns):
        plt.figure(figsize=(8, 5))
        plt.plot(history_df["epoch"], history_df["accuracy"], label="train")
        plt.plot(history_df["epoch"], history_df["val_accuracy"], label="val")
        plt.title(f"{model_name} accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(model_plots_dir / "accuracy_curve.png", dpi=200)
        plt.close()

    if {"loss", "val_loss"}.issubset(history_df.columns):
        plt.figure(figsize=(8, 5))
        plt.plot(history_df["epoch"], history_df["loss"], label="train")
        plt.plot(history_df["epoch"], history_df["val_loss"], label="val")
        plt.title(f"{model_name} loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(model_plots_dir / "loss_curve.png", dpi=200)
        plt.close()

    return history_df


def _compute_class_weights(class_names: List[str]) -> Dict[int, float]:
    # Compute balanced class weights from train split labels
    train_df = load_split_dataframe("train")

    # Macro scores are useful, but they can still hide that rare classes are being ignored.
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(class_names),
        y=train_df["label"].values,
    )

    class_weights_df = pd.DataFrame(
        {
            "class_index": np.arange(len(class_names)),
            "class_name": class_names,
            "weight": class_weights,
        }
    )
    class_weights_df.to_csv(METRICS_GLOBAL_DIR / "class_weights.csv", index=False)

    return {idx: float(weight) for idx, weight in enumerate(class_weights)}


def _train_model(
    model: tf.keras.Model,
    model_name: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Dict[int, float],
    epochs: int,
) -> Dict[str, float]:
    # Train one model and return summary metrics for the experiment log
    callbacks, checkpoint_path = _build_callbacks(model_name)

    start_time = perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    training_time_sec = perf_counter() - start_time

    # Load best checkpoint when available, then save final model file
    if (MODELS_DIR / f"{model_name}_best.keras").exists():
        model = tf.keras.models.load_model(checkpoint_path)

    model.save(MODELS_DIR / f"{model_name}.keras")

    history_df = _save_history_and_plots(model_name, history)

    best_epoch_idx = history_df["val_loss"].idxmin() if "val_loss" in history_df else history_df.index[-1]
    best_row = history_df.loc[best_epoch_idx]

    return {
        "model_name": model_name,
        "epochs_ran": int(len(history_df)),
        "best_epoch": int(best_row["epoch"]),
        "best_val_accuracy": float(best_row.get("val_accuracy", np.nan)),
        "best_val_loss": float(best_row.get("val_loss", np.nan)),
        "training_time_sec": float(round(training_time_sec, 3)),
        "parameter_count": int(model.count_params()),
    }


def run_training() -> None:
    # Prepare folders and deterministic behavior
    ensure_directories()
    set_global_seed()

    # Load datasets and derive class weights from label distribution
    train_ds, val_ds, _, class_names = load_datasets()
    num_classes = len(class_names)

    class_weights = _compute_class_weights(class_names)
    experiment_logs: List[Dict[str, float]] = []

    # Train baseline CNN
    baseline_model = build_baseline_cnn(num_classes=num_classes)
    experiment_logs.append(
        _train_model(
            model=baseline_model,
            model_name=MODEL_NAMES[0],
            train_ds=train_ds,
            val_ds=val_ds,
            class_weights=class_weights,
            epochs=EPOCHS_FROZEN,
        )
    )

    # Train MobileNetV2 with frozen backbone
    mobilenet_frozen = build_mobilenet_frozen(num_classes=num_classes, model_name=MODEL_NAMES[1])
    experiment_logs.append(
        _train_model(
            model=mobilenet_frozen,
            model_name=MODEL_NAMES[1],
            train_ds=train_ds,
            val_ds=val_ds,
            class_weights=class_weights,
            epochs=EPOCHS_FROZEN,
        )
    )

    # Fine-tune the best frozen MobileNet checkpoint
    mobilenet_finetuned = tf.keras.models.load_model(MODELS_DIR / f"{MODEL_NAMES[1]}.keras")
    # We lower the learning rate here because fine-tuning can easily erase pretrained features.
    mobilenet_finetuned = enable_mobilenet_fine_tuning(mobilenet_finetuned, unfreeze_last_n_layers=20)
    experiment_logs.append(
        _train_model(
            model=mobilenet_finetuned,
            model_name=MODEL_NAMES[2],
            train_ds=train_ds,
            val_ds=val_ds,
            class_weights=class_weights,
            epochs=EPOCHS_FINE,
        )
    )

    run_timestamp = datetime.now().isoformat(timespec="seconds")
    for row in experiment_logs:
        row["run_timestamp"] = run_timestamp

    # Append current run rows to the cumulative experiment log
    experiment_log_path = LOGS_DIR / "experiment_log.csv"
    new_logs_df = pd.DataFrame(experiment_logs)

    if experiment_log_path.exists():
        existing_logs_df = pd.read_csv(experiment_log_path)
        new_logs_df = pd.concat([existing_logs_df, new_logs_df], ignore_index=True)

    new_logs_df.to_csv(experiment_log_path, index=False)

    # This keeps prediction aligned with the model we treat as final in the dissertation.
    (MODELS_DIR / "final_model.txt").write_text("mobilenet_finetuned_best.keras\n", encoding="utf-8")
    print("Training complete. Models and logs are saved.")


if __name__ == "__main__":
    run_training()
