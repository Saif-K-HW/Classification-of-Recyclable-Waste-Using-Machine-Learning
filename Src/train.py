"""
Model Training
Trains baseline and transfer-learning models and logs experiment artifacts
"""

from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

import config
from config import EPOCHS_FROZEN, MODEL_NAMES, set_global_seed
from data_loader import load_datasets, load_split_dataframe
from experiment_utils import OutputPaths, get_default_output_paths
from model_io import load_saved_model
from model import build_baseline_cnn, build_transfer_model, enable_backbone_fine_tuning


def _safe_tag(value: str) -> str:
    # Keep filenames predictable and filesystem-safe when experiments add run labels.
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_")


def _resolve_use_augmentation(value: str | bool | None) -> bool:
    # No-augmentation still keeps shuffle/cache/prefetch; we only skip the random transform stage.
    if isinstance(value, bool):
        return value
    if value is None:
        return str(config.AUGMENTATION).lower() != "none"
    return str(value).lower() != "none"


def _resolve_training_plan(run_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Allow experiments to define multiple comparable runs in one config file.
    explicit_plan = run_config.get("training_plan")
    if isinstance(explicit_plan, list) and explicit_plan:
        return [dict(item) for item in explicit_plan if isinstance(item, dict)]

    backbone_list = run_config.get("backbones")
    if isinstance(backbone_list, list) and backbone_list:
        return [{"name": str(backbone).lower(), "backbone": str(backbone).lower()} for backbone in backbone_list]

    return [
        {
            "name": "main",
            "backbone": str(run_config.get("backbone", config.BACKBONE)).lower(),
        }
    ]


def _resolve_transfer_model_names(backbone: str, run_name: str, use_default_names: bool) -> Tuple[str, str]:
    # Stable pipeline keeps canonical names for the default backbone.
    backbone_key = backbone.lower()
    run_tag = _safe_tag(run_name.lower())

    if use_default_names and backbone_key == str(config.BACKBONE).lower():
        return MODEL_NAMES[1], MODEL_NAMES[2]

    if run_tag in {"", "main", "default", backbone_key}:
        return f"{backbone_key}_frozen", f"{backbone_key}_finetuned"

    return f"{run_tag}_{backbone_key}_frozen", f"{run_tag}_{backbone_key}_finetuned"


def _build_callbacks(model_name: str, output_paths: OutputPaths) -> Tuple[List[tf.keras.callbacks.Callback], str]:
    # Configure callbacks for early stopping, checkpointing, and LR scheduling
    checkpoint_path = str(output_paths.models_dir / f"{model_name}_best.keras")

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


def _save_history_and_plots(
    model_name: str,
    history: tf.keras.callbacks.History,
    output_paths: OutputPaths,
) -> pd.DataFrame:
    # Save raw training history and learning curves for each model
    model_plots_dir = output_paths.model_plots_dir(model_name)
    model_plots_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(output_paths.logs_dir / f"{model_name}_history.csv", index=False)

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


def _compute_class_weights(
    class_names: List[str],
    class_weight_cap: float | None,
    output_paths: OutputPaths,
    run_name: str,
) -> Dict[int, float]:
    # Compute balanced class weights from train split labels
    train_df = load_split_dataframe("train")

    # Macro scores are useful, but they can still hide that rare classes are being ignored.
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(class_names),
        y=train_df["label"].values,
    )

    if class_weight_cap is not None:
        class_weights = np.minimum(class_weights, class_weight_cap)

    class_weights_df = pd.DataFrame(
        {
            "class_index": np.arange(len(class_names)),
            "class_name": class_names,
            "weight": class_weights,
        }
    )
    class_weights_df["cap"] = class_weight_cap

    file_suffix = "" if run_name in {"", "main", "default"} else f"_{_safe_tag(run_name)}"
    class_weights_df.to_csv(output_paths.metrics_global_dir / f"class_weights{file_suffix}.csv", index=False)

    return {idx: float(weight) for idx, weight in enumerate(class_weights)}


def _train_model(
    model: tf.keras.Model,
    model_name: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: Dict[int, float],
    epochs: int,
    output_paths: OutputPaths,
) -> Dict[str, float]:
    # Train one model and return summary metrics for the experiment log
    callbacks, checkpoint_path = _build_callbacks(model_name, output_paths=output_paths)

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
    checkpoint_file = output_paths.models_dir / f"{model_name}_best.keras"
    if checkpoint_file.exists():
        model = load_saved_model(checkpoint_path, compile_model=False)

    model.save(output_paths.models_dir / f"{model_name}.keras", include_optimizer=False)

    history_df = _save_history_and_plots(model_name, history, output_paths=output_paths)

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


def run_training(
    output_paths: OutputPaths | None = None,
    run_config: Dict[str, Any] | None = None,
) -> List[str]:
    # Prepare folders and deterministic behavior
    output_paths = output_paths or get_default_output_paths()
    run_config = run_config or {}

    training_plan = _resolve_training_plan(run_config)
    if not training_plan:
        raise ValueError("Training plan is empty. Add at least one run in config_override.")

    model_names_hint = list(MODEL_NAMES)
    output_paths.ensure_directories(model_names=model_names_hint)
    set_global_seed()

    # Keep stable names in the default pipeline so scripts and docs match main defaults.
    is_default_output = output_paths.models_dir.resolve() == config.MODELS_DIR.resolve()
    use_default_names = is_default_output and len(training_plan) == 1 and training_plan[0].get("backbone", "") in {
        "",
        str(config.BACKBONE).lower(),
    }

    include_baseline = run_config.get("include_baseline")
    if include_baseline is None:
        include_baseline = use_default_names

    experiment_logs: List[Dict[str, float]] = []
    trained_model_names: List[str] = []
    baseline_trained = False

    for plan_idx, plan in enumerate(training_plan, start=1):
        run_name = str(plan.get("name", f"run_{plan_idx}"))
        backbone = str(plan.get("backbone", run_config.get("backbone", config.BACKBONE))).lower()
        loss_name = str(plan.get("loss", run_config.get("loss", config.LOSS))).lower()
        class_weight_cap = plan.get("class_weight_cap", run_config.get("class_weight_cap", config.CLASS_WEIGHT_CAP))
        label_smoothing = float(plan.get("label_smoothing", run_config.get("label_smoothing", config.LABEL_SMOOTHING)))

        finetune_unfreeze_layers = int(
            plan.get(
                "finetune_unfreeze_layers",
                run_config.get("finetune_unfreeze_layers", config.FINETUNE_UNFREEZE_LAYERS),
            )
        )
        finetune_lr = float(plan.get("finetune_lr", run_config.get("finetune_lr", config.FINETUNE_LR)))
        finetune_epochs = int(plan.get("finetune_epochs", run_config.get("finetune_epochs", config.FINETUNE_EPOCHS)))
        frozen_epochs = int(plan.get("frozen_epochs", run_config.get("frozen_epochs", EPOCHS_FROZEN)))

        augmentation_value = plan.get("use_augmentation", plan.get("augmentation", run_config.get("augmentation")))
        use_augmentation = _resolve_use_augmentation(augmentation_value)

        # Exp02 uses this switch to isolate augmentation effects without changing data order/pipeline speedups.
        train_ds, val_ds, _, class_names = load_datasets(
            use_augmentation=use_augmentation,
            output_paths=output_paths,
        )
        num_classes = len(class_names)

        class_weights = _compute_class_weights(
            class_names,
            class_weight_cap=class_weight_cap,
            output_paths=output_paths,
            run_name=run_name,
        )

        if include_baseline and not baseline_trained:
            baseline_name = MODEL_NAMES[0] if use_default_names else f"{_safe_tag(run_name)}_baseline_cnn"
            baseline_model = build_baseline_cnn(
                num_classes=num_classes,
                loss_name=loss_name,
                label_smoothing=label_smoothing,
            )
            baseline_row = _train_model(
                model=baseline_model,
                model_name=baseline_name,
                train_ds=train_ds,
                val_ds=val_ds,
                class_weights=class_weights,
                epochs=frozen_epochs,
                output_paths=output_paths,
            )
            baseline_row.update(
                {
                    "run_name": run_name,
                    "backbone": "baseline_cnn",
                    "loss": loss_name,
                    "class_weight_cap": class_weight_cap,
                    "augmentation": "none" if not use_augmentation else "default",
                }
            )
            experiment_logs.append(baseline_row)
            trained_model_names.append(baseline_name)
            baseline_trained = True

        frozen_name, finetuned_name = _resolve_transfer_model_names(backbone, run_name, use_default_names)

        transfer_frozen = build_transfer_model(
            num_classes=num_classes,
            backbone=backbone,
            model_name=frozen_name,
            loss_name=loss_name,
            label_smoothing=label_smoothing,
        )
        frozen_row = _train_model(
            model=transfer_frozen,
            model_name=frozen_name,
            train_ds=train_ds,
            val_ds=val_ds,
            class_weights=class_weights,
            epochs=frozen_epochs,
            output_paths=output_paths,
        )
        frozen_row.update(
            {
                "run_name": run_name,
                "backbone": backbone,
                "loss": loss_name,
                "class_weight_cap": class_weight_cap,
                "augmentation": "none" if not use_augmentation else "default",
            }
        )
        experiment_logs.append(frozen_row)
        trained_model_names.append(frozen_name)

        # Fine-tuning uses a smaller LR so pretrained features adapt instead of collapsing.
        finetuned_model = load_saved_model(output_paths.models_dir / f"{frozen_name}.keras", compile_model=False)
        finetuned_model = enable_backbone_fine_tuning(
            model=finetuned_model,
            backbone=backbone,
            unfreeze_last_n_layers=finetune_unfreeze_layers,
            learning_rate=finetune_lr,
            loss_name=loss_name,
            label_smoothing=label_smoothing,
        )
        finetuned_row = _train_model(
            model=finetuned_model,
            model_name=finetuned_name,
            train_ds=train_ds,
            val_ds=val_ds,
            class_weights=class_weights,
            epochs=finetune_epochs,
            output_paths=output_paths,
        )
        finetuned_row.update(
            {
                "run_name": run_name,
                "backbone": backbone,
                "loss": loss_name,
                "class_weight_cap": class_weight_cap,
                "augmentation": "none" if not use_augmentation else "default",
            }
        )
        experiment_logs.append(finetuned_row)
        trained_model_names.append(finetuned_name)

    run_timestamp = datetime.now().isoformat(timespec="seconds")
    for row in experiment_logs:
        row["run_timestamp"] = run_timestamp

    # Append current run rows to the cumulative experiment log
    experiment_log_path = output_paths.logs_dir / "experiment_log.csv"
    new_logs_df = pd.DataFrame(experiment_logs)

    if experiment_log_path.exists():
        existing_logs_df = pd.read_csv(experiment_log_path)
        new_logs_df = pd.concat([existing_logs_df, new_logs_df], ignore_index=True)

    new_logs_df.to_csv(experiment_log_path, index=False)

    if trained_model_names:
        # This pointer keeps prediction/calibration scripts aligned with the latest trained model in this run scope.
        final_pointer = f"{trained_model_names[-1]}_best.keras"
        if is_default_output and use_default_names:
            final_pointer = f"{MODEL_NAMES[2]}_best.keras"
        (output_paths.models_dir / "final_model.txt").write_text(f"{final_pointer}\n", encoding="utf-8")

    print("Training complete. Models and logs are saved.")
    return trained_model_names


if __name__ == "__main__":
    run_training()
