"""
Model Evaluation
Evaluates trained models and saves metrics, reports, and comparison artifacts
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config import MODEL_NAMES, set_global_seed
from data_loader import load_datasets, load_split_dataframe
from experiment_utils import OutputPaths, get_default_output_paths
from model_io import load_saved_model


def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, output_path: Path) -> None:
    # Render and save confusion matrix heatmap for one model
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def _evaluate_model(
    model_name: str,
    class_names: List[str],
    class_to_idx: Dict[str, int],
    test_df: pd.DataFrame,
    test_ds: tf.data.Dataset,
    output_paths: OutputPaths,
) -> Dict[str, float]:
    # Load trained model and ensure per-model output folders exist
    model_path = output_paths.models_dir / f"{model_name}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_metrics_dir = output_paths.model_metrics_dir(model_name)
    model_plots_dir = output_paths.model_plots_dir(model_name)
    model_metrics_dir.mkdir(parents=True, exist_ok=True)
    model_plots_dir.mkdir(parents=True, exist_ok=True)

    model = load_saved_model(model_path, compile_model=False)

    # Run inference and compute aggregate classification metrics
    y_true = test_df["label"].map(class_to_idx).to_numpy()

    start = perf_counter()
    probabilities = model.predict(test_ds, verbose=0)
    elapsed = perf_counter() - start

    y_pred = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # Macro F1 matters here because each class gets equal weight, even if some are rare.
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(model_metrics_dir / "classification_report.csv")

    # Save confusion matrix and deployment-oriented runtime/size metrics
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion_matrix(
        cm,
        class_names,
        title=f"Confusion Matrix - {model_name}",
        output_path=model_plots_dir / "confusion_matrix.png",
    )

    inference_row = {
        "model_name": model_name,
        "num_samples": int(len(y_true)),
        "total_inference_time_sec": float(round(elapsed, 6)),
        "avg_inference_time_ms": float(round((elapsed / max(1, len(y_true))) * 1000, 4)),
    }

    size_row = {
        "model_name": model_name,
        "model_size_mb": float(round(model_path.stat().st_size / (1024 * 1024), 4)),
    }

    deployment_df = pd.DataFrame([{**inference_row, **size_row}])
    deployment_df.to_csv(model_metrics_dir / "deployment_metrics.csv", index=False)

    metrics_row = {
        "model_name": model_name,
        "accuracy": float(round(accuracy, 6)),
        "macro_precision": float(round(precision, 6)),
        "macro_recall": float(round(recall, 6)),
        "macro_f1": float(round(f1, 6)),
    }

    return {**metrics_row, **inference_row, **size_row}


def _save_model_comparison(evaluation_df: pd.DataFrame, output_paths: OutputPaths) -> None:
    # Merge evaluation summary with latest training metadata for comparison table
    experiment_log_path = output_paths.logs_dir / "experiment_log.csv"
    if not experiment_log_path.exists():
        return

    train_log_df = pd.read_csv(experiment_log_path)
    if "run_timestamp" in train_log_df.columns:
        train_log_df = train_log_df.sort_values("run_timestamp")
    latest_train_df = train_log_df.drop_duplicates(subset=["model_name"], keep="last")

    comparison_df = evaluation_df.merge(
        latest_train_df[["model_name", "parameter_count", "training_time_sec"]],
        on="model_name",
        how="left",
    )

    comparison_df = comparison_df.rename(
        columns={
            "model_name": "Model",
            "accuracy": "Test Acc",
            "macro_f1": "Macro F1",
            "parameter_count": "Params",
            "training_time_sec": "Training Time",
        }
    )[["Model", "Test Acc", "Macro F1", "Params", "Training Time"]]

    comparison_df.to_csv(output_paths.metrics_global_dir / "model_comparison.csv", index=False)
    print("\nModel comparison:")
    print(comparison_df.to_string(index=False))


def run_evaluation(
    output_paths: OutputPaths | None = None,
    model_names: List[str] | None = None,
) -> None:
    # Prepare folders and deterministic evaluation behavior
    output_paths = output_paths or get_default_output_paths()
    eval_model_names = list(model_names or MODEL_NAMES)
    output_paths.ensure_directories(model_names=eval_model_names)
    set_global_seed()

    # Load test data and label mapping
    _, _, test_ds, class_names = load_datasets(output_paths=output_paths)
    test_df = load_split_dataframe("test")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    evaluation_rows = []
    inference_rows = []
    model_size_rows = []

    # Evaluate each configured model and collect summary rows
    for model_name in eval_model_names:
        metrics = _evaluate_model(model_name, class_names, class_to_idx, test_df, test_ds, output_paths=output_paths)

        evaluation_rows.append(
            {
                "model_name": metrics["model_name"],
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
            }
        )

        inference_rows.append(
            {
                "model_name": metrics["model_name"],
                "num_samples": metrics["num_samples"],
                "total_inference_time_sec": metrics["total_inference_time_sec"],
                "avg_inference_time_ms": metrics["avg_inference_time_ms"],
            }
        )

        model_size_rows.append(
            {
                "model_name": metrics["model_name"],
                "model_size_mb": metrics["model_size_mb"],
            }
        )

    # Save global evaluation outputs used by error analysis and reporting
    evaluation_df = pd.DataFrame(evaluation_rows)
    evaluation_df.to_csv(output_paths.metrics_global_dir / "evaluation_summary.csv", index=False)
    pd.DataFrame(inference_rows).to_csv(output_paths.metrics_global_dir / "inference_time.csv", index=False)
    pd.DataFrame(model_size_rows).to_csv(output_paths.metrics_global_dir / "model_size.csv", index=False)
    _save_model_comparison(evaluation_df, output_paths=output_paths)

    print("Evaluation complete. Metrics and confusion matrices are saved.")


if __name__ == "__main__":
    run_evaluation()
