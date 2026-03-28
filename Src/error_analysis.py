"""
Error Analysis
Inspects model mistakes and saves confusion-focused artifacts for discussion
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix

from config import (
    METRICS_GLOBAL_DIR,
    MISCLASSIFIED_DIR,
    MODELS_DIR,
    get_model_plots_dir,
    ensure_directories,
    set_global_seed,
)
from data_loader import load_datasets, load_split_dataframe


def _safe_name(value: str) -> str:
    # Convert labels to filesystem-safe fragments for output filenames
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-") or "unknown"


def _save_misclassified_images(misclassified_df: pd.DataFrame, model_name: str) -> None:
    # Save each misclassified sample with true/pred labels in the filename
    output_dir = MISCLASSIFIED_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in misclassified_df.iterrows():
        src_path = Path(row["filepath"])
        if not src_path.exists():
            continue

        true_label = _safe_name(row["true_label"])
        pred_label = _safe_name(row["pred_label"])
        confidence = f"{row['confidence']:.3f}"

        target_name = f"{true_label}__{pred_label}__{confidence}.jpg"
        target_path = output_dir / target_name

        suffix = 1
        while target_path.exists():
            target_path = output_dir / f"{true_label}__{pred_label}__{confidence}__{suffix}.jpg"
            suffix += 1

        with Image.open(src_path) as image:
            image.convert("RGB").save(target_path, format="JPEG")


def _save_confusion_example_grid(misclassified_df: pd.DataFrame, model_name: str) -> None:
    # Build a quick visual grid of the most common confusion pairs
    if misclassified_df.empty:
        return

    pair_counts = (
        misclassified_df.groupby(["true_label", "pred_label"]).size().rename("pair_count").reset_index()
    )
    grid_df = misclassified_df.merge(pair_counts, on=["true_label", "pred_label"], how="left")
    grid_df = grid_df.sort_values("pair_count", ascending=False).head(9)

    n_images = len(grid_df)
    n_cols = 3
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (_, row) in enumerate(grid_df.iterrows()):
        image_path = Path(row["filepath"])
        axes[idx].imshow(Image.open(image_path).convert("RGB"))
        axes[idx].set_title(f"{row['true_label']} -> {row['pred_label']}")
        axes[idx].axis("off")

    for idx in range(n_images, len(axes)):
        axes[idx].axis("off")

    model_plots_dir = get_model_plots_dir(model_name)
    plt.tight_layout()
    plt.savefig(model_plots_dir / "common_confusions.png", dpi=220)
    plt.close(fig)


def _select_analysis_model(preferred_model: str | None = None) -> str:
    # Use the caller preference first, then fallback to best macro-F1 model
    if preferred_model:
        return preferred_model

    summary_path = METRICS_GLOBAL_DIR / "evaluation_summary.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        if not summary_df.empty and "macro_f1" in summary_df.columns:
            best_row = summary_df.sort_values("macro_f1", ascending=False).iloc[0]
            return str(best_row["model_name"])

    return "mobilenet_finetuned"


def run_error_analysis(model_name: str | None = None) -> None:
    # Prepare folders and deterministic behavior
    ensure_directories()
    set_global_seed()

    # Load test data and label mappings used for decoding predictions
    _, _, test_ds, class_names = load_datasets()
    test_df = load_split_dataframe("test").reset_index(drop=True)
    class_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(class_names)}
    idx_to_class: Dict[int, str] = {idx: label for label, idx in class_to_idx.items()}

    y_true = test_df["label"].map(class_to_idx).to_numpy()

    # Select model, run inference, and compute per-sample confidence
    analysis_model_name = _select_analysis_model(model_name)
    model_path = MODELS_DIR / f"{analysis_model_name}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)

    mis_mask = y_pred != y_true
    misclassified_df = test_df.loc[mis_mask].copy()
    misclassified_df["true_label"] = [idx_to_class[idx] for idx in y_true[mis_mask]]
    misclassified_df["pred_label"] = [idx_to_class[idx] for idx in y_pred[mis_mask]]
    misclassified_df["confidence"] = confidence[mis_mask]

    # Save example mistakes for qualitative inspection
    _save_misclassified_images(misclassified_df, analysis_model_name)
    _save_confusion_example_grid(misclassified_df, analysis_model_name)

    # Build confusion matrix and derive worst-recall classes
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    # Error patterns tell us whether we need better data, better labels, or better augmentation.
    recall_per_class = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    worst_indices = np.argsort(recall_per_class)[: min(5, len(class_names))]
    worst_rows: List[Dict[str, object]] = []
    for idx in worst_indices:
        worst_rows.append(
            {
                "model_name": analysis_model_name,
                "label": idx_to_class[int(idx)],
                "recall": float(round(recall_per_class[int(idx)], 6)),
            }
        )

    confusion_rows = []
    for true_idx in range(len(class_names)):
        for pred_idx in range(len(class_names)):
            count = int(cm[true_idx, pred_idx])
            if true_idx == pred_idx or count == 0:
                continue
            confusion_rows.append(
                {
                    "model_name": analysis_model_name,
                    "true_label": idx_to_class[true_idx],
                    "pred_label": idx_to_class[pred_idx],
                    "count": count,
                }
            )

    # Save ranked confusion tables for reporting
    confusion_df = pd.DataFrame(confusion_rows).sort_values("count", ascending=False)
    confusion_df.to_csv(METRICS_GLOBAL_DIR / "confusion_counts.csv", index=False)
    confusion_df.head(10).to_csv(METRICS_GLOBAL_DIR / "top_confusions.csv", index=False)
    pd.DataFrame(worst_rows).to_csv(METRICS_GLOBAL_DIR / "worst_classes.csv", index=False)

    print(f"Error analysis complete for {analysis_model_name}. Misclassifications and confusion summaries are saved.")


if __name__ == "__main__":
    run_error_analysis()
