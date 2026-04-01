"""
Calibration Analysis
Checks whether model confidence matches real correctness on the test set
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import config
from config import set_global_seed
from data_loader import load_datasets, load_split_dataframe
from experiment_utils import OutputPaths, get_default_output_paths
from model_io import load_saved_model


def _resolve_model_path(model_path: str | None, output_paths: OutputPaths) -> Path:
    # Calibration needs a concrete model file; we resolve explicit paths first.
    if model_path:
        raw_path = Path(model_path)
        candidates = [raw_path]
        if not raw_path.is_absolute():
            candidates.append(config.PROJECT_ROOT / raw_path)
            candidates.append(output_paths.models_dir / raw_path)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(f"Model not found. Checked: {[str(path) for path in candidates]}")

    # Reuse the run-local final model pointer when available.
    final_pointer = output_paths.models_dir / "final_model.txt"
    if final_pointer.exists():
        pointer_value = final_pointer.read_text(encoding="utf-8").strip()
        if pointer_value:
            pointer_path = Path(pointer_value)
            pointer_candidates = [pointer_path]
            if not pointer_path.is_absolute():
                pointer_candidates.append(output_paths.models_dir / pointer_path)
                pointer_candidates.append(config.PROJECT_ROOT / pointer_path)
            for candidate in pointer_candidates:
                if candidate.exists():
                    return candidate.resolve()

    default_candidates = [
        output_paths.models_dir / "resnet50_finetuned_best.keras",
        config.MODELS_DIR / "resnet50_finetuned_best.keras",
    ]
    for candidate in default_candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not resolve a model for calibration. Provide --model_path or train a model first."
    )


def _build_reliability_table(confidence: np.ndarray, correct: np.ndarray, num_bins: int = 10) -> pd.DataFrame:
    # Reliability bins show where confidence is over-optimistic or under-confident.
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(confidence, bins=bin_edges[1:-1], right=False)

    rows: List[Dict[str, float]] = []
    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        count = int(np.sum(mask))

        if count == 0:
            rows.append(
                {
                    "bin": bin_idx,
                    "bin_lower": float(bin_edges[bin_idx]),
                    "bin_upper": float(bin_edges[bin_idx + 1]),
                    "count": 0,
                    "avg_confidence": np.nan,
                    "accuracy": np.nan,
                    "gap": np.nan,
                }
            )
            continue

        avg_conf = float(np.mean(confidence[mask]))
        acc = float(np.mean(correct[mask]))
        rows.append(
            {
                "bin": bin_idx,
                "bin_lower": float(bin_edges[bin_idx]),
                "bin_upper": float(bin_edges[bin_idx + 1]),
                "count": count,
                "avg_confidence": avg_conf,
                "accuracy": acc,
                "gap": float(acc - avg_conf),
            }
        )

    return pd.DataFrame(rows)


def _compute_ece(reliability_df: pd.DataFrame) -> float:
    # ECE summarizes calibration mismatch into one weighted score.
    total = float(reliability_df["count"].sum())
    if total <= 0:
        return 0.0

    valid_df = reliability_df[reliability_df["count"] > 0].copy()
    weighted_gap = (valid_df["count"] / total) * (valid_df["accuracy"] - valid_df["avg_confidence"]).abs()
    return float(weighted_gap.sum())


def _compute_brier_score(probabilities: np.ndarray, y_true: np.ndarray, num_classes: int) -> float:
    y_true_one_hot = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((probabilities - y_true_one_hot) ** 2, axis=1)))


def _save_reliability_plot(reliability_df: pd.DataFrame, output_paths: OutputPaths) -> None:
    # Confidence calibration matters because high confidence can still be wrong.
    plot_df = reliability_df[reliability_df["count"] > 0].copy()

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")

    if not plot_df.empty:
        plt.plot(
            plot_df["avg_confidence"],
            plot_df["accuracy"],
            marker="o",
            linewidth=2,
            label="Model",
        )

    plt.title("Reliability curve")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_paths.plots_global_dir / "reliability_curve.png", dpi=220)
    plt.close()


def _save_confidence_examples(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    class_names: List[str],
    output_paths: OutputPaths,
) -> None:
    # These examples make calibration failures easy to inspect qualitatively.
    idx_to_class = {idx: label for idx, label in enumerate(class_names)}

    examples_df = test_df.copy()
    examples_df = examples_df.reset_index(drop=True)
    examples_df["true_label"] = [idx_to_class[int(idx)] for idx in y_true]
    examples_df["pred_label"] = [idx_to_class[int(idx)] for idx in y_pred]
    examples_df["confidence"] = confidence
    examples_df["is_correct"] = y_true == y_pred

    high_correct = examples_df[examples_df["is_correct"]].sort_values("confidence", ascending=False).head(10).copy()
    high_correct["set"] = "high_confidence_correct"

    high_wrong = examples_df[~examples_df["is_correct"]].sort_values("confidence", ascending=False).head(10).copy()
    high_wrong["set"] = "high_confidence_wrong"

    combined = pd.concat([high_correct, high_wrong], ignore_index=True)
    keep_cols = ["set", "filepath", "true_label", "pred_label", "confidence", "is_correct"]
    combined[keep_cols].to_csv(output_paths.metrics_global_dir / "confidence_examples.csv", index=False)


def run_calibration(model_path: str | None = None, output_paths: OutputPaths | None = None) -> None:
    # Prepare folders and deterministic behavior.
    output_paths = output_paths or get_default_output_paths()
    output_paths.ensure_directories()
    set_global_seed()

    # Load test split and model predictions.
    _, _, test_ds, class_names = load_datasets(output_paths=output_paths)
    test_df = load_split_dataframe("test").reset_index(drop=True)

    class_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y_true = test_df["label"].map(class_to_idx).to_numpy()

    resolved_model_path = _resolve_model_path(model_path=model_path, output_paths=output_paths)
    model = load_saved_model(resolved_model_path, compile_model=False)
    probabilities = model.predict(test_ds, verbose=0)

    y_pred = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = (y_pred == y_true).astype(np.float32)

    reliability_df = _build_reliability_table(confidence=confidence, correct=correct, num_bins=10)
    reliability_df.to_csv(output_paths.metrics_global_dir / "reliability_bins.csv", index=False)

    ece = _compute_ece(reliability_df)
    brier_score = _compute_brier_score(probabilities, y_true=y_true, num_classes=len(class_names))

    calibration_metrics_df = pd.DataFrame(
        [
            {
                "model_name": resolved_model_path.stem,
                "num_samples": int(len(y_true)),
                "ece": float(round(ece, 6)),
                "brier_score": float(round(brier_score, 6)),
                "mean_confidence": float(round(float(np.mean(confidence)), 6)),
                "accuracy": float(round(float(np.mean(correct)), 6)),
            }
        ]
    )
    calibration_metrics_df.to_csv(output_paths.metrics_global_dir / "calibration_metrics.csv", index=False)

    _save_reliability_plot(reliability_df, output_paths=output_paths)
    _save_confidence_examples(
        test_df=test_df,
        y_true=y_true,
        y_pred=y_pred,
        confidence=confidence,
        class_names=class_names,
        output_paths=output_paths,
    )

    print(f"Calibration complete for {resolved_model_path.name}. Metrics and reliability outputs are saved.")


if __name__ == "__main__":
    run_calibration()
