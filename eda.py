"""
Exploratory Data Analysis
Summarizes dataset balance and saves global sanity-check plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from config import METRICS_GLOBAL_DIR, PLOTS_GLOBAL_DIR, SEED, ensure_directories, set_global_seed
from make_splits import collect_image_records


def _save_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Save per-class counts and percentages for reporting
    class_counts = df["label"].value_counts().sort_values(ascending=False)
    summary_df = class_counts.rename_axis("label").reset_index(name="count")
    summary_df["percentage"] = (summary_df["count"] / len(df)).round(4)
    summary_df.to_csv(METRICS_GLOBAL_DIR / "dataset_summary.csv", index=False)
    return summary_df


def _plot_class_distribution(summary_df: pd.DataFrame) -> None:
    # Visualize dataset imbalance across classes
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df["label"], summary_df["count"])
    plt.title("Class distribution in raw dataset")
    plt.xlabel("Class")
    plt.ylabel("Image count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_GLOBAL_DIR / "class_distribution.png", dpi=200)
    plt.close()


def _sample_for_grid(df: pd.DataFrame, max_images: int = 12) -> pd.DataFrame:
    # Picking one image per class first makes the grid more informative at a glance.
    base_samples = df.groupby("label", group_keys=False).sample(n=1, random_state=SEED)

    remaining_slots = max(0, max_images - len(base_samples))
    if remaining_slots == 0:
        return base_samples.reset_index(drop=True)

    remaining_pool = df.drop(base_samples.index)
    extra_samples = remaining_pool.sample(n=min(remaining_slots, len(remaining_pool)), random_state=SEED)
    return pd.concat([base_samples, extra_samples], ignore_index=True)


def _save_sample_grid(sample_df: pd.DataFrame) -> None:
    # Skip plotting when there are no samples to display
    if sample_df.empty:
        return

    n_images = len(sample_df)
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        image_path = Path(row["filepath"])
        axes[idx].imshow(Image.open(image_path).convert("RGB"))
        axes[idx].set_title(row["label"])
        axes[idx].axis("off")

    for idx in range(n_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(PLOTS_GLOBAL_DIR / "sample_grid.png", dpi=200)
    plt.close()


def run_eda() -> None:
    # Prepare folders and deterministic sampling
    ensure_directories()
    set_global_seed()

    # Collect image inventory and create summary artifacts
    df = collect_image_records()
    summary_df = _save_dataset_summary(df)
    _plot_class_distribution(summary_df)
    _save_sample_grid(_sample_for_grid(df))

    print(f"Total images: {len(df)}")
    print(f"Number of classes: {df['label'].nunique()}")


if __name__ == "__main__":
    run_eda()
