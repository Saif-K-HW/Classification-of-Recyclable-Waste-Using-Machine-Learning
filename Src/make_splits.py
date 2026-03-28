"""
Dataset Split Creation
Builds stratified train/validation/test CSV indexes from raw image folders
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    IMAGE_EXTENSIONS,
    METRICS_GLOBAL_DIR,
    PLOTS_GLOBAL_DIR,
    RAW_DIR,
    SEED,
    SPLIT_RATIOS,
    SPLITS_DIR,
    ensure_directories,
    set_global_seed,
)


def _resolve_dataset_root(raw_dir: Path) -> Path:
    """
    Descend through wrapper folders until we reach the level where class folders live.
    """
    current = raw_dir
    while True:
        children = [path for path in current.iterdir() if path.is_dir()]
        direct_images = [
            file_path
            for file_path in current.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(children) == 1 and not direct_images:
            current = children[0]
            continue
        return current


def collect_image_records(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    # Locate the directory level where class folders live
    dataset_root = _resolve_dataset_root(raw_dir)
    class_dirs = sorted([path for path in dataset_root.iterdir() if path.is_dir()])

    # Build one record per image with absolute file path and class label
    records = []
    for class_dir in class_dirs:
        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({"filepath": str(image_path.resolve()), "label": class_dir.name})

    if not records:
        raise ValueError(f"No images were found under {raw_dir}.")

    return pd.DataFrame(records)


def _create_stratified_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Validate split ratios before splitting
    train_ratio, val_ratio, test_ratio = SPLIT_RATIOS
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("SPLIT_RATIOS must add up to 1.0.")

    # We stratify here so smaller classes don't disappear in validation or test.
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=df["label"],
        random_state=SEED,
    )

    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_fraction_of_temp),
        stratify=temp_df["label"],
        random_state=SEED,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _save_split_summary(splits: Dict[str, pd.DataFrame]) -> None:
    # Save per-class counts for each split as a long-form table
    summary_rows = []
    for split_name, split_df in splits.items():
        class_counts = split_df["label"].value_counts().sort_index()
        for label, count in class_counts.items():
            summary_rows.append({"split": split_name, "label": label, "count": int(count)})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(METRICS_GLOBAL_DIR / "split_summary.csv", index=False)


def _plot_split_distribution(splits: Dict[str, pd.DataFrame]) -> None:
    # Build grouped bar chart data showing class balance by split
    plot_rows = []
    for split_name, split_df in splits.items():
        counts = split_df["label"].value_counts().sort_index()
        for label, count in counts.items():
            plot_rows.append({"label": label, "split": split_name, "count": int(count)})

    plot_df = pd.DataFrame(plot_rows)
    pivot_df = plot_df.pivot(index="label", columns="split", values="count").fillna(0)
    pivot_df = pivot_df[["train", "val", "test"]]

    pivot_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Class distribution across train/val/test splits")
    plt.xlabel("Class")
    plt.ylabel("Image count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_GLOBAL_DIR / "split_class_distribution.png", dpi=200)
    plt.close()


def create_splits() -> Dict[str, pd.DataFrame]:
    # Prepare folders and deterministic random seed
    ensure_directories()
    set_global_seed()

    # Generate and save train/val/test split CSV files
    full_df = collect_image_records()
    train_df, val_df, test_df = _create_stratified_splits(full_df)

    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    splits = {"train": train_df, "val": val_df, "test": test_df}

    # Save summary tables and visual checks for split quality
    _save_split_summary(splits)
    _plot_split_distribution(splits)

    return splits


def main() -> None:
    splits = create_splits()
    for split_name, split_df in splits.items():
        print(f"{split_name}: {len(split_df)} images")


if __name__ == "__main__":
    main()
