"""
Pipeline Entry Point
Parses CLI arguments and routes to the selected pipeline task
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# Set TensorFlow log controls before importing modules that load TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from config import SPLITS_DIR
from eda import run_eda
from error_analysis import run_error_analysis
from evaluate import run_evaluation
from make_splits import create_splits
from predict import run_prediction
from train import run_training


def _build_parser() -> argparse.ArgumentParser:
    # Configure command-line interface for all supported pipeline modes
    parser = argparse.ArgumentParser(description="Recyclable waste classification pipeline controller")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["make_splits", "eda", "train", "evaluate", "error_analysis", "predict", "run_all"],
        help="Pipeline mode to execute",
    )

    # Optional arguments used by predict mode
    parser.add_argument("--image_path", help="Required for predict mode")
    parser.add_argument("--model_path", help="Optional override model path for predict mode")
    parser.add_argument("--class_names_path", help="Optional override class names path for predict mode")
    parser.add_argument("--save_csv", help="Optional CSV path to append prediction results in predict mode")
    return parser


def _resolve_predict_smoke_image() -> str | None:
    # Reuse the first test image path for a lightweight end-to-end smoke check.
    test_csv_path = SPLITS_DIR / "test.csv"
    if not test_csv_path.exists():
        print("Test split file not found - skipping optional predict smoke test")
        return None

    try:
        with test_csv_path.open("r", encoding="utf-8", newline="") as file_obj:
            first_row = next(csv.DictReader(file_obj), None)
    except Exception as exc:
        print(f"Could not read test split ({exc}) - skipping optional predict smoke test")
        return None

    if not first_row:
        print("Test split is empty - skipping optional predict smoke test")
        return None

    image_path = str(first_row.get("filepath", "")).strip()
    if not image_path:
        print("Missing filepath in first test row - skipping optional predict smoke test")
        return None

    if not Path(image_path).exists():
        print("First test image does not exist - skipping optional predict smoke test")
        return None

    return image_path


def _run_all(args: argparse.Namespace) -> None:
    # Skipping existing split files avoids accidental split drift between runs.
    split_paths = [SPLITS_DIR / "train.csv", SPLITS_DIR / "val.csv", SPLITS_DIR / "test.csv"]
    if all(path.exists() for path in split_paths):
        print("Splits already exist - skipping make_splits")
    else:
        print("-> Running make_splits...")
        create_splits()
        print("OK: make_splits complete")

    print("-> Running EDA...")
    run_eda()
    print("OK: EDA complete")

    print("-> Running training...")
    run_training()
    print("OK: training complete")

    print("-> Running evaluation...")
    run_evaluation()
    print("OK: evaluation complete")

    print("-> Running error analysis...")
    run_error_analysis()
    print("OK: error analysis complete")

    smoke_image_path = _resolve_predict_smoke_image()
    if not smoke_image_path:
        return

    # Keep predict output quiet in CLI when TensorFlow starts.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    print("-> Running optional predict smoke test...")
    run_prediction(
        image_path=smoke_image_path,
        model_path=args.model_path,
        class_names_path=args.class_names_path,
        save_csv=args.save_csv,
    )
    print("OK: optional predict smoke test complete")


def main() -> None:
    # Parse command-line arguments
    args = _build_parser().parse_args()

    # Create stratified train/validation/test split files
    if args.mode == "make_splits":
        create_splits()
        return

    # Run exploratory data analysis pipeline
    if args.mode == "eda":
        run_eda()
        return

    # Train baseline and transfer-learning models
    if args.mode == "train":
        run_training()
        return

    # Evaluate trained models and save metrics/plots
    if args.mode == "evaluate":
        run_evaluation()
        return

    # Analyze misclassifications from evaluation outputs
    if args.mode == "error_analysis":
        run_error_analysis()
        return

    # Run single-image prediction with optional CSV logging
    if args.mode == "predict":
        if not args.image_path:
            raise ValueError("--image_path is required when --mode predict")
        run_prediction(
            image_path=args.image_path,
            model_path=args.model_path,
            class_names_path=args.class_names_path,
            save_csv=args.save_csv,
        )
        return

    # Run the full pipeline end to end in one command.
    if args.mode == "run_all":
        _run_all(args)
        return


if __name__ == "__main__":
    main()
