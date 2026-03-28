"""
Single Image Prediction
Runs inference with the final trained model and optional CSV logging
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List

# Reduce noisy TensorFlow startup logs in CLI output
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import tensorflow as tf

from config import IMAGE_SIZE, MODELS_DIR
from data_loader import load_class_names


def _preprocess_image(image_path: Path) -> tf.Tensor:
    # Load image, resize to model input shape, and normalize to [0, 1]
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return tf.expand_dims(image, axis=0)


def _resolve_model_path(model_path: str | None = None) -> Path:
    # Prefer explicit model path, then final_model.txt, then default best checkpoint
    if model_path:
        return Path(model_path)

    final_pointer = MODELS_DIR / "final_model.txt"
    if final_pointer.exists():
        pointer_value = final_pointer.read_text(encoding="utf-8").strip()
        if pointer_value:
            candidate = Path(pointer_value)
            if not candidate.is_absolute():
                candidate = MODELS_DIR / pointer_value
            if candidate.exists():
                return candidate

    return MODELS_DIR / "mobilenet_finetuned_best.keras"


def predict_image(
    image_path: str,
    model_path: str | None = None,
    class_names_path: str | None = None,
) -> Dict[str, object]:
    # Resolve file paths and validate required artifacts
    image_file = Path(image_path)
    model_file = _resolve_model_path(model_path)

    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}. Train first or update models/final_model.txt.")

    class_names = load_class_names(Path(class_names_path) if class_names_path else None)
    model = tf.keras.models.load_model(model_file)

    # Run inference and collect top-3 predicted classes
    image_tensor = _preprocess_image(image_file)
    probabilities = model.predict(image_tensor, verbose=0)[0]

    top_indices = np.argsort(probabilities)[::-1][:3]
    top3: List[Dict[str, float]] = [
        {
            "label": class_names[int(idx)],
            "confidence": float(round(probabilities[int(idx)] * 100, 2)),
        }
        for idx in top_indices
    ]

    result = {
        "model_name": model_file.name,
        "model_path": str(model_file),
        "image_name": image_file.name,
        "top_prediction": top3[0]["label"],
        "confidence_percent": top3[0]["confidence"],
        "top_3": top3,
    }
    return result


def _append_prediction_csv(save_csv: str, image_path: str, result: Dict[str, object]) -> None:
    # Append one prediction row so repeated CLI calls build an experiment log
    csv_path = Path(save_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    top3_str = "; ".join(
        f"{item['label']} ({item['confidence']:.2f}%)"
        for item in result["top_3"]
    )

    row_df = pd.DataFrame(
        [
            {
                "image_path": image_path,
                "predicted_class": result["top_prediction"],
                "confidence": result["confidence_percent"],
                "top3_classes": top3_str,
                "model_name": result["model_name"],
            }
        ]
    )

    row_df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def _safe_console_text(value: object) -> str:
    # Replace unsupported characters to avoid terminal encoding errors
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(value).encode(encoding, errors="replace").decode(encoding, errors="replace")


def run_prediction(
    image_path: str,
    model_path: str | None = None,
    class_names_path: str | None = None,
    save_csv: str | None = None,
) -> Dict[str, object]:
    # Run prediction and print readable CLI summary
    result = predict_image(image_path=image_path, model_path=model_path, class_names_path=class_names_path)

    print(f"Model: {_safe_console_text(result['model_name'])}")
    print(f"Image: {_safe_console_text(result['image_name'])}")
    print(f"Top prediction: {_safe_console_text(result['top_prediction'])}")
    print(f"Confidence: {result['confidence_percent']:.2f}%")
    print("Top 3 classes:")
    for item in result["top_3"]:
        print(f"  - {_safe_console_text(item['label'])}: {item['confidence']:.2f}%")

    if save_csv:
        _append_prediction_csv(save_csv=save_csv, image_path=image_path, result=result)
        print(f"Saved prediction row to: {save_csv}")

    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    # Configure CLI options for one-image inference
    parser = argparse.ArgumentParser(description="Predict recyclable waste class for one image.")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument(
        "--model_path",
        default=None,
        help="Optional model path. If omitted, we use final_model.txt then mobilenet_finetuned_best.keras.",
    )
    parser.add_argument(
        "--class_names_path",
        default=str(MODELS_DIR / "class_names.json"),
        help="Path to class_names.json",
    )
    parser.add_argument(
        "--save_csv",
        default=None,
        help="Optional CSV path to append prediction rows for appendix tables.",
    )
    return parser


def main() -> None:
    # Parse CLI args and execute prediction
    args = _build_arg_parser().parse_args()
    run_prediction(
        image_path=args.image_path,
        model_path=args.model_path,
        class_names_path=args.class_names_path,
        save_csv=args.save_csv,
    )


if __name__ == "__main__":
    main()
