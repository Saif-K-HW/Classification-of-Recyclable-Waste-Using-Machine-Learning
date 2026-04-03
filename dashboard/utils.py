"""
Dashboard Utilities
Shared helpers for artifact discovery, model selection, and dashboard-triggered inference.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Define project-relative root folders used across all dashboard scopes.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class DashboardScope:
    id: str
    label: str
    results_dir: Path
    models_dir: Path
    class_names_path: Path


def _resolve_pointer_model(models_dir: Path) -> Path | None:
    # Read final_model.txt when present so dashboard defaults match pipeline best-model pointer.
    pointer_path = models_dir / "final_model.txt"
    if not pointer_path.exists():
        return None

    pointer_value = pointer_path.read_text(encoding="utf-8").strip()
    if not pointer_value:
        return None

    candidate = Path(pointer_value)
    if not candidate.is_absolute():
        candidate = models_dir / candidate

    if candidate.exists():
        return candidate.resolve()
    return None


def discover_scopes() -> List[DashboardScope]:
    # Always include the stable pipeline scope rooted at results/ and models/.
    scopes = [
        DashboardScope(
            id="stable",
            label="Stable Pipeline (results/)",
            results_dir=DEFAULT_RESULTS_DIR,
            models_dir=DEFAULT_MODELS_DIR,
            class_names_path=DEFAULT_MODELS_DIR / "class_names.json",
        )
    ]

    # Add experiment scopes dynamically when experiments/<name>/results exists.
    if EXPERIMENTS_DIR.exists():
        for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
            if not exp_dir.is_dir():
                continue

            results_dir = exp_dir / "results"
            models_dir = results_dir / "models"
            class_names_path = models_dir / "class_names.json"

            if results_dir.exists():
                scopes.append(
                    DashboardScope(
                        id=exp_dir.name,
                        label=f"Experiment: {exp_dir.name}",
                        results_dir=results_dir,
                        models_dir=models_dir,
                        class_names_path=class_names_path,
                    )
                )

    return scopes


def get_scope_map() -> Dict[str, DashboardScope]:
    # Build a fast id -> scope lookup map for stateful UI controls.
    return {scope.id: scope for scope in discover_scopes()}


def list_model_files(scope: DashboardScope) -> List[Path]:
    # Return model checkpoints newest-first so latest runs appear first in selectors.
    if not scope.models_dir.exists():
        return []

    model_files = list(scope.models_dir.glob("*.keras"))
    return sorted(model_files, key=lambda path: path.stat().st_mtime, reverse=True)


def resolve_default_model(scope: DashboardScope) -> Path | None:
    # Prefer explicit pointer model first, then fall back to most recently updated checkpoint.
    pointer_model = _resolve_pointer_model(scope.models_dir)
    if pointer_model is not None:
        return pointer_model

    models = list_model_files(scope)
    if models:
        return models[0]

    return None


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    # Read optional CSV artifacts safely without breaking dashboard rendering on parse errors.
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def path_last_updated(path: Path) -> str:
    # Format file timestamp for artifact status cards.
    if not path.exists():
        return "Missing"
    return pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")


def get_classification_reports(scope: DashboardScope) -> Dict[str, Path]:
    # Collect per-model classification reports while skipping shared global metrics folder.
    reports: Dict[str, Path] = {}
    metrics_dir = scope.results_dir / "metrics"
    if not metrics_dir.exists():
        return reports

    for model_dir in sorted(metrics_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "global":
            continue
        report_path = model_dir / "classification_report.csv"
        if report_path.exists():
            reports[model_dir.name] = report_path

    return reports


def get_confusion_images(scope: DashboardScope) -> Dict[str, Path]:
    # Map each model name to its confusion matrix image when available.
    images: Dict[str, Path] = {}
    plots_dir = scope.results_dir / "plots"
    if not plots_dir.exists():
        return images

    for model_dir in sorted(plots_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "global":
            continue
        image_path = model_dir / "confusion_matrix.png"
        if image_path.exists():
            images[model_dir.name] = image_path

    return images


def get_common_confusion_images(scope: DashboardScope) -> Dict[str, Path]:
    # Map each model name to the common-confusions visual when available.
    images: Dict[str, Path] = {}
    plots_dir = scope.results_dir / "plots"
    if not plots_dir.exists():
        return images

    for model_dir in sorted(plots_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "global":
            continue
        image_path = model_dir / "common_confusions.png"
        if image_path.exists():
            images[model_dir.name] = image_path

    return images


def list_misclassified_images(scope: DashboardScope, model_name: str, limit: int = 12) -> List[Path]:
    # Return newest misclassified examples for the selected model gallery.
    target_dir = scope.results_dir / "misclassified" / model_name
    if not target_dir.exists():
        return []

    images = [
        path
        for path in target_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ]
    images = sorted(images, key=lambda path: path.stat().st_mtime, reverse=True)
    return images[:limit]


def list_any_misclassified_images(scope: DashboardScope, limit: int = 12) -> List[Path]:
    # Fallback gallery loader across all model folders when the selected one is empty.
    misclassified_root = scope.results_dir / "misclassified"
    if not misclassified_root.exists():
        return []

    collected: List[Path] = []
    for model_dir in sorted(misclassified_root.iterdir()):
        if not model_dir.is_dir():
            continue

        model_images = [
            path
            for path in model_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ]
        collected.extend(model_images)

    collected = sorted(collected, key=lambda path: path.stat().st_mtime, reverse=True)
    return collected[:limit]


def run_uploaded_prediction(
    image_bytes: bytes,
    model_path: Path,
    class_names_path: Path | None,
) -> Dict[str, object]:
    # Keep heavy prediction import local so utility module stays lightweight at dashboard startup.
    from predict import predict_image

    # Write upload bytes to a temporary image path compatible with existing predict pipeline.
    suffix = ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(image_bytes)
        temp_path = Path(temp_file.name)

    try:
        # Reuse src/predict.py logic so dashboard inference matches CLI behavior.
        return predict_image(
            image_path=str(temp_path),
            model_path=str(model_path),
            class_names_path=str(class_names_path) if class_names_path and class_names_path.exists() else None,
        )
    finally:
        # Always clean up temporary upload files after inference.
        temp_path.unlink(missing_ok=True)


def append_prediction_row(csv_path: Path, image_name: str, result: Dict[str, object]) -> None:
    # Ensure destination directory exists before appending dashboard prediction logs.
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Format top-3 predictions into one compact text field for quick review.
    top3 = result.get("top_3", [])
    top3_text = "; ".join(
        f"{item.get('label', 'unknown')} ({float(item.get('confidence', 0.0)):.2f}%)"
        for item in top3
    )

    # Build a one-row frame and append it without rewriting existing history.
    row = pd.DataFrame(
        [
            {
                "image_name": image_name,
                "predicted_class": result.get("top_prediction", ""),
                "confidence": result.get("confidence_percent", 0.0),
                "top3_classes": top3_text,
                "model_name": result.get("model_name", ""),
            }
        ]
    )

    row.to_csv(csv_path, mode="a", index=False, header=not csv_path.exists())


def get_dashboard_artifact_paths(scope: DashboardScope) -> Dict[str, Path]:
    # Centralize artifact paths used by overview and artifact-browser pages.
    metrics_global = scope.results_dir / "metrics" / "global"
    plots_global = scope.results_dir / "plots" / "global"

    return {
        "Model comparison": metrics_global / "model_comparison.csv",
        "Evaluation summary": metrics_global / "evaluation_summary.csv",
        "Top confusions": metrics_global / "top_confusions.csv",
        "Worst classes": metrics_global / "worst_classes.csv",
        "Calibration metrics": metrics_global / "calibration_metrics.csv",
        "Reliability bins": metrics_global / "reliability_bins.csv",
        "Reliability curve": plots_global / "reliability_curve.png",
    }


def list_missing_calibration_artifacts(scope: DashboardScope) -> List[Path]:
    # Check calibration outputs expected by both calibration and overview pages.
    metrics_global = scope.results_dir / "metrics" / "global"
    plots_global = scope.results_dir / "plots" / "global"
    required_paths = [
        metrics_global / "calibration_metrics.csv",
        metrics_global / "reliability_bins.csv",
        plots_global / "reliability_curve.png",
    ]
    return [path for path in required_paths if not path.exists()]


def _build_scope_output_paths(scope: DashboardScope):
    # Build output-path object expected by calibration runner for stable and experiment scopes.
    from experiment_utils import OutputPaths, get_default_output_paths

    if scope.id == "stable":
        return get_default_output_paths()

    return OutputPaths(
        results_dir=scope.results_dir,
        metrics_dir=scope.results_dir / "metrics",
        metrics_global_dir=scope.results_dir / "metrics" / "global",
        plots_dir=scope.results_dir / "plots",
        plots_global_dir=scope.results_dir / "plots" / "global",
        logs_dir=scope.results_dir / "logs",
        misclassified_dir=scope.results_dir / "misclassified",
        models_dir=scope.models_dir,
        class_names_path=scope.class_names_path,
    )


def ensure_calibration_artifacts(scope: DashboardScope) -> tuple[bool, str]:
    # Short-circuit when all required calibration files already exist.
    missing_paths = list_missing_calibration_artifacts(scope)
    if not missing_paths:
        return True, "Calibration artifacts already exist."

    # Resolve a checkpoint path to run calibration against.
    model_path = resolve_default_model(scope)
    if model_path is None:
        return False, "No model checkpoint found to run calibration."

    try:
        # Trigger calibration generation using scope-aware output paths.
        from calibration import run_calibration

        output_paths = _build_scope_output_paths(scope)
        run_calibration(model_path=str(model_path), output_paths=output_paths)
    except Exception as exc:
        return False, f"Calibration generation failed: {exc}"

    # Confirm expected outputs exist after the run.
    remaining_missing = list_missing_calibration_artifacts(scope)
    if remaining_missing:
        return False, "Calibration ran but some outputs are still missing."

    return True, f"Calibration artifacts generated using {model_path.name}."
