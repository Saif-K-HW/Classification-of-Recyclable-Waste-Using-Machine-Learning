"""
Experiment Utilities
Loads experiment overrides and builds isolated output paths for experiment runs
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import config

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


@dataclass(frozen=True)
class OutputPaths:
    # Group all output directories used by train/evaluate/error analysis/calibration.
    results_dir: Path
    metrics_dir: Path
    metrics_global_dir: Path
    plots_dir: Path
    plots_global_dir: Path
    logs_dir: Path
    misclassified_dir: Path
    models_dir: Path
    class_names_path: Path

    def model_metrics_dir(self, model_name: str) -> Path:
        return self.metrics_dir / model_name

    def model_plots_dir(self, model_name: str) -> Path:
        return self.plots_dir / model_name

    def ensure_directories(self, model_names: List[str] | None = None) -> None:
        # Keep every run self-contained so experiment outputs never touch the stable pipeline results.
        base_paths = [
            self.results_dir,
            self.metrics_dir,
            self.metrics_global_dir,
            self.plots_dir,
            self.plots_global_dir,
            self.logs_dir,
            self.misclassified_dir,
            self.models_dir,
        ]

        for path in base_paths:
            path.mkdir(parents=True, exist_ok=True)

        for model_name in model_names or []:
            self.model_metrics_dir(model_name).mkdir(parents=True, exist_ok=True)
            self.model_plots_dir(model_name).mkdir(parents=True, exist_ok=True)


def get_default_output_paths() -> OutputPaths:
    # This preserves the current behavior for normal pipeline modes.
    return OutputPaths(
        results_dir=config.RESULTS_DIR,
        metrics_dir=config.METRICS_DIR,
        metrics_global_dir=config.METRICS_GLOBAL_DIR,
        plots_dir=config.PLOTS_DIR,
        plots_global_dir=config.PLOTS_GLOBAL_DIR,
        logs_dir=config.LOGS_DIR,
        misclassified_dir=config.MISCLASSIFIED_DIR,
        models_dir=config.MODELS_DIR,
        class_names_path=config.MODELS_DIR / "class_names.json",
    )


def resolve_experiment_config_path(exp_name: str, override_path: str | None = None) -> Path:
    # Allow a custom path, but default to the convention requested in each experiment folder.
    if override_path:
        return Path(override_path)

    experiment_dir = config.EXPERIMENTS_DIR / exp_name
    default_candidates = [
        experiment_dir / "config_override.json",
        experiment_dir / "config_override.yaml",
        experiment_dir / "config_override.yml",
    ]
    for candidate in default_candidates:
        if candidate.exists():
            return candidate

    # Keep a stable default filename in the message so setup issues are easy to spot.
    return experiment_dir / "config_override.json"


def load_override_config(override_path: str | Path) -> Dict[str, Any]:
    # Support JSON by default and YAML when PyYAML is available.
    config_path = Path(override_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Override config not found: {config_path}")

    suffix = config_path.suffix.lower()
    raw_text = config_path.read_text(encoding="utf-8")

    if suffix == ".json":
        data = json.loads(raw_text)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML override files.")
        data = yaml.safe_load(raw_text)
    else:
        raise ValueError("Override config must be .json, .yaml, or .yml")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Override config must be a JSON/YAML object")
    return data


def create_experiment_output_paths(exp_name: str, results_dir_override: str | None = None) -> OutputPaths:
    # The requested directory layout lives under experiments/<exp_name>/results/...
    experiment_root = config.EXPERIMENTS_DIR / exp_name

    if results_dir_override:
        results_dir = Path(results_dir_override)
        if not results_dir.is_absolute():
            results_dir = (experiment_root / results_dir).resolve()
    else:
        results_dir = experiment_root / "results"

    paths = OutputPaths(
        results_dir=results_dir,
        metrics_dir=results_dir / "metrics",
        metrics_global_dir=results_dir / "metrics" / "global",
        plots_dir=results_dir / "plots",
        plots_global_dir=results_dir / "plots" / "global",
        logs_dir=results_dir / "logs",
        misclassified_dir=results_dir / "misclassified",
        models_dir=results_dir / "models",
        class_names_path=(results_dir / "models" / "class_names.json"),
    )
    paths.ensure_directories()
    return paths
