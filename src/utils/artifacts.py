"""Shared model artifact loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def load_model_artifacts(models_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load best model pipeline and training metadata from a models directory."""
    metadata_path = models_dir / "training_metadata.json"
    model_path = models_dir / "best_model.joblib"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Training metadata not found: {metadata_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found: {model_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pipeline = joblib.load(model_path)
    return pipeline, metadata
