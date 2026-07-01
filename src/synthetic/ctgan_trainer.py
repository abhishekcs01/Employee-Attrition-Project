"""CTGAN synthetic data training utilities."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def train_ctgan(
    real_data: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    output_dir: Path,
    model_path: Path,
) -> Any:
    """Train a CTGAN synthesizer on real tabular data."""
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer

    from src.utils.hardware import get_hardware_profile

    ctgan_cfg = cfg.get("ctgan", {})
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)

    hardware = get_hardware_profile(probe_ml_backends=False)
    enable_gpu = bool(ctgan_cfg.get("enable_gpu", hardware.uses_accelerated))

    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=int(ctgan_cfg.get("epochs", 300)),
        batch_size=int(ctgan_cfg.get("batch_size", 500)),
        verbose=bool(ctgan_cfg.get("verbose", False)),
        enable_gpu=enable_gpu,
    )

    logger.info("Training CTGAN on %d rows", len(real_data))
    synthesizer.fit(real_data)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(synthesizer, f)
    logger.info("Saved CTGAN model: %s", model_path)

    return synthesizer


def generate_synthetic_datasets(
    synthesizer: Any,
    sizes: list[int],
    output_dir: Path,
    *,
    run_id: str,
) -> dict[int, Path]:
    """Generate synthetic datasets at multiple sizes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}

    for size in sizes:
        synthetic = synthesizer.sample(num_rows=size)
        versioned_path = output_dir / f"synthetic_{size}_{run_id}.csv"
        synthetic.to_csv(versioned_path, index=False)
        paths[size] = versioned_path
        logger.info("Generated synthetic data: %s (%d rows)", versioned_path, size)

    return paths
