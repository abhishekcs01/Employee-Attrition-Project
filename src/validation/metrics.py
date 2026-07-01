"""Synthetic data validation metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _ks_scores(real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for col in columns:
        if col in real.columns and col in synthetic.columns:
            stat, _ = stats.ks_2samp(real[col], synthetic[col])
            scores[col] = float(stat)
    return scores


def _chi2_scores(
    real: pd.DataFrame, synthetic: pd.DataFrame, columns: list[str]
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for col in columns:
        if col not in real.columns or col not in synthetic.columns:
            continue
        real_counts = real[col].value_counts(normalize=True)
        syn_counts = synthetic[col].value_counts(normalize=True)
        categories = sorted(set(real_counts.index) | set(syn_counts.index))
        observed = np.array([syn_counts.get(c, 0.0) for c in categories])
        expected = np.array([real_counts.get(c, 0.0) for c in categories])
        if expected.sum() == 0:
            continue
        expected = expected / expected.sum()
        observed = observed / observed.sum() if observed.sum() > 0 else observed
        chi2 = float(((observed - expected) ** 2 / (expected + 1e-9)).sum())
        scores[col] = chi2
    return scores


def validate_synthetic_dataset(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    """Compute validation metrics for a synthetic dataset."""
    ks = _ks_scores(real, synthetic, numerical_columns)
    chi2 = _chi2_scores(real, synthetic, categorical_columns)

    ks_mean = float(np.mean(list(ks.values()))) if ks else 1.0
    composite_score = max(0.0, 1.0 - ks_mean)

    return {
        "dataset_size": len(synthetic),
        "ks_scores": ks,
        "chi2_scores": chi2,
        "composite_score": round(composite_score, 4),
        "passed": composite_score >= 0.85,
    }


def select_best_synthetic(
    real: pd.DataFrame,
    synthetic_paths: dict[int, Path],
    cfg: dict[str, Any],
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    """Validate all synthetic sizes and select the best dataset."""
    all_results: dict[str, Any] = {}
    best_size: int | None = None
    best_score = -1.0
    best_path: Path | None = None

    for size, path in synthetic_paths.items():
        if not path.exists():
            logger.warning("Synthetic file not found: %s", path)
            continue
        synthetic = pd.read_csv(path)
        result = validate_synthetic_dataset(real, synthetic, numerical_columns, categorical_columns)
        all_results[str(size)] = result
        if result["composite_score"] > best_score:
            best_score = result["composite_score"]
            best_size = size
            best_path = path

    pass_threshold = float(cfg.get("pass_threshold", 0.85))
    return {
        "best_size": best_size,
        "best_path": str(best_path) if best_path else None,
        "composite_score": round(best_score, 4),
        "passed": best_score >= pass_threshold,
        "all_results": all_results,
    }
