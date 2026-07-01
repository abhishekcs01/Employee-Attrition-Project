"""Outlier detection and boxplot analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.eda.plot_utils import get_figsize, save_matplotlib_fig

logger = logging.getLogger(__name__)


def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = 1.5,
) -> tuple[int, float, float]:
    """Detect outliers using IQR method. Returns count, lower bound, upper bound."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    outlier_count = int(((series < lower) | (series > upper)).sum())
    return outlier_count, float(lower), float(upper)


def plot_outlier_boxplots(
    df: pd.DataFrame,
    numerical_columns: list[str],
    plots_dir: Path,
    cfg: dict[str, Any],
) -> tuple[list[Path], pd.DataFrame]:
    """Generate boxplots and IQR outlier summary for numerical features."""
    saved_paths: list[Path] = []
    dpi = int(cfg.get("dpi", 150))
    figsize = get_figsize(cfg)
    plot_format = cfg.get("plot_format", "png")
    multiplier = float(cfg.get("outlier", {}).get("iqr_multiplier", 1.5))

    outlier_records: list[dict[str, Any]] = []

    for col in numerical_columns:
        if col not in df.columns:
            continue

        outlier_count, lower, upper = detect_outliers_iqr(df[col].dropna(), multiplier)
        outlier_records.append(
            {
                "feature": col,
                "outlier_count": outlier_count,
                "outlier_pct": round(outlier_count / len(df) * 100, 2),
                "lower_bound": lower,
                "upper_bound": upper,
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
        )

        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(y=df[col].dropna(), ax=ax, color="coral")
        ax.set_title(f"Boxplot - {col} (Outliers: {outlier_count})")
        ax.set_ylabel(col)
        png_path = save_matplotlib_fig(fig, plots_dir / f"boxplot_{col}.{plot_format}", dpi=dpi)
        saved_paths.append(png_path)

    summary_df = pd.DataFrame(outlier_records)
    summary_path = plots_dir.parent / "outlier_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved %d boxplots and outlier summary", len(saved_paths))
    return saved_paths, summary_df
