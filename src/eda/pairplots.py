"""Pair plot analysis for top correlated features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import seaborn as sns

from src.eda.plot_utils import save_matplotlib_fig

logger = logging.getLogger(__name__)


def select_top_correlated_features(
    corr_matrix: pd.DataFrame,
    target_col: str | None = None,
    max_features: int = 6,
) -> list[str]:
    """Select top features by absolute correlation with target or mean correlation."""
    if target_col and target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].drop(target_col, errors="ignore").abs()
        return target_corr.nlargest(max_features).index.tolist()

    mean_corr = corr_matrix.abs().mean().sort_values(ascending=False)
    return mean_corr.head(max_features).index.tolist()


def plot_pairplot(
    df: pd.DataFrame,
    features: list[str],
    target_column: str,
    plots_dir: Path,
    cfg: dict[str, Any],
) -> list[Path]:
    """Generate pair plot for selected features colored by target."""
    saved_paths: list[Path] = []
    dpi = int(cfg.get("dpi", 150))
    plot_format = cfg.get("plot_format", "png")
    pairplot_cfg = cfg.get("pairplot", {})
    sample_size = int(pairplot_cfg.get("sample_size", 500))

    plot_cols = [c for c in features if c in df.columns]
    if target_column in df.columns and target_column not in plot_cols:
        plot_cols.append(target_column)

    if len(plot_cols) < 2:
        logger.warning("Insufficient features for pair plot")
        return saved_paths

    sample_df = df[plot_cols].sample(n=min(sample_size, len(df)), random_state=42)

    g = sns.pairplot(
        sample_df,
        hue=target_column if target_column in sample_df.columns else None,
        diag_kind="kde",
        corner=False,
        plot_kws={"alpha": 0.6, "s": 20},
    )
    g.fig.suptitle("Pair Plot - Top Correlated Features", y=1.02)
    png_path = save_matplotlib_fig(g.fig, plots_dir / f"pairplot.{plot_format}", dpi=dpi)
    saved_paths.append(png_path)
    logger.info("Saved pair plot with %d features", len(plot_cols))
    return saved_paths
