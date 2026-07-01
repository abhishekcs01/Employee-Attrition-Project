"""Numerical feature distribution analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from src.eda.plot_utils import get_figsize, save_matplotlib_fig, save_plotly_fig

logger = logging.getLogger(__name__)


def plot_numerical_distributions(
    df: pd.DataFrame,
    numerical_columns: list[str],
    plots_dir: Path,
    cfg: dict[str, Any],
) -> list[Path]:
    """Generate histograms for all numerical features."""
    saved_paths: list[Path] = []
    dpi = int(cfg.get("dpi", 150))
    figsize = get_figsize(cfg)
    plot_format = cfg.get("plot_format", "png")

    for col in numerical_columns:
        if col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        png_path = save_matplotlib_fig(fig, plots_dir / f"hist_{col}.{plot_format}", dpi=dpi)
        saved_paths.append(png_path)

        plotly_fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        html_path = save_plotly_fig(plotly_fig, plots_dir / f"hist_{col}.html")
        saved_paths.append(html_path)

    logger.info("Saved %d distribution plots", len(saved_paths))
    return saved_paths
