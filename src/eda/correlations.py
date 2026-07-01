"""Correlation analysis and heatmaps."""

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


def plot_correlation_heatmaps(
    df: pd.DataFrame,
    numerical_columns: list[str],
    plots_dir: Path,
    cfg: dict[str, Any],
) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    """Generate Pearson and Spearman correlation heatmaps."""
    saved_paths: list[Path] = []
    dpi = int(cfg.get("dpi", 150))
    figsize = get_figsize(cfg)
    plot_format = cfg.get("plot_format", "png")
    corr_cfg = cfg.get("correlation", {})
    annot = bool(corr_cfg.get("annot", True))
    cmap = corr_cfg.get("cmap", "RdBu_r")

    num_df = df[numerical_columns].select_dtypes(include="number")
    pearson_corr = num_df.corr(method="pearson")
    spearman_corr = num_df.corr(method="spearman")

    for method, corr_matrix in [("pearson", pearson_corr), ("spearman", spearman_corr)]:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=annot,
            fmt=".2f",
            cmap=cmap,
            center=0,
            square=True,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(f"{method.capitalize()} Correlation Heatmap")
        png_path = save_matplotlib_fig(
            fig, plots_dir / f"correlation_{method}.{plot_format}", dpi=dpi
        )
        saved_paths.append(png_path)

        plotly_fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title=f"{method.capitalize()} Correlation Heatmap",
            aspect="auto",
        )
        html_path = save_plotly_fig(plotly_fig, plots_dir / f"correlation_{method}.html")
        saved_paths.append(html_path)

    pearson_corr.to_csv(plots_dir.parent / "correlation_pearson.csv")
    spearman_corr.to_csv(plots_dir.parent / "correlation_spearman.csv")

    logger.info("Saved correlation heatmaps")
    return saved_paths, pearson_corr, spearman_corr
