"""Categorical feature frequency analysis."""

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


def plot_categorical_frequencies(
    df: pd.DataFrame,
    categorical_columns: list[str],
    target_column: str,
    plots_dir: Path,
    cfg: dict[str, Any],
) -> list[Path]:
    """Generate count plots for categorical features and attrition breakdowns."""
    saved_paths: list[Path] = []
    dpi = int(cfg.get("dpi", 150))
    figsize = get_figsize(cfg)
    plot_format = cfg.get("plot_format", "png")

    freq_records: list[dict[str, Any]] = []

    for col in categorical_columns:
        if col not in df.columns:
            continue

        value_counts = df[col].value_counts()
        for val, count in value_counts.items():
            freq_records.append({"feature": col, "category": str(val), "count": int(count)})

        fig, ax = plt.subplots(figsize=figsize)
        order = value_counts.index.tolist()
        sns.countplot(data=df, x=col, order=order, ax=ax, palette="viridis")
        ax.set_title(f"Frequency Distribution - {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        png_path = save_matplotlib_fig(fig, plots_dir / f"catfreq_{col}.{plot_format}", dpi=dpi)
        saved_paths.append(png_path)

        plotly_fig = px.bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            title=f"Frequency Distribution - {col}",
            labels={"x": col, "y": "Count"},
        )
        html_path = save_plotly_fig(plotly_fig, plots_dir / f"catfreq_{col}.html")
        saved_paths.append(html_path)

        if target_column in df.columns:
            fig, ax = plt.subplots(figsize=figsize)
            ct = pd.crosstab(df[col], df[target_column], normalize="index") * 100
            ct.plot(kind="bar", stacked=False, ax=ax, colormap="Set2")
            ax.set_title(f"Attrition Rate by {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Percentage (%)")
            ax.legend(title=target_column)
            plt.xticks(rotation=45, ha="right")
            attr_path = save_matplotlib_fig(
                fig, plots_dir / f"attrition_by_{col}.{plot_format}", dpi=dpi
            )
            saved_paths.append(attr_path)

    freq_df = pd.DataFrame(freq_records)
    freq_df.to_csv(plots_dir.parent / "categorical_frequencies.csv", index=False)
    logger.info("Saved %d categorical frequency plots", len(saved_paths))
    return saved_paths
