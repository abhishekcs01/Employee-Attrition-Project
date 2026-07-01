"""Target variable and class imbalance analysis."""

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


def analyze_class_imbalance(
    df: pd.DataFrame,
    target_column: str,
    plots_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Analyze and plot class imbalance for the target variable."""
    dpi = int(cfg.get("dpi", 150))
    figsize = get_figsize(cfg)
    plot_format = cfg.get("plot_format", "png")
    positive_class = cfg.get("target_analysis", {}).get("positive_class", "Yes")

    counts = df[target_column].value_counts()
    total = len(df)
    positive_count = int(counts.get(positive_class, 0))
    negative_count = total - positive_count
    imbalance_ratio = round(
        max(positive_count, negative_count) / max(min(positive_count, negative_count), 1), 2
    )

    stats = {
        "target_column": target_column,
        "total_samples": total,
        "class_counts": counts.to_dict(),
        "positive_class": positive_class,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate_pct": round(positive_count / total * 100, 2),
        "imbalance_ratio": imbalance_ratio,
    }

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 1.5, figsize[1]))

    sns.countplot(data=df, x=target_column, ax=axes[0], palette="Set2")
    axes[0].set_title("Class Distribution")
    axes[0].set_xlabel(target_column)
    axes[0].set_ylabel("Count")

    axes[1].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=sns.color_palette("Set2", len(counts)),
        startangle=90,
    )
    axes[1].set_title(f"Class Imbalance (Ratio: {imbalance_ratio}:1)")

    save_matplotlib_fig(fig, plots_dir / f"class_imbalance.{plot_format}", dpi=dpi)

    plotly_fig = px.bar(
        x=counts.index.astype(str),
        y=counts.values,
        title="Class Distribution",
        labels={"x": target_column, "y": "Count"},
    )
    save_plotly_fig(plotly_fig, plots_dir / "class_imbalance.html")

    logger.info(
        "Attrition rate: %.2f%% (imbalance ratio %s:1)", stats["positive_rate_pct"], imbalance_ratio
    )
    return stats
