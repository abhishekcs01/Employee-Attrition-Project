"""Plotting helpers for Phase 1 notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from notebooks._shared.config import ATTRITION_COLORS, POSITIVE_CLASS, TARGET


def setup_plot_style() -> None:
    """Configure matplotlib and seaborn defaults for consistent report-style plots."""
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def plot_distribution(
    data: pd.DataFrame,
    col: str,
    hue: str | None = None,
    kind: str = "kde",
    bins: int = 30,
) -> plt.Figure:
    """Plot a numeric distribution, optionally split by attrition status."""
    fig, ax = plt.subplots(figsize=(10, 6))
    title = f"{col} by Attrition Status" if hue else f"Distribution of {col}"
    if kind == "kde" and hue:
        sns.kdeplot(
            data=data,
            x=col,
            hue=hue,
            palette=ATTRITION_COLORS,
            ax=ax,
            fill=True,
            alpha=0.4,
        )
    elif kind == "hist":
        if hue:
            for level in data[hue].unique():
                subset = data[data[hue] == level]
                ax.hist(
                    subset[col],
                    bins=bins,
                    alpha=0.6,
                    label=level,
                    color=ATTRITION_COLORS.get(level),
                    edgecolor="white",
                )
            ax.legend(title=hue)
        else:
            ax.hist(data[col], bins=bins, color="#2E86AB", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("Density" if kind == "kde" else "Count")
    plt.tight_layout()
    return fig


def attrition_rate_by(data: pd.DataFrame, column: str) -> pd.Series:
    """Return attrition rate (%) grouped by a categorical column."""
    return (
        data.groupby(column)[TARGET]
        .apply(lambda values: (values == POSITIVE_CLASS).mean() * 100)
        .sort_values(ascending=False)
        .round(1)
    )
