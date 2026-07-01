"""Shared plotting utilities for EDA modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go


def ensure_plot_dir(plot_dir: Path) -> Path:
    """Create plot output directory if needed."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def save_matplotlib_fig(
    fig: plt.Figure,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Save matplotlib figure and close it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def save_plotly_fig(fig: go.Figure, output_path: Path) -> Path:
    """Save plotly figure as HTML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    return output_path


def get_figsize(cfg: dict[str, Any]) -> tuple[float, float]:
    """Extract figure size from config."""
    figsize = cfg.get("figsize", [10, 6])
    return float(figsize[0]), float(figsize[1])
