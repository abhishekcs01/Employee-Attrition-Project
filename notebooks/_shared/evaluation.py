"""Evaluation plotting helpers for Phase 1 notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline


def plot_coefficients(
    pipeline: Pipeline,
    top_n: int = 15,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot top logistic regression coefficients by absolute magnitude."""
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = pipeline.named_steps["classifier"].coef_.flatten()
    coef_df = (
        pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
        .assign(AbsCoefficient=lambda frame: frame["Coefficient"].abs())
        .sort_values("AbsCoefficient", ascending=False)
        .head(top_n)
        .sort_values("Coefficient")
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#E94F37" if value > 0 else "#2E86AB" for value in coef_df["Coefficient"]]
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Top {top_n} Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient (log-odds impact on attrition)")
    plt.tight_layout()
    return fig, coef_df
