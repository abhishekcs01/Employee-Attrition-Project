"""EDA helpers for Phase 1 notebooks."""

from __future__ import annotations

import pandas as pd


def iqr_outlier_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Summarize IQR-based outliers for numeric columns."""
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        rows.append(
            {
                "Feature": col,
                "Q1": round(q1, 2),
                "Q3": round(q3, 2),
                "IQR": round(iqr, 2),
                "Lower Bound": round(lower, 2),
                "Upper Bound": round(upper, 2),
                "Outlier Count": int(mask.sum()),
                "Outlier %": round(100 * mask.mean(), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("Outlier Count", ascending=False)
