"""Preprocessing helpers for Phase 1 notebooks (I/O and column roles only)."""

from __future__ import annotations

import pandas as pd

from notebooks._shared.config import (
    DATA_PATH,
    DROP_COLS,
    ENGINEERED_CATEGORICAL,
    ORIGINAL_CATEGORICAL,
    TARGET,
)


def load_raw_dataset(path=None) -> pd.DataFrame:
    """Load the raw IBM HR Analytics dataset."""
    return pd.read_csv(path or DATA_PATH)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifiers and constant columns."""
    return df.drop(columns=[col for col in DROP_COLS if col in df.columns])


def get_feature_columns(
    df: pd.DataFrame,
    *,
    include_engineered: bool = False,
) -> tuple[list[str], list[str]]:
    """Return categorical and numerical feature column lists."""
    engineered_cats = ENGINEERED_CATEGORICAL if include_engineered else []
    categorical_cols = [
        col for col in ORIGINAL_CATEGORICAL + engineered_cats if col in df.columns
    ]
    exclude = set(categorical_cols + [TARGET])
    numerical_cols = [col for col in df.columns if col not in exclude]
    return categorical_cols, numerical_cols


def encode_targets(y: pd.Series) -> pd.Series:
    """Encode attrition labels as binary integers (Yes=1, No=0)."""
    return y.map({"Yes": 1, "No": 0})
