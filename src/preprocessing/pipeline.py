"""Leakage-safe preprocessing and stratified data splitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataSplit:
    """Container for stratified train/validation/test partitions."""

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    categorical_columns: list[str]
    numerical_columns: list[str]

    @property
    def train_rows(self) -> int:
        return len(self.x_train)

    @property
    def test_rows(self) -> int:
        return len(self.x_test)


def _encode_target(y: pd.Series, positive_class: str, negative_class: str) -> np.ndarray:
    mapping = {positive_class: 1, negative_class: 0}
    return y.map(mapping).to_numpy()


def _infer_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: list[str],
    categorical_columns: list[str] | None,
) -> tuple[list[str], list[str]]:
    drop_set = set(drop_columns) | {target_column}
    feature_df = df.drop(columns=[c for c in drop_set if c in df.columns])
    categorical = list(categorical_columns or [])
    categorical = [col for col in categorical if col in feature_df.columns]
    numerical = [col for col in feature_df.columns if col not in categorical]
    return numerical, categorical


def build_preprocessing_pipeline(
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for numeric scaling and categorical encoding."""
    transformers: list[tuple[str, Any, list[str]]] = []
    if numerical_columns:
        transformers.append(
            (
                "num",
                Pipeline([("scaler", StandardScaler())]),
                numerical_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipeline(
    preprocessor: ColumnTransformer,
    estimator: Any,
    smote_cfg: dict[str, Any],
) -> ImbPipeline:
    """Build a leakage-safe pipeline with optional SMOTE inside CV folds."""
    steps: list[tuple[str, Any]] = [("preprocess", preprocessor)]
    if smote_cfg.get("enabled", True):
        steps.append(
            (
                "smote",
                SMOTE(
                    random_state=int(smote_cfg.get("random_state", 42)),
                    k_neighbors=int(smote_cfg.get("k_neighbors", 5)),
                ),
            )
        )
    steps.append(("model", estimator))
    return ImbPipeline(steps=steps)


def split_dataframe(
    df: pd.DataFrame,
    target_column: str,
    cfg: dict[str, Any],
) -> DataSplit:
    """Create stratified train/validation/test splits without leakage."""
    split_cfg = cfg["split"]
    drop_columns = list(cfg.get("drop_columns", []))
    categorical_columns = list(cfg.get("categorical_columns", []))

    numerical_columns, categorical_columns = _infer_feature_columns(
        df,
        target_column,
        drop_columns,
        categorical_columns,
    )

    feature_columns = numerical_columns + categorical_columns
    x = df[feature_columns].copy()
    y = df[target_column].copy()

    train_size = float(split_cfg["train_size"])
    val_size = float(split_cfg["val_size"])
    test_size = float(split_cfg["test_size"])
    random_state = int(split_cfg.get("random_state", 42))
    stratify = y if split_cfg.get("stratify", True) else None

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=(1.0 - train_size),
        random_state=random_state,
        stratify=stratify,
    )

    relative_val = val_size / (val_size + test_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=(1.0 - relative_val),
        random_state=random_state,
        stratify=y_temp if stratify is not None else None,
    )

    positive_rate = _encode_target(y_train, cfg["positive_class"], cfg["negative_class"]).mean()
    logger.info(
        "Data split: train=%d, val=%d, test=%d, positive_rate_train=%.2f%%",
        len(x_train),
        len(x_val),
        len(x_test),
        positive_rate * 100,
    )

    return DataSplit(
        x_train=x_train.reset_index(drop=True),
        x_val=x_val.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        feature_names=feature_columns,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
    )


def prepare_datasets(
    dataset_path: Path,
    cfg: dict[str, Any],
) -> tuple[DataSplit, ColumnTransformer, dict[str, Any]]:
    """Load engineered data and return splits plus preprocessing transformer."""
    target_column = cfg["target_column"]
    df = pd.read_csv(dataset_path)
    data_split = split_dataframe(df, target_column, cfg)
    preprocessor = build_preprocessing_pipeline(
        data_split.numerical_columns,
        data_split.categorical_columns,
    )
    target_meta = {
        "positive_class": cfg["positive_class"],
        "negative_class": cfg["negative_class"],
    }
    return data_split, preprocessor, target_meta


def encode_targets(
    data_split: DataSplit, target_meta: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode train/val/test targets as binary arrays."""
    y_train = _encode_target(
        data_split.y_train,
        target_meta["positive_class"],
        target_meta["negative_class"],
    )
    y_val = _encode_target(
        data_split.y_val,
        target_meta["positive_class"],
        target_meta["negative_class"],
    )
    y_test = _encode_target(
        data_split.y_test,
        target_meta["positive_class"],
        target_meta["negative_class"],
    )
    return y_train, y_val, y_test


def get_feature_names_after_fit(
    preprocessor: ColumnTransformer, x_sample: pd.DataFrame
) -> list[str]:
    """Return feature names produced by a fitted preprocessor."""
    preprocessor.fit(x_sample)
    return preprocessor.get_feature_names_out().tolist()
