"""Tests for preprocessing pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.preprocessing.pipeline import (
    build_model_pipeline,
    build_preprocessing_pipeline,
    encode_targets,
    split_dataframe,
)


def _preprocessing_cfg(*, stratify: bool = True) -> dict:
    return {
        "split": {
            "train_size": 0.70,
            "val_size": 0.15,
            "test_size": 0.15,
            "random_state": 42,
            "stratify": stratify,
        },
        "drop_columns": ["EmployeeNumber", "Over18", "StandardHours", "EmployeeCount"],
        "categorical_columns": [
            "BusinessTravel",
            "Department",
            "EducationField",
            "Gender",
            "JobRole",
            "MaritalStatus",
            "OverTime",
        ],
        "positive_class": "Yes",
        "negative_class": "No",
    }


def test_split_ratios(expanded_hr_dataframe):
    cfg = _preprocessing_cfg()
    split = split_dataframe(expanded_hr_dataframe, "Attrition", cfg)
    total = split.train_rows + len(split.x_val) + split.test_rows
    assert total == len(expanded_hr_dataframe)
    assert split.train_rows > split.test_rows


def test_smote_pipeline_runs_inside_cv(expanded_hr_dataframe):
    cfg = _preprocessing_cfg()
    split = split_dataframe(expanded_hr_dataframe, "Attrition", cfg)
    preprocessor = build_preprocessing_pipeline(split.numerical_columns, split.categorical_columns)
    pipeline = build_model_pipeline(
        preprocessor,
        LogisticRegression(max_iter=1000, solver="saga"),
        {"enabled": False},
    )
    y_train, _, _ = encode_targets(split, {"positive_class": "Yes", "negative_class": "No"})
    pipeline.fit(split.x_train, y_train)
    assert hasattr(pipeline, "predict_proba")


def test_encode_targets_binary(expanded_hr_dataframe):
    cfg = _preprocessing_cfg()
    split = split_dataframe(expanded_hr_dataframe, "Attrition", cfg)
    y_train, y_val, y_test = encode_targets(
        split, {"positive_class": "Yes", "negative_class": "No"}
    )
    for arr in (y_train, y_val, y_test):
        assert set(np.unique(arr)).issubset({0, 1})
