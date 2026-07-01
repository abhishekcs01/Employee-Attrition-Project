"""Tests for validation metrics."""

from __future__ import annotations

from src.validation.metrics import validate_synthetic_dataset


def test_validate_synthetic_dataset_returns_scores(sample_hr_dataframe):
    synthetic = sample_hr_dataframe.copy()
    numerical = [
        "Age",
        "MonthlyIncome",
        "TotalWorkingYears",
        "YearsAtCompany",
    ]
    categorical = ["Department", "OverTime"]

    result = validate_synthetic_dataset(
        sample_hr_dataframe,
        synthetic,
        numerical_columns=numerical,
        categorical_columns=categorical,
    )

    assert "composite_score" in result
    assert "ks_scores" in result
    assert result["dataset_size"] == len(synthetic)
