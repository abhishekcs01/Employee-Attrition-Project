"""Tests for schema validation module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.eda.schema_validator import SchemaValidationError, SchemaValidator


def test_valid_schema_passes(sample_hr_dataframe, schema_config):
    validator = SchemaValidator(schema_config)
    result = validator.validate(sample_hr_dataframe, strict=False)
    assert result.is_valid
    assert result.row_count == 3
    assert not result.missing_columns


def test_missing_columns_detected(sample_hr_dataframe, schema_config):
    df = sample_hr_dataframe.drop(columns=["Age", "MonthlyIncome"])
    validator = SchemaValidator(schema_config)
    result = validator.validate(df, strict=False)
    assert not result.is_valid
    assert "Age" in result.missing_columns
    assert "MonthlyIncome" in result.missing_columns


def test_strict_mode_raises(sample_hr_dataframe, schema_config):
    df = sample_hr_dataframe.drop(columns=["Age"])
    validator = SchemaValidator(schema_config)
    with pytest.raises(SchemaValidationError):
        validator.validate(df, strict=True)


def test_duplicate_count(sample_hr_dataframe, schema_config):
    df = pd.concat([sample_hr_dataframe, sample_hr_dataframe.iloc[[0]]], ignore_index=True)
    validator = SchemaValidator(schema_config)
    result = validator.validate(df, strict=False)
    assert result.duplicate_count == 1


def test_constant_columns_detected(schema_config):
    data = {col: [1, 1, 1] for col in schema_config["columns"]}
    data["Attrition"] = ["Yes", "No", "Yes"]
    data["Department"] = ["Sales", "HR", "IT"]
    data["Gender"] = ["M", "F", "M"]
    data["JobRole"] = ["A", "B", "C"]
    data["MaritalStatus"] = ["Single", "Married", "Single"]
    data["BusinessTravel"] = ["Rarely", "Often", "Rarely"]
    data["EducationField"] = ["LS", "LS", "Other"]
    data["OverTime"] = ["Yes", "No", "Yes"]
    data["Over18"] = ["Y", "Y", "Y"]
    df = pd.DataFrame(data)
    validator = SchemaValidator(schema_config)
    result = validator.validate(df, strict=False)
    assert "EmployeeCount" in result.constant_columns
