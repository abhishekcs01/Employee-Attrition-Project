"""Schema validation for IBM HR Analytics dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of schema validation against expected configuration."""

    is_valid: bool
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    dtype_mismatches: dict[str, tuple[str, str]] = field(default_factory=dict)
    null_counts: dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    row_count: int = 0
    column_count: int = 0
    constant_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize validation result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "missing_columns": self.missing_columns,
            "extra_columns": self.extra_columns,
            "dtype_mismatches": {
                k: {"expected": v[0], "actual": v[1]} for k, v in self.dtype_mismatches.items()
            },
            "null_counts": self.null_counts,
            "duplicate_count": self.duplicate_count,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "constant_columns": self.constant_columns,
        }


class SchemaValidationError(Exception):
    """Raised when schema validation fails in strict mode."""


class SchemaValidator:
    """Validate dataset against expected IBM HR schema."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema
        self.expected_columns = schema.get("columns", {})
        self.numerical_columns = set(schema.get("numerical_columns", []))
        self.categorical_columns = set(schema.get("categorical_columns", []))

    @classmethod
    def from_yaml(cls, schema_path: Path) -> SchemaValidator:
        """Load validator from YAML schema file."""
        with schema_path.open("r", encoding="utf-8") as f:
            schema = yaml.safe_load(f)
        return cls(schema)

    def _check_dtype(self, series: pd.Series, expected: str) -> bool:
        """Check if series dtype matches expected schema type."""
        actual = str(series.dtype)
        if expected == "int64":
            return pd.api.types.is_integer_dtype(series)
        if expected == "float64":
            return pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series)
        if expected == "object":
            return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
        return actual == expected

    def validate(self, df: pd.DataFrame, strict: bool = True) -> ValidationResult:
        """
        Validate DataFrame against schema.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to validate.
        strict : bool
            If True, raise SchemaValidationError when validation fails.
        """
        expected_cols = set(self.expected_columns.keys())
        actual_cols = set(df.columns)

        missing = sorted(expected_cols - actual_cols)
        extra = sorted(actual_cols - expected_cols)

        dtype_mismatches: dict[str, tuple[str, str]] = {}
        for col, expected_dtype in self.expected_columns.items():
            if col in df.columns and not self._check_dtype(df[col], expected_dtype):
                dtype_mismatches[col] = (expected_dtype, str(df[col].dtype))

        null_counts = {col: int(df[col].isna().sum()) for col in df.columns if col in expected_cols}
        duplicate_count = int(df.duplicated().sum())

        constant_columns = [
            col for col in df.columns if col in expected_cols and df[col].nunique(dropna=False) <= 1
        ]

        is_valid = not missing and not dtype_mismatches

        result = ValidationResult(
            is_valid=is_valid,
            missing_columns=missing,
            extra_columns=extra,
            dtype_mismatches=dtype_mismatches,
            null_counts=null_counts,
            duplicate_count=duplicate_count,
            row_count=len(df),
            column_count=len(df.columns),
            constant_columns=constant_columns,
        )

        if not is_valid:
            logger.warning(
                "Schema validation failed: missing=%s, dtype_mismatches=%s",
                missing,
                list(dtype_mismatches.keys()),
            )
            if strict:
                raise SchemaValidationError(
                    f"Schema validation failed. Missing: {missing}. "
                    f"Dtype mismatches: {list(dtype_mismatches.keys())}"
                )
        else:
            logger.info("Schema validation passed for %d rows", len(df))

        return result
