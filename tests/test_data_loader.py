"""Tests for data loading utilities."""

from __future__ import annotations

import pytest

from src.utils.data_loader import DataLoadError, get_feature_columns, load_raw_dataset
from src.utils.paths import PathResolver


def test_path_resolver_from_config(project_root):
    cfg = {"paths": {"project_root": str(project_root)}}
    resolver = PathResolver.from_config(cfg)
    assert resolver.project_root == project_root.resolve()
    assert resolver.data_raw == project_root.resolve() / "data" / "raw"


def test_path_resolver_ensure_dirs(project_root, tmp_path):
    cfg = {"paths": {"project_root": str(tmp_path)}}
    resolver = PathResolver.from_config(cfg)
    resolver.ensure_dirs()
    assert (tmp_path / "data" / "raw").exists()
    assert (tmp_path / "reports").exists()


def test_load_raw_dataset_from_fixture(sample_hr_dataframe, tmp_path):
    csv_path = tmp_path / "test_data.csv"
    sample_hr_dataframe.to_csv(csv_path, index=False)
    df = load_raw_dataset(csv_path, auto_download=False)
    assert len(df) == 3
    assert "Attrition" in df.columns


def test_load_missing_dataset_raises(tmp_path):
    missing_path = tmp_path / "nonexistent.csv"
    with pytest.raises(DataLoadError):
        load_raw_dataset(missing_path, auto_download=False)


def test_get_feature_columns(schema_config):
    numerical, categorical, drop_cols = get_feature_columns(schema_config)
    assert "MonthlyIncome" in numerical
    assert "Department" in categorical
    assert "EmployeeNumber" in drop_cols
