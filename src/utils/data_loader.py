"""Data loading utilities for IBM HR Analytics dataset."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when dataset loading fails."""


def load_schema_config(schema_path: Path) -> dict:
    """Load the dataset schema YAML configuration."""
    with schema_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_from_kaggle(dataset: str, output_dir: Path, filename: str) -> Path:
    """Download dataset from Kaggle using the Kaggle API."""
    username = os.environ.get("KAGGLE_USERNAME")
    api_key = os.environ.get("KAGGLE_KEY")

    if not username or not api_key:
        raise DataLoadError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
            "environment variables, or place the CSV manually in data/raw/."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / filename

    if target_path.exists():
        logger.info("Dataset already exists at %s", target_path)
        return target_path

    logger.info("Downloading dataset %s from Kaggle...", dataset)
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(output_dir), "--unzip"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise DataLoadError("Kaggle CLI not installed. Install with: pip install kaggle") from exc
    except subprocess.CalledProcessError as exc:
        raise DataLoadError(f"Kaggle download failed: {exc.stderr}") from exc

    if not target_path.exists():
        csv_files = list(output_dir.glob("*.csv"))
        if not csv_files:
            raise DataLoadError(f"Download completed but {filename} not found in {output_dir}")
        return csv_files[0]

    return target_path


def load_raw_dataset(
    raw_path: Path,
    schema_path: Path | None = None,
    auto_download: bool = True,
) -> pd.DataFrame:
    """
    Load the IBM HR Analytics dataset.

    Attempts Kaggle download if file is missing and credentials are available.
    """
    if not raw_path.exists():
        if auto_download and schema_path and schema_path.exists():
            schema = load_schema_config(schema_path)
            kaggle_cfg = schema.get("kaggle", {})
            dataset = kaggle_cfg.get("dataset", "pavansubhasht/ibm-hr-analytics-attrition-dataset")
            filename = kaggle_cfg.get("file", raw_path.name)
            raw_path = download_from_kaggle(dataset, raw_path.parent, filename)
        else:
            raise DataLoadError(
                f"Dataset not found at {raw_path}. "
                "Download from https://www.kaggle.com/datasets/pavansubhasht/"
                "ibm-hr-analytics-attrition-dataset and place in data/raw/."
            )

    logger.info("Loading dataset from %s", raw_path)
    df = pd.read_csv(raw_path)
    logger.info("Loaded %d rows and %d columns", len(df), len(df.columns))
    return df


def get_feature_columns(schema: dict) -> tuple[list[str], list[str], list[str]]:
    """Return numerical, categorical, and drop column lists from schema."""
    numerical = list(schema.get("numerical_columns", []))
    categorical = list(schema.get("categorical_columns", []))
    drop_cols = list(schema.get("drop_columns", []))
    return numerical, categorical, drop_cols
