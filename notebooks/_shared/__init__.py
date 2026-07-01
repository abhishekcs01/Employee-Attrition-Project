"""Shared utilities for Phase 1 notebooks (independent of src/)."""

from notebooks._shared.config import (
    DATA_PATH,
    DROP_COLS,
    ENGINEERED_CATEGORICAL,
    FEATURE_CFG,
    INTERIM_DIR,
    LIKERT_COLS,
    MODEL_DIR,
    ORIGINAL_CATEGORICAL,
    PROJECT_ROOT,
    RANDOM_STATE,
    SATISFACTION_INDEX_COLS,
    TARGET,
    TEST_SIZE,
)
from notebooks._shared.eda import iqr_outlier_summary
from notebooks._shared.evaluation import plot_coefficients
from notebooks._shared.features import FEATURE_DEFINITIONS
from notebooks._shared.plotting import attrition_rate_by, plot_distribution, setup_plot_style
from notebooks._shared.preprocessing import (
    clean_dataset,
    encode_targets,
    get_feature_columns,
    load_raw_dataset,
)

__all__ = [
    "DATA_PATH",
    "DROP_COLS",
    "ENGINEERED_CATEGORICAL",
    "FEATURE_CFG",
    "FEATURE_DEFINITIONS",
    "INTERIM_DIR",
    "LIKERT_COLS",
    "MODEL_DIR",
    "ORIGINAL_CATEGORICAL",
    "PROJECT_ROOT",
    "RANDOM_STATE",
    "SATISFACTION_INDEX_COLS",
    "TARGET",
    "TEST_SIZE",
    "attrition_rate_by",
    "clean_dataset",
    "encode_targets",
    "get_feature_columns",
    "iqr_outlier_summary",
    "load_raw_dataset",
    "plot_coefficients",
    "plot_distribution",
    "setup_plot_style",
]
