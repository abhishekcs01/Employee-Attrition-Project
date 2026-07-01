"""Model factory with automatic GPU/CPU configuration."""

from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.utils.hardware import (
    HardwareProfile,
    get_catboost_params,
    get_hardware_profile,
    get_lightgbm_params,
    get_xgboost_params,
)


def _base_kwargs(random_state: int) -> dict[str, Any]:
    return {
        "random_state": random_state,
        "n_jobs": -1,
    }


def create_logistic_regression(random_state: int = 42) -> LogisticRegression:
    """Create a logistic regression baseline model."""
    return LogisticRegression(
        max_iter=2000,
        solver="saga",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def create_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Create a random forest classifier."""
    return RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def create_xgboost(
    random_state: int = 42,
    hardware: HardwareProfile | None = None,
    **params: Any,
) -> XGBClassifier:
    """Create an XGBoost classifier with automatic device selection."""
    hardware = hardware or get_hardware_profile(probe_ml_backends=False)
    model_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
        **get_xgboost_params(hardware),
        **params,
    }
    return XGBClassifier(**model_params)


def create_lightgbm(
    random_state: int = 42,
    hardware: HardwareProfile | None = None,
    **params: Any,
) -> LGBMClassifier:
    """Create a LightGBM classifier with automatic device selection."""
    hardware = hardware or get_hardware_profile(probe_ml_backends=False)
    model_params = {
        "objective": "binary",
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
        "force_col_wise": True,
        **get_lightgbm_params(hardware),
        **params,
    }
    return LGBMClassifier(**model_params)


def create_catboost(
    random_state: int = 42,
    hardware: HardwareProfile | None = None,
    **params: Any,
) -> CatBoostClassifier:
    """Create a CatBoost classifier with automatic task type."""
    hardware = hardware or get_hardware_profile(probe_ml_backends=False)
    model_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
        "auto_class_weights": "Balanced",
        **get_catboost_params(hardware),
        **params,
    }
    return CatBoostClassifier(**model_params)


MODEL_BUILDERS = {
    "logistic_regression": create_logistic_regression,
    "random_forest": create_random_forest,
    "xgboost": create_xgboost,
    "lightgbm": create_lightgbm,
    "catboost": create_catboost,
}


def get_model_builder(model_name: str):
    """Return the factory function for a model name."""
    if model_name not in MODEL_BUILDERS:
        raise KeyError(f"Unsupported model: {model_name}")
    return MODEL_BUILDERS[model_name]


def get_search_space(model_name: str, trial: Any) -> dict[str, Any]:
    """Return Optuna search space for a model."""
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        }
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
    raise KeyError(f"No search space for model: {model_name}")
