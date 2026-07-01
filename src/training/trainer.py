"""Model training orchestration with Optuna and MLflow."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

from src.preprocessing.pipeline import DataSplit, build_model_pipeline, encode_targets
from src.training.models import get_model_builder, get_search_space
from src.utils.hardware import get_hardware_profile

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ModelResult:
    """Training result for a single model."""

    name: str
    best_params: dict[str, Any]
    cv_score: float
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    estimator: Any
    artifact_path: Path | None = None


@dataclass
class TrainingResult:
    """Aggregate training outputs."""

    model_results: list[ModelResult] = field(default_factory=list)
    comparison_table: pd.DataFrame | None = None
    best_model_name: str | None = None
    best_model_path: Path | None = None
    feature_names: list[str] = field(default_factory=list)


class ModelTrainer:
    """Train, tune, compare, and persist attrition models."""

    def __init__(
        self,
        cfg: dict[str, Any],
        data_split: DataSplit,
        preprocessor: Any,
        target_meta: dict[str, Any],
        output_dir: Path,
        models_dir: Path,
        models_latest_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg
        self.data_split = data_split
        self.preprocessor = preprocessor
        self.target_meta = target_meta
        self.output_dir = output_dir
        self.models_dir = models_dir
        self.models_latest_dir = models_latest_dir or models_dir.parent / "latest"
        self.hardware = get_hardware_profile()
        self.random_state = int(cfg.get("random_state", 42))
        self.metric = cfg.get("metric", "roc_auc")
        self.cv_folds = int(cfg.get("cv_folds", 5))
        self.smote_cfg = cfg.get("smote", {"enabled": True})
        self.threshold = float(cfg.get("classification_threshold", 0.5))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models_latest_dir.mkdir(parents=True, exist_ok=True)

        self.y_train, self.y_val, self.y_test = encode_targets(data_split, target_meta)
        self.x_train = data_split.x_train
        self.x_val = data_split.x_val
        self.x_test = data_split.x_test

        fitted_preprocessor = clone(preprocessor)
        fitted_preprocessor.fit(self.x_train)
        self.feature_names = fitted_preprocessor.get_feature_names_out().tolist()

    def _cv_splitter(self) -> StratifiedKFold:
        return StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

    def _build_pipeline(self, model_name: str) -> Any:
        builder = get_model_builder(model_name)
        if model_name in {"xgboost", "lightgbm", "catboost"}:
            estimator = builder(random_state=self.random_state, hardware=self.hardware)
        else:
            estimator = builder(random_state=self.random_state)
        return build_model_pipeline(clone(self.preprocessor), estimator, self.smote_cfg)

    def _evaluate(self, pipeline: Any, x: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
        y_prob = pipeline.predict_proba(x)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y, y_prob)),
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
        }

    def _param_distributions_from_config(self, model_name: str) -> dict[str, list[Any]]:
        model_cfg = self.cfg.get(model_name, {})
        search_space = model_cfg.get("search_space", {})
        return {f"model__{key}": values for key, values in search_space.items()}

    def _tune_with_random_search(
        self,
        pipeline: Any,
        param_distributions: dict[str, list[Any]],
        n_iter: int,
    ) -> tuple[Any, dict[str, Any], float]:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=self.metric,
            cv=self._cv_splitter(),
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0,
        )
        search.fit(self.x_train, self.y_train)
        return search.best_estimator_, dict(search.best_params_), float(search.best_score_)

    def _tune_with_optuna(
        self, model_name: str, pipeline: Any, n_trials: int
    ) -> tuple[Any, dict[str, Any], float]:
        logger.info("Running Optuna HPO for %s", model_name)

        def objective(trial: optuna.Trial) -> float:
            params = get_search_space(model_name, trial)
            model_params = {f"model__{key}": value for key, value in params.items()}
            candidate = clone(pipeline)
            candidate.set_params(**model_params)
            scores = cross_val_score(
                candidate,
                self.x_train,
                self.y_train,
                cv=self._cv_splitter(),
                scoring=self.metric,
                n_jobs=-1,
            )
            return float(np.mean(scores))

        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction=self.cfg.get("optuna", {}).get("direction", "maximize"),
            sampler=sampler,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=int(self.cfg.get("optuna", {}).get("n_jobs", 1)),
        )
        best_params = {f"model__{k}": v for k, v in study.best_params.items()}
        best_estimator = clone(pipeline)
        best_estimator.set_params(**best_params)
        best_estimator.fit(self.x_train, self.y_train)
        return best_estimator, best_params, float(study.best_value)

    def train_model(self, model_name: str) -> ModelResult:
        """Train and tune a single model."""
        logger.info("Training model: %s", model_name)
        pipeline = self._build_pipeline(model_name)

        with mlflow.start_run(run_name=model_name, nested=True):
            mlflow.set_tag("model_name", model_name)

            if model_name == "logistic_regression":
                param_distributions = self._param_distributions_from_config(model_name)
                if not param_distributions:
                    param_distributions = {
                        "model__C": np.logspace(-3, 2, 20).tolist(),
                        "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
                    }
                n_iter = int(self.cfg.get("logistic_regression", {}).get("n_iter", 20))
                best_estimator, best_params, cv_score = self._tune_with_random_search(
                    pipeline,
                    param_distributions,
                    n_iter,
                )
            elif model_name == "random_forest":
                param_distributions = self._param_distributions_from_config(model_name)
                if not param_distributions:
                    param_distributions = {
                        "model__n_estimators": list(range(100, 501, 50)),
                        "model__max_depth": list(range(3, 21)),
                        "model__min_samples_split": list(range(2, 11)),
                        "model__min_samples_leaf": list(range(1, 6)),
                    }
                n_iter = int(self.cfg.get("random_forest", {}).get("n_iter", 20))
                best_estimator, best_params, cv_score = self._tune_with_random_search(
                    pipeline,
                    param_distributions,
                    n_iter,
                )
            else:
                n_trials = int(self.cfg.get(model_name, {}).get("n_trials", 20))
                best_estimator, best_params, cv_score = self._tune_with_optuna(
                    model_name,
                    pipeline,
                    n_trials,
                )

            val_metrics = self._evaluate(best_estimator, self.x_val, self.y_val)
            test_metrics = self._evaluate(best_estimator, self.x_test, self.y_test)

            mlflow.log_params(best_params)
            mlflow.log_metric("cv_score", cv_score)
            for key, value in val_metrics.items():
                mlflow.log_metric(f"val_{key}", value)
            for key, value in test_metrics.items():
                mlflow.log_metric(f"test_{key}", value)

            artifact_path = None
            if self.cfg.get("save_artifacts", True):
                artifact_path = self.models_dir / f"{model_name}.joblib"
                joblib.dump(best_estimator, artifact_path)
                try:
                    mlflow.log_artifact(str(artifact_path))
                except Exception as exc:
                    logger.warning("MLflow artifact logging failed: %s", exc)

            logger.info(
                "%s complete | cv_%s=%.4f | val_%s=%.4f | test_%s=%.4f",
                model_name,
                self.metric,
                cv_score,
                self.metric,
                val_metrics[self.metric],
                self.metric,
                test_metrics[self.metric],
            )

            return ModelResult(
                name=model_name,
                best_params=best_params,
                cv_score=cv_score,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                estimator=best_estimator,
                artifact_path=artifact_path,
            )

    def _publish_latest_artifacts(self, best_result: ModelResult, comparison_path: Path) -> None:
        """Copy run artifacts to models/latest for deployment."""
        for src in self.models_dir.glob("*.joblib"):
            shutil.copy2(src, self.models_latest_dir / src.name)
        metadata = {
            "run_id": self.cfg.get("run_id"),
            "best_model_name": best_result.name,
            "feature_names": self.feature_names,
            "raw_feature_names": self.data_split.feature_names,
            "target_meta": self.target_meta,
            "comparison_path": str(comparison_path),
            "classification_threshold": self.threshold,
        }
        metadata_path = self.models_dir / "training_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        shutil.copy2(metadata_path, self.models_latest_dir / "training_metadata.json")
        shutil.copy2(comparison_path, self.models_latest_dir / "model_comparison.csv")

    def run(self, model_names: list[str]) -> TrainingResult:
        """Train all configured models and select the best performer."""
        mlflow.set_experiment(self.cfg.get("experiment_name", "employee_attrition"))
        results: list[ModelResult] = []

        with mlflow.start_run(run_name=f"training_{self.cfg.get('run_id', 'run')}"):
            for model_name in model_names:
                results.append(self.train_model(model_name))

            comparison = pd.DataFrame(
                [
                    {
                        "model": result.name,
                        "cv_score": result.cv_score,
                        **{f"val_{k}": v for k, v in result.val_metrics.items()},
                        **{f"test_{k}": v for k, v in result.test_metrics.items()},
                    }
                    for result in results
                ]
            ).sort_values("val_roc_auc", ascending=False)

            best_row = comparison.iloc[0]
            best_model_name = str(best_row["model"])
            best_result = next(r for r in results if r.name == best_model_name)

            comparison_path = self.output_dir / "model_comparison.csv"
            comparison.to_csv(comparison_path, index=False)

            best_model_path = self.models_dir / "best_model.joblib"
            joblib.dump(best_result.estimator, best_model_path)

            self._publish_latest_artifacts(best_result, comparison_path)

            try:
                mlflow.log_artifact(str(comparison_path))
                mlflow.log_artifact(str(best_model_path))
                mlflow.log_artifact(str(self.models_dir / "training_metadata.json"))
                mlflow.set_tag("best_model", best_model_name)
            except Exception as exc:
                logger.warning("MLflow artifact logging failed: %s", exc)

            logger.info(
                "Best model: %s (val_roc_auc=%.4f)", best_model_name, best_row["val_roc_auc"]
            )

            return TrainingResult(
                model_results=results,
                comparison_table=comparison,
                best_model_name=best_model_name,
                best_model_path=best_model_path,
                feature_names=self.feature_names,
            )
