"""Post-training model evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.preprocessing.pipeline import split_dataframe

logger = logging.getLogger(__name__)


from src.utils.artifacts import load_model_artifacts
def evaluate_model(
    pipeline: Any,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute test-set metrics."""
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def save_evaluation_plots(
    pipeline: Any,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    output_dir: Path,
    *,
    plot_format: str = "png",
    dpi: int = 150,
) -> dict[str, Path]:
    """Save ROC curve and confusion matrix plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test, pipeline.predict_proba(x_test)[:, 1], ax=ax)
    roc_path = output_dir / f"roc_curve.{plot_format}"
    fig.savefig(roc_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths["roc_curve"] = roc_path

    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, pipeline.predict(x_test), ax=ax)
    cm_path = output_dir / f"confusion_matrix.{plot_format}"
    fig.savefig(cm_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    paths["confusion_matrix"] = cm_path

    return paths


def run_evaluation(cfg: dict[str, Any], preprocessing_cfg: dict[str, Any]) -> Path:
    """Execute evaluation workflow and return report path."""
    models_dir = Path(str(cfg["models_dir"]))
    output_dir = Path(str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline, metadata = load_model_artifacts(models_dir)
    threshold = float(
        cfg.get("classification_threshold", metadata.get("classification_threshold", 0.5))
    )

    df = pd.read_csv(cfg["dataset_path"])
    data_split = split_dataframe(df, cfg["target_column"], preprocessing_cfg)
    target_meta = {
        "positive_class": cfg.get("positive_class", "Yes"),
        "negative_class": cfg.get("negative_class", "No"),
    }
    from src.preprocessing.pipeline import encode_targets

    _, _, y_test = encode_targets(data_split, target_meta)

    metrics = evaluate_model(pipeline, data_split.x_test, y_test, threshold=threshold)
    plot_paths = save_evaluation_plots(pipeline, data_split.x_test, y_test, output_dir)

    report = {
        "run_id": cfg.get("run_id"),
        "best_model_name": metadata.get("best_model_name"),
        "metrics": metrics,
        "plots": {k: str(v) for k, v in plot_paths.items()},
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Evaluation report: %s", report_path)
    return report_path
