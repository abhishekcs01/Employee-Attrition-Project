"""SHAP explainability report generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.preprocessing.pipeline import split_dataframe

logger = logging.getLogger(__name__)


def generate_shap_report(
    pipeline: Any,
    x_sample: pd.DataFrame,
    output_dir: Path,
    *,
    max_display: int = 20,
    plot_format: str = "png",
    dpi: int = 150,
) -> dict[str, str]:
    """Generate SHAP summary and bar plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model = pipeline.named_steps["model"]
    x_transformed = pipeline.named_steps["preprocess"].transform(x_sample)

    explainer = shap.Explainer(model, x_transformed)
    shap_values = explainer(x_transformed)

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, x_transformed, feature_names=feature_names, show=False, max_display=max_display
    )
    summary_path = output_dir / f"shap_summary.{plot_format}"
    plt.savefig(summary_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    bar_path = output_dir / f"shap_bar.{plot_format}"
    plt.savefig(bar_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return {"summary": str(summary_path), "bar": str(bar_path)}


def run_explainability(cfg: dict[str, Any], preprocessing_cfg: dict[str, Any]) -> Path:
    """Execute explainability workflow."""
    models_dir = Path(str(cfg["models_dir"]))
    output_dir = Path(str(cfg["output_dir"]))
    metadata_path = models_dir / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    pipeline = joblib.load(models_dir / "best_model.joblib")
    df = pd.read_csv(cfg["dataset_path"])
    data_split = split_dataframe(df, cfg["target_column"], preprocessing_cfg)

    sample_size = int(cfg.get("sample_size", 200))
    x_sample = data_split.x_train.sample(
        n=min(sample_size, len(data_split.x_train)),
        random_state=int(cfg.get("random_state", 42)),
    )

    plots = generate_shap_report(
        pipeline,
        x_sample,
        output_dir,
        max_display=int(cfg.get("max_display", 20)),
        plot_format=cfg.get("plot_format", "png"),
        dpi=int(cfg.get("dpi", 150)),
    )

    report = {
        "run_id": cfg.get("run_id"),
        "best_model_name": metadata.get("best_model_name"),
        "sample_size": len(x_sample),
        "plots": plots,
    }
    report_path = output_dir / "explainability_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Explainability report: %s", report_path)
    return report_path
