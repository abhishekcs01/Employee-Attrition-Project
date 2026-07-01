"""FastAPI prediction service."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from src.utils.artifacts import load_model_artifacts
from src.utils.hydra_utils import run_with_hydra
from src.utils.pipeline_bootstrap import bootstrap_pipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="Employee Attrition API", version="1.0.0")

_pipeline: Any = None
_metadata: dict[str, Any] = {}
_raw_features: list[str] = []


class PredictionRequest(BaseModel):
    """Employee feature payload for attrition prediction."""

    features: dict[str, Any] = Field(..., description="Raw feature name-value pairs")


class PredictionResponse(BaseModel):
    """Attrition prediction response."""

    attrition_probability: float
    attrition_prediction: str


def load_artifacts(models_dir: Path) -> None:
    """Load model pipeline and metadata into module state."""
    global _pipeline, _metadata, _raw_features

    _pipeline, _metadata = load_model_artifacts(models_dir)
    _raw_features = list(_metadata.get("raw_feature_names", _metadata.get("feature_names", [])))
    logger.info("Loaded model: %s", _metadata.get("best_model_name"))


@app.on_event("startup")
def startup() -> None:
    """Load model on application startup if MODELS_DIR is set."""
    models_dir = Path(os.environ.get("MODELS_DIR", "models/latest"))
    if models_dir.exists():
        load_artifacts(models_dir)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict employee attrition probability."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    threshold = float(_metadata.get("classification_threshold", 0.5))
    positive = _metadata.get("target_meta", {}).get("positive_class", "Yes")
    negative = _metadata.get("target_meta", {}).get("negative_class", "No")

    row = {col: request.features.get(col) for col in _raw_features}
    df = pd.DataFrame([row])
    prob = float(_pipeline.predict_proba(df)[0, 1])
    label = positive if prob >= threshold else negative

    return PredictionResponse(attrition_probability=prob, attrition_prediction=label)


def run_api(cfg: DictConfig) -> None:
    """Start the FastAPI server."""
    bootstrap_pipeline(cfg, caller_file=__file__, log_filename="api.log")
    deployment_cfg = OmegaConf.to_container(cfg.deployment, resolve=True)
    api_cfg = deployment_cfg.get("api", deployment_cfg)

    models_dir = Path(str(api_cfg["models_dir"]))
    os.environ["MODELS_DIR"] = str(models_dir)
    load_artifacts(models_dir)

    uvicorn.run(
        "src.deployment.api.main:app",
        host=str(api_cfg.get("host", "0.0.0.0")),
        port=int(api_cfg.get("port", 8000)),
        reload=bool(api_cfg.get("reload", False)),
    )


def main() -> None:
    """Hydra-backed CLI entry point."""
    run_with_hydra(run_api, caller_file=__file__)


if __name__ == "__main__":
    main()
