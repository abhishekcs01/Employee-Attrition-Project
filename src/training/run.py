"""CLI entry point for model training pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.preprocessing.pipeline import prepare_datasets
from src.training.trainer import ModelTrainer
from src.utils.hydra_utils import run_with_hydra
from src.utils.pipeline_bootstrap import bootstrap_pipeline

logger = logging.getLogger(__name__)


def run_training_pipeline(cfg: DictConfig) -> None:
    """Execute the full model training workflow."""
    ctx = bootstrap_pipeline(cfg, caller_file=__file__, log_filename="training.log", seed=cfg.seed)
    cfg = ctx.cfg

    training_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    preprocessing_cfg = OmegaConf.to_container(cfg.preprocessing, resolve=True)
    preprocessing_cfg["target_column"] = cfg.data.target_column

    output_dir = Path(str(training_cfg["output_dir"]))
    models_run_dir = Path(str(training_cfg["models_run_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    models_run_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MLFLOW_TRACKING_URI", str(training_cfg["mlflow_tracking_uri"]))
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    dataset_path = Path(str(preprocessing_cfg["dataset_path"]))
    data_split, preprocessor, target_meta = prepare_datasets(dataset_path, preprocessing_cfg)

    training_cfg["smote"] = preprocessing_cfg.get("smote", {})
    training_cfg["classification_threshold"] = preprocessing_cfg.get(
        "classification_threshold", 0.5
    )

    trainer = ModelTrainer(
        cfg=training_cfg,
        data_split=data_split,
        preprocessor=preprocessor,
        target_meta=target_meta,
        output_dir=output_dir,
        models_dir=models_run_dir,
        models_latest_dir=Path(str(training_cfg["models_latest_dir"])),
    )

    result = trainer.run(list(training_cfg["models"]))

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("Best model: %s", result.best_model_name)
    logger.info("Best model path: %s", result.best_model_path)
    logger.info("Comparison table: %s", output_dir / "model_comparison.csv")
    logger.info("=" * 60)


def main() -> None:
    """Hydra-backed CLI entry point."""
    run_with_hydra(run_training_pipeline, caller_file=__file__)


if __name__ == "__main__":
    main()
