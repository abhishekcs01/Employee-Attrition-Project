"""CLI entry point for model evaluation."""

from __future__ import annotations

import logging

from omegaconf import DictConfig, OmegaConf

from src.evaluation.evaluator import run_evaluation
from src.utils.hydra_utils import run_with_hydra
from src.utils.pipeline_bootstrap import bootstrap_pipeline

logger = logging.getLogger(__name__)


def run_evaluation_pipeline(cfg: DictConfig) -> None:
    """Evaluate the best trained model on the held-out test set."""
    ctx = bootstrap_pipeline(
        cfg, caller_file=__file__, log_filename="evaluation.log", seed=cfg.seed
    )
    cfg = ctx.cfg

    evaluation_cfg = OmegaConf.to_container(cfg.evaluation, resolve=True)
    preprocessing_cfg = OmegaConf.to_container(cfg.preprocessing, resolve=True)
    preprocessing_cfg["target_column"] = cfg.data.target_column

    run_evaluation(evaluation_cfg, preprocessing_cfg)
    logger.info("Evaluation pipeline complete")


def main() -> None:
    """Hydra-backed CLI entry point."""
    run_with_hydra(run_evaluation_pipeline, caller_file=__file__)


if __name__ == "__main__":
    main()
