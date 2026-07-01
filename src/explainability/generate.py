"""CLI entry point for SHAP explainability."""

from __future__ import annotations

import logging

from omegaconf import DictConfig, OmegaConf

from src.explainability.generator import run_explainability
from src.utils.hydra_utils import run_with_hydra
from src.utils.pipeline_bootstrap import bootstrap_pipeline

logger = logging.getLogger(__name__)


def run_explainability_pipeline(cfg: DictConfig) -> None:
    """Generate SHAP explainability reports for the best model."""
    ctx = bootstrap_pipeline(
        cfg, caller_file=__file__, log_filename="explainability.log", seed=cfg.seed
    )
    cfg = ctx.cfg

    explain_cfg = OmegaConf.to_container(cfg.explainability, resolve=True)
    preprocessing_cfg = OmegaConf.to_container(cfg.preprocessing, resolve=True)
    preprocessing_cfg["target_column"] = cfg.data.target_column

    run_explainability(explain_cfg, preprocessing_cfg)
    logger.info("Explainability pipeline complete")


def main() -> None:
    """Hydra-backed CLI entry point."""
    run_with_hydra(run_explainability_pipeline, caller_file=__file__)


if __name__ == "__main__":
    main()
