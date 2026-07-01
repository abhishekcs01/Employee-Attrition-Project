"""CLI entry point for model comparison across training runs."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.utils.hydra_utils import run_with_hydra
from src.utils.pipeline_bootstrap import bootstrap_pipeline

logger = logging.getLogger(__name__)


def run_model_comparison(cfg: DictConfig) -> None:
    """Load and compare model comparison tables from training runs."""
    ctx = bootstrap_pipeline(cfg, caller_file=__file__, log_filename="compare.log", seed=cfg.seed)
    cfg = ctx.cfg

    training_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    reports_dir = ctx.path_resolver.reports / "training"

    comparison_files = sorted(reports_dir.glob("*/model_comparison.csv"))
    if not comparison_files:
        latest = Path(str(training_cfg["models_latest_dir"])) / "model_comparison.csv"
        if latest.exists():
            comparison_files = [latest]
        else:
            logger.error("No model comparison files found under %s", reports_dir)
            return

    frames: list[pd.DataFrame] = []
    for path in comparison_files:
        df = pd.read_csv(path)
        df["run_id"] = path.parent.name if path.parent.name != "latest" else "latest"
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    output_dir = Path(str(training_cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cross_run_comparison.csv"
    combined.to_csv(output_path, index=False)

    logger.info("Cross-run comparison saved: %s", output_path)
    logger.info(
        "Top models by val_roc_auc:\n%s",
        combined.sort_values("val_roc_auc", ascending=False).head(),
    )


def main() -> None:
    """Hydra-backed CLI entry point."""
    run_with_hydra(run_model_comparison, caller_file=__file__)


if __name__ == "__main__":
    main()
