"""CLI entry point for CTGAN synthetic data generation."""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.synthetic.ctgan_trainer import generate_synthetic_datasets, train_ctgan
from src.utils.data_loader import load_raw_dataset
from src.utils.hydra_utils import run_with_hydra
from src.utils.pipeline_bootstrap import bootstrap_pipeline

logger = logging.getLogger(__name__)


def run_synthetic_training(cfg: DictConfig) -> None:
    """Train CTGAN and generate synthetic datasets at configured sizes."""
    ctx = bootstrap_pipeline(cfg, caller_file=__file__, log_filename="synthetic.log", seed=cfg.seed)
    cfg = ctx.cfg

    synthetic_cfg = OmegaConf.to_container(cfg.synthetic, resolve=True)
    schema_path = ctx.project_root / "configs" / "data" / "schema.yaml"

    raw_path = Path(str(synthetic_cfg["raw_path"]))
    real_data = load_raw_dataset(raw_path, schema_path=schema_path, auto_download=True)

    output_dir = Path(str(synthetic_cfg["output_dir"]))
    model_path = Path(str(synthetic_cfg["model_path"]))

    synthesizer = train_ctgan(
        real_data, synthetic_cfg, output_dir=output_dir, model_path=model_path
    )
    generate_synthetic_datasets(
        synthesizer,
        list(synthetic_cfg["sizes"]),
        output_dir,
        run_id=str(synthetic_cfg["run_id"]),
    )

    logger.info("Synthetic data generation complete")


def main() -> None:
    """Hydra-backed CLI entry point."""
    run_with_hydra(run_synthetic_training, caller_file=__file__)


if __name__ == "__main__":
    main()
