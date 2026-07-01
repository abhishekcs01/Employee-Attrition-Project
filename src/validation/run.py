"""CLI entry point for synthetic data validation."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.utils.data_loader import load_raw_dataset, load_schema_config
from src.utils.hydra_utils import run_with_hydra
from src.utils.pipeline_bootstrap import bootstrap_pipeline
from src.validation.metrics import select_best_synthetic

logger = logging.getLogger(__name__)


def run_validation(cfg: DictConfig) -> None:
    """Validate synthetic datasets and select the best candidate."""
    ctx = bootstrap_pipeline(
        cfg, caller_file=__file__, log_filename="validation.log", seed=cfg.seed
    )
    cfg = ctx.cfg

    validation_cfg = OmegaConf.to_container(cfg.validation, resolve=True)
    schema_path = ctx.project_root / "configs" / "data" / "schema.yaml"
    schema = load_schema_config(schema_path)

    real_path = Path(str(validation_cfg["real_path"]))
    real_data = load_raw_dataset(real_path, schema_path=schema_path, auto_download=True)

    synthetic_dir = Path(str(validation_cfg["synthetic_dir"]))
    pattern = validation_cfg.get("synthetic_pattern", "synthetic_{size}.csv")
    synthetic_paths = {
        size: synthetic_dir / pattern.format(size=size) for size in validation_cfg["sizes"]
    }

    report = select_best_synthetic(
        real_data,
        synthetic_paths,
        validation_cfg,
        numerical_columns=list(schema.get("numerical_columns", [])),
        categorical_columns=list(schema.get("categorical_columns", [])),
    )

    output_dir = Path(str(validation_cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(str(validation_cfg["report_path"]))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Validation report: %s", report_path)

    if report.get("best_path"):
        best_dest = Path(str(validation_cfg["best_synthetic_path"]))
        best_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report["best_path"], best_dest)
        logger.info("Best synthetic dataset: %s", best_dest)


def main() -> None:
    """Hydra-backed CLI entry point."""
    run_with_hydra(run_validation, caller_file=__file__)


if __name__ == "__main__":
    main()
