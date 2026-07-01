"""CLI entry point for Phase 2+ EDA pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.eda.analyzer import EDAAnalyzer
from src.eda.schema_validator import SchemaValidator
from src.utils.data_loader import load_raw_dataset, load_schema_config
from src.utils.hydra_utils import run_with_hydra
from src.utils.pipeline_bootstrap import bootstrap_pipeline

logger = logging.getLogger(__name__)


def run_eda_pipeline(cfg: DictConfig) -> None:
    """Run the complete EDA pipeline."""
    ctx = bootstrap_pipeline(cfg, caller_file=__file__, log_filename="eda.log", seed=cfg.seed)
    cfg = ctx.cfg

    logger.info("EDA run ID: %s", cfg.eda.run_id)

    schema_path = ctx.project_root / "configs" / "data" / "schema.yaml"
    schema = load_schema_config(schema_path)
    schema["target_column"] = cfg.data.target_column

    raw_path = Path(str(cfg.data.raw_path))
    df = load_raw_dataset(raw_path, schema_path=schema_path, auto_download=True)

    validator = SchemaValidator(schema)
    validation_result = validator.validate(df, strict=True)

    eda_cfg = OmegaConf.to_container(cfg.eda, resolve=True)
    analyzer = EDAAnalyzer(df, schema, eda_cfg, validation_result)
    result = analyzer.run(generate_profile=True)

    logger.info("=" * 60)
    logger.info("EDA PIPELINE COMPLETE")
    logger.info("Output directory: %s", result.output_dir)
    logger.info("Plots: %d files in %s", len(result.plot_paths), result.plots_dir)
    if result.profile_report_path:
        logger.info("Profile report: %s", result.profile_report_path)
    if result.business_insights_path:
        logger.info("Business insights: %s", result.business_insights_path)
    logger.info("Summary: %s", result.summary_path)
    logger.info("=" * 60)


def main() -> None:
    """Hydra-backed CLI entry point."""
    run_with_hydra(run_eda_pipeline, caller_file=__file__)


if __name__ == "__main__":
    main()
