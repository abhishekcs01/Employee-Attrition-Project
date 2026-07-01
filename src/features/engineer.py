"""CLI entry point for feature engineering."""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.features.engineering import engineer_features, write_feature_dictionary
from src.utils.data_loader import load_raw_dataset
from src.utils.hydra_utils import run_with_hydra
from src.utils.pipeline_bootstrap import bootstrap_pipeline

logger = logging.getLogger(__name__)


def run_feature_engineering(cfg: DictConfig) -> None:
    """Engineer features from raw HR data and persist outputs."""
    ctx = bootstrap_pipeline(cfg, caller_file=__file__, log_filename="features.log", seed=cfg.seed)
    cfg = ctx.cfg

    features_cfg = OmegaConf.to_container(cfg.features, resolve=True)
    schema_path = ctx.project_root / "configs" / "data" / "schema.yaml"

    input_path = Path(str(features_cfg["input_path"]))
    df = load_raw_dataset(input_path, schema_path=schema_path, auto_download=True)

    engineered = engineer_features(df, features_cfg)

    output_path = Path(str(features_cfg["output_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(output_path, index=False)
    logger.info("Wrote engineered features: %s", output_path)

    dictionary_path = Path(str(features_cfg["dictionary_path"]))
    write_feature_dictionary(dictionary_path)
    logger.info("Wrote feature dictionary: %s", dictionary_path)


def main() -> None:
    """Hydra-backed CLI entry point."""
    run_with_hydra(run_feature_engineering, caller_file=__file__)


if __name__ == "__main__":
    main()
