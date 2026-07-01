"""Logging configuration utilities."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml


def setup_logging(
    config_path: Path | None = None,
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """Configure application logging from YAML or defaults."""
    if config_path and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if log_file:
            config["handlers"]["file"]["filename"] = str(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            if "file" not in config["root"]["handlers"]:
                config["root"]["handlers"].append("file")

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
