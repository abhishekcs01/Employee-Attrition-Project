"""Shared pipeline bootstrap for all ML lifecycle entry points."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.utils.hydra_utils import merge_project_paths, resolve_project_root
from src.utils.logging import setup_logging
from src.utils.paths import PathResolver

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineContext:
    """Resolved runtime context for a pipeline stage."""

    project_root: Path
    path_resolver: PathResolver
    cfg: DictConfig


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)


def bootstrap_pipeline(
    cfg: DictConfig,
    *,
    caller_file: str | Path,
    log_filename: str,
    seed: int | None = None,
) -> PipelineContext:
    """
    Initialize a pipeline stage with resolved paths, directories, and logging.

    Parameters
    ----------
    cfg:
        Hydra configuration object (struct will be disabled for path injection).
    caller_file:
        ``__file__`` from the calling entry-point module.
    log_filename:
        Log file name under ``outputs/``.
    seed:
        Optional global random seed. When provided, numpy and random are seeded.
    """
    project_root = resolve_project_root(caller_file)
    cfg = merge_project_paths(cfg, project_root)

    path_resolver = PathResolver.from_config(OmegaConf.to_container(cfg.paths, resolve=True))
    path_resolver.ensure_dirs()

    if seed is not None:
        set_global_seed(seed)

    setup_logging(level=cfg.logging.level, log_file=path_resolver.outputs / log_filename)
    logger.info("Project root: %s", project_root)

    return PipelineContext(
        project_root=project_root,
        path_resolver=path_resolver,
        cfg=cfg,
    )
