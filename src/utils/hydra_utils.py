"""Hydra configuration bootstrap for CLI entry points."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


def resolve_project_root(caller_file: str | Path | None = None) -> Path:
    """Determine project root from env or caller module location."""
    if env_root := os.environ.get("PROJECT_ROOT"):
        return Path(env_root).resolve()
    if caller_file is None:
        caller_file = Path.cwd()
    caller_path = Path(caller_file).resolve()
    if caller_path.name in {
        "run.py",
        "train.py",
        "engineer.py",
        "evaluate.py",
        "generate.py",
        "main.py",
        "compare.py",
    }:
        parent = caller_path.parent.name
        if parent in {
            "eda",
            "synthetic",
            "validation",
            "features",
            "training",
            "evaluation",
            "explainability",
        }:
            return caller_path.parents[2]
        if parent == "api" and caller_path.parents[1].name == "deployment":
            return caller_path.parents[3]
        if parent == "streamlit_app" and caller_path.parents[1].name == "deployment":
            return caller_path.parents[3]
    return caller_path


def config_dir(project_root: Path | None = None) -> str:
    """Return absolute Hydra config directory path."""
    root = project_root or resolve_project_root()
    return str(root / "configs")


def run_with_hydra(
    pipeline: Callable[[DictConfig], None],
    *,
    config_name: str = "config",
    caller_file: str | Path | None = None,
) -> None:
    """
    Execute a pipeline with Hydra configuration.

    Works reliably from console script entry points and ``python -m`` invocations.
    """
    root = resolve_project_root(caller_file)
    os.environ.setdefault("PROJECT_ROOT", str(root))

    overrides = list(sys.argv[1:])
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir(root), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
        pipeline(cfg)


def merge_project_paths(cfg: DictConfig, project_root: Path) -> DictConfig:
    """Attach resolved project-root paths to a Hydra config object."""
    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    cfg.paths.project_root = str(project_root)
    return cfg
