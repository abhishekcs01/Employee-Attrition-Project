"""Tests for pipeline bootstrap utilities."""

from __future__ import annotations

from omegaconf import OmegaConf

from src.utils.pipeline_bootstrap import set_global_seed


def test_set_global_seed_does_not_raise():
    set_global_seed(42)


def test_merge_project_paths():
    from pathlib import Path

    from src.utils.hydra_utils import merge_project_paths

    cfg = OmegaConf.create({"paths": {"project_root": "."}})
    result = merge_project_paths(cfg, Path("/tmp/project"))
    assert result.paths.project_root == "/tmp/project"
