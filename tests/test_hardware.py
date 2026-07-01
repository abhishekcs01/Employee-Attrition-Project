"""Tests for automatic hardware detection utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.hardware import (
    HardwareProfile,
    detect_hardware,
    get_catboost_params,
    get_lightgbm_params,
    get_xgboost_params,
    log_hardware_configuration,
)


def test_detect_hardware_returns_profile():
    profile = detect_hardware(probe_ml_backends=False)
    assert isinstance(profile, HardwareProfile)
    assert profile.device_type in ("accelerated", "cpu")
    assert profile.torch_device_name in ("cuda", "cpu")
    assert profile.xgboost_device in ("cuda", "cpu")


def test_backend_param_helpers():
    profile = detect_hardware(probe_ml_backends=False)
    xgb = get_xgboost_params(profile)
    cat = get_catboost_params(profile)
    lgb = get_lightgbm_params(profile)

    assert xgb["tree_method"] == "hist"
    assert xgb["device"] in ("cuda", "cpu")
    assert cat["task_type"] in ("GPU", "CPU")
    assert lgb["device"] in ("gpu", "cpu")


def test_profile_to_dict_excludes_hardware_details():
    profile = detect_hardware(probe_ml_backends=False)
    data = profile.to_dict()
    assert "device_type" in data
    assert "xgboost_device" in data
    assert "gpu_name" not in data
    assert "cuda_version" not in data


def test_log_hardware_configuration_is_silent(caplog):
    import logging

    with caplog.at_level(logging.DEBUG):
        profile = log_hardware_configuration(detect_hardware(probe_ml_backends=False))
    assert "Compute backends resolved (automatic)" in caplog.text
    assert "Hardware Configuration" not in caplog.text
    assert "CUDA" not in caplog.text
    assert profile.device_type in ("accelerated", "cpu")


@patch("src.utils.hardware._import_torch")
def test_detect_hardware_without_torch(mock_import_torch):
    mock_import_torch.return_value = None
    profile = detect_hardware(probe_ml_backends=False)
    assert profile.device_type == "cpu"
    assert profile.torch_device_name == "cpu"


@patch("src.utils.hardware._accelerated_available")
@patch("src.utils.hardware._import_torch")
def test_detect_hardware_accelerated_path(mock_import_torch, mock_accelerated):
    mock_import_torch.return_value = MagicMock()
    mock_accelerated.return_value = True

    profile = detect_hardware(probe_ml_backends=False)
    assert profile.device_type == "accelerated"
    assert profile.torch_device_name == "cuda"


@pytest.fixture
def mock_torch_module():
    torch_mod = MagicMock()
    torch_mod.__version__ = "2.1.0"
    torch_mod.cuda.is_available.return_value = True
    if hasattr(torch_mod.cuda, "is_built_with_cuda"):
        torch_mod.cuda.is_built_with_cuda.return_value = True
    torch_mod.cuda.device_count.return_value = 1
    return torch_mod
