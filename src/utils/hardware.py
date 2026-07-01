"""Automatic hardware detection and compute backend configuration for ML workloads.

Detects the best available execution device at runtime without hardcoding device types.
Provides backend-specific parameters for PyTorch, XGBoost, CatBoost, and LightGBM.
Hardware details remain an internal implementation detail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["accelerated", "cpu"]


@dataclass(frozen=True)
class HardwareProfile:
    """Internal snapshot of per-library compute backend configuration."""

    device_type: DeviceType
    torch_device_name: str
    xgboost_device: str
    catboost_task_type: str
    lightgbm_device: str
    backend_notes: dict[str, str] = field(default_factory=dict)

    @property
    def uses_accelerated(self) -> bool:
        """Return True when accelerated compute is available for PyTorch workloads."""
        return self.device_type == "accelerated"

    def to_dict(self) -> dict[str, Any]:
        """Serialize backend configuration (no hardware-identifying details)."""
        return {
            "device_type": self.device_type,
            "xgboost_device": self.xgboost_device,
            "catboost_task_type": self.catboost_task_type,
            "lightgbm_device": self.lightgbm_device,
        }


def _import_torch() -> Any | None:
    """Import PyTorch if installed."""
    try:
        import torch

        return torch
    except ImportError:
        return None


def _is_pytorch_cuda_build(torch_mod: Any) -> bool:
    """Return True when PyTorch was compiled with GPU support."""
    if hasattr(torch_mod.cuda, "is_built_with_cuda"):
        return bool(torch_mod.cuda.is_built_with_cuda())
    if hasattr(torch_mod.backends, "cuda") and hasattr(torch_mod.backends.cuda, "is_built"):
        return bool(torch_mod.backends.cuda.is_built())
    version = getattr(torch_mod, "__version__", "")
    return "+cu" in version


def _accelerated_available(torch_mod: Any | None) -> bool:
    """Return True when PyTorch can use an accelerated device."""
    if torch_mod is None:
        return False
    if not torch_mod.cuda.is_available():
        return False
    if not _is_pytorch_cuda_build(torch_mod):
        return False
    return torch_mod.cuda.device_count() > 0


def _probe_xgboost_gpu(accelerated: bool) -> tuple[str, str | None]:
    """Determine XGBoost device and optional note when CPU is used."""
    if not accelerated:
        return "cpu", None

    try:
        import numpy as np
        import xgboost as xgb

        x_train = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        y_train = np.array([0, 1], dtype=np.float32)
        model = xgb.XGBClassifier(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )
        model.fit(x_train, y_train)
        return "cuda", None
    except Exception:
        return "cpu", None


def _probe_catboost_gpu(accelerated: bool) -> tuple[str, str | None]:
    """Determine CatBoost task_type and optional note when CPU is used."""
    if not accelerated:
        return "CPU", None

    try:
        import numpy as np
        from catboost import CatBoostClassifier

        x_train = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        y_train = np.array([0, 1], dtype=np.int32)
        model = CatBoostClassifier(
            iterations=1,
            depth=1,
            task_type="GPU",
            devices="0",
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x_train, y_train)
        return "GPU", None
    except Exception:
        return "CPU", None


def _probe_lightgbm_gpu(accelerated: bool) -> tuple[str, str | None]:
    """Determine LightGBM device and optional note when CPU is used."""
    if not accelerated:
        return "cpu", None

    try:
        import lightgbm as lgb
        import numpy as np

        x_train = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        y_train = np.array([0, 1], dtype=np.float32)
        train_data = lgb.Dataset(x_train, label=y_train)
        params = {
            "objective": "binary",
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
            "verbosity": -1,
            "num_iterations": 1,
            "max_depth": 1,
        }
        lgb.train(params, train_data, num_boost_round=1)
        return "gpu", None
    except Exception:
        return "cpu", None


def detect_hardware(*, probe_ml_backends: bool = True) -> HardwareProfile:
    """
    Detect available compute backends and configure ML library devices.

    Parameters
    ----------
    probe_ml_backends:
        When True, run lightweight fit probes for GPU-capable libraries.
        Disable in unit tests for faster execution.
    """
    torch_mod = _import_torch()
    accelerated = _accelerated_available(torch_mod)
    device_type: DeviceType = "accelerated" if accelerated else "cpu"
    torch_device_name = "cuda" if accelerated else "cpu"

    backend_notes: dict[str, str] = {}

    if probe_ml_backends:
        xgb_device, _ = _probe_xgboost_gpu(accelerated)
        catboost_task, _ = _probe_catboost_gpu(accelerated)
        lgb_device, _ = _probe_lightgbm_gpu(accelerated)
    else:
        xgb_device = "cuda" if accelerated else "cpu"
        catboost_task = "GPU" if accelerated else "CPU"
        lgb_device = "gpu" if accelerated else "cpu"

    return HardwareProfile(
        device_type=device_type,
        torch_device_name=torch_device_name,
        xgboost_device=xgb_device,
        catboost_task_type=catboost_task,
        lightgbm_device=lgb_device,
        backend_notes=backend_notes,
    )


def log_hardware_configuration(
    profile: HardwareProfile | None = None,
    *,
    log_level: int = logging.DEBUG,
) -> HardwareProfile:
    """
    Log that compute backends were resolved automatically.

    Returns the detected hardware profile. No hardware-identifying details are logged.
    """
    profile = profile or detect_hardware()
    logger.log(log_level, "Compute backends resolved (automatic)")
    return profile


def get_torch_device(profile: HardwareProfile | None = None) -> Any:
    """Return a ``torch.device`` for the active compute backend."""
    torch_mod = _import_torch()
    if torch_mod is None:
        raise ImportError("PyTorch is required but not installed.")

    profile = profile or detect_hardware(probe_ml_backends=False)
    return torch_mod.device(profile.torch_device_name)


def get_xgboost_params(profile: HardwareProfile | None = None) -> dict[str, Any]:
    """Return XGBoost training parameters with automatic device selection."""
    profile = profile or detect_hardware(probe_ml_backends=False)
    return {"tree_method": "hist", "device": profile.xgboost_device}


def get_catboost_params(profile: HardwareProfile | None = None) -> dict[str, Any]:
    """Return CatBoost training parameters with automatic task type."""
    profile = profile or detect_hardware(probe_ml_backends=False)
    params: dict[str, Any] = {"task_type": profile.catboost_task_type}
    if profile.catboost_task_type == "GPU":
        params["devices"] = "0"
    return params


def get_lightgbm_params(profile: HardwareProfile | None = None) -> dict[str, Any]:
    """Return LightGBM training parameters with automatic device selection."""
    profile = profile or detect_hardware(probe_ml_backends=False)
    params: dict[str, Any] = {"device": profile.lightgbm_device}
    if profile.lightgbm_device == "gpu":
        params.update({"gpu_platform_id": 0, "gpu_device_id": 0})
    return params


_HARDWARE_PROFILE: HardwareProfile | None = None


def get_hardware_profile(
    *, refresh: bool = False, probe_ml_backends: bool = True
) -> HardwareProfile:
    """Return cached hardware profile, detecting on first access."""
    global _HARDWARE_PROFILE
    if _HARDWARE_PROFILE is None or refresh:
        _HARDWARE_PROFILE = detect_hardware(probe_ml_backends=probe_ml_backends)
    return _HARDWARE_PROFILE
