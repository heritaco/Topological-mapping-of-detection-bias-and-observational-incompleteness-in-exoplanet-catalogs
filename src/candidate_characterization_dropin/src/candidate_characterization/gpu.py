"""GPU detection and optional accelerator helpers.

The project must run on CPU-only machines, but uses GPU-backed libraries when they
are installed and usable. Every GPU path is guarded by runtime checks and fit-time
fallbacks.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import importlib.util
import os
from typing import Any, Dict


@dataclass(frozen=True)
class AcceleratorInfo:
    prefer_gpu: bool
    torch_installed: bool
    torch_cuda_available: bool
    torch_device_count: int
    xgboost_installed: bool
    cupy_installed: bool
    cudf_installed: bool
    selected_device: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def detect_accelerator(prefer_gpu: bool = True) -> AcceleratorInfo:
    torch_installed = _module_exists("torch")
    torch_cuda_available = False
    torch_device_count = 0
    # Importing torch and probing CUDA can be slow on some CPU-only systems.
    # Skip that probe entirely when GPU was explicitly disabled.
    if prefer_gpu and torch_installed:
        try:
            import torch  # type: ignore

            torch_cuda_available = bool(torch.cuda.is_available())
            torch_device_count = int(torch.cuda.device_count()) if torch_cuda_available else 0
        except Exception:
            torch_cuda_available = False
            torch_device_count = 0

    xgboost_installed = _module_exists("xgboost")
    cupy_installed = _module_exists("cupy")
    cudf_installed = _module_exists("cudf")

    selected_device = "cuda" if prefer_gpu and torch_cuda_available else "cpu"
    return AcceleratorInfo(
        prefer_gpu=prefer_gpu,
        torch_installed=torch_installed,
        torch_cuda_available=torch_cuda_available,
        torch_device_count=torch_device_count,
        xgboost_installed=xgboost_installed,
        cupy_installed=cupy_installed,
        cudf_installed=cudf_installed,
        selected_device=selected_device,
    )


def xgb_tree_params(prefer_gpu: bool = True, random_state: int = 42) -> Dict[str, Any]:
    """Return conservative XGBoost params with GPU enabled when plausible.

    XGBoost changed GPU parameters across versions. Modern XGBoost accepts
    ``tree_method='hist', device='cuda'``. If a local installation rejects these
    parameters, model wrappers catch the failure and retry on CPU.
    """
    info = detect_accelerator(prefer_gpu)
    params: Dict[str, Any] = {
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": max(1, min(os.cpu_count() or 1, 8)),
    }
    if prefer_gpu and info.torch_cuda_available:
        params["device"] = "cuda"
    return params
