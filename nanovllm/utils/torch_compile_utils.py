

import os
import torch
import functools
import logging
logger = logging.getLogger(__name__)


def optional_torch_compile(fn):
    """
    Decorator to optionally apply torch.compile to a function based on device and environment variable.
    Usage:
        @optional_torch_compile
        def forward(...):
            ...
    Control with env var USE_TORCH_COMPILE (default: on for CUDA, off for MPS/CPU).
    """
    if _should_use_torch_compile():
        return torch.compile(fn)
    return fn


@functools.lru_cache(maxsize=1)
def _should_use_torch_compile() -> bool:
    """
    Determines if torch.compile should be applied based on environment and device.
    Returns:
        bool: True if torch.compile should be used, False otherwise.
    """
    if not hasattr(torch, "compile"):
        logger.info("torch.compile is not available in this version of torch.")
        return False
    use_torch_compile = os.environ.get("USE_TORCH_COMPILE")
    if use_torch_compile is not None:
        enabled = use_torch_compile.lower() in ("1", "true", "yes", "on")
        logger.info(f"USE_TORCH_COMPILE env var set to '{use_torch_compile}', enabled={enabled}")
        return enabled
    is_tegra = _is_tegra_platform()
    is_mps = torch.backends.mps.is_available()
    is_nvidia = torch.cuda.is_available()
    logger.info(f"is_tegra={is_tegra}, is_mps={is_mps}, is_nvidia={is_nvidia}")
    enabled = (not is_tegra) and (not is_mps) and is_nvidia
    logger.info(f"torch.compile enabled by device logic: {enabled}")
    return enabled

def _is_tegra_platform() -> bool:
    """
    Checks if the current platform is NVIDIA Tegra/Orin (Jetson).
    Returns:
        bool: True if running on Tegra/Orin, False otherwise.
    """
    try:
        with open("/proc/device-tree/compatible", "rb") as f:
            compat = f.read().lower()
            if b"nvidia,tegra" in compat or b"nvidia,orin" in compat:
                return True
    except Exception:
        pass
    return False
