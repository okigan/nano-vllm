import os
import torch


def optional_torch_compile(fn):
    """
    Decorator to optionally apply torch.compile to a function based on device and environment variable.
    Usage:
        @optional_torch_compile
        def forward(...):
            ...
    Control with env var USE_TORCH_COMPILE (default: on for CUDA, off for MPS/CPU).
    """

    if not hasattr(torch, "compile"):
        return fn

    use_torch_compile = os.environ.get("USE_TORCH_COMPILE")

    if use_torch_compile is not None:
        enabled = use_torch_compile.lower() in ("1", "true", "yes", "on")
    else:
        enabled = False

    if enabled:
        return torch.compile(fn)
        
    is_tegra = _is_tegra_platform()
    is_mps = torch.backends.mps.is_available()
    is_nvidia = torch.cuda.is_available()

    enabled = (not is_tegra) and (not is_mps) and (is_nvidia)

    if enabled:
        return torch.compile(fn)
    return fn

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
