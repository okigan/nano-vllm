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
        
    # Disable by default on NVIDIA Orin/Tegra (Jetson) platforms
    is_tegra = False
    try:
        with open("/proc/device-tree/compatible", "rb") as f:
            compat = f.read().lower()
            if b"nvidia,tegra" in compat or b"nvidia,orin" in compat:
                is_tegra = True
    except Exception:
        pass

    is_mps = False
    if torch.backends.mps.is_available():
        is_mps = True

    is_nvidia = False
    if torch.cuda.is_available():
        is_nvidia = True

    if use_torch_compile is not None:
        enabled = use_torch_compile.lower() in ("1", "true", "yes", "on")
    else:
        enabled = (not is_tegra) and (not is_mps) and (is_nvidia)

    if enabled:
        return torch.compile(fn)
    return fn
