"""Accelerator detection helpers (NVIDIA CUDA, AMD ROCm/HIP, CPU).

PyTorch's ROCm builds expose AMD GPUs through the ``torch.cuda`` namespace, so
``device="cuda"`` and ``torch.amp.autocast("cuda")`` work unchanged on e.g. an
RX 9070 XT. What does *not* carry over is FlashInfer, which only ships CUDA
kernels; on ROCm the model has to use the SDPA attention backend instead.
"""

import torch


def is_rocm() -> bool:
    """True when this PyTorch build targets AMD ROCm/HIP rather than NVIDIA CUDA."""
    return getattr(torch.version, "hip", None) is not None


def describe_accelerator() -> str:
    """Human-readable summary of the GPU PyTorch will use (or 'CPU')."""
    if not torch.cuda.is_available():
        return "CPU (no GPU visible to PyTorch)"
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if is_rocm():
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "?")
        return f"AMD ROCm {torch.version.hip}: {name} ({arch}, {total_gb:.1f} GB)"
    return f"NVIDIA CUDA {torch.version.cuda}: {name} ({total_gb:.1f} GB)"


def pick_inference_dtype(device: torch.device, requested: str = "auto") -> torch.dtype:
    """Choose the autocast dtype.

    ``requested`` is one of ``auto``, ``bf16``, ``fp16`` or ``fp32``. ``auto`` keeps
    the original CUDA rule (bf16 on Ampere+ / sm_80+, else fp16), uses bf16 on
    ROCm (CDNA and RDNA3/RDNA4 handle it natively), and fp32 on CPU.
    """
    explicit = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if requested in explicit:
        return explicit[requested]
    if device.type != "cuda":
        return torch.float32
    if is_rocm():
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def resolve_use_sdpa(use_sdpa: bool) -> bool:
    """Return True when the SDPA attention backend must be used.

    FlashInfer is the default backend, but it is NVIDIA-only, so fall back to
    SDPA automatically on ROCm, on CPU, or when FlashInfer is not installed
    instead of failing deep inside model construction.
    """
    if use_sdpa:
        return True
    if is_rocm():
        print("ROCm build of PyTorch detected: FlashInfer is CUDA-only, using the SDPA backend.")
        return True
    if not torch.cuda.is_available():
        print("No GPU available: FlashInfer needs CUDA, using the SDPA backend.")
        return True
    from lingbot_map.layers.attention import FLASHINFER_AVAILABLE
    if not FLASHINFER_AVAILABLE:
        print("FlashInfer is not installed: using the SDPA backend.")
        return True
    return False
