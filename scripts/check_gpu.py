"""Check that PyTorch can drive this machine's GPU the way LingBot-Map needs.

Works for NVIDIA (CUDA) and AMD (ROCm) builds of PyTorch. Reports which
scaled-dot-product-attention kernels are usable: if only "math" works on an AMD
card, attention will be slow and memory-hungry (see docs/AMD_ROCM.md).

Usage:
    python scripts/check_gpu.py
"""

import os
import time

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from lingbot_map.utils.device import describe_accelerator, is_rocm, pick_inference_dtype


def main():
    print(f"PyTorch {torch.__version__} | CUDA build: {torch.version.cuda} | ROCm/HIP build: {torch.version.hip}")
    print(f"Accelerator: {describe_accelerator()}")
    if not torch.cuda.is_available():
        if is_rocm():
            print("ROCm PyTorch is installed but sees no GPU. Check `rocminfo`, that your user is in the "
                  "'render' and 'video' groups, and that the ROCm version supports your GPU.")
        return

    device = torch.device("cuda")
    dtype = pick_inference_dtype(device)
    print(f"Inference dtype (auto): {dtype}")

    # Shapes close to one streaming step: ~1000 query tokens attending to a few frames of KV.
    q = torch.randn(1, 16, 1024, 64, device=device, dtype=dtype)
    kv = torch.randn(1, 16, 8 * 1024, 64, device=device, dtype=dtype)
    for name, backend in [("flash", SDPBackend.FLASH_ATTENTION),
                          ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                          ("math", SDPBackend.MATH)]:
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, kv, kv)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(10):
                    F.scaled_dot_product_attention(q, kv, kv)
                torch.cuda.synchronize()
            print(f"  SDPA {name:>13}: OK  ({(time.time() - t0) * 100:.2f} ms/call)")
        except RuntimeError as e:
            print(f"  SDPA {name:>13}: unavailable ({str(e).splitlines()[0][:100]})")

    a = torch.randn(4096, 4096, device=device, dtype=dtype)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    tflops = 20 * 2 * 4096 ** 3 / (time.time() - t0) / 1e12
    print(f"  {dtype} matmul: {tflops:.1f} TFLOPS")


if __name__ == "__main__":
    main()
