# Running LingBot-Map on AMD GPUs (RX 9070 XT / RDNA4, ROCm)

**Status: feasible for `demo.py`, `live_demo.py` and `gct_profile.py --backend sdpa`.**
The offline renderer (`demo_render/batch_demo.py`) is **not** supported: it needs NVIDIA
Kaolin, which has no ROCm build.

## Why it works

PyTorch's ROCm build exposes AMD GPUs through the `torch.cuda` API, so the model code
(`device="cuda"`, `torch.amp.autocast("cuda")`, bf16) runs unchanged. The one hard
NVIDIA dependency is **FlashInfer** (paged-KV attention, CUDA kernels only). The model
already has a pure-PyTorch alternative, the **SDPA** backend, which this branch now
selects automatically on ROCm (or whenever FlashInfer is missing).

Changes on this branch:

- `lingbot_map/utils/device.py`: detects ROCm, picks the dtype (bf16 on ROCm), and falls back to SDPA instead of crashing with "FlashInfer is not available".
- `demo.py` / `live_demo.py` set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (ignored on NVIDIA). Without it, PyTorch can leave consumer RDNA cards on the slow `math` attention kernel instead of the AOTriton flash / memory-efficient kernels.
- `--device`, `--dtype` flags for troubleshooting (`--dtype fp16` if bf16 misbehaves).
- `scripts/check_gpu.py`: verifies the GPU is visible and which attention kernels work.

## Install (Linux — recommended)

The RX 9070 XT (gfx1201) is supported from ROCm 6.4.1; ROCm 7.x is the better choice now.

```bash
# 1. ROCm driver/runtime: follow AMD's ROCm install guide for your distro, then
sudo usermod -aG render,video $USER   # log out/in afterwards
rocminfo | grep gfx                   # should list gfx1201

# 2. Python env
conda create -n lingbot-map python=3.12 -y && conda activate lingbot-map

# 3. PyTorch ROCm wheel. Take the exact command from https://pytorch.org/get-started/locally/
#    (Linux > Pip > ROCm). Example for torch 2.8 / ROCm 6.4:
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/rocm6.4

# 4. LingBot-Map (do NOT install flashinfer)
pip install -e ".[vis]"

# 5. Sanity check
python scripts/check_gpu.py
```

`scripts/check_gpu.py` should report `AMD ROCm ...: AMD Radeon RX 9070 XT (gfx1201...)`
and at least `mem_efficient` or `flash` as OK. If only `math` works, attention will be
several times slower and use much more VRAM; upgrade PyTorch/ROCm.

`onnxruntime` (sky masking) runs on the CPU, so `--mask_sky` works unchanged.

## Install (Windows)

AMD ships "PyTorch on Windows" builds (ROCm 7.2, PyTorch 2.9, Python 3.12) that list
the RX 9070 XT as supported — follow AMD's release notes for the install command, then
steps 4–5 above. Expect it to be less mature than Linux, and do not use `--compile`
(it needs Triton/HIP graphs). Running Ubuntu under WSL2 with AMD's ROCm-on-WSL
packages is the alternative.

## Run

```bash
python demo.py --model_path lingbot-map.pt --image_folder example/courthouse --mask_sky
# SDPA is selected automatically; add --offload_to_cpu for long sequences.
```

## What to expect on a 16 GB card

- **Speed**: the "~20 FPS" in the README is NVIDIA + FlashInfer (+ `--compile`). On a 9070 XT with SDPA expect noticeably lower throughput; measure with
  `python gct_profile.py --backend sdpa --dtype bf16`.
- **Memory**: weights are ~2–3 GB in bf16. The KV cache for the default 64-frame sliding window is several GB. If you hit out-of-memory:
  `--offload_to_cpu`, `--kv_cache_sliding_window 32`, `--num_scale_frames 2`, or a larger `--keyframe_interval`.
- `--compile` (HIP graphs) is experimental on ROCm; drop it if warmup fails.
- `HSA_OVERRIDE_GFX_VERSION` is **not** needed for gfx1201 on a ROCm version that supports it.

## Untested

This branch was checked on CPU (no AMD GPU available to the author); the GPU-specific
paths follow PyTorch's documented ROCm behavior. Please run `scripts/check_gpu.py` and the
courthouse example and report anything that fails.
