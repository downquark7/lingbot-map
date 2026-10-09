"""Run LingBot-Map on a folder of frames and SAVE per-frame predictions (no viewer, no Kaolin).

Run from anywhere inside this repo (it imports the root demo.py). Works on NVIDIA (FlashInfer if
installed) and AMD ROCm / CPU (SDPA auto-selected). Unlike demo.py it keeps the frame tensor on the CPU
(demo.py:494 moves all frames to the GPU) and unlike demo_render/batch_demo.py it casts the aggregator to
bf16 (saves ~2-3 GB VRAM, demo.py:485-492) and auto-picks the keyframe interval for streaming.

Output: OUT/frame_%06d.npz with extrinsic (3,4) W2C OpenCV, intrinsic (3,3) at model res, depth (H,W,1),
depth_conf (H,W), images (3,H,W) in [0,1], pose_enc (9,) ; OUT/frames.txt = source image path per index.
"""
import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
# tools/splat_map/ -> repo root, so `from demo import ...` finds the root demo.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from demo import load_images, load_model, postprocess, prepare_for_visualization
from lingbot_map.utils.device import pick_inference_dtype, resolve_use_sdpa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_folder", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--mode", default="streaming", choices=["streaming", "windowed"])
    p.add_argument("--keyframe_interval", type=int, default=None)
    p.add_argument("--window_size", type=int, default=128)
    p.add_argument("--overlap_keyframes", type=int, default=16)
    p.add_argument("--num_scale_frames", type=int, default=8)
    p.add_argument("--kv_cache_sliding_window", type=int, default=64)
    p.add_argument("--stride", type=int, default=1)
    a = p.parse_args()
    # Remaining fields read by demo.load_model (demo.py:140-173)
    a.image_size, a.patch_size, a.enable_3d_rope, a.max_frame_num = 518, 14, True, 1024
    a.camera_num_iterations, a.trust_checkpoint = 4, False

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.use_sdpa = resolve_use_sdpa(dev.type != "cuda")
    images, paths, _ = load_images(image_folder=a.image_folder, stride=a.stride)  # CPU [S,3,H,W]
    model = load_model(a, dev)
    dtype = pick_inference_dtype(dev)
    if dtype != torch.float32:
        model.aggregator = model.aggregator.to(dtype=dtype)

    S = images.shape[0]
    kf = a.keyframe_interval or (1 if (a.mode == "windowed" or S <= 320) else -(-S // 320))
    cpu = torch.device("cpu")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=dev.type == "cuda"):
        if a.mode == "streaming":
            pred = model.inference_streaming(images, num_scale_frames=a.num_scale_frames,
                                             keyframe_interval=kf, output_device=cpu)
        else:
            pred = model.inference_windowed(images, window_size=a.window_size,
                                            overlap_keyframes=a.overlap_keyframes,
                                            num_scale_frames=a.num_scale_frames,
                                            keyframe_interval=kf, output_device=cpu)
    imgs = pred["images"]
    pred, imgs = postprocess(pred, imgs)            # adds 'extrinsic' (= inverse of decoded pose = W2C)
    pred = prepare_for_visualization(pred, imgs)    # numpy, batch dim removed
    os.makedirs(a.out_dir, exist_ok=True)
    for i in range(S):
        np.savez(os.path.join(a.out_dir, f"frame_{i:06d}.npz"),
                 extrinsic=pred["extrinsic"][i], intrinsic=pred["intrinsic"][i],
                 depth=pred["depth"][i], depth_conf=pred["depth_conf"][i],
                 images=pred["images"][i], pose_enc=pred["pose_enc"][i])
    meta = {k: np.asarray(pred[k]) for k in ("frame_type", "is_keyframe", "chunk_scales", "chunk_transforms")
            if k in pred}
    if meta:
        np.savez(os.path.join(a.out_dir, "meta.npz"), **meta)
    with open(os.path.join(a.out_dir, "frames.txt"), "w") as f:
        f.write("\n".join(paths) + "\n")
    print(f"saved {S} frames (mode={a.mode}, keyframe_interval={kf}) to {a.out_dir}")


if __name__ == "__main__":
    main()
