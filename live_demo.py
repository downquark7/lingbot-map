"""LingBot-MAP live demo: streaming 3D reconstruction from a live camera feed.

The model runs on this machine's GPU; the camera can be anything OpenCV can open:
a local webcam index, an MJPEG-over-HTTP stream (e.g. an ESP32-CAM's
``http://<ip>:81/stream``), an RTSP URL, or a phone running an IP-camera app.

The first ``--num_scale_frames`` frames are processed together to fix the scene
scale; after that each new frame goes through the model once, reusing the KV
cache, and its points are pushed to the browser viewer as soon as they are ready.

Usage:
    # ESP32-CAM / any MJPEG stream on the local network
    python live_demo.py --model_path lingbot-map.pt --source http://192.168.1.50:81/stream

    # Local webcam, viewer reachable from a phone on the same Wi-Fi
    python live_demo.py --model_path lingbot-map.pt --source 0 --host 0.0.0.0
"""

import argparse
import os
import sys
import threading
import time

if "--compile" not in sys.argv:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# See demo.py: enables AOTriton SDPA kernels on consumer AMD GPUs (no-op on CUDA).
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lingbot_map.utils.checkpoint import load_checkpoint_state_dict
from lingbot_map.utils.device import describe_accelerator, pick_inference_dtype, resolve_use_sdpa
from lingbot_map.utils.geometry import closed_form_inverse_se3_general, unproject_depth_map_to_point_map
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


class LatestFrameReader:
    """Reads a capture source on a background thread and keeps only the newest frame.

    Inference is usually slower than the camera, and network streams buffer
    otherwise, so the reconstruction would fall further and further behind.
    """

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source {source!r}")
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self, last_seq, timeout=10.0):
        """Block until a frame newer than ``last_seq`` arrives; return (frame_bgr, seq)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._seq != last_seq:
                    return self._frame, self._seq
            time.sleep(0.005)
        raise TimeoutError("No new frame from the camera source")

    def close(self):
        self._stopped = True
        self._thread.join(timeout=1.0)
        self.cap.release()


def preprocess_frame(frame_bgr, image_size, patch_size):
    """Mirror load_and_preprocess_images(mode="crop") for a single in-memory frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)[None]
    h, w = img.shape[-2:]
    new_w = image_size
    new_h = round(h * (new_w / w) / patch_size) * patch_size
    img = F.interpolate(img, size=(new_h, new_w), mode="bicubic", align_corners=False).clamp_(0, 1)[0]
    if new_h > image_size:
        start = (new_h - image_size) // 2
        img = img[:, start:start + image_size, :]
    return img  # [3, H, W]


def frame_to_points(output, image, conf_threshold, downsample):
    """Turn one frame's predictions into world-space points/colors (same math as demo.py + viewer)."""
    H, W = image.shape[-2:]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(output["pose_enc"].float(), (H, W))
    ext4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    ext4[..., :3, :4] = extrinsic
    ext4[..., 3, 3] = 1.0
    c2w = closed_form_inverse_se3_general(ext4)[..., :3, :4]

    depth = output["depth"][0].float().cpu().numpy()          # [S, H, W, 1]
    conf = output["depth_conf"][0].float().cpu().numpy()      # [S, H, W]
    c2w = c2w[0].cpu().numpy()
    intrinsic = intrinsic[0].cpu().numpy()
    points = unproject_depth_map_to_point_map(depth, c2w, intrinsic)  # [S, H, W, 3]

    colors = (image.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)  # [S, H, W, 3]
    results = []
    for i in range(points.shape[0]):
        keep = np.zeros(conf[i].shape, dtype=bool)
        keep[::downsample, ::downsample] = True
        keep &= conf[i] > conf_threshold
        results.append((points[i][keep].astype(np.float32), colors[i][keep], c2w[i], intrinsic[i]))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=str, required=True,
                        help="Webcam index (e.g. 0) or stream URL (http://.../stream, rtsp://...)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--trust_checkpoint", action="store_true",
                        help="Allow full (code-executing) pickle loading of a non-plain .pt checkpoint")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Max frames per second fed to the model (lower = slower camera motion tolerated)")
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--keyframe_interval", type=int, default=2,
                        help="Every N-th streamed frame is kept in the KV cache")
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Stop after this many frames (default: --max_frame_num)")
    parser.add_argument("--camera_num_iterations", type=int, default=4)
    parser.add_argument("--use_sdpa", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Viewer bind address (0.0.0.0 to watch from a phone on the same network)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--downsample", type=int, default=4, help="Keep every N-th pixel (per axis) in the viewer")
    parser.add_argument("--point_size", type=float, default=0.002)
    parser.add_argument("--max_vis_frames", type=int, default=300,
                        help="Only keep the newest N frames' points in the viewer")
    args = parser.parse_args()
    max_frames = min(args.max_frames or args.max_frame_num, args.max_frame_num)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {describe_accelerator() if device.type == 'cuda' else 'CPU'}")
    use_sdpa = resolve_use_sdpa(args.use_sdpa or device.type != "cuda")
    dtype = pick_inference_dtype(device, args.dtype)

    from lingbot_map.models.gct_stream import GCTStream
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=True,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )
    state_dict = load_checkpoint_state_dict(args.model_path, "cpu", args.trust_checkpoint)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    model = model.to(device).eval()
    if dtype != torch.float32:
        model.aggregator = model.aggregator.to(dtype=dtype)

    import viser
    import viser.transforms as tf
    server = viser.ViserServer(host=args.host, port=args.port)
    status = server.gui.add_text("Status", "waiting for camera...")
    print(f"Viewer: http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}")

    source = int(args.source) if args.source.isdigit() else args.source
    reader = LatestFrameReader(source)
    shown = []

    def publish(idx, pts, cols, c2w, K, H, W):
        server.scene.add_point_cloud(f"/frames/{idx}/points", points=pts, colors=cols,
                                     point_size=args.point_size)
        server.scene.add_camera_frustum(
            f"/frames/{idx}/camera", fov=2 * np.arctan(H / (2 * K[1, 1])), aspect=W / H, scale=0.03,
            wxyz=tf.SO3.from_matrix(c2w[:, :3]).wxyz, position=c2w[:, 3])
        shown.append(idx)
        while len(shown) > args.max_vis_frames:
            server.scene.remove_by_name(f"/frames/{shown.pop(0)}")

    def grab(last_seq):
        frame, seq = reader.read(last_seq)
        return preprocess_frame(frame, args.image_size, args.patch_size), seq

    period = 1.0 / args.fps if args.fps > 0 else 0.0
    seq = -1
    try:
        # Phase 1: scale frames, processed together.
        scale = []
        while len(scale) < args.num_scale_frames:
            t0 = time.time()
            img, seq = grab(seq)
            scale.append(img)
            status.value = f"collecting scale frames {len(scale)}/{args.num_scale_frames}"
            time.sleep(max(0.0, period - (time.time() - t0)))
        scale_images = torch.stack(scale)[None].to(device)
        H, W = scale_images.shape[-2:]

        model.clean_kv_cache()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
            out = model.forward(scale_images, num_frame_for_scale=len(scale),
                                num_frame_per_block=len(scale), causal_inference=True)
        for i, item in enumerate(frame_to_points(out, scale_images[0], args.conf_threshold, args.downsample)):
            publish(i, *item, H, W)

        # Phase 2: one frame at a time with the KV cache.
        n = len(scale)
        t_start = time.time()
        while n < max_frames:
            t0 = time.time()
            img, seq = grab(seq)
            if img.shape[-2:] != (H, W):
                print("Camera resolution changed; skipping frame.")
                continue
            frame = img[None, None].to(device)
            is_keyframe = args.keyframe_interval <= 1 or (n - len(scale)) % args.keyframe_interval == 0
            if not is_keyframe:
                model._set_skip_append(True)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                out = model.forward(frame, num_frame_for_scale=len(scale),
                                    num_frame_per_block=1, causal_inference=True)
            if not is_keyframe:
                model._set_skip_append(False)
            publish(n, *frame_to_points(out, frame[0], args.conf_threshold, args.downsample)[0], H, W)
            n += 1
            fps = (n - len(scale)) / (time.time() - t_start)
            status.value = f"frame {n}/{max_frames}, {fps:.1f} FPS"
            time.sleep(max(0.0, period - (time.time() - t0)))
        print(f"Reached {max_frames} frames; stopping capture (viewer stays up, Ctrl+C to exit).")
        status.value = f"done: {n} frames"
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()


if __name__ == "__main__":
    main()
