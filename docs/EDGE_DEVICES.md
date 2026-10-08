# Phones and Wi-Fi microcontroller cameras: feasibility

Short answer: **the model cannot run on a phone or a microcontroller in any useful way,
but both work well as the *camera* (and the phone as the *viewer*) with the model running
on a PC GPU.** `live_demo.py` on this branch implements that setup.

## How heavy the model is

- ~1.2 B parameters: DINOv2 ViT-L patch encoder (24 blocks) + 24 frame-attention + 24
  global-attention blocks at width 1024, plus camera and DPT depth heads (counted on
  this branch by instantiating the model). ≈ 2.4 GB of weights in bf16/fp16.
- One 518×378 frame is ~1 000 tokens. The transformer trunk alone is ≈ 2 TFLOP per frame,
  and global attention over the KV cache (8 scale frames + 64-frame sliding window ≈
  72 000 cached tokens × 24 layers) adds several TFLOP more once the window is full.
- The KV cache for that window is ≈ 100 MB per cached frame in bf16, i.e. several GB.
- The README's ~20 FPS is on a datacenter-class NVIDIA GPU with FlashInfer.

## Running it on a phone

| Option | Verdict |
|---|---|
| On-device, real time | **Not feasible.** A flagship phone GPU/NPU delivers a few dense fp16 TFLOPS in practice — seconds per frame at best before the KV cache fills, and the weights + cache (≥ 5 GB) exceed what Android/iOS let one app allocate. |
| On-device, offline (record now, reconstruct slowly) | **Technically possible, not practical.** Would need exporting the streaming model (custom 3D RoPE, dynamic KV cache, keyframe logic) to ExecuTorch / ONNX Runtime Mobile / Core ML / QNN, plus int8/int4 quantization and a much smaller sliding window, with unknown accuracy loss. Weeks of engineering for minutes-per-frame results. PyTorch CPU in Termux/proot would take on the order of minutes per frame. |
| **Phone as camera, PC does the work** | **Feasible now.** Run an IP-camera app on the phone (e.g. "IP Webcam" on Android exposes `http://<phone-ip>:8080/video`), then `python live_demo.py --model_path ... --source http://<phone-ip>:8080/video`. Or record a video and run `demo.py --video_path`. |
| **Phone as viewer** | **Feasible now.** The viewer is a web page: run with `--host 0.0.0.0` and open `http://<pc-ip>:8080` in the phone's browser (trusted network only — see `SECURITY_AUDIT.md`). For use away from home, use a VPN such as Tailscale instead of port-forwarding. |

## Camera on a Wi-Fi microcontroller (ESP32-CAM, ESP32-S3 + OV2640/OV5640, etc.)

**Running the model on the microcontroller: impossible** (hundreds of KB of SRAM and a
few MB of PSRAM vs. GB of weights).

**Using it as the camera for the PC: feasible**, with caveats:

- The standard `CameraWebServer` firmware serves MJPEG at `http://<esp-ip>:81/stream`,
  which OpenCV opens directly:
  ```bash
  python live_demo.py --model_path lingbot-map.pt --source http://192.168.1.50:81/stream --fps 8
  ```
  `live_demo.py` reads the stream on a background thread and always uses the newest frame,
  so Wi-Fi jitter or slow inference drops frames instead of building up lag.
- Use VGA (640×480) or SVGA (800×600). The model resizes to 518 px wide anyway, so higher
  resolutions only cost frame rate. Realistic MJPEG rates over Wi-Fi are roughly 10–20 FPS
  at VGA, less at higher resolutions or with weak signal.
- Quality limits, roughly in order of impact: motion blur and rolling shutter (move the
  camera slowly), heavy JPEG compression (raise the firmware's JPEG quality), auto
  exposure / white balance jumps, and lens distortion. Standard ~65° lenses are fine;
  fisheye modules will hurt pose accuracy because the model assumes a pinhole camera.
- The stream is unauthenticated plain HTTP; keep the camera on your own network.
- If you are choosing hardware, a Raspberry Pi Zero 2 W + Pi camera module (not a
  microcontroller, but similar price) gives far better image quality, global-shutter
  options and H.264/RTSP, and works with the same `--source rtsp://...` path.

`demo.py --video_path` cannot take a live URL: it reads until end-of-stream (never, for a
camera) and writes frames next to the "file" path. Use `live_demo.py` for live sources.

### live_demo.py notes

- Collects `--num_scale_frames` frames (default 8) to fix the scene scale, then processes one
  frame at a time with the KV cache and pushes each frame's points to the viewer immediately.
- `--keyframe_interval` (default 2) controls how many frames enter the KV cache; raise it for
  long sessions. Capture stops after `--max_frames` (capped by `--max_frame_num`, default 1024).
- The viewer keeps the newest `--max_vis_frames` frames (default 300) so the browser stays responsive.
- Sky masking and the full `PointCloudViewer` controls are not wired into the live script;
  record the stream to a file and run `demo.py` for the polished offline result.
