# Security audit: running LingBot-Map on a personal PC without a container

Scope: everything in this repository (`lingbot_map/`, `demo.py`, `gct_profile.py`,
`demo_render/`, `benchmark/`, `preprocess/`, `scripts/`, shell scripts, HTML/JS
templates). Reviewed by searching for every process-spawn, network, deserialization,
file-deletion and dynamic-import call site and reading each one in context.

## Verdict

**Reasonably safe to run on your main PC**, provided you (a) get the checkpoint from the
official source and (b) keep the 3D viewer off untrusted networks. Nothing in the code
is obfuscated, phones home, collects telemetry, uses `shell=True`, `eval`/`exec`, or
downloads and runs code. The real risks are the standard ones for ML research code,
listed below with what this branch changes.

## Findings

| # | Severity | Finding | Status on this branch |
|---|----------|---------|-----------------------|
| 1 | High (if the checkpoint is untrusted) | `demo.py`, `demo_render/demo.py` and `benchmark/methods/lingbot_map.py` load the checkpoint with `torch.load(..., weights_only=False)`. That runs the full pickle unpickler, so a tampered `.pt` file can execute arbitrary code as your user. | **Fixed** for `demo.py`, `live_demo.py`, `demo_render/`: loads with `weights_only=True` (tensors only); full unpickling needs an explicit `--trust_checkpoint`. `.safetensors` is supported and `scripts/convert_checkpoint_to_safetensors.py` converts once. Benchmark harness left as-is. |
| 2 | Medium | The point-cloud viewer (`viser`) listened on `0.0.0.0` with no authentication. Anyone on the same network could open it, see your reconstruction (your rooms, from your photos), **and use its GUI "Output Path" text boxes (screenshot / GLB / video export) to write or overwrite files anywhere your user can write.** **Fixed (file writes) / by design (viewing)**: export file names are now confined to `--export_dir` (default `./viewer_exports/`); absolute paths, `..` and symlinks leading out of it are rejected. The viewer still listens on `0.0.0.0` by default (owner's choice, for phone viewing), so anyone on the network can *see* the reconstruction; pass `--host 127.0.0.1` on public or shared Wi-Fi. |
| 3 | Medium | `demo_render/interactive_viewer/server.py` (aiohttp) defaults to `--host 0.0.0.0`; `benchmark/viewer.py` binds your LAN IP. Read-only routes (no file-path inputs), but they expose your data to the network. | Not changed — pass `--host 127.0.0.1` to the interactive viewer. |
| 4 | Low | `--mask_sky` auto-downloads `skyseg.onnx` from Hugging Face over HTTPS with no checksum (`lingbot_map/vis/sky_segmentation.py`, `demo_render/rgbd_render/data/sky.py`, `lingbot_map/vis/glb_export.py`). ONNX is a graph run by onnxruntime, not Python, so a swapped file is far less dangerous than a pickle. | Not changed. Download it yourself and pass `--sky_model` if you prefer. |
| 5 | Low | `np.load(..., allow_pickle=True)` on prediction `.npz` files (`demo_render/demo.py --load_predictions`, `demo_render/batch_demo.py`, `benchmark/viewer.py` cache). Same pickle risk as #1, but only for `.npz` files someone else gives you. | Not changed. Only load `.npz` files you produced. |
| 6 | Low (data loss) | `shutil.rmtree` on output dirs: `<output>_render_frames` after rendering, benchmark `--force`/`--clean` result and report dirs, viewer cache "Clear cache". All paths are derived from arguments you pass. | Not changed. Don't point `--output` / benchmark output dirs at folders holding other data. |
| 7 | Info | Subprocesses: `ffmpeg`, `colmap`, `conda run` — all list-form arguments, no shell, no injection path. | OK |
| 8 | Info | `demo.py --video_path x.mp4` writes extracted frames to `x_frames/` next to the video. | OK (side effect to know about) |
| 9 | Info | FlashInfer / `torch.compile` / `render_cuda_ext` JIT-compile native kernels into `~/.cache` or the build dir; `benchmark/envs/*.sh` create/remove *conda envs* by name (`--force` deletes the env). | OK |
| 10 | Info | Dependencies are unpinned PyPI packages (normal supply-chain exposure). `huggingface_hub` is declared but unused. | OK |

`benchmark/viewer.py`'s `socket.connect(("8.8.8.8", 1))` sends no packets; it is the
usual trick to discover which local IP the default route uses.

## Recommended way to run it on your PC

1. Use a dedicated venv/conda env (isolates Python packages, not a security boundary, but keeps your system Python clean).
2. Download the checkpoint only from `huggingface.co/robbyant/lingbot-map` (or ModelScope `Robbyant/lingbot-map`).
3. Run without `--trust_checkpoint` first. If the official file is refused, it contains non-tensor objects; decide whether you trust it, then convert once to `.safetensors` and use that from then on.
4. The viewer is reachable from your network by default. On public or shared Wi-Fi, pass `--host 127.0.0.1`.
