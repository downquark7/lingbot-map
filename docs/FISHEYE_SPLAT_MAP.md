# Long looping S25 Ultra fisheye video → 3D Gaussian splat map

How to turn **one long video** from the Galaxy S25 Ultra's 0.6x ultrawide ("fisheye") camera of a
**windowless indoor floor** — many loops, glossy/reflective surfaces, fast movement with motion
blur — into a single, loop-consistent **Gaussian splat map**, on an **AMD RX 9070 XT (16 GB)**,
with an NVIDIA rental as the alternative.

Researched 2026-10-09: 8 parallel researchers, a critic pass and 2 follow-ups. Versions, flags and
file/line references below were checked against source code or release assets that day unless
marked otherwise. Several vendor sites, arxiv.org and huggingface.co were blocked during research,
so some product claims are from search extracts (marked *medium*). **Nothing here has been run on
an RX 9070 XT**; every AMD GPU step comes with a validation gate to run first.

---

## TL;DR — the recommended pipeline

```
video ──► 0. inspect (ffprobe, straight-line test)
      ──► 1. keep the sharpest frame per short window (pick_sharp.py)        CPU
      ──► 2. choose the camera model (OPENCV vs OPENCV_FISHEYE)
      ──► 3. poses WITH loop closure: COLMAP 4.2.1 sequential + loop        CPU (or hloc on GPU)
             detection → global_mapper
      ──► 4. prove the loops closed (model_analyzer, loop_check.py, top-down view)
      ──► 5. (optional) mask mirrors / glass
      ──► 6. train the splat: gsplat RDNA4 fork (Linux, gated) or Brush main  AMD GPU
      ──► 7. clean + compress (SuperSplat, splat-transform)
```

The decisions that matter most, in order:

1. **Loop closure happens in the pose stage, never in the splat trainer.** Doubled walls come from
   poses whose revisits were never linked. The trainer's pose refinement only polishes small
   residuals. Get a single COLMAP model in which every revisit shows up as verified loop pairs.
2. **LingBot-Map is not the pose source for this video.** It has no place recognition, pose graph
   or bundle adjustment; the paper says so itself ("does not incorporate explicit loop-closure
   detection"), and it is pinhole-only. Use it for a fast preview / coverage check, or as optional
   initialization (§8).
3. **Frame selection is the single best fix for blur *and* rolling shutter.** Both scale with
   angular speed, so the sharpest frame of each short window is also the least skewed.
4. **Find out whether the footage is actually fisheye.** Samsung's camera app most likely
   lens-corrects ultrawide video (rectilinear output). Straight door frames at the image edge →
   `OPENCV`; bowed lines → `OPENCV_FISHEYE`, which in COLMAP ≥ 4.2 *requires* a focal prior.
5. **Per-image appearance compensation** (bilateral grid / PPISP) matters for auto-exposure phone
   video, and is one reason gsplat beats Brush when it works on your GPU.

| Stage | AMD Linux (recommended) | AMD Windows | Rented NVIDIA |
|---|---|---|---|
| Frames | `pick_sharp.py` (CPU) | same | same |
| Poses + loops | COLMAP 4.2.1 (conda-forge) SIFT on CPU, or hloc on ROCm | COLMAP 4.2.1 `nocuda` zip; or RealityScan 2.2 (GUI) | COLMAP 4.2.1 CUDA with ALIKED + LightGlue |
| Splat training | gsplat RDNA4 fork (Docker) with MCMC + bilateral grid + pose_opt, after gates; fallback Brush main | Brush main (build from source) | gsplat main with PPISP; or LichtFeld Studio with depth/normal priors |
| Cleanup / delivery | SuperSplat, splat-transform | same | same |

A rented RTX 4090 costs about $0.40–0.89/h (Vast.ai / RunPod catalogs, 2026-09/10), so renting
just for training is a realistic option if the AMD gates fail.

---

## 0. Inspect the clip (5 minutes)

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,color_transfer,color_primaries,color_space:stream_side_data=rotation:format=duration,bit_rate \
  -of json VIDEO.mp4
```

| Field | What it tells you |
|---|---|
| `color_transfer=arib-std-b67` | 10-bit **HLG HDR** (the S25 default per DXOMARK). Tone-map once with a fixed curve — `pick_sharp.py` does it automatically. `smpte2084` = PQ/HDR10, also handled. |
| `r_frame_rate` ≠ `avg_frame_rate` | Variable frame rate (Auto FPS or drops). Work in frame indices, not timestamps. |
| `rotation` = ±90 | Portrait. ffmpeg auto-rotates on decode. Portrait is fine for COLMAP and the splat; LingBot-Map would crop it (§8). |
| width × height | Resolution. 4K is plenty; frames for training get downscaled to ~1920 px anyway. |

Then grab a frame with a door frame or wall/ceiling edge close to the **left or right border**:

```bash
ffmpeg -ss 60 -i VIDEO.mp4 -frames:v 1 -q:v 1 check_lines.jpg   # repeat at a few timestamps
```

- **Edge lines straight, corners look stretched** → lens-corrected, rectilinear footage (most likely
  for the stock Camera app). Use **`OPENCV`** (§2).
- **Edge lines bow outward** → real fisheye/barrel projection (some third-party apps, Blackmagic
  Camera with *Lens correction: Off*, or a clip-on fisheye). Use **`OPENCV_FISHEYE`** (§2).

`exiftool -G1 -a -u -U -ee VIDEO.mp4` shows Android keys (model, capture fps), but no lens or
stabilization tag was found for Samsung video. Infer the lens from the fitted focal length instead:
COLMAP's `fx / width` ≈ **0.38** for the ultrawide at 16:9 (13 mm-equivalent, rectilinear), ≈ 0.69
for the main camera.

> **Video stabilization (EIS / "VDIS") unknown?** It crops and warps each frame differently, which
> breaks the "one shared camera" assumption. There is no tag for it. If the reconstruction has high
> reprojection error or bends even with loops closed, EIS was probably on — re-shoot (§10).

---

## 1. Frame selection: sharpest frame per window

Don't use `ns-process-data video` (verified in source: it extracts only 300 frames by default,
picks "representative" frames with ffmpeg's `thumbnail` filter, and matches without loop detection)
and don't use LingBot-Map's every-Nth `--fps` loader. Neither looks at sharpness.

```bash
python tools/splat_map/pick_sharp.py VIDEO.mp4 frames_sel --window 10 --score-width 960 --ext jpg
```

- One decode pass, full-resolution output, automatic HLG/PQ → SDR tone-mapping (needs ffmpeg with
  `zscale`). Writes `frames_sel/frame_<index>.jpg` (zero-padded, so names sort in time order, which
  COLMAP's `sequential_matcher` requires) and `scores.csv`.
- `--window` = source frames per kept frame. Aim for **3–5 kept fps** with fast motion:
  `--window 10` at 30 fps → 3 fps; `--window 12` at 60 fps → 5 fps.
- Target **2,000–5,000 frames** total. A 20-minute walk at 3 fps is ~3,600.
- Selection is window-relative on purpose: a global sharpness threshold throws away every sharp
  frame of a blank wall.
- Alternatives (also verified): `sharp-frames` 0.4.0
  (`sharp-frames VIDEO.mp4 frames_sel --fps 30 --selection-method batched --batch-size 6 --batch-buffer 0 --format jpg`;
  disk-heavy at 4K) or ffmpeg `blurdetect` (higher `lavfi.blur` = blurrier).

After poses are solved, drop the few frames that are still outliers (high reprojection error or
visibly smeared) before splat training. Do **not** run neural deblurring or temporal denoising on
the frames: per-frame hallucinated detail makes views disagree. If noise visibly hurts matching,
test spatial-only `hqdn3d=luma_spatial=2:chroma_spatial=1.5:luma_tmp=0:chroma_tmp=0` on a short
segment first.

---

## 2. Camera model and focal prior

Always **one shared camera** for the whole video (`--ImageReader.single_camera 1`). Self-calibrating
per image on a long forward-moving sequence absorbs focal errors as scale drift and bending, which is
exactly what stops loops from closing.

| Observation | COLMAP model | Focal prior |
|---|---|---|
| Straight edge lines (lens-corrected, stock app) | `OPENCV` (fx, fy, cx, cy, k1, k2, p1, p2) | Optional. COLMAP's shared-focal solver works without one. A good start is fx = fy ≈ 0.38·W (13 mm-eq., ~106° HFOV at 16:9), cx = W/2, cy = H/2, k = p = 0. |
| Bowed lines (true fisheye projection) | `OPENCV_FISHEYE` (fx, fy, cx, cy, k1–k4) | **Required.** Since COLMAP 4.2.0, fisheye pairs without a focal prior are marked DEGENERATE and nothing registers (reproduced on synthetic data). Equidistant guess: f ≈ (W/2) / (HFOV/2 in radians); for 120° horizontal that is ≈ 0.48·W. |
| Clip-on fisheye (>150°) | `OPENCV_FISHEYE` + a circular mask (`--ImageReader.camera_mask_path`) | Calibrate (below). |

How good does the prior need to be? In a synthetic test, a focal guess 8% off with k = 0 still
registered 80/80 frames and refined to within 0.1%. The two candidate ultrawide figures (0.38·W vs
0.48·W) differ by ~25%, so if unsure, run the quick self-calibration below with each and keep the
one with lower reprojection error.

**Quick self-calibration** (no calibration board): take ~300 consecutive frames with good parallax
and one loop, solve them, read the camera, and fix it for the full run.

```bash
mkdir -p work/calib_sparse work/calib_txt
colmap feature_extractor --database_path work/calib.db --image_path work/calib_frames \
  --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV_FISHEYE \
  --ImageReader.camera_params "FX,FY,CX,CY,0,0,0,0" \
  --FeatureExtraction.max_image_size 1600 --FeatureExtraction.use_gpu 0
colmap sequential_matcher --database_path work/calib.db --SequentialMatching.overlap 15 \
  --FeatureMatching.use_gpu 0
colmap mapper --database_path work/calib.db --image_path work/calib_frames \
  --output_path work/calib_sparse --Mapper.init_min_tri_angle 8
colmap model_converter --input_path work/calib_sparse/0 --output_path work/calib_txt --output_type TXT
cat work/calib_txt/cameras.txt     # -> MODEL W H params...
```

(For rectilinear footage use `OPENCV` with `"FX,FY,CX,CY,0,0,0,0"`.)

**Best: a ChArUco calibration clip** recorded with the *same* lens, resolution, fps and stabilization
setting, the board held at the scan's typical distance and moved into every corner:

```bash
python -c "import cv2; d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100); b=cv2.aruco.CharucoBoard((11,8),0.03,0.022,d); cv2.imwrite('board.png', b.generateImage((3300,2400), marginSize=60))"
python -I tools/splat_map/calib_fisheye_charuco.py calib.mp4 --cols 11 --rows 8 \
  --square 0.03 --marker 0.022 --dict DICT_5X5_100 --every 5 --out calib.json
```

Measure the printed square/marker sizes and pass the real values. The script works on OpenCV 4.x
and 5.0 (5.0 moved the fisheye flags and rejects the usual `(N,1,3)` point layout), aims for < 0.5 px
RMS, and prints `colmap_camera_params` with the +0.5 px COLMAP principal-point convention applied.

Avoid COLMAP's `FOV` model (Brush silently falls back to pinhole, gsplat raises, 3DGRUT asserts) and
`EUCM` (no trainer reads it).

---

## 3. Poses with loop closure

All of these end in **COLMAP 4.2.1** (released 2026-09-29). GLOMAP was merged into COLMAP 4.0 as
`global_mapper` and is the right mapper for thousands of frames. Flags below were checked against
the **4.2.1 tag** (`src/colmap/controllers/option_manager.cc`).

### Getting COLMAP 4.2.1 on AMD

- **Linux:** `micromamba create -y -n colmap -c conda-forge "colmap=4.2.1"` (CPU build with ONNX,
  vocab-tree download and OpenGL; also pulls ~1 GB of unused CUDA libs). Ubuntu apt is too old
  (3.9–3.12, no `global_mapper`). The official Docker image is NVIDIA-only.
- **Windows:** `colmap-x64-windows-nocuda.zip` from the 4.2.1 GitHub release; run `COLMAP.bat`.
- GPU SIFT on AMD: OpenGL SiftGPU needs a display and is **untested on Radeon** — try
  `--FeatureExtraction.use_gpu 1` on ~100 frames and keep it only if feature counts match CPU.
  HIP SIFT exists only on COLMAP `main` (PR #4795, 2026-10-01, never tested on gfx1201). CPU SIFT
  is the dependable default.
- COLMAP's built-in learned features (ALIKED / LightGlue / LoMa, via ONNX) only use the GPU on
  CUDA or CoreML; on AMD they run on the CPU at ~0.3–0.4 s per pair, i.e. **tens of hours** for this
  video. The PyPI `pycolmap` wheel has no ONNX at all. For learned features on AMD use hloc (route B).

### Route A — COLMAP only, CPU SIFT (fewest moving parts; works on Linux and Windows)

```bash
# Prepare the layout every trainer expects: scene/images + scene/sparse/0
mkdir -p scene && mv frames_sel scene/images && mkdir -p work scene/sparse

colmap feature_extractor --database_path work/database.db --image_path scene/images \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV \
  --ImageReader.camera_params "FX,FY,CX,CY,0,0,0,0" \
  --FeatureExtraction.max_image_size 1600 \
  --FeatureExtraction.use_gpu 0 \
  --SiftExtraction.max_num_features 8192
#  fisheye footage: --ImageReader.camera_model OPENCV_FISHEYE --ImageReader.camera_params "FX,FY,CX,CY,K1,K2,K3,K4"
#  mirror/glass masks (§5): --ImageReader.mask_path masks

colmap sequential_matcher --database_path work/database.db \
  --FeatureMatching.use_gpu 0 \
  --SequentialMatching.overlap 10 \
  --SequentialMatching.quadratic_overlap 1 \
  --SequentialMatching.loop_detection 1 \
  --SequentialMatching.loop_detection_period 5 \
  --SequentialMatching.loop_detection_num_images 30 \
  --SequentialMatching.loop_detection_min_index_distance 50 \
  --TwoViewGeometry.min_num_inliers 30 \
  --TwoViewGeometry.min_inlier_ratio 0.25 \
  --TwoViewGeometry.max_error 1.5

colmap global_mapper --database_path work/database.db --image_path scene/images \
  --output_path scene/sparse \
  --GlobalMapper.ba_refine_focal_length 1 \
  --GlobalMapper.ba_refine_principal_point 0 \
  --GlobalMapper.ba_refine_extra_params 1
```

Notes:
- `SiftExtraction.max_image_size` no longer exists (renamed to `FeatureExtraction.max_image_size`
  in 3.13; the CLI rejects unknown options). `SiftMatching.use_gpu` is now `FeatureMatching.use_gpu`.
- **Vocab tree:** with `vocab_tree_path` empty, COLMAP picks and downloads the FAISS tree for the
  feature type (into `~/.cache/colmap`) — confirmed in the conda-forge build, the Windows zip and the
  pycolmap wheel. Offline: download
  `https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin`
  (72,412,636 bytes, sha256 `96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc`) and pass
  `--SequentialMatching.vocab_tree_path`. Old FLANN trees are incompatible.
- `loop_detection_min_index_distance` counts *selected frames* (50 ≈ 15 s at 3 fps); it stops
  "loops" between neighbours that are already matched sequentially.
- Two-view thresholds: COLMAP's own global pipeline uses `max_error 1.0`, `min_num_inliers 30`,
  `min_inlier_ratio 0.25` (CLI defaults are 4.0 / 15 / 0). The stricter inlier settings guard
  against false loops between look-alike corridors; 1.5 px is relaxed for blur and rolling shutter
  (COLMAP has no rolling-shutter model). If too few frames register, try 2–4 px.
- With a calibrated camera, freeze it: set all three `GlobalMapper.ba_refine_*` to 0.
- `global_mapper` does **not** run `view_graph_calibrator`. For `OPENCV` without `camera_params`,
  run `colmap view_graph_calibrator --database_path work/database.db` first (it skips fisheye
  cameras, which must have a prior anyway).
- Since 4.2.0 each disconnected component is written as its own model (`sparse/0`, `sparse/1`, …).
  More than one model means some part of the floor never linked up — see §4.
- Fallback if the global mapper folds the map: incremental
  `colmap mapper ... --Mapper.init_min_tri_angle 8 --Mapper.ba_global_frames_ratio 1.4 --Mapper.ba_global_points_ratio 1.4`
  (hours on CPU).
- **Runtime estimate** (measured 0.64 s/image SIFT and 0.075 s/pair matching on 4 slow cores,
  extrapolated to an 8–16-core desktop): ~5–20 min to extract 3–5k frames, ~15–60 min for ~100k
  pairs, then tens of minutes to ~1.5 h for `global_mapper`.

### Route B — hloc on ROCm PyTorch (better on blur and blank walls; Linux)

Learned ALIKED features + LightGlue matching handle blur and low texture better than SIFT, and
MegaLoc retrieval finds loop candidates much better than a SIFT vocabulary tree.
`tools/splat_map/hloc_pose_stage.py` builds a COLMAP database from: a sequential window, MegaLoc
top-k pairs with near-in-time pairs masked out, ALIKED-n16 + LightGlue matches, one shared camera
with your prior. Then `colmap global_mapper` (route A's last command) does the rest. Tested end to
end with hloc `c13273b` + pycolmap 4.2.1 on CPU (MegaLoc stubbed: Hugging Face was blocked).

```bash
python3.12 -m venv ~/venvs/hloc && source ~/venvs/hloc/bin/activate
# ROCm torch FIRST, otherwise hloc's requirements pull CUDA torch from PyPI.
# Index from AMD TheRock RELEASES.md (could not be fetched to confirm) — or use the
# Linux/Pip/ROCm command from pytorch.org (see docs/AMD_ROCM.md):
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1201]" "torchvision[device-gfx1201]"
pip install "pycolmap==4.2.1" huggingface_hub safetensors
git clone https://github.com/cvg/Hierarchical-Localization && cd Hierarchical-Localization \
  && git checkout c13273b && pip install -e . && cd ..
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Smoke test: ALIKED needs deform_conv2d, LightGlue needs fp16 SDPA. Expect a diff of ~1e-5.
python -c "import torch,torchvision,torch.nn.functional as F; d='cuda'; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0)); x=torch.randn(1,8,64,64,device=d); w=torch.randn(8,8,3,3,device=d); o=torch.zeros(1,18,64,64,device=d); print('deform_conv2d max|diff| vs conv2d:', (torchvision.ops.deform_conv2d(x,o,w,padding=1)-F.conv2d(x,w,padding=1)).abs().max().item()); q=torch.randn(1,4,4096,64,device=d).half(); print('sdpa', F.scaled_dot_product_attention(q,q,q).shape)"

python tools/splat_map/hloc_pose_stage.py --images scene/images --out work \
  --camera_model OPENCV --camera_params "FX,FY,CX,CY,0,0,0,0" \
  --resize_max 1600 --max_kp 4096 --seq_overlap 10 --num_loop 20 --min_gap 30
colmap global_mapper --database_path work/database.db --image_path scene/images --output_path scene/sparse
```

- If the smoke test fails or is wildly off, hloc is unusable on this stack: fall back to route A
  (or run route B on a rented NVIDIA box, where it's the same commands with CUDA torch).
- MegaLoc code comes from `torch.hub` (`gmberton/MegaLoc`; answer `y` to the trust prompt), weights
  from Hugging Face.
- On CPU this route takes tens of hours. On the 9070 XT it is unmeasured (guess: 1–3 h for ~100k
  pairs); on an RTX 4090, ~0.5–1.5 h.

### Route C — RealityScan 2.2 (Windows, GUI) *(medium confidence, from search extracts)*

RealityScan 2.2 (Epic, 2026-06-24) reportedly added full GPU acceleration on AMD Radeon including
the RX 9070 XT. It is free under $1M revenue, imports video directly, has a *Division* distortion
model documented as working for fisheye (GoPro) optics, out-of-core alignment with no hard image
limit, and **control points** to tie loop revisits by hand. Since 2.1 its COLMAP export references
the original distorted images with `FULL_OPENCV`. It does not train splats: export COLMAP, then train
in Brush (Windows) or gsplat. Not evaluated head-to-head with routes A/B — worth trying if you prefer
a GUI, and its manual tie points are a unique fallback for a revisit that won't link automatically.

### Route D — rented NVIDIA

```bash
nvidia-smi   # 'CUDA Version' must be >= 12.9
micromamba create -y -n sfm -c conda-forge "colmap=4.2.1=cuda_129*" python=3.12
micromamba activate sfm && colmap -h | head -n 3     # expect 'with CUDA'
```

Pin the build string, or an old driver makes the solver silently fall back to the CPU build. Then
route A with GPU learned features (the matching ALIKED vocab tree and ONNX models auto-download):

```bash
colmap feature_extractor ... --FeatureExtraction.type ALIKED_N16ROT --FeatureExtraction.use_gpu 1 \
  --AlikedExtraction.max_num_features 4096
colmap sequential_matcher ... --FeatureMatching.type ALIKED_LIGHTGLUE --FeatureMatching.use_gpu 1 \
  --SequentialMatching.loop_detection 1 ...
```

(`LOMA_B` for both is the newer alternative. Don't mix feature types in one database.)

### What not to use as the pose source here

- **LingBot-Map** — see §8.
- MASt3R-SLAM, DROID-SLAM, DPV-SLAM, MegaSaM, VGGSfM, InstantSfM: custom CUDA kernels, mostly
  pinhole-only, several output keyframes only. In VGGT-Long's experiments MASt3R-SLAM lost tracking
  on long sequences.
- MapAnything, VGGT, π3: process all views at once (MapAnything ~140 GB for 2,000 views).
- VGGT-Long / Pi-Long / DA3-Streaming: do have chunk-level retrieval + Sim(3) loop closure and can
  export COLMAP, but no bundle adjustment — at most a backup initialization, refined with COLMAP.
- New 2026 loop-closing streaming models (ABot-Recon has code; HorizonStream, Scal3R, others) are
  worth watching but pinhole-only and unverified for this use.

---

## 4. Prove the loops closed (before any training)

```bash
colmap model_analyzer --path scene/sparse/0
python tools/splat_map/loop_check.py work/database.db 300 50   # verified pairs >= 300 frames apart, >= 50 inliers
colmap gui --import_path scene/sparse/0 --database_path work/database.db --image_path scene/images
```

Pass criteria:
- **One model** (`scene/sparse/0` only) containing nearly all frames.
- Mean reprojection error **under ~1 px**.
- `loop_check.py` lists edges for **every revisit** you remember making — and no implausible ones
  between look-alike corridors.
- In a top-down view, **walls are single thin lines**. Doubled walls = a loop that didn't close.

If something failed:
- **Doubled walls / several models:** more loop candidates (`loop_detection_period 3`,
  `loop_detection_num_images 50`), route B's MegaLoc retrieval, or re-shoot the missing revisit
  (same direction of travel — place recognition often fails on opposite-direction revisits).
- **False loops** (look-alike corridors stitched together): stricter verification
  (`--TwoViewGeometry.min_num_inliers 50`). Doppelgangers++ classifies true vs false pairs on a
  MASt3R backbone (CC BY-NC-SA, script written for COLMAP 3.10; adapting it to these databases is
  untested).
- **Ghost points under glossy floors:** harmless to poses; remove later in the splat.

Optional final refinement once the model is well constrained:
`colmap bundle_adjuster --input_path scene/sparse/0 --output_path scene/sparse_ba --BundleAdjustment.refine_principal_point 1`.

---

## 5. Reflections (optional masking)

3DGS reconstructs a mirror as a window into a "ghost room" behind it. Mirrors and glass also create
false feature matches.

- **Mirrors / glass walls:** segment them (SAM 3 with text prompts like "mirror", "glass door";
  needs a CUDA GPU per its README, so do it on a rental or by hand), save `masks/<image name>.png`
  (white = keep, black = ignore), and pass `--ImageReader.mask_path masks` to COLMAP (file name
  `masks/frame_0000123.jpg.png` for `frame_0000123.jpg`). For training, either keep the masks (the
  mirror becomes a hole; Brush reads a `masks/` folder, nerfstudio `--masks-path`; gsplat's COLMAP
  parser needs a small patch for per-image masks) or train unmasked and delete the ghost Gaussians.
- **Glossy floors: don't mask** — that throws away the floor. Floor reflections produce "mirror
  world" Gaussians below the floor plane; crop them in SuperSplat, and use depth/normal priors where
  available (LichtFeld, §6).
- Reflection-aware splatting methods (3DGS-DR, Ref-Gaussian, GaussianShader, Spec-Gaussian, EnvGS)
  are research code tested on single objects or rooms; not practical for a whole floor yet.
- A circular polarizer on the phone cuts reflections from non-metallic surfaces, but unevenly
  across a 120° frame and at ~1.5 stops of light (re-shoot only).

---

## 6. Train the splat

Train **one model** with a capped Gaussian budget (MCMC), appearance compensation and light pose
refinement. A single floor is not a "city-scale" problem: hierarchical / chunked / anchor-based
methods add CUDA-only complexity and custom viewers without helping here. gsplat's Zip-NeRF fisheye
house recipe (`examples/benchmarks/fisheye/mcmc_zipnerf.sh`: 4 indoor scenes, 1–2k fisheye photos)
is the closest public analog and the base for the commands below.

Downscale training images to ~1920 px: put them in `scene/images_2` (gsplat `--data_factor 2`) or
use Brush's `--max-resolution 1920`. Remove frames flagged as outliers in §4.

### AMD Linux: gsplat RDNA4 fork — best features, must pass gates first

Upstream gsplat is CUDA-first; AMD's official ROCm/gsplat fork targets Instinct (wave64) and is
**silently wrong on RDNA's wave32** without a patch. The community repo
**charyang-ai/gsplat-rocm-rdna4** (commit `34c1ca8`, 2026-09-15; no licence file, 3 stars) packages
ROCm/gsplat 1.5.3 (`b01acd4`) for gfx1201 with a wave32 `WARP_SIZE` patch, Triton fused SSIM and a
tile-size-16 wrapper. Its author validated the pinhole `default` strategy on a Radeon AI PRO R9700
(same gfx1201 chip, 32 GB). MCMC, fisheye and pose refinement have **not** been validated by anyone
on gfx1201; static reading shows they're pure PyTorch or use the patched code, hence the gates.

```bash
git clone https://github.com/charyang-ai/gsplat-rocm-rdna4.git && cd gsplat-rocm-rdna4 \
  && git checkout 34c1ca83ac4b72e32f58db5fc9190ae730f4b690 \
  && docker build -f Dockerfile.gfx1201 --build-arg TRISSIM_REF=22cc02814fd795225144098f27c8d753e70f8889 \
     -t gsplat-rocm:gfx1201-rocm72 .
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=16g -v $PWD/../scene:/data/scene -v $PWD/../subset:/data/subset \
  -v $PWD/../out:/data/out -w /opt/gsplat gsplat-rocm:gfx1201-rocm72 /bin/bash
```

(The host needs an amdgpu driver compatible with ROCm 7.2.1. On Fedora, use `podman` the same way.)

Inside the container, **stop at the first gate that fails** and use Brush instead:

```bash
# Gate 1: wave32 build correctness
python /opt/gsplat/smoke_test.py rasterize && python /opt/gsplat/correctness_test.py
# Gate 2: HIP vs torch reference for projection (incl. fisheye) and pose (v_viewmats) gradients
pip install pytest && python -m pytest -q tests/test_basic.py \
  -k "test_quat_scale_to_covar_preci or test_proj or test_fully_fused_projection_packed or test_isect or test_sh"
# Gate 3: MCMC strategy smoke test
python -m pytest -q tests/test_strategy.py
# Gate 4: on a ~300-frame subset, default vs mcmc should be within ~1 dB PSNR, no NaNs
GSPLAT_TILE_SIZE=16 python /opt/gsplat/run_simple_trainer.py default --data_dir /data/subset --data_factor 1 \
  --result_dir /data/out/ab_default --max_steps 7000 --test_every 20 --disable_viewer
GSPLAT_TILE_SIZE=16 python /opt/gsplat/run_simple_trainer.py mcmc --data_dir /data/subset --data_factor 1 \
  --result_dir /data/out/ab_mcmc --max_steps 7000 --test_every 20 --strategy.cap-max 1000000 --disable_viewer
# compare psnr in /data/out/*/stats/val_step6999.json; repeat mcmc with --use_bilateral_grid, then --pose_opt
```

Full-floor training (rectilinear / `OPENCV` model — the loader undistorts automatically):

```bash
GSPLAT_TILE_SIZE=16 python /opt/gsplat/run_simple_trainer.py mcmc --data_dir /data/scene --data_factor 2 \
  --result_dir /data/out/floor --steps_scaler 2 --strategy.cap-max 3000000 \
  --opacity_reg 0.001 --init_scale 0.5 --use_bilateral_grid --pose_opt \
  --test_every 50 --save_ply --disable_viewer
```

- This fork is the gsplat **1.5.3** API: `--use_bilateral_grid` (not `--post_processing`), no PPISP,
  no `--cache_images`. Don't use `--use_fused_bilagrid` (CUDA extension).
- `--steps_scaler 2` → 60k steps for ~3k frames (rule of thumb ≈ 20 × number of images).
- VRAM (estimates): ~6–9 GB at a 3M cap. Watch `rocm-smi`; raise to 4–5M only with headroom, don't
  exceed ~6M on 16 GB. PLY lands in `/data/out/floor/ply/`.
- **Fisheye model** (`OPENCV_FISHEYE`): add `--camera_model fisheye`, only after Gate 2 passes with
  fisheye (`-k fisheye`). Known parser bug (upstream too): the fisheye remap uses `W//2, H//2`
  instead of `cx, cy`, shifting the image when the principal point is off-centre. Keep cx, cy
  fixed at the centre in COLMAP (`ba_refine_principal_point 0`, the default), or patch inside the
  container:
  `sed -i 's|fx \* x1 \* r + width // 2|fx * x1 * r + cx|; s|fy \* y1 \* r + height // 2|fy * y1 * r + cy|' /opt/gsplat/examples/datasets/colmap.py`
  (untested). 3DGUT (`--with_ut --with_eval3d`) exists but is unvalidated on gfx1201.
- Avoid `--app_opt` for the final export: its PLY bakes view-independent colour and drops SH.

### AMD (Linux or Windows): Brush main — portable fallback, fisheye-native

Brush (Rust + wgpu: Vulkan/DX12/Metal) runs on any GPU. **Build `main` at a commit**: the last
release (v0.3.0, Sep 2025) is pinhole-only; `main` (`1388f74`, 2026-10-03; Cargo version 1.0.0 but
unreleased) renders COLMAP `OPENCV`/`FULL_OPENCV` (radial-tangential), `OPENCV_FISHEYE` (KB4) and
`THIN_PRISM_FISHEYE` natively. Its MCMC-like densification matched gsplat MCMC on Mip-NeRF 360 with
about half the splats. **Weaknesses:** no exposure/white-balance compensation and no pose refinement
on `main` (PR #483 adds `--ppisp-grid`/`--bilateral-grid`, experimental, older base), and the AMD
Vulkan path isn't CI-tested.

```bash
# Linux deps (Ubuntu names; Fedora: vulkan-loader mesa-vulkan-drivers gtk3-devel libxkbcommon-devel openssl-devel)
rustup toolchain install 1.95.0
git clone https://github.com/ArthurBrussee/brush.git && cd brush \
  && git checkout 1388f74c6fe0236f68ee4915564bf00e9d2e3747 \
  && cargo +1.95.0 build --release --locked -p brush-cli
# validate the renderer and camera-model gradients on this GPU first
cargo +1.95.0 test --release --locked -p brush-render
cargo +1.95.0 test --release --locked -p brush-bench-test --test finite_diff
# train
./target/release/brush-cli ../scene --total-train-iters 60000 --growth-stop-iter 30000 \
  --max-splats 4000000 --max-resolution 1920 --eval-split-every 50 --eval-every 5000 \
  --export-every 10000 --export-path ../out/brush
```

- Windows: same commands in PowerShell (needs the MSVC build tools for Rust);
  `.\target\release\brush-cli.exe D:\scene ...`. `-p brush-app` builds the GUI viewer (`brush`).
- Scene folder: `images/`, exactly one `sparse/` model, optional `masks/` (white = keep;
  `--invert-masks` flips), and **no stray `.ply` files** — any `init.ply` (or else the alphabetically
  last `.ply`) becomes the initialization. Random init is currently broken (#555), so keep COLMAP's
  `points3D` (or supply a dense `init.ply`, e.g. from §8).
- v0.3.0 used `--total-steps`; `main` uses `--total-train-iters`.
- Memory: start at `--max-splats 4000000` on 16 GB. Hard ceiling ≈ 11M splats at SH3 (a 2 GiB
  buffer limit, issue #500). Known issues: #395 (crashes on 8–10 GB cards), #555 (wgpu OOM during
  autotune on Windows Radeon).
- Optional `--render-mode mip` for anti-aliased training.

**Not recommended on AMD: OpenSplat.** Its HIP reduction hardcodes `WARP_SIZE 64`
(`rasterizer/gsplat/reduce.cuh:5`, likely wrong gradients on wave32), its COLMAP reader rejects
`OPENCV_FISHEYE`, and it has no appearance model or MCMC.

### Rented NVIDIA (24–48 GB)

**gsplat main** (pin `512d366`, 2026-09-19; PyPI is still 1.5.3):

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git@512d366b67073d77ca099ede742683c165dfc23b
git clone https://github.com/nerfstudio-project/gsplat.git && cd gsplat && git checkout 512d366b67073d77ca099ede742683c165dfc23b
pip install -r examples/requirements.txt --no-build-isolation   # install matching CUDA torch first (pins torch 2.9.1)
cd examples && python simple_trainer.py mcmc --data_dir /data/scene --data_factor 2 \
  --result_dir /data/out/floor --strategy.cap-max 6000000 --opacity_reg 0.001 --init_scale 0.5 \
  --post_processing ppisp --pose_opt --cache_images --steps_scaler 2 --test_every 50 --save_ply --disable_viewer
```

- `--post_processing ppisp` (per-frame exposure/colour + vignetting/response; batch size 1, single
  GPU) or `bilateral_grid`. Fisheye: `--camera_model fisheye`, optionally `--with_ut --with_eval3d`
  (3DGUT, MCMC only; same `W//2` parser caveat). Caps: ~6M on 24 GB, 8–10M on 48 GB.
  `--cache_images` costs ~6.2 MB host RAM per 1080p frame.

**LichtFeld Studio** (v0.5.3; CUDA 12.8+) — the best built-in answer to blank walls and glossy
floors, via MoGe depth/normal priors:

```bash
./build/LichtFeld-Studio preprocess /data/scene --mode both
./build/LichtFeld-Studio --headless -d /data/scene -o /data/out/lfs --strategy mcmc --max-cap 6000000 \
  --steps-scaler 2 --exposure-correction --use-depth-loss --use-normal-loss --export ply,sog
```

Add `--undistort` for distorted cameras (depth/normal losses don't work with `--gut`). It has no
pose-refinement flag in the current CLI. **NVIDIA 3DGRUT** (v2.0.0) trains 3DGUT with native
`OPENCV_FISHEYE`, MCMC and PPISP (`train.py --config-name apps/colmap_3dgut_mcmc.yaml ...`).
**Postshot** (Windows + NVIDIA, commercial; photometric compensation; imports COLMAP) is the easiest
GUI trainer — import poses from §3 rather than trusting its own tracker on a looping floor.

**Skip for this job:** nerfstudio splatfacto (stale, pins gsplat 1.4.0), Inria 3DGS (pinhole only,
no Gaussian cap), Hierarchical-3DGS / Octree-GS / Scaffold-GS / CityGaussian (complex, custom
viewers or non-commercial licences), blur/rolling-shutter-aware splatting (BAD-Gaussians,
Deblur-GS, "Gaussian Splatting on the Move": 2024 CUDA research code tested on ~20–100 images).

---

## 7. Clean up and deliver

- **SuperSplat** (browser): crop the "mirror world" under the floor, delete floaters and ghost rooms
  behind mirrors.
- **PlayCanvas splat-transform** (v3.10.0) for compression and streamed level-of-detail:

```bash
npm install -g @playcanvas/splat-transform
splat-transform -w floor.ply --filter-nan floor_clean.ply
splat-transform -w floor_clean.ply floor.sog
splat-transform -w floor_clean.ply -d 50% floor_L1.ply && splat-transform -w floor_L1.ply -d 50% floor_L2.ply
splat-transform -w floor_clean.ply -l 0 floor_L1.ply -l 1 floor_L2.ply -l 2 lod/lod-meta.json
```

  `-F` (floater) and `-C` (cluster) filters use world-unit defaults, so scale to metres (`-s`) first.
- **Spark** (three.js, v2.3.1): `npm run build-lod -- floor.ply --quality` → paged `.rad`. Default LoD
  budgets are ~1M splats on Android and 2.5M on desktop — relevant for viewing on the phone.
- gsplat exports in a normalized world frame by default; Brush tags its PLY for anti-aliasing.

---

## 8. Where LingBot-Map fits (and where it doesn't)

**What it is:** a fast feed-forward visual-odometry + depth front end. Per frame it gives a pose, a
pinhole FoV and dense depth with confidence. Benchmarks: robust to blur (7-Scenes ATE 0.08 m) and
long sequences; an independent August 2026 evaluation found it globally stable but locally jittery
(RPE-rot 2.29° vs ~0.1° for the best methods) and failing on the longest paths.

**Why it can't be the pose source here:**
- **No loop closure.** Streaming mode keeps the 8 anchor frames and the last 64 keyframes in full,
  and only 6 tokens per older frame. A revisit after a long excursion can at best snap the *current*
  pose; the drifted segment stays drifted — doubled walls. Windowed mode aligns each window to the
  previous one only (`gct_stream_window.py:756-956`) and never links revisits. Upstream issues #60
  (duplicated layers on a 2,873-frame loop) and #78 are open with no fix.
- **Pinhole only.** No distortion terms, principal point fixed at the centre, trained on 40–100°
  HFOV. Undistort to ≤ ~100° HFOV first (`undistort_fisheye_to_pinhole.py` for fisheye footage).
- **Length:** stable to ~3,000 frames in streaming mode; keep it in streaming mode with ≤ 3,000
  sampled frames rather than windowed.
- **Portrait input** is centre-cropped to 518×518 (loses ~44% of the vertical FoV).
- No COLMAP/3DGS exporter upstream (issue #87); 3DGS collapsing on exported poses is reported (#35).

**Useful roles:**
1. **Preview / coverage check** in minutes before committing hours to SfM:
   `python demo.py --model_path lingbot-map.pt --image_folder frames_sel --offload_to_cpu`.
   Holes in coverage show up immediately; drift on revisits is expected.
2. **Dense initialization / extra loop candidates.** Save predictions and export a COLMAP text model
   + `transforms.json` + a dense PLY:

   ```bash
   python tools/splat_map/lingbot_run_save.py --image_folder frames_pinhole --out_dir pred \
     --model_path lingbot-map.pt --mode streaming
   python -I tools/splat_map/lingbot_export.py pred frames_pinhole lingbot_colmap \
     --full_w W --full_h H --shared_camera --conf 1.5 --pts_step 8
   ```

   `points_lingbot.ply` can seed Brush (`init.ply`) only after it is aligned to the COLMAP frame
   (the two reconstructions have different coordinate frames and scales) — not automated here.
   Camera centres can also feed `colmap spatial_matcher` as extra pair candidates; never use them as
   tight pose priors, which would lock in the drift.
3. **Pose convention trap:** the decoded `pose_enc` is camera-to-world; the saved `extrinsic` is
   **world-to-camera** (the "Convert w2c to c2w" comment in `demo.py`/`demo_render/demo.py` is
   misleading). Units are normalized to the anchor frames, not metres.
4. `lingbot-map-long.pt` exists on Hugging Face (4.63 GB) but is undocumented in this repo; it was
   not evaluated.

On AMD, `demo_render/batch_demo.py` needs `--use_sdpa` passed explicitly and casts nothing to bf16;
`lingbot_run_save.py` avoids both issues and keeps frames on the CPU.

---

## 9. Turnkey alternatives (worth one cheap try)

No turnkey product reliably turns this kind of video into a drift-free whole-floor splat on an AMD
GPU (as of 2026-10). Their SfM is opaque: when revisits fail you get doubled walls and nothing to
fix them with. Still, a one-hour trial tells you whether "good enough" is achievable:

| Product | Accepts this file? | Limits (medium confidence) |
|---|---|---|
| Polycam (web upload) | Yes | Business: 30 min / 16 GB video; splats capped at 1,000 images |
| Varjo Teleport (web upload) | Yes, if ≤ 15 min | 30 s–15 min, ≤ 5 GB; dev plan 3,000 images / 4M splats |
| KIRI Engine | Barely | ~3 min at ≤ 1080p |
| Scaniverse | No (live capture or 360° video only) | Re-shoot option |
| Postshot | Yes (drag in video) | Windows + NVIDIA RTX only; export needs Indie (€17/mo) |
| RealityScan 2.2 | Yes (frames from video) | Windows; alignment only — see §3 route C |

Feed-forward "video → splat" models (AnySplat, WorldMirror 2.0, DA3 Gaussian head, Long-LRM,
InstantSplat…) handle 2–64 views, don't close loops, and assume pinhole cameras: room-sized previews
only.

---

## 10. If you can re-shoot (big quality gain)

Camera settings (stock app *Pro Video* on the 0.6x lens, or Blackmagic Camera):
- **Video stabilization OFF**, Super Steady OFF, Auto FPS OFF, HDR / Log OFF, auto lens switching OFF.
- **4K 60 fps**, landscape, fixed white balance (Kelvin), manual focus ~1.5–3 m (also avoids the
  ultrawide's reported AF shake).
- **Shutter:** first point at a white wall at 1/500 s. No moving bands → 1/250–1/500 s (blur at a
  90°/s turn: ~38 px at 1/60 vs ~4–9 px at 1/250–1/500). Bands (LED/fluorescent flicker) → 1/100
  (50 Hz mains) or 1/120 (60 Hz) and move about half as fast. Shorter exposure does **not** reduce
  rolling-shutter skew; slower turning does. *Unverified:* whether Pro Video allows manual shutter on
  the ultrawide.
- Never zoom, switch lenses or apps mid-capture; projection must be constant.

Walking pattern:
- Chest height, phone tilted 10–20° down, two hands or a mechanical gimbal (doesn't warp pixels).
- ~0.5 m/s, 1–3 m from walls; turns as slow arcs while walking, not fast pans in place
  (≤ ~30°/s).
- Rooms: walk the perimeter facing inward. Junctions: a slow (~10 s) 360° turn.
- **Close loops in the same direction of travel**; finish where you started and re-film the first
  10–20 s of the path.
- Keep lights on and doors fixed; avoid filming straight into mirrors and glass.
- Before the scan: a 30–60 s ChArUco calibration clip with identical settings (§2) and short test
  clips (flicker, exposure, straight lines at the frame edge).

IMU recording only pays off with a visual-inertial pipeline (e.g. Spectacular AI's app + `sai-cli`,
NVIDIA-oriented, non-commercial licence); COLMAP, the trainers and LingBot-Map ignore it.

---

## 11. Known unknowns

- Nothing here was run on an RX 9070 XT: OpenGL SiftGPU on Radeon, hloc (ALIKED/LightGlue) on ROCm
  gfx1201, MCMC/fisheye/pose_opt in the gsplat RDNA4 fork, and Brush's Vulkan path on RDNA4 are all
  gated, not confirmed.
- Whether the stock S25U ultrawide video is fully rectilinear, and how Samsung's EIS warps it
  (rotation-only vs mesh warp, crop size). The follow-up researching this hit the session limit;
  use the straight-line test (§0) and the reconstruction's reprojection error as the evidence.
- RealityScan 2.2's AMD acceleration, fisheye handling and COLMAP export are from search extracts.
- `lingbot-map-long.pt` is undocumented.

---

## Helper scripts (`tools/splat_map/`)

| Script | Purpose | Tested |
|---|---|---|
| `pick_sharp.py` | Sharpest frame per window, one decode pass, HDR tone-mapping, `scores.csv` | Synthetic blurred / rotated / HLG clips |
| `calib_fisheye_charuco.py` | ChArUco → OPENCV_FISHEYE intrinsics (+COLMAP params) | Synthetic KB camera, OpenCV 4.14 and 5.0 |
| `undistort_fisheye_to_pinhole.py` | Fisheye frames → pinhole at an exact HFOV, centred principal point | OpenCV 4.14 and 5.0 |
| `hloc_pose_stage.py` | hloc ALIKED + LightGlue + MegaLoc → COLMAP database | End to end with pycolmap 4.2.1 on CPU (MegaLoc stubbed) |
| `loop_check.py` | Lists verified long-range (loop) pairs in a COLMAP database | Real COLMAP database |
| `lingbot_run_save.py` | Run LingBot-Map on a frame folder and save per-frame predictions | End to end with `lingbot_export.py` on CPU with random weights (mechanics only) |
| `lingbot_export.py` | LingBot predictions → COLMAP text + `transforms.json` + PLY | Synthetic data (reprojection error ~1e-4 px) |

`undistort_fisheye_to_pinhole.py` expects OpenCV-convention intrinsics (as written by
`calib_fisheye_charuco.py`). If your intrinsics come from COLMAP's `cameras.txt`, subtract 0.5 from
cx and cy first.

## Key sources

- COLMAP 4.2.1 options: https://github.com/colmap/colmap/blob/4.2.1/src/colmap/controllers/option_manager.cc ·
  release assets: https://github.com/colmap/colmap/releases/tag/4.2.1 · FAQ (camera models,
  known-pose triangulation): https://github.com/colmap/colmap/blob/main/doc/faq.rst · HIP SIFT:
  https://github.com/colmap/colmap/pull/4795
- hloc: https://github.com/cvg/Hierarchical-Localization · MegaLoc: https://github.com/gmberton/MegaLoc
- gsplat: https://github.com/nerfstudio-project/gsplat (fisheye recipe
  `examples/benchmarks/fisheye/mcmc_zipnerf.sh`) · ROCm/gsplat: https://github.com/ROCm/gsplat ·
  RDNA4 fork: https://github.com/charyang-ai/gsplat-rocm-rdna4
- Brush: https://github.com/ArthurBrussee/brush · OpenSplat: https://github.com/pierotofy/OpenSplat
- LichtFeld Studio: https://github.com/MrNeRF/LichtFeld-Studio · 3DGRUT / 3DGUT:
  https://github.com/nv-tlabs/3dgrut, https://arxiv.org/abs/2412.12507
- VGGT-Long: https://arxiv.org/abs/2507.16443 · Doppelgangers++:
  https://github.com/doppelgangers25/doppelgangers-plusplus · SAM 3:
  https://github.com/facebookresearch/sam3 · MoGe: https://github.com/microsoft/MoGe
- sharp-frames: https://github.com/Reflct/sharp-frames-python · splat-transform:
  https://github.com/playcanvas/splat-transform · Spark: https://github.com/sparkjsdev/spark
- AMD gsplat docs: https://rocm.docs.amd.com/projects/gsplat/en/latest/ · TheRock:
  https://github.com/ROCm/TheRock
- LingBot-Map paper (Limitations, long-sequence modes): `lingbot-map_paper.pdf` in this repo;
  issues #35, #60, #78, #87 upstream.
