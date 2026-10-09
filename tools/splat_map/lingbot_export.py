"""Export LingBot-Map per-frame predictions (tools/splat_map/lingbot_run_save.py, or
demo_render/batch_demo.py --save_predictions) to a COLMAP text model + nerfstudio transforms.json,
using the FULL-RES frames on disk.

Reads  <pred_dir>/frame_XXXXXX.npz  with keys (per frame, written by batch_demo.save_predictions_npz):
  extrinsic  (3,4)   world->camera (W2C), OpenCV axes (x right, y down, z forward)
  intrinsic  (3,3)   pinhole K at MODEL resolution (cx = W/2, cy = H/2)
  depth      (H,W,1) z-depth at model resolution, arbitrary (non-metric) scale
  depth_conf (H,W)   confidence (>= 1, higher is better)
  images     (3,H,W) float RGB in [0,1] at model resolution
  pose_enc   (9,)    [c2w translation(3), c2w quat XYZW(4), fov_h, fov_w (radians)]
Usage:
  python -I lingbot_export.py PRED_DIR FRAMES_DIR OUT_DIR --full_w 3840 --full_h 2160 \
      [--stride 1] [--first_k 0] [--conf 1.5] [--pts_step 8] [--frame_glob 'frame_%06d.png']
"""
import argparse, glob, json, os
import numpy as np


def rot_to_qvec_wxyz(R):
    # Hamilton quaternion (w, x, y, z) from a rotation matrix (as COLMAP expects).
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def full_res_K(K_model, Hm, Wm, Hf, Wf, image_size=518, patch=14):
    """Map model-res K to full-res K, undoing load_and_preprocess_images(mode='crop')."""
    Hr = round(Hf * (image_size / Wf) / patch) * patch      # resized height before the center crop
    sx, sy = Wm / Wf, Hr / Hf                                # per-axis resize factors (slightly anisotropic)
    assert Wm == image_size and (Hm == Hr or Hm == image_size), (Hm, Wm, Hr)
    return np.array([[K_model[0, 0] / sx, 0, Wf / 2.0],
                     [0, K_model[1, 1] / sy, Hf / 2.0],
                     [0, 0, 1.0]])


def unproject(depth, K, c2w, step):
    H, W = depth.shape
    v, u = np.mgrid[0:H:step, 0:W:step]
    z = depth[v, u]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], -1)
    return pc @ c2w[:3, :3].T + c2w[:3, 3], v, u


def write_ply(path, xyz, rgb):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.empty(len(xyz), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = xyz, rgb
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir"); ap.add_argument("frames_dir"); ap.add_argument("out_dir")
    ap.add_argument("--full_w", type=int, required=True); ap.add_argument("--full_h", type=int, required=True)
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--first_k", type=int, default=0)
    ap.add_argument("--conf", type=float, default=1.5); ap.add_argument("--pts_step", type=int, default=8)
    ap.add_argument("--frame_glob", default="frame_%06d.png")
    ap.add_argument("--shared_camera", action="store_true", help="one PINHOLE camera with median focal")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.pred_dir, "frame_*.npz")))
    listed = os.path.join(a.pred_dir, "frames.txt")   # written by lingbot_run_save.py
    names = [os.path.basename(l.strip()) for l in open(listed) if l.strip()] if os.path.exists(listed) else None
    os.makedirs(os.path.join(a.out_dir, "sparse", "0"), exist_ok=True)
    Wf, Hf = a.full_w, a.full_h
    recs, pts, cols = [], [], []
    for i, fpath in enumerate(files):
        d = np.load(fpath)
        w2c = np.eye(4); w2c[:3] = d["extrinsic"]
        c2w = np.linalg.inv(w2c)
        depth = d["depth"][..., 0]
        Hm, Wm = depth.shape
        Kf = full_res_K(d["intrinsic"], Hm, Wm, Hf, Wf)
        src_idx = (i * a.stride)  # index into the un-strided extracted frames (after first_k)
        name = names[i] if names else a.frame_glob % src_idx
        recs.append((name, w2c, c2w, Kf))
        if a.pts_step > 0:
            p, v, u = unproject(depth, d["intrinsic"], c2w, a.pts_step)
            keep = d["depth_conf"][v, u] > a.conf
            img = (np.transpose(d["images"], (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
            pts.append(p[keep]); cols.append(img[v, u][keep])

    fx_med = np.median([r[3][0, 0] for r in recs]); fy_med = np.median([r[3][1, 1] for r in recs])
    sp = os.path.join(a.out_dir, "sparse", "0")
    with open(os.path.join(sp, "cameras.txt"), "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        if a.shared_camera:
            f.write(f"1 PINHOLE {Wf} {Hf} {fx_med:.6f} {fy_med:.6f} {Wf/2:.6f} {Hf/2:.6f}\n")
        else:
            for j, (_, _, _, K) in enumerate(recs, 1):
                f.write(f"{j} PINHOLE {Wf} {Hf} {K[0,0]:.6f} {K[1,1]:.6f} {K[0,2]:.6f} {K[1,2]:.6f}\n")
    with open(os.path.join(sp, "images.txt"), "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for j, (name, w2c, _, _) in enumerate(recs, 1):
            q = rot_to_qvec_wxyz(w2c[:3, :3]); t = w2c[:3, 3]
            cam = 1 if a.shared_camera else j
            f.write(f"{j} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {cam} {name}\n\n")
    xyz = np.concatenate(pts) if pts else np.zeros((0, 3)); rgb = np.concatenate(cols) if cols else np.zeros((0, 3), np.uint8)
    with open(os.path.join(sp, "points3D.txt"), "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for k, (p, c) in enumerate(zip(xyz, rgb), 1):
            f.write(f"{k} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]} 0\n")
    write_ply(os.path.join(a.out_dir, "points_lingbot.ply"), xyz.astype(np.float32), rgb)
    # nerfstudio: c2w in OpenGL camera axes (flip y and z columns), per-frame intrinsics.
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    frames = [{"file_path": os.path.join(os.path.relpath(a.frames_dir, a.out_dir), n),
               "transform_matrix": (c2w @ flip).tolist(),
               "fl_x": K[0, 0], "fl_y": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]} for n, _, c2w, K in recs]
    json.dump({"camera_model": "OPENCV", "w": Wf, "h": Hf, "frames": frames,
               "ply_file_path": "points_lingbot.ply"},
              open(os.path.join(a.out_dir, "transforms.json"), "w"), indent=1)
    print(f"{len(recs)} images, {len(xyz)} points -> {a.out_dir}")


if __name__ == "__main__":
    main()
