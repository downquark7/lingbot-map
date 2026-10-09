#!/usr/bin/env python3
"""Reproject OpenCV-fisheye (Kannala-Brandt / COLMAP OPENCV_FISHEYE) frames to an ideal pinhole
camera with a chosen horizontal FoV and a CENTERED principal point, and write the pinhole intrinsics
(COLMAP cameras.txt line + JSON). Camera poses are unchanged by this (R = identity), so poses
estimated on the pinhole frames are valid for the original fisheye frames and vice versa.

Usage:
  python undistort_fisheye_to_pinhole.py calib.json frames_fisheye/ frames_pinhole/ \
      --hfov 100 --width 1600 --height 1200
"""
import argparse, glob, json, math, os
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calib")
    ap.add_argument("src_dir")
    ap.add_argument("dst_dir")
    ap.add_argument("--hfov", type=float, default=100.0, help="output horizontal FoV in degrees (< ~110)")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--interp", default="INTER_CUBIC", choices=["INTER_LINEAR", "INTER_CUBIC", "INTER_LANCZOS4"])
    a = ap.parse_args()

    c = json.load(open(a.calib))
    K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], dtype=np.float64)
    D = np.array([c["k1"], c["k2"], c["k3"], c["k4"]], dtype=np.float64).reshape(4, 1)

    W, H = a.width, a.height
    f = (W / 2.0) / math.tan(math.radians(a.hfov) / 2.0)   # square pixels, fx = fy
    # exactly centered principal point: (W-1)/2 in OpenCV convention == W/2 in COLMAP convention
    P = np.array([[f, 0, (W - 1) / 2.0], [0, f, (H - 1) / 2.0], [0, 0, 1]], dtype=np.float64)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, (W, H), cv2.CV_16SC2)

    # valid-pixel mask (pixels that map outside the source frame are black)
    ones = np.full((c["height"], c["width"]), 255, np.uint8)
    valid = cv2.remap(ones, map1, map2, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    frac = float((valid > 0).mean())
    print(f"pinhole f={f:.2f}px  HFOV={a.hfov:.1f}  VFOV={2*math.degrees(math.atan(H/2/f)):.1f}  "
          f"valid pixels={100*frac:.1f}%")
    if frac < 0.999:
        print("NOTE: output has black borders -> lower --hfov/--height, or train with the mask below")

    os.makedirs(a.dst_dir, exist_ok=True)
    interp = getattr(cv2, a.interp)
    paths = sorted(glob.glob(os.path.join(a.src_dir, "*.png")) + glob.glob(os.path.join(a.src_dir, "*.jpg")))
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        out = cv2.remap(img, map1, map2, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(os.path.join(a.dst_dir, os.path.splitext(os.path.basename(p))[0] + ".png"), out)
    cv2.imwrite(os.path.join(a.dst_dir, "..", "valid_mask.png"), valid)

    # COLMAP text format: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy  (COLMAP pixel convention)
    line = f"1 PINHOLE {W} {H} {f} {f} {W/2.0} {H/2.0}"
    with open(os.path.join(a.dst_dir, "..", "cameras_pinhole.txt"), "w") as fh:
        fh.write("# Camera list with one line of data per camera:\n"
                 "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + line + "\n")
    json.dump({"model": "PINHOLE", "width": W, "height": H, "fx": f, "fy": f, "cx": W / 2.0, "cy": H / 2.0,
               "hfov_deg": a.hfov, "valid_fraction": frac},
              open(os.path.join(a.dst_dir, "..", "pinhole_intrinsics.json"), "w"), indent=2)
    print(line)


if __name__ == "__main__":
    main()
