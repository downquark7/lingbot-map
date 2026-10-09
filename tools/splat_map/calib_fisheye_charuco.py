#!/usr/bin/env python3
"""Calibrate a fisheye / ultrawide lens (Kannala-Brandt = OpenCV fisheye = COLMAP OPENCV_FISHEYE)
from a ChArUco calibration video recorded with the SAME lens, resolution, fps and stabilization
setting as the scan video.

Usage:
  python calib_fisheye_charuco.py calib.mp4 --cols 11 --rows 8 --square 0.03 --marker 0.022 \
      --dict DICT_5X5_100 --every 10 --out calib.json

Works with opencv-python 4.x and 5.x (flag names moved in 5.x).
"""
import argparse, json, sys
import cv2
import numpy as np


def F(name):
    """Fisheye calib flag. OpenCV 4.x: cv2.fisheye.CALIB_* (different bit values from the pinhole
    flags!). OpenCV 5.x: unified cv2.CALIB_* (cv2.fisheye.CALIB_* no longer exported)."""
    return getattr(cv2.fisheye, name) if hasattr(cv2.fisheye, name) else getattr(cv2, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--cols", type=int, required=True, help="squares along X")
    ap.add_argument("--rows", type=int, required=True, help="squares along Y")
    ap.add_argument("--square", type=float, required=True, help="square side (m)")
    ap.add_argument("--marker", type=float, required=True, help="marker side (m)")
    ap.add_argument("--dict", default="DICT_5X5_100")
    ap.add_argument("--every", type=int, default=10, help="use every Nth frame")
    ap.add_argument("--min-corners", type=int, default=12)
    ap.add_argument("--max-views", type=int, default=150)
    ap.add_argument("--legacy", action="store_true", help="board printed with OpenCV < 4.6")
    ap.add_argument("--out", default="calib.json")
    a = ap.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a.dict))
    board = cv2.aruco.CharucoBoard((a.cols, a.rows), a.square, a.marker, dictionary)
    if a.legacy:
        board.setLegacyPattern(True)
    detector = cv2.aruco.CharucoDetector(board)

    cap = cv2.VideoCapture(a.video)
    obj_pts, img_pts, size, idx = [], [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % a.every:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]  # (w, h)
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) < a.min_corners:
            continue
        o, i = board.matchImagePoints(ch_corners, ch_ids)
        # (1, N, 3)/(1, N, 2) float64 works on OpenCV 4.x AND 5.0; the (N, 1, 3) layout used in
        # most tutorials raises "Sizes of input arguments do not match" on 5.0.0.
        obj_pts.append(o.reshape(1, -1, 3).astype(np.float64))
        img_pts.append(i.reshape(1, -1, 2).astype(np.float64))
    if len(obj_pts) < 10:
        sys.exit(f"only {len(obj_pts)} usable views - record a longer / closer calibration video")
    if len(obj_pts) > a.max_views:  # subsample evenly, BA cost grows with views
        keep = np.linspace(0, len(obj_pts) - 1, a.max_views).astype(int)
        obj_pts = [obj_pts[k] for k in keep]
        img_pts = [img_pts[k] for k in keep]

    flags = F("CALIB_RECOMPUTE_EXTRINSIC") | F("CALIB_FIX_SKEW")
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-9)
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(obj_pts, img_pts, size, K, D, None, None, flags, crit)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, k3, k4 = D.ravel().tolist()
    # cx, cy below are in OpenCV convention (pixel centers at integers). COLMAP puts the origin at
    # the top-left pixel CORNER, so COLMAP cx/cy = OpenCV cx/cy + 0.5 (COLMAP FAQ).
    res = {
        "model": "OPENCV_FISHEYE", "width": size[0], "height": size[1],
        "fx": fx, "fy": fy, "cx": cx, "cy": cy, "k1": k1, "k2": k2, "k3": k3, "k4": k4,
        "rms_px": rms, "num_views": len(obj_pts),
        "colmap_camera_params": f"{fx},{fy},{cx + 0.5},{cy + 0.5},{k1},{k2},{k3},{k4}",
    }
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    if rms > 1.0:
        print("WARNING: RMS > 1 px - blurry views, rolling shutter or too few edge/corner views", file=sys.stderr)


if __name__ == "__main__":
    main()
