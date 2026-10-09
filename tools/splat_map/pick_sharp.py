#!/usr/bin/env python3
"""Pick the sharpest frame in every window of N consecutive video frames.

One decode pass (ffmpeg -> raw BGR pipe), full-resolution output, HLG/PQ tone-mapping
when the clip is HDR. Writes frames + a CSV with every frame's sharpness score.

usage: pick_sharp.py VIDEO OUTDIR [--window 10] [--score-width 960] [--ext jpg|png]
"""
import argparse, csv, json, os, subprocess, sys
import numpy as np
import cv2


def probe(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate,color_transfer,nb_frames:"
         "stream_side_data=rotation", "-of", "json", video],
        check=True, capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    w, h = int(s["width"]), int(s["height"])
    if abs(rot) % 180 == 90:  # ffmpeg auto-rotates on decode
        w, h = h, w
    return w, h, s.get("color_transfer", ""), s


HDR_TO_SDR = {  # verified chain (same as reflct/sharp-frames 0.4.0), needs zscale (libzimg)
    "arib-std-b67": "zscale=tin=arib-std-b67:min=bt2020nc:pin=bt2020:t=linear:npl=100,"
                    "format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
                    "zscale=t=bt709:m=bt709:r=tv",
    "smpte2084": "zscale=tin=smpte2084:min=bt2020nc:pin=bt2020:t=linear:npl=100,"
                 "format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
                 "zscale=t=bt709:m=bt709:r=tv",
}


def sharpness(bgr, score_width):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if score_width and g.shape[1] > score_width:
        s = score_width / g.shape[1]
        g = cv2.resize(g, (score_width, int(round(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (3, 3), 0)  # suppress sensor/compression noise
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("outdir")
    ap.add_argument("--window", type=int, default=10, help="frames per window (30fps/10 = 3 fps out)")
    ap.add_argument("--score-width", type=int, default=960)
    ap.add_argument("--ext", choices=["jpg", "png"], default="jpg")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    w, h, trc, s = probe(a.video)
    vf = HDR_TO_SDR.get(trc)
    cmd = ["ffmpeg", "-v", "error", "-i", a.video]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    print(f"{w}x{h} transfer={trc or 'unknown'} fps={s.get('avg_frame_rate')} -> {' '.join(cmd)}", file=sys.stderr)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    fsize = w * h * 3
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if a.ext == "jpg" else []
    rows, best, n, kept = [], None, 0, 0
    with open(os.path.join(a.outdir, "scores.csv"), "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["frame", "score", "selected"])
        while True:
            buf = p.stdout.read(fsize)
            if len(buf) < fsize:
                break
            img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            sc = sharpness(img, a.score_width)
            rows.append([n, sc, 0])
            if best is None or sc > best[1]:
                best = (n, sc, img.copy())
            if (n + 1) % a.window == 0:
                cv2.imwrite(os.path.join(a.outdir, f"frame_{best[0]:07d}.{a.ext}"), best[2], params)
                rows[best[0]][2] = 1; kept += 1; best = None
            n += 1
        if best is not None:
            cv2.imwrite(os.path.join(a.outdir, f"frame_{best[0]:07d}.{a.ext}"), best[2], params)
            rows[best[0]][2] = 1; kept += 1
        wr.writerows(rows)
    p.wait()
    print(f"decoded {n} frames, kept {kept}", file=sys.stderr)


if __name__ == "__main__":
    main()
