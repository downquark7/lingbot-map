"""hloc (ALIKED-n16 + LightGlue + MegaLoc) -> COLMAP database for `colmap global_mapper`.

Tested against: hloc master c13273b (2025-12-10), LightGlue eb42fee (2026-02-18),
pycolmap 4.2.1. Usage:
  python hloc_pose_stage.py --images frames/ --out work/ \
      --camera_model OPENCV_FISHEYE --camera_params "fx,fy,cx,cy,k1,k2,k3,k4"
Then:
  colmap global_mapper --database_path work/database.db --image_path frames/ \
      --output_path work/sparse
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import pycolmap

from hloc import extract_features, match_features
from hloc import reconstruction as R
from hloc import triangulation as T
from hloc.pairs_from_retrieval import get_descriptors, pairs_from_score_matrix

p = argparse.ArgumentParser()
p.add_argument("--images", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--camera_model", default="OPENCV_FISHEYE")
p.add_argument("--camera_params", default="")  # empty -> no focal prior
p.add_argument("--resize_max", type=int, default=1600)
p.add_argument("--max_kp", type=int, default=4096)
p.add_argument("--seq_overlap", type=int, default=10)  # i -> i+1..i+K
p.add_argument("--num_loop", type=int, default=20)  # retrieval pairs per frame
p.add_argument("--min_gap", type=int, default=30)  # frames; ignore near-in-time
p.add_argument("--skip_retrieval", action="store_true")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

names = sorted(x.name for x in a.images.iterdir() if x.is_file())  # temporal order

# 1) local features (ALIKED-n16; hloc default resize_max=1024, max_kp=-1)
fconf = copy.deepcopy(extract_features.confs["aliked-n16"])
fconf["preprocessing"]["resize_max"] = a.resize_max
fconf["model"]["max_num_keypoints"] = a.max_kp
feats = extract_features.main(
    fconf, a.images, a.out, image_list=names, feature_path=a.out / "feats.h5"
)

# 2) pairs = sequential window + MegaLoc loop candidates far apart in time
pairs = set()
for i in range(len(names)):
    for j in range(i + 1, min(i + 1 + a.seq_overlap, len(names))):
        pairs.add((names[i], names[j]))
if not a.skip_retrieval:
    gconf = extract_features.confs["megaloc"]  # torch.hub gmberton/MegaLoc + HF weights
    gdesc = extract_features.main(
        gconf, a.images, a.out, image_list=names, feature_path=a.out / "megaloc.h5"
    )
    d = get_descriptors(names, gdesc)  # (N, D) float tensor
    sim = d @ d.T
    idx = np.arange(len(names))
    invalid = np.abs(idx[:, None] - idx[None, :]) < a.min_gap
    for i, j in pairs_from_score_matrix(sim, invalid, a.num_loop, min_score=0):
        i, j = int(i), int(j)
        pairs.add((names[min(i, j)], names[max(i, j)]))
pairs_path = a.out / "pairs.txt"
pairs_path.write_text("\n".join(f"{x} {y}" for x, y in sorted(pairs)))
print(f"{len(pairs)} pairs")

# 3) LightGlue matching
matches = match_features.main(
    match_features.confs["aliked+lightglue"],
    pairs_path,
    features=feats,
    matches=a.out / "matches.h5",
)

# 4) COLMAP database with ONE shared camera (+ optional fixed intrinsics prior)
db = a.out / "database.db"
R.create_empty_db(db)
opts = {"camera_model": a.camera_model}
if a.camera_params:
    opts["camera_params"] = a.camera_params
R.import_images(a.images, db, pycolmap.CameraMode.SINGLE, names, opts)
ids = R.get_image_ids(db)
with pycolmap.Database.open(db) as d_:
    T.import_features(ids, d_, feats)
    T.import_matches(ids, d_, pairs_path, matches, None, False)
T.estimation_and_geometric_verification(db, pairs_path)
print("database ready:", db)
