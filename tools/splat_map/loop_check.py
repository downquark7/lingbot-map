# List verified image pairs that are far apart in time (= loop closures).
# usage: python loop_check.py database.db 300 50
import sqlite3, sys
db, min_gap, min_inl = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
con = sqlite3.connect(db)
names = dict(con.execute("SELECT image_id, name FROM images"))
order = {iid: k for k, iid in enumerate(sorted(names, key=lambda i: names[i]))}
loops = []
for pair_id, rows in con.execute("SELECT pair_id, rows FROM two_view_geometries"):
    i2 = pair_id % 2147483647
    i1 = (pair_id - i2) // 2147483647
    gap = abs(order[i1] - order[i2])
    if gap >= min_gap and rows >= min_inl:
        loops.append((order[i1], order[i2], rows, names[i1], names[i2]))
loops.sort()
print(f"{len(loops)} loop edges (gap >= {min_gap} frames, >= {min_inl} inliers)")
for a, b, n, na, nb in loops:
    print(f"{a:6d} <-> {b:6d}  inliers={n:5d}  {na}  {nb}")
