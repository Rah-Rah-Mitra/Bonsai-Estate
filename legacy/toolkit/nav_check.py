#!/usr/bin/env python3
"""
nav_check.py - can a person-sized agent walk from the street to every room and onto the roof?

Method (a small Recast-style test on the exported geometry, not on the generator's intentions):
  1. Tessellate the IFC (IfcOpenShell) and voxelise every solid surface at 0.1 m.
     Door leaves of passable doors are ignored (they open); lift landing doors stay closed.
  2. A voxel is walkable when there is solid floor under it, 0.4 m of free space for the feet
     (so steps up to 0.4 m are fine) and a free cylinder of the agent's radius from 0.4 m to 1.8 m.
  3. Walkable voxels connect to horizontal neighbours whose floor is within 0.4 m (one step).
  4. Every IfcSpace and the roof is checked for a walkable voxel connected to the street.

Usage: python nav_check.py hdb_block.ifc --radius 0.3 0.4
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from scipy import ndimage
from shapely.ops import unary_union
import ifcopenshell.util.element as uel

from ifc_mesh import load

P = 0.1                      # voxel size (m)
HEIGHT, STEP = 1.8, 0.4      # agent height, max step


def voxelise(tris, lo, shape):
    S = np.zeros(shape, dtype=bool)
    e = np.maximum(np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1),
                   np.maximum(np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1), np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)))
    n = np.maximum(1, np.ceil(e / (P * 0.45))).astype(int)
    for nn in np.unique(n):
        grp = tris[n == nn]
        i, j = np.meshgrid(np.arange(nn + 1), np.arange(nn + 1), indexing="ij")
        keep = (i + j) <= nn
        uv = np.stack([i[keep], j[keep]], 1) / nn                      # (k, 2) barycentric grid
        chunk = max(1, int(4e6 // len(uv)))
        for s in range(0, len(grp), chunk):
            g = grp[s:s + chunk]
            pts = g[:, None, 0] + uv[None, :, :1] * (g[:, None, 1] - g[:, None, 0]) + uv[None, :, 1:] * (g[:, None, 2] - g[:, None, 0])
            idx = np.floor((pts.reshape(-1, 3) - lo) / P).astype(np.int64)
            ok = np.all((idx >= 0) & (idx < shape), axis=1)
            idx = idx[ok]
            S[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return S


def walkable(S, r_vox):
    H, St = int(round(HEIGHT / P)), int(round(STEP / P))
    E = ~S
    yy, xx = np.mgrid[-r_vox:r_vox + 1, -r_vox:r_vox + 1]
    disk = (xx ** 2 + yy ** 2) <= r_vox ** 2
    Eb = ndimage.binary_erosion(E, structure=disk[:, :, None], border_value=1)
    nz = S.shape[2]
    cE = np.concatenate([np.zeros(S.shape[:2] + (1,), np.uint16), np.cumsum(E, axis=2, dtype=np.uint16)], axis=2)
    cB = np.concatenate([np.zeros(S.shape[:2] + (1,), np.uint16), np.cumsum(Eb, axis=2, dtype=np.uint16)], axis=2)
    W = np.zeros_like(S)
    k = np.arange(1, nz - H)
    feet = (cE[:, :, k + St] - cE[:, :, k]) == St
    body = (cB[:, :, k + H] - cB[:, :, k + St]) == (H - St)
    W[:, :, k] = S[:, :, k - 1] & feet & body
    D = W.copy()
    for j in range(1, St + 1):           # vertical thickening = "step up to 0.4 m" connectivity
        D[:, :, j:] |= W[:, :, :-j]
    lab, nlab = ndimage.label(D)
    return W, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ifc", nargs="?", default="hdb_block.ifc")
    ap.add_argument("--radius", type=float, nargs="+", default=[0.3, 0.4])
    ap.add_argument("--start", type=float, nargs=2, default=[5.5, -12.0])
    a = ap.parse_args()
    f, meshes = load(a.ifc)
    doors_open = {d.GlobalId for d in f.by_type("IfcDoor")
                  if uel.get_psets(d).get("Navigation", {}).get("Passable", True)}
    obstacles = [d for g, d in meshes.items() if d["cls"] != "IfcSpace" and g not in doors_open]
    tris = np.concatenate([d["verts"][d["faces"]] for d in obstacles])

    lo = np.array([-18.0, -14.0, -1.0]) - P / 2
    hi = np.array([18.0, 14.0, 40.0])
    shape = tuple(np.ceil((hi - lo) / P).astype(int))
    S = voxelise(tris, lo, shape)
    print(f"voxel grid {shape}  ({np.prod(shape) / 1e6:.0f} M cells), solid {S.mean() * 100:.1f}%")

    storeys = {s.Name: s.Elevation for s in f.by_type("IfcBuildingStorey")}
    spaces = []
    for sp in f.by_type("IfcSpace"):
        st = uel.get_aggregate(sp)
        d = meshes[sp.GlobalId]
        poly = unary_union([shapely.Polygon(t[:, :2]) for t in d["verts"][d["faces"]] if shapely.Polygon(t[:, :2]).area > 1e-8])
        spaces.append((sp, st.Name, storeys[st.Name], poly))
    roof = [s for s in f.by_type("IfcSlab") if s.Name == "Roof slab"][0]
    rd = meshes[roof.GlobalId]
    roof_poly = unary_union([shapely.Polygon(t[:, :2]) for t in rd["verts"][rd["faces"]] if shapely.Polygon(t[:, :2]).area > 1e-8])
    spaces.append((roof, "RF", storeys["RF"], roof_poly))

    report = {}
    for radius in a.radius:
        r_vox = int(round(radius / P))
        W, lab = walkable(S, r_vox)
        i0 = np.floor((np.array([*a.start, 0.0]) - lo) / P).astype(int)
        col = lab[i0[0], i0[1], :]
        start_labels = set(col[col > 0].tolist())
        start = max(start_labels, key=lambda L: (lab == L).sum()) if start_labels else 0
        res = []
        xs = lo[0] + (np.arange(shape[0]) + 0.5) * P
        ys = lo[1] + (np.arange(shape[1]) + 0.5) * P
        for sp, sname, z, poly in spaces:
            k0 = int(np.floor((z - lo[2]) / P))
            band = lab[:, :, max(0, k0 - 1):k0 + 3]
            minx, miny, maxx, maxy = poly.bounds
            ix = np.where((xs > minx) & (xs < maxx))[0]
            iy = np.where((ys > miny) & (ys < maxy))[0]
            sub = band[np.ix_(ix, iy)]
            gx, gy = np.meshgrid(xs[ix], ys[iy], indexing="ij")
            inside = shapely.contains_xy(poly.buffer(-0.05), gx, gy)
            labels = sub[inside[:, :, None].repeat(sub.shape[2], 2)]
            walk = (labels > 0).sum()
            ok = bool((labels == start).any())
            res.append(dict(name=sp.Name, long_name=getattr(sp, "LongName", None) or sp.Name, storey=sname,
                            reachable=ok, walkable_cells=int(walk)))
        n_ok = sum(r["reachable"] for r in res)
        print(f"\nagent radius {radius:.2f} m: {n_ok}/{len(res)} spaces reachable from the street")
        bad = [r for r in res if not r["reachable"]]
        by_room = {}
        for r in bad:
            by_room.setdefault(r["long_name"], []).append(r["name"])
        for k, v in sorted(by_room.items()):
            print(f"   unreachable: {k:<28} x{len(v)}  e.g. {v[0]}")
        report[f"{radius:.2f}"] = res
        # visual for one typical floor
        for sname in ("L5",):
            z = storeys[sname]
            k0 = int(np.floor((z - lo[2]) / P))
            band = lab[:, :, k0 - 1:k0 + 3]
            reach = (band == start).any(axis=2)
            other = ((band > 0) & (band != start)).any(axis=2)
            wall = S[:, :, int(np.floor((z + 1.2 - lo[2]) / P))]
            img = np.ones(shape[:2] + (3,))
            img[wall] = (0.25, 0.25, 0.25)
            img[reach] = (0.45, 0.78, 0.45)
            img[other] = (0.90, 0.40, 0.35)
            fig, ax = plt.subplots(figsize=(13, 10), dpi=100)
            ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=(lo[0], hi[0], lo[1], hi[1]))
            ax.set_title(f"{sname}: walkable floor reachable from the street (green), walkable but cut off (red); "
                         f"agent r={radius:.2f} m, h={HEIGHT} m, step {STEP} m", fontsize=10)
            ax.set_aspect("equal")
            ax.set_xlim(-15.5, 15.5)
            ax.set_ylim(-11.5, 11.5)
            fig.tight_layout()
            fig.savefig(f"nav_{sname}_r{int(radius * 100)}.png", facecolor="white")
            plt.close(fig)
    with open("nav_report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    print("\nwrote nav_report.json and nav_L5_r*.png")


if __name__ == "__main__":
    main()
