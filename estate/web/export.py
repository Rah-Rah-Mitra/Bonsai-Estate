"""The web stage's work for one building: <ID>_walk.bin (SN5W, estate/web/walk.py) and <ID>_web.json.

Inputs are the building IFC and its engine JSON (the glb stage's <ID>_engine.json: transform, storeys, rooms and
the doors' leaves). <ID>_web.json, block-local, Z up, metres:

    {"schema": "sample-town-n5/web/1", "site": "BLK_509", "frame": "block-local, Z up, m",
     "walk": {file, cell, radius, step, band_pad, coarse, layers, ref, cells, overflow, stamped, leaves,
              leaf_blocked, leaf_narrowed, leaf_passthrough, leaf_overhead, doors_impassable, doors_unlinked,
              doors_cut},
     "stairs": [{name, room, storey, to, from_ffl, to_ffl,
                 flights: [{start, end, width, risers, riser, going}], landings: [{z, polygon}], path: [[x, y, z]]}],
     "doors": [{leaf_node, storey, motion, open, blocked, grid, angle (swing leaves)}]}

``doors`` lists every leaf of every passable door (lift landing doors are not passable and are absent): ``open`` is
the row-major 3 x 4 matrix (block-local, Z up) that takes the closed leaf to its opened pose
(nav_doorpose.open_matrix), ``angle`` a swing leaf's opening in degrees (90, or less where a full swing would cut
off part of a room: estate/web/walk.py block_leaves), ``blocked`` the plan of the opened leaf (undilated, a closed
ring; empty for a leaf that opens above head height), ``grid`` what the walk grid does with it: "blocked",
"passthrough" (left unblocked because blocking it would cut its doorway or a room off) or "overhead". Rings repeat
their first point, as the engine JSON's rooms do.
Coordinates are rounded to the millimetre, the matrices to 1e-6; nothing depends on a clock.
"""
from __future__ import annotations

import gc
import gzip
import json
import time
import traceback
from pathlib import Path

import numpy as np

SCHEMA = "sample-town-n5/web/1"
FRAME = "block-local, Z up, m"


def _ring(pts) -> list:
    pts = [[round(float(x), 3) + 0.0, round(float(y), 3) + 0.0] for x, y in pts]
    return pts + pts[:1] if pts else []


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def build(ifc, engine_json, out_dir, stem, cfg=None, log=None) -> dict:
    """Write <out_dir>/<stem>_walk.bin and <stem>_web.json for one building; returns their paths and the walk
    figures. Raises on failure."""
    from estate.validate import nav3d
    from estate.validate import nav_doorpose as doorpose
    from estate.web import stairs as stairs_mod
    from estate.web import walk
    say = log or (lambda *_: None)
    t0 = time.time()
    wc = walk.WalkConfig.of(cfg)
    out_dir = Path(out_dir)
    eng = json.loads(Path(engine_json).read_text(encoding="utf-8"))
    T = np.asarray(eng["transform"], float)
    if not np.allclose(T[:3, :3], np.eye(3), atol=1e-9):
        raise ValueError(f"{stem}: the walk grid needs a building placed by a translation, not {T[:3, :3].tolist()}")
    offset = T[:3, 3]
    storeys = sorted(eng["storeys"].items(), key=lambda kv: (kv[1], kv[0]))
    M = nav3d.read_model(ifc)
    stairs = stairs_mod.stairs(M.file, M.meshes, T, eng["storeys"], eng.get("rooms", []))
    ground = float(next(iter(M.storeys.values())) if M.storeys else 0.0)
    blo, bhi = M.bounds
    lo = np.array([blo[0] - wc.margin, blo[1] - wc.margin, min(blo[2], ground) - 1.0])
    hi = np.array([bhi[0] + wc.margin, bhi[1] + wc.margin, bhi[2] + wc.height + 0.5])
    g = nav3d.Grid.around(lo, hi, wc.cell)
    obst = [M.meshes[gid]["verts"][M.meshes[gid]["faces"]] for gid in M.obstacles] + \
        [nav3d.jamb_tris(d) for d in M.doors]
    tris = np.concatenate(obst) if obst else np.zeros((0, 3, 3))
    elem = np.repeat(np.arange(len(obst)), [len(x) for x in obst])
    del obst
    M.meshes = M.file = None
    gc.collect()
    w = walk.solve(tris, elem, g, ground, M.doors, wc, log=say)
    del tris, elem
    gc.collect()
    leaves = walk.leaf_records(eng, offset, wc.height)
    door_index = {d["guid"]: i for i, d in enumerate(M.doors)}
    for lf in leaves:
        lf["door"] = door_index.get(lf["guid"], -1)
    blocked = walk.block_leaves(w, M.doors, leaves, wc)
    keep = walk.reachable(w, blocked)
    nx, ny, origin, lays = walk.layers(w, keep, storeys, offset, wc.band_pad)
    data, ref, coarse = walk.encode(nx, ny, origin, lays, wc)
    walk_path, web_path = out_dir / f"{stem}_walk.bin", out_dir / f"{stem}_web.json"
    st = w.stats
    info = dict(file=walk_path.name, cell=round(wc.cell * (2 if coarse else 1), 3), radius=wc.radius, step=wc.step,
                band_pad=wc.band_pad, coarse=coarse, layers=len(lays), ref=lays[ref]["tag"], cells=int(keep.sum()),
                overflow=sum(len(lay["overflow"]) for lay in lays), stamped=int(w.stamped[keep].sum()),
                **{k: int(st[k]) for k in ("leaves", "leaf_blocked", "leaf_narrowed", "leaf_passthrough",
                                           "leaf_overhead", "doors_impassable", "doors_unlinked", "doors_cut")})
    doors = []
    for lf in leaves:
        rec = lf["rec"]
        O = doorpose.open_matrix(rec, lf["angle"])
        entry = dict(leaf_node=rec["node"], storey=lf["storey"], motion=rec["motion"],
                     open=[round(float(v), 6) + 0.0 for v in O[:3, :4].reshape(-1)],
                     blocked=_ring(walk.leaf_polygon(lf, lf["angle"])[1]) if lf["obstructs"] else [], grid=lf["grid"])
        if rec["motion"] == "swing":
            entry["angle"] = float(lf["angle"] if lf["angle"] is not None else rec.get("max_angle_deg", 90.0))
        doors.append(entry)
    doc = dict(schema=SCHEMA, site=stem, frame=FRAME, walk=info, stairs=stairs, doors=doors)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(walk_path, data)
    _write(web_path, (json.dumps(doc, indent=1) + "\n").encode("utf-8"))
    say(f"  {stem}: {nx} x {ny} cells, {len(lays)} layers (ref {lays[ref]['tag']}), {len(data):,} B "
        f"({len(gzip.compress(data, 9, mtime=0)):,} B gzipped){', coarse' if coarse else ''}")
    return dict(stem=stem, ok=True, files=[str(walk_path), str(web_path)], walk=info, bytes=len(data),
                gz_bytes=len(gzip.compress(data, 9, mtime=0)), stairs=len(stairs), seconds=round(time.time() - t0, 1))


def web_one(task) -> dict:
    """Pool worker: build() that never raises (a failure comes back as ok False with the error)."""
    ifc, engine_json, out_dir, stem, cfg = task
    t0 = time.time()
    try:
        return build(ifc, engine_json, out_dir, stem, cfg)
    except Exception as e:  # noqa: BLE001
        return dict(stem=stem, ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:],
                    seconds=round(time.time() - t0, 1))


def line(r) -> str:
    if not r.get("ok"):
        return f"FAILED {r.get('error')}"
    w = r["walk"]
    return (f"{w['layers']} layers (ref {w['ref']}), {w['cells']:,} cells, {w['overflow']} overflow, {w['stamped']} "
            f"stamped, leaves {w['leaf_blocked']} blocked / {w['leaf_passthrough']} passthrough / "
            f"{w['leaf_overhead']} overhead, {r['stairs']} stairs, {r['gz_bytes'] / 1024:.0f} KB gzipped"
            f"{' (coarse)' if w['coarse'] else ''}, {r['seconds']} s")
