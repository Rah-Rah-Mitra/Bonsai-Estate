"""Walk grids for a browser viewer: per-storey floor heights of one building at r = 0.20 m (numpy, shapely).

The grid is nav3d's own walk test (estate/validate/nav3d.py) run once more for a smaller agent, with nothing in
nav3d changed: the same voxeliser over the same solid (every element but the leaves of passable doors, whose
linings stay), the same walkability test (solid below, free feet zone, free disc of the radius up to the agent
height), the same spans and the same fine door-band test. Then, for the viewer:

1. Doorways stamped walkable. Every door the band test passes has walkable cells on both sides
   (nav3d._door_sides); the straight line of cells between the side cells nearest the opening's centre is made
   walkable at the floor height interpolated between them, wherever the 0.1 m grid had no floor there (narrow
   openings fall between its cells). ``stamped`` counts those cells.
2. Opened leaves blocked. A viewer bakes every passable leaf open (nav_doorpose.open_matrix, the engine JSON's
   leaves); the cells whose centre lies within the radius of an opened leaf's plan (nav_doorpose.opened_footprint)
   at its door's level are blocked, so a camera cannot walk through it. A leaf whose blocking would cut its door's
   two sides apart within DOOR_REACH of the opening is left unblocked (``leaf_passthrough``), and so is a leaf
   that, with the others, cuts off an area of POCKET_M2 or more the street reached before (a shelter door in the
   sweep of the flat's main door: the leaf whose release reconnects it). A leaf that opens above head height (a
   rolled-up shutter) blocks nothing (``leaf_overhead``).
3. Only cells reachable from the street are kept: components of the grid (neighbours whose floors differ by at
   most a step) that hold a start cell on the grid's margin at ground level. Table tops, sills, rail tops and
   closed shafts drop out.
4. Storey bands. Storey s owns the floors from FFL_s - band_pad up to FFL_{s+1} - band_pad (the lowest storey
   everything below, the top one everything above). In each cell the floor closest to FFL_s is the layer's main
   floor and any other floor in the band an overflow record (a stair flight above a landing).
5. Block-local, int16 mm above the storey's FFL, written as SN5W (estate/web/sn5w.py). The reference layer is the
   typical storey (not the lowest, not the top) with the most walkable cells; every other layer is raw, delta or
   the same as it. If the gzipped file would exceed ``max_walk_gz`` the grid falls back to 0.2 m cells, each
   walkable only where all four 0.1 m cells under it are.

Building placements must be pure translations (every estate site is); the grid is built in the estate frame
like nav3d's and moved to block-local by the translation.
"""
from __future__ import annotations

import gc
import gzip
import math
from dataclasses import dataclass, field

import numpy as np
import shapely

from estate.validate import nav3d
from estate.validate import nav_doorpose as doorpose
from estate.web import sn5w

DOOR_REACH = 1.5            # m around a door's opening within which its two sides must stay connected
LEAF_LEVEL = 0.35           # m: cells whose floor is this close to the door's sill are the ones a leaf can block
DOOR_LEVEL = 1.0            # m: the vertical window of a door's local connectivity check
POCKET_M2 = 0.25            # m2: a cut-off area this large is a place to walk (a cupboard-sized shelter), not a corner
                            # behind an opened leaf: no leaf may cut it off
NARROW = (75.0, 60.0, 45.0)  # degrees a swing leaf may open instead of its 90 when that keeps an area reachable


@dataclass
class WalkConfig:
    radius: float = 0.20
    cell: float = 0.10
    band_pad: float = 0.25
    max_walk_gz: int = 131072
    height: float = 1.80
    step: float = 0.40
    margin: float = 2.5
    level_tol: float = 0.015
    door_voxel: float = nav3d.DOOR_VOXEL

    @classmethod
    def of(cls, cfg: dict | None) -> "WalkConfig":
        """From config/estate.toml: [web] radius, cell, band_pad, max_walk_gz; [agent] height and step."""
        cfg = cfg or {}
        w, a = cfg.get("web", {}), cfg.get("agent", {})
        return cls(radius=float(w.get("radius", 0.20)), cell=float(w.get("cell", 0.10)),
                   band_pad=float(w.get("band_pad", 0.25)), max_walk_gz=int(w.get("max_walk_gz", 131072)),
                   height=float(a.get("height", 1.80)), step=float(a.get("step", 0.40)))


# ----------------------------------------------------------------------------- grid graph
def grid_edges(ix, iy, k, g: nav3d.Grid, St: int):
    """Edges (index pairs into the given arrays) between 4-neighbouring walkable spans whose floor cells differ by
    at most St voxels: nav3d.connect's rule, for any set of spans."""
    ix, iy, k = (np.asarray(v, np.int64) for v in (ix, iy, k))
    key = (ix * g.ny + iy) * g.nz + k
    order = np.argsort(key, kind="stable")
    skey = key[order]
    n = len(key)
    ea, eb = [], []
    if not n:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    for dx, dy in ((1, 0), (0, 1)):
        jx, jy = ix + dx, iy + dy
        ok = (jx < g.nx) & (jy < g.ny)
        base = (jx * g.ny + jy) * g.nz
        pos = np.searchsorted(skey, base + k - St)
        pc = np.minimum(pos, n - 1)
        hit = ok & (pos < n) & (skey[pc] <= base + k + St)
        ea.append(np.nonzero(hit)[0])
        eb.append(order[pc[hit]])
    return np.concatenate(ea), np.concatenate(eb)


class ColumnIndex:
    """Cells sorted by column (ix, iy): the cells of a plan box are a few contiguous runs."""

    def __init__(self, ix, iy, ny):
        key = np.asarray(ix, np.int64) * ny + iy
        self.order = np.argsort(key, kind="stable")
        self.key = key[self.order]
        self.ny = ny

    def box(self, i0, i1, j0, j1) -> np.ndarray:
        rows = np.arange(i0, i1 + 1, dtype=np.int64) * self.ny
        lo = np.searchsorted(self.key, rows + j0)
        hi = np.searchsorted(self.key, rows + j1, side="right")
        idx, _ = nav3d._expand(lo, np.maximum(hi - lo, 0))
        return self.order[idx]

    def column(self, ix, iy) -> np.ndarray:
        q = int(ix) * self.ny + int(iy)
        return self.order[np.searchsorted(self.key, q):np.searchsorted(self.key, q, side="right")]


@dataclass
class Walk:
    """Walkable spans of one building (estate frame), stamped doorway cells appended last."""
    g: nav3d.Grid
    ix: np.ndarray
    iy: np.ndarray
    k: np.ndarray
    fz: np.ndarray
    stamped: np.ndarray                       # bool per span
    start: np.ndarray                         # span indices on the street ring
    sides: list                               # per door: (a, b) span indices, or None (impassable / no cells)
    St: int
    doors_impassable: int = 0
    stats: dict = field(default_factory=dict)

    def cx(self, idx=slice(None)):
        return self.g.cx(self.ix[idx])

    def cy(self, idx=slice(None)):
        return self.g.cy(self.iy[idx])


def _line_cells(g: nav3d.Grid, p, q):
    """4-connected cells (ix, iy) of the segment p -> q (estate xy), in order, with the parameter t of each."""
    n = max(2, int(math.ceil(float(np.hypot(*(np.asarray(q) - p))) / (g.P / 4))) + 1)
    t = np.linspace(0.0, 1.0, n)
    xy = np.asarray(p)[None] + t[:, None] * (np.asarray(q) - np.asarray(p))[None]
    ci = np.floor((xy[:, 0] - g.lo[0]) / g.P).astype(np.int64)
    cj = np.floor((xy[:, 1] - g.lo[1]) / g.P).astype(np.int64)
    out, tt = [(int(ci[0]), int(cj[0]))], [0.0]
    for a, b, s in zip(ci[1:], cj[1:], t[1:]):
        a, b = int(a), int(b)
        if (a, b) == out[-1]:
            continue
        if a != out[-1][0] and b != out[-1][1]:          # a diagonal step: go round one corner
            out.append((a, out[-1][1]))
            tt.append(float(s))
        out.append((a, b))
        tt.append(float(s))
    return out, tt


def solve(T, elem, g: nav3d.Grid, ground: float, doors: list, wc: WalkConfig, log=None) -> Walk:
    """Walkable spans at wc.radius over the obstacle triangles T (elements ``elem``), nav3d's door records
    ``doors`` (passable doors with their linings already among T), with their doorways stamped."""
    say = log or (lambda *_: None)
    P = g.P
    H, St = int(round(wc.height / P)), int(round(wc.step / P))
    S, up, _ = nav3d.voxelise(T, elem, g, ground)
    band = []
    if wc.door_voxel:
        tris, cache = nav3d.Tris.of(T, elem), {}
        for d in doors:
            res, _ = nav3d.door_band(d, tris, [wc.radius], wc.height, wc.step, wc.level_tol, wc.door_voxel, cache,
                                     ground)
            band.append(res[f"{wc.radius:.2f}"])
        del tris, cache
    k, ix, iy = nav3d.walkable_cells(S, g, nav3d.voxel_radius(wc.radius, P), H, St)
    del S
    gc.collect()
    cells = nav3d.connect(k, ix, iy, g, St, up)
    del k, ix, iy, up
    index = nav3d.CellIndex.of(cells, g)
    cols = ColumnIndex(cells.ix, cells.iy, g.ny)
    sides, impassable = [], 0
    new, stamped_k = {}, {}                               # (ix, iy, k) -> fz of stamped cells; (ix, iy) -> [k]
    for i, d in enumerate(doors):
        b = band[i] if band else dict(passable=True, zA=0.0, zB=0.0)
        s = (nav3d._door_sides(cells, g, d, d["z"] + b["zA"], d["z"] + b["zB"], 0.05, index)
             if b["passable"] else None)
        sides.append(s)
        if s is None:
            impassable += 1
            continue
        # the side cells nearest the middle of the opening, and the line of cells between them
        Mw = d["M"]
        ends = []
        for side in s:
            loc = (np.stack([g.cx(cells.ix[side]), g.cy(cells.iy[side])], 1) - Mw[:2, 3]) @ Mw[:2, :2]
            dist = (loc[:, 0] - d["w"] / 2) ** 2 + (loc[:, 1] - d["yc"]) ** 2
            ends.append(int(side[np.argmin(dist)]))
        pa, pb = ends
        line, tt = _line_cells(g, (g.cx(cells.ix[pa]), g.cy(cells.iy[pa])), (g.cx(cells.ix[pb]), g.cy(cells.iy[pb])))
        for (ci, cj), t in zip(line[1:-1], tt[1:-1]):
            z = float(cells.fz[pa] + t * (cells.fz[pb] - cells.fz[pa]))
            kk = g.kz(z)
            have = cols.column(ci, cj)
            if len(have) and np.any(np.abs(cells.k[have].astype(np.int64) - kk) <= St):
                continue
            if any(abs(k0 - kk) <= St for k0 in stamped_k.get((ci, cj), ())):    # stamped by an earlier door
                continue
            new[(ci, cj, kk)] = z
            stamped_k.setdefault((ci, cj), []).append(kk)
    stamped = sorted(new)
    sx = np.array([s[0] for s in stamped], np.int32)
    sy = np.array([s[1] for s in stamped], np.int32)
    sk = np.array([s[2] for s in stamped], np.int32)
    sz = np.array([new[s] for s in stamped], float)
    ring = max(1, int(round(0.5 / P)))
    ix = np.concatenate([cells.ix, sx]).astype(np.int32)
    iy = np.concatenate([cells.iy, sy]).astype(np.int32)
    k = np.concatenate([cells.k, sk]).astype(np.int32)
    fz = np.concatenate([cells.fz, sz])
    edge = (ix < ring) | (ix >= g.nx - ring) | (iy < ring) | (iy >= g.ny - ring)
    start = np.nonzero(edge & (np.abs(fz - ground) <= 0.6))[0]
    say(f"  walk: {len(cells.key)} walkable cells at r={wc.radius}, {len(stamped)} stamped in "
        f"{len(doors) - impassable} doorways, {impassable} doors impassable")
    return Walk(g, ix, iy, k, fz, np.concatenate([np.zeros(len(cells.key), bool), np.ones(len(stamped), bool)]),
                start, sides, St, impassable)


# ----------------------------------------------------------------------------- opened leaves
def _local(w: Walk, cols: ColumnIndex, x, y, z, reach, level):
    """Spans within ``reach`` (plan) of (x, y) whose floor is within ``level`` of z."""
    g = w.g
    idx = cols.box(max(0, g.ix(x - reach)), min(g.nx - 1, g.ix(x + reach)),
                   max(0, g.iy(y - reach)), min(g.ny - 1, g.iy(y + reach)))
    m = (np.hypot(w.cx(idx) - x, w.cy(idx) - y) <= reach) & (np.abs(w.fz[idx] - z) <= level)
    return idx[m]


def connected(w: Walk, region, a, b, blocked) -> bool:
    """Do some unblocked cells of a and b fall into one component of the unblocked cells of ``region``?"""
    region = np.union1d(region, np.concatenate([a, b]))
    alive = region[~blocked[region]]
    a, b = a[~blocked[a]], b[~blocked[b]]
    if not len(a) or not len(b):
        return False
    ea, eb = grid_edges(w.ix[alive], w.iy[alive], w.k[alive], w.g, w.St)
    lab = nav3d.components(len(alive), ea, eb)
    pos = {int(c): i for i, c in enumerate(alive)}
    la = {int(lab[pos[int(c)]]) for c in a}
    return any(int(lab[pos[int(c)]]) in la for c in b)


def leaf_polygon(lf: dict, angle=None):
    """(estate-frame plan polygon, block-local ring) of a leaf opened to ``angle`` degrees (None: as recorded)."""
    ring = doorpose.opened_footprint(lf["rec"], lf["along"], lf["across"], angle)
    ox, oy = lf["offset"][0], lf["offset"][1]
    return shapely.Polygon([(x + ox, y + oy) for x, y in ring]), ring


def _leaf_cells(w: Walk, cols: ColumnIndex, lf: dict, angle, wc: WalkConfig) -> np.ndarray:
    """Spans within the radius of the opened leaf's plan, at its door's level (none for a leaf above head height)."""
    if not lf["obstructs"]:
        return np.zeros(0, np.int64)
    poly = leaf_polygon(lf, angle)[0].buffer(wc.radius)
    x0, y0, x1, y1 = poly.bounds
    g = w.g
    idx = cols.box(max(0, g.ix(x0)), min(g.nx - 1, g.ix(x1)), max(0, g.iy(y0)), min(g.ny - 1, g.iy(y1)))
    idx = idx[np.abs(w.fz[idx] - lf["z"]) <= LEAF_LEVEL]
    return idx[shapely.contains_xy(poly, w.cx(idx), w.cy(idx))]


def block_leaves(w: Walk, doors: list, leaves: list, wc: WalkConfig) -> np.ndarray:
    """Blocked spans: the opened leaves (``leaves``: leaf_records with ``door``, the index of their nav3d door or
    -1), each dilated by the radius.

    1. A leaf whose blocking cuts its door's sides apart within DOOR_REACH goes through (``passthrough``); then any
       leaf blocking near a door that is still cut.
    2. Two leaves can close a room off together while each door still connects on its own (a household shelter door
       in the sweep of the flat's main door), and a leaf swung across a shallow room cuts off the part behind it:
       every area of POCKET_M2 or more the street reached without the leaves must still be reached. A swing leaf
       touching it opens less (NARROW, largest first) if that joins it again and keeps the leaf's own door
       connected; else a touching leaf whose release alone joins it goes through, else every touching leaf does.

    Sets each leaf's ``grid`` ('blocked', 'passthrough', 'overhead') and ``angle`` (None, or the narrowed swing in
    degrees) and w.stats."""
    cols = ColumnIndex(w.ix, w.iy, w.g.ny)
    n = len(w.ix)
    for lf in leaves:
        lf["grid"], lf["angle"] = ("blocked" if lf["obstructs"] else "overhead"), None
    cells_of = [_leaf_cells(w, cols, lf, None, wc) for lf in leaves]
    active = np.array([lf["grid"] == "blocked" for lf in leaves], bool)

    def cover():
        c = np.zeros(n, np.int32)
        for i in np.nonzero(active)[0]:
            np.add.at(c, cells_of[i], 1)
        return c

    by_door = {}
    for i, lf in enumerate(leaves):
        by_door.setdefault(lf["door"], []).append(i)
    regions, base_ok = {}, {}
    none = np.zeros(n, bool)
    for di, s in enumerate(w.sides):
        if s is None:
            continue
        d = doors[di]
        c = d["M"] @ np.array([d["w"] / 2, d["yc"], 0.0, 1.0])
        regions[di] = _local(w, cols, c[0], c[1], d["z"], DOOR_REACH, DOOR_LEVEL)
        base_ok[di] = connected(w, regions[di], s[0], s[1], none)

    def release(ids):
        for i in ids:
            active[i] = False
            leaves[i]["grid"] = "passthrough"

    blocked = cover() > 0
    for widen in (False, True):               # first a door's own leaves, then any leaf blocking near it
        changed = True
        while changed:
            changed = False
            for di in sorted(regions):
                if not base_ok[di] or connected(w, regions[di], *w.sides[di], blocked):
                    continue
                if widen:
                    near = set(regions[di].tolist())
                    culprits = [i for i in np.nonzero(active)[0] if near.intersection(cells_of[i].tolist())]
                else:
                    culprits = [i for i in by_door.get(di, []) if active[i]]
                release(culprits)
                if culprits:
                    changed = True
                    blocked = cover() > 0
    before = reachable(w, none)
    pocket = max(1, int(round(POCKET_M2 / w.g.P ** 2)))
    for _ in range(64):
        count = cover()
        blocked = count > 0
        after = reachable(w, blocked)
        lost = np.nonzero(before & ~after & ~blocked)[0]
        if not len(lost):
            break
        ea, eb = grid_edges(w.ix[lost], w.iy[lost], w.k[lost], w.g, w.St)
        lab = nav3d.components(len(lost), ea, eb)
        sizes = np.bincount(lab, minlength=len(lost))
        fixes = [_repair(w, cols, lost[lab == root], after, count, leaves, cells_of, active, regions, wc)
                 for root in np.unique(lab[sizes[lab] >= pocket])]
        if not any(ids for _, ids, _, _ in fixes):
            break
        for kind, ids, angle, cells in fixes:
            if kind == "narrow" and active[ids[0]]:
                leaves[ids[0]]["angle"], cells_of[ids[0]] = angle, cells
            else:
                release([i for i in ids if active[i]])
    blocked = cover() > 0
    w.stats.update(leaves=len(leaves), leaf_blocked=int(active.sum()),
                   leaf_narrowed=sum(1 for i, lf in enumerate(leaves) if active[i] and lf["angle"] is not None),
                   leaf_passthrough=sum(lf["grid"] == "passthrough" for lf in leaves),
                   leaf_overhead=sum(lf["grid"] == "overhead" for lf in leaves),
                   doors_unlinked=sum(1 for di in regions if not base_ok[di]),
                   doors_cut=sum(1 for di in regions if base_ok[di]
                                 and not connected(w, regions[di], *w.sides[di], blocked)),
                   doors_impassable=w.doors_impassable)
    return blocked


def _repair(w: Walk, cols: ColumnIndex, comp, after, count, leaves, cells_of, active, regions, wc):
    """How to join the cut-off spans ``comp`` to the reached ones again: ("narrow", [leaf], angle, its cells) for the
    first touching swing leaf (fewest blocked cells, then list order) that a smaller opening (NARROW) lets through
    while its own door stays connected; else ("release", [leaf], ...) for the first touching leaf whose release alone
    does; else ("release", every touching leaf)."""
    g = w.g
    i0, i1 = int(w.ix[comp].min()), int(w.ix[comp].max())
    j0, j1 = int(w.iy[comp].min()), int(w.iy[comp].max())
    pad = int(math.ceil(DOOR_REACH / g.P))
    region = cols.box(max(0, i0 - pad), min(g.nx - 1, i1 + pad), max(0, j0 - pad), min(g.ny - 1, j1 + pad))
    region = region[np.abs(w.fz[region] - float(np.median(w.fz[comp]))) <= DOOR_LEVEL]
    inner = np.isin(region, comp)
    near = {}
    for c in comp.tolist():
        near.setdefault((int(w.ix[c]), int(w.iy[c])), []).append(float(w.fz[c]))
    touching = []
    for i in np.nonzero(active)[0]:
        c = cells_of[i]
        if not len(c) or w.ix[c].max() < i0 - 1 or w.ix[c].min() > i1 + 1 or w.iy[c].max() < j0 - 1 \
                or w.iy[c].min() > j1 + 1:
            continue
        for x, y, f in zip(w.ix[c].tolist(), w.iy[c].tolist(), w.fz[c].tolist()):
            if any(abs(f - f2) <= w.St * g.P + 1e-6 for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
                   for f2 in near.get((x + dx, y + dy), ())):
                touching.append(int(i))
                break
    touching.sort(key=lambda i: (len(cells_of[i]), i))

    def joins(i, cells):
        """Does ``comp`` reach a reached span within the region once leaf i blocks ``cells`` instead?"""
        held = count[region] - np.isin(region, cells_of[i]) + np.isin(region, cells) > 0
        alive = region[~held]
        ea, eb = grid_edges(w.ix[alive], w.iy[alive], w.k[alive], g, w.St)
        lab = nav3d.components(len(alive), ea, eb)
        inn = inner[~held]
        return bool(np.intersect1d(lab[inn], lab[after[alive] & ~inn]).size)

    for i in touching:
        lf = leaves[i]
        if lf["rec"]["motion"] != "swing":
            continue
        current = lf["angle"] or lf["rec"].get("max_angle_deg", doorpose.OPEN_DEG)
        for angle in (a for a in NARROW if a < current - 1e-9):
            cells = _leaf_cells(w, cols, lf, angle, wc)
            if not joins(i, cells):
                continue
            di = lf["door"]
            if di in regions:                 # the narrowed leaf must not cut its own door
                held = count > 0
                held[cells_of[i]] = count[cells_of[i]] > 1
                held[cells] = True
                if not connected(w, regions[di], *w.sides[di], held):
                    continue
            return ("narrow", [i], angle, cells)
    for i in touching:
        if joins(i, np.zeros(0, np.int64)):
            return ("release", [i], None, None)
    return ("release", touching, None, None)


def reachable(w: Walk, blocked: np.ndarray) -> np.ndarray:
    """Spans in a component (of the unblocked spans) that holds a street start cell."""
    alive = np.nonzero(~blocked)[0]
    ea, eb = grid_edges(w.ix[alive], w.iy[alive], w.k[alive], w.g, w.St)
    lab = nav3d.components(len(alive), ea, eb)
    pos = np.full(len(w.ix), -1, np.int64)
    pos[alive] = np.arange(len(alive))
    st = pos[w.start]
    st = st[st >= 0]
    out = np.zeros(len(w.ix), bool)
    if len(st):
        out[alive[np.isin(lab, np.unique(lab[st]))]] = True
    return out


# ----------------------------------------------------------------------------- layers and the file
def band_of(z, ffls, pad):
    """Storey index owning floor height z (block-local): FFL_s - pad <= z < FFL_{s+1} - pad (clamped)."""
    i = np.searchsorted(np.asarray(ffls, float) - pad, np.asarray(z, float), side="right") - 1
    return np.clip(i, 0, len(ffls) - 1)


def layers(w: Walk, keep: np.ndarray, storeys: list, offset, pad: float):
    """Per-storey main rasters and overflow records of the spans ``keep`` (``storeys``: [(name, block-local FFL)]
    bottom up; ``offset``: the block -> estate translation). Returns (nx, ny, origin (block-local corner of cell
    (0, 0)), [dict(tag, ffl, raster (int16, nx * ny), overflow [(ix, iy, mm)])], cropped to the kept cells."""
    idx = np.nonzero(keep)[0]
    if not len(idx):
        raise ValueError("no walkable cell is reachable from the street")
    ffl = np.array([f for _, f in storeys], float)
    z = w.fz[idx] - offset[2]
    s = band_of(z, ffl, pad)
    mm = np.round((z - ffl[s]) * 1000.0).astype(np.int64)
    if mm.min() < -32768 or mm.max() >= sn5w.BLOCKED:
        raise ValueError(f"a floor lies {mm.min()} .. {mm.max()} mm from its storey's FFL: beyond int16")
    i0, j0 = int(w.ix[idx].min()), int(w.iy[idx].min())
    nx, ny = int(w.ix[idx].max()) - i0 + 1, int(w.iy[idx].max()) - j0 + 1
    cx = w.ix[idx].astype(np.int64) - i0
    cy = w.iy[idx].astype(np.int64) - j0
    order = np.lexsort((mm, np.abs(mm), cx, cy, s))       # per (storey, column): the floor closest to FFL first
    s, cx, cy, mm = s[order], cx[order], cy[order], mm[order]
    first = np.ones(len(s), bool)
    first[1:] = (s[1:] != s[:-1]) | (cx[1:] != cx[:-1]) | (cy[1:] != cy[:-1])
    out = []
    for li, (name, f) in enumerate(storeys):
        m = s == li
        r = np.full(nx * ny, sn5w.BLOCKED, np.int16)
        main = m & first
        r[cy[main] * nx + cx[main]] = mm[main]
        ov = m & ~first
        out.append(dict(tag=name, ffl=float(f), raster=r,
                        overflow=list(zip(cx[ov].tolist(), cy[ov].tolist(), mm[ov].tolist()))))
    g = w.g
    origin = (float(g.lo[0] + i0 * g.P - offset[0]), float(g.lo[1] + j0 * g.P - offset[1]))
    return nx, ny, origin, out


def reference(lays: list) -> int:
    """The typical storey (not the lowest, not the top one) with the most walkable cells; the lowest index wins a
    tie. With fewer than three storeys, any storey."""
    cand = list(range(1, len(lays) - 1)) or list(range(len(lays)))
    counts = [int((lays[i]["raster"] != sn5w.BLOCKED).sum()) for i in cand]
    return cand[int(np.argmax(counts))]


def coarsen(nx, ny, lays):
    """The 0.2 m fallback: a coarse cell is walkable where all four 0.1 m cells under it are, at the highest of their
    four floors (main), and keeps an overflow floor where every one of the four has a floor within 0.2 m of it."""
    nx2, ny2 = (nx + 1) // 2, (ny + 1) // 2
    out = []
    for lay in lays:
        R = np.full((2 * ny2, 2 * nx2), sn5w.BLOCKED, np.int32)
        R[:ny, :nx] = np.asarray(lay["raster"], np.int32).reshape(ny, nx)
        quads = np.stack([R[0::2, 0::2], R[0::2, 1::2], R[1::2, 0::2], R[1::2, 1::2]])
        ok = (quads != sn5w.BLOCKED).all(axis=0)
        main = np.where(ok, quads.max(axis=0), sn5w.BLOCKED).astype(np.int16).reshape(-1)
        floors = {}
        for ix, iy, mm in lay["overflow"]:
            floors.setdefault((ix, iy), []).append(mm)
        ov = set()
        for (ix, iy), mms in floors.items():
            cx, cy = ix // 2, iy // 2
            subs = [(2 * cx + a, 2 * cy + b) for b in (0, 1) for a in (0, 1)]
            for mm in mms:
                hits = []
                for sx, sy in subs:
                    have = list(floors.get((sx, sy), []))
                    if sx < nx and sy < ny and R[sy, sx] != sn5w.BLOCKED:
                        have.append(int(R[sy, sx]))
                    near = [h for h in have if abs(h - mm) <= 200]
                    if not near:
                        break
                    hits.append(max(near))
                else:
                    top = max(hits)
                    if top != int(main[cy * nx2 + cx]):
                        ov.add((cx, cy, top))
        out.append(dict(lay, raster=main, overflow=sorted(ov)))
    return nx2, ny2, out


def encode(nx, ny, origin, lays, wc: WalkConfig):
    """SN5W bytes of the layers, the coarse fallback when the gzipped fine grid exceeds wc.max_walk_gz. Returns
    (bytes, ref index, coarse)."""
    ref = reference(lays)
    data = sn5w.encode(nx, ny, wc.cell, wc.radius, wc.step, origin, lays, ref)
    if len(gzip.compress(data, 9, mtime=0)) <= wc.max_walk_gz:
        return data, ref, False
    nx2, ny2, clays = coarsen(nx, ny, lays)
    ref = reference(clays)
    data = sn5w.encode(nx2, ny2, 2 * wc.cell, wc.radius, wc.step, origin, clays, ref, coarse=True)
    size = len(gzip.compress(data, 9, mtime=0))
    if size > wc.max_walk_gz:
        raise ValueError(f"walk grid {size} B gzipped even at {2 * wc.cell} m cells (cap {wc.max_walk_gz} B)")
    return data, ref, True


def leaf_records(eng: dict, offset, height: float) -> list:
    """Every leaf of every passable door of an engine JSON: dict(guid, storey, rec (the JSON leaf), along, across
    (the door's axes), offset (block -> estate), z (its pivot, the sill: estate frame), obstructs: the opened leaf
    comes lower than the agent's ``height`` above the sill; a rolled-up shutter does not)."""
    out = []
    for d in sorted(eng.get("doors", []), key=lambda d: d["guid"]):
        if not d.get("passable", True):
            continue
        for rec in d.get("leaves", []):
            zlo, _ = doorpose.opened_z(rec, d["along_wall"], d["swing_side"])
            pz = float(rec["pivot"][2])
            out.append(dict(guid=d["guid"], storey=d.get("storey"), rec=rec, along=d["along_wall"],
                            across=d["swing_side"], offset=np.asarray(offset, float), z=pz + float(offset[2]),
                            obstructs=zlo - pz < height, angle=None))
    return out
