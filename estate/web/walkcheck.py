"""The exported walk grid as a viewer reads it, and what every building's stair paths must hold on it.

``Grid`` reads an SN5W file (estate/web/sn5w.py) the way the portfolio viewer does (plan 8.4, lib/estate/walk.ts),
from the file's own f32 header values:

- ``floor_at(x, y, z)``: floorAt. In the cell floor((x - origin_x) / cell), floor((y - origin_y) / cell), the floor
  closest to the feet height z within +-step (0.40 m), raster and overflow records alike, from the layers whose band
  meets [z - step, z + step]; at equal distance the higher floor. None for a blocked or off-grid cell.
- ``nearest(x, y, z, reach)``: nearestWalkable. The point's own cell if it has a floor, else the cell centre nearest
  the point (ties to the lower iy, then ix) that has one, no further than ``reach``.
- ``band(z)``: the layer owning height z: FFL_s - band_pad <= z < FFL_s+1 - band_pad, in whole millimetres, as
  estate/web/walk.py files the floors.

- ``crossed(p, q)``: every cell the plan segment p -> q crosses, an exact supercover (the cells an Amanatides-Woo
  traversal walks, plus the ones it only touches at a corner), each cell's square grown by EDGE (0.1 mm) on every
  side: a segment running along a cell edge, or through a corner, crosses the cells on both sides, whichever way
  float rounding puts it. ``gaps(p, q)``: those of its cells without a floor for feet at the segment's height over
  the whole stretch of it inside the cell (``holds``).

``stair_errors(grid, stairs)`` holds the web JSON's stair paths to that reading (an empty list when they keep it):

1. every point of a path snaps to a walkable cell within SNAP (0.1 m) on the grid of its own storey band: a floor
   of layer band(z) within +-step of z, in a cell whose centre is within SNAP of the point (or the point's own cell);
2. every cell every segment crosses (``crossed``, both ends included) has a floor within +-step of the path's height
   all the way across it, so a viewer walking or auto-climbing the path never leaves the grid wherever it samples
   it, not even where the line clips the corner of a blocked cell by a few cm (samples every 0.05 m missed NC_514's
   two shop-block stairs, whose last segment crossed 35.7 mm of a cell an obstruction on the landing blocks);
3. the first point stands on its storey's layer and the last on the ``to`` storey's layer, each within END_DZ of
   that storey's FFL, on a floor of their own cell;
4. heights never fall and never rise more than a step between consecutive points.

``band_errors(grid)`` reports floors filed outside their layer's band (a viewer skips layers by band, so it could
miss one), and ``site_errors`` is both for one building. The web stage runs stair_errors on the bytes it is about to
write (estate/web/export.py), the release site_errors on the files it ships (estate/web/release.py), and
tests/test_web.py both on the test block and on every current building under model/.
"""
from __future__ import annotations

import math

import numpy as np

from estate.web import sn5w

SNAP = 0.10                 # m: how far a path point may sit from the centre of a walkable cell of its band
EDGE = 1e-4                 # m: a segment this close to a cell (an edge or a corner of it) crosses it; paths are in mm
END_DZ = 0.05               # m: the first and last points' floors against the FFLs of their storeys
EPS = 1e-9                  # the viewer's tolerance on floor distances (lib/estate/walk.ts)
PRUNE_SLACK = 0.001         # m: the viewer's slack when it skips the layers a feet height cannot reach
BAND_PAD = 0.25             # m below an FFL where a storey's band starts (web.json walk.band_pad; the viewer's too)


def _clip(lo: float, hi: float, a: float, d: float):
    """The stretch (t0, t1) of [0, 1] where lo <= a + t d <= hi, or None."""
    if d == 0.0:
        return (0.0, 1.0) if lo <= a <= hi else None
    t0, t1 = (lo - a) / d, (hi - a) / d
    if t0 > t1:
        t0, t1 = t1, t0
    t0, t1 = max(t0, 0.0), min(t1, 1.0)
    return (t0, t1) if t0 <= t1 else None


def crossed(origin, cell: float, p, q, eps: float = EDGE) -> list[tuple[int, int, float, float]]:
    """Every cell of a grid (``origin``: the corner of cell (0, 0); square cells of side ``cell``) that the plan
    segment p -> q crosses: [(ix, iy, t0, t1)], t0 <= t1 the stretch of the segment (t = 0 at p, 1 at q) inside the
    cell's square grown by ``eps`` (m) on every side, in order along the segment (then iy, ix). Exact, nothing is
    sampled: column by column the segment is clipped to the column's (grown) strip, then to each cell of the column
    whose grown square that piece meets. A cell the segment only touches at a corner is crossed (t0 == t1), and so is
    every cell within ``eps`` of it: both cells of an edge it runs along, all four round a corner it passes through.
    A point (p == q) crosses the cell(s) it stands in."""
    ox, oy = float(origin[0]), float(origin[1])
    ax, ay = (float(p[0]) - ox) / cell, (float(p[1]) - oy) / cell
    bx, by = (float(q[0]) - ox) / cell, (float(q[1]) - oy) / cell
    dx, dy, e = bx - ax, by - ay, eps / cell
    out = []
    for ix in range(math.floor(min(ax, bx) - e) - 1, math.floor(max(ax, bx) + e) + 2):
        tx = _clip(ix - e, ix + 1 + e, ax, dx)
        if tx is None:
            continue
        y0, y1 = sorted((ay + dy * tx[0], ay + dy * tx[1]))
        for iy in range(math.floor(y0 - e) - 1, math.floor(y1 + e) + 2):
            ty = _clip(iy - e, iy + 1 + e, ay, dy)
            if ty is not None and max(tx[0], ty[0]) <= min(tx[1], ty[1]):
                out.append((ix, iy, max(tx[0], ty[0]), min(tx[1], ty[1])))
    return sorted(out, key=lambda c: (c[2], c[3], c[1], c[0]))


class Grid:
    """An SN5W file decoded for floor lookups (numpy): rasters resolved to absolute mm, overflow records by cell."""

    def __init__(self, data: bytes, band_pad: float = BAND_PAD):
        t = sn5w.read_table(data)
        data = bytes(data)
        self.nx, self.ny, self.cell, self.step = t["nx"], t["ny"], float(t["cell"]), float(t["step"])
        self.ox, self.oy = (float(v) for v in t["origin"])
        lays = t["layers"]
        self.tags = [lay["tag"] for lay in lays]
        self.ffl = [float(lay["ffl"]) for lay in lays]
        n = self.nx * self.ny
        stored = {i: np.frombuffer(data, "<i2", n, lay["raster_off"]).astype(np.int32)
                  for i, lay in enumerate(lays) if lay["mode"] != "same"}
        base = stored[t["ref"]]
        self.rasters = []
        for i, lay in enumerate(lays):
            r = base if lay["mode"] == "same" else stored[i]
            if lay["mode"] == "delta":
                r = (r + base + 32768) % 65536 - 32768
            self.rasters.append(r.astype(np.int16))
        self.overflow = []
        for lay in lays:
            recs = {}
            o = lay["overflow_off"]
            for ix, iy, mm, _ in sn5w._OVERFLOW.iter_unpack(data[o:o + 8 * lay["overflow_count"]]):
                recs.setdefault(iy * self.nx + ix, []).append(mm)
            self.overflow.append(recs)
        pad = float(band_pad)
        self.band_lo = [-math.inf] + [f - pad for f in self.ffl[1:]]
        self.band_hi = [f - pad for f in self.ffl[1:]] + [math.inf]
        self._edges_mm = [int(round(f * 1000)) - int(round(pad * 1000)) for f in self.ffl]

    # ------------------------------------------------------------------ cells
    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        """The cell of block-local (x, y), computed as the viewer does (times the reciprocal of the cell, floored)."""
        inv = 1.0 / self.cell
        return math.floor((x - self.ox) * inv), math.floor((y - self.oy) * inv)

    def centre(self, ix: int, iy: int) -> tuple[float, float]:
        return self.ox + (ix + 0.5) * self.cell, self.oy + (iy + 0.5) * self.cell

    def floors(self, ix: int, iy: int, layers=None) -> list[tuple[float, int]]:
        """Every floor of cell (ix, iy): [(height m, layer)], of ``layers`` only when given."""
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return []
        key = iy * self.nx + ix
        out = []
        for i in (range(len(self.ffl)) if layers is None else layers):
            v = int(self.rasters[i][key])
            if v != sn5w.BLOCKED:
                out.append((self.ffl[i] + v / 1000.0, i))
            out += [(self.ffl[i] + mm / 1000.0, i) for mm in self.overflow[i].get(key, ())]
        return out

    def floor_in_cell(self, ix: int, iy: int, z: float, layers=None):
        """(floor, layer) of cell (ix, iy) closest to z within +-step, from the layers whose band meets that reach
        (and among ``layers`` when given); None when there is none."""
        if not (0 <= ix < self.nx and 0 <= iy < self.ny) or not math.isfinite(z):
            return None
        lo, hi = z - self.step - EPS, z + self.step + EPS
        cand = [i for i in range(len(self.ffl)) if not (self.band_lo[i] > hi + PRUNE_SLACK
                                                         or self.band_hi[i] < lo - PRUNE_SLACK)]
        if layers is not None:
            cand = [i for i in cand if i in layers]
        best = None
        for f, i in self.floors(ix, iy, cand):
            if lo <= f <= hi and (best is None or abs(f - z) < abs(best[0] - z) - EPS
                                  or (abs(f - z) <= abs(best[0] - z) + EPS and f > best[0])):
                best = (f, i)
        return best

    def floor_at(self, x: float, y: float, z: float, layers=None):
        """floorAt: (floor, layer) under block-local (x, y) for feet at z, or None."""
        return self.floor_in_cell(*self.cell_of(x, y), z, layers)

    def nearest(self, x: float, y: float, z: float, reach: float, layers=None):
        """nearestWalkable: (x, y, floor, layer) of the point itself if its cell has a floor for feet at z, else of
        the nearest cell centre within ``reach`` that has one (ties to the lower iy, then ix); None."""
        cx, cy = self.cell_of(x, y)
        own = self.floor_in_cell(cx, cy, z, layers)
        if own is not None:
            return x, y, own[0], own[1]
        if not reach > 0:
            return None
        k = math.ceil(reach / self.cell) + 1
        limit, best = reach * reach + EPS, None
        for iy in range(max(0, cy - k), min(self.ny - 1, cy + k) + 1):
            for ix in range(max(0, cx - k), min(self.nx - 1, cx + k) + 1):
                px, py = self.centre(ix, iy)
                d2 = (px - x) ** 2 + (py - y) ** 2
                if d2 > limit or (best is not None and d2 >= best[0]):
                    continue
                f = self.floor_in_cell(ix, iy, z, layers)
                if f is not None:
                    best = (d2, px, py, f[0], f[1])
        return None if best is None else best[1:]

    def holds(self, ix: int, iy: int, z0: float, z1: float) -> bool:
        """Cell (ix, iy) has a floor within +-step of every feet height between z0 and z1, so floorAt finds one there
        wherever the feet are on that stretch. Only floors inside their own layer's band count: floorAt can never
        prune those (a layer whose band holds a floor within a step of the feet meets their reach)."""
        lo, hi = min(z0, z1), max(z0, z1)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return False
        reach = self.step + EPS
        for f in sorted(f for f, i in self.floors(ix, iy) if self.band(f) == i):
            if f + reach < lo:
                continue                    # wholly below what is still to be covered
            if f - reach > lo:
                return False                # lo is out of every floor's reach
            lo = f + reach
            if lo >= hi:
                return True
        return False

    def crossed(self, p, q, eps: float = EDGE) -> list[tuple[int, int, float, float]]:
        """crossed() on this grid's own (f32) origin and cell, the cells a viewer computes."""
        return crossed((self.ox, self.oy), self.cell, p, q, eps)

    def gaps(self, p, q, eps: float = EDGE) -> list[tuple[int, int, float, float, float]]:
        """The cells the segment p -> q ([x, y, z], z interpolated) crosses (``crossed`` with ``eps``) that do not
        hold it (``holds``, from the height where it enters the cell to the one where it leaves): [(ix, iy, x, y, z)]
        with the point where it enters each, in order along the segment. p == q checks a point."""
        p, q = [float(v) for v in p], [float(v) for v in q]
        out = []
        for ix, iy, t0, t1 in self.crossed(p, q, eps):
            z0, z1 = p[2] + t0 * (q[2] - p[2]), p[2] + t1 * (q[2] - p[2])
            if not self.holds(ix, iy, z0, z1):
                out.append((ix, iy, p[0] + t0 * (q[0] - p[0]), p[1] + t0 * (q[1] - p[1]), z0))
        return out

    def band(self, z: float) -> int:
        """The layer owning height z (whole mm; the lowest takes everything below, the top everything above)."""
        mm = int(round(z * 1000))
        i = 0
        while i + 1 < len(self._edges_mm) and mm >= self._edges_mm[i + 1]:
            i += 1
        return i

    def spans(self) -> dict:
        """Every floor of the grid, raster and overflow: numpy arrays ix, iy, mm (above its layer's FFL), layer."""
        cols = {k: [] for k in ("ix", "iy", "mm", "layer")}
        for i, r in enumerate(self.rasters):
            idx = np.nonzero(r != sn5w.BLOCKED)[0]
            keys = np.array(sorted(self.overflow[i]), np.int64)
            reps = np.array([len(self.overflow[i][k]) for k in keys.tolist()], np.int64)
            okey = np.repeat(keys, reps)
            omm = np.array([mm for k in keys.tolist() for mm in self.overflow[i][k]], np.int64)
            key = np.concatenate([idx, okey])
            cols["ix"].append(key % self.nx)
            cols["iy"].append(key // self.nx)
            cols["mm"].append(np.concatenate([r[idx].astype(np.int64), omm]))
            cols["layer"].append(np.full(len(key), i, np.int64))
        return {k: np.concatenate(v) if v else np.zeros(0, np.int64) for k, v in cols.items()}


def band_errors(grid: Grid) -> list[str]:
    """Floors stored outside their layer's band (FFL_s - band_pad <= FFL_s + mm < FFL_s+1 - band_pad, whole mm; no
    lower edge for the lowest layer, no upper one for the top): a viewer prunes layers by band, so such a floor can
    go unseen. One line per layer that has any."""
    s, out = grid.spans(), []
    ffl = [int(round(f * 1000)) for f in grid.ffl]
    for i, tag in enumerate(grid.tags):
        a = ffl[i] + s["mm"][s["layer"] == i]
        lo = grid._edges_mm[i] if i else -(1 << 40)
        hi = grid._edges_mm[i + 1] if i + 1 < len(ffl) else 1 << 40
        bad = a[(a < lo) | (a >= hi)]
        if len(bad):
            out.append(f"layer {tag}: {len(bad)} floor(s) outside its band [{lo if i else '-inf'}, "
                       f"{hi if i + 1 < len(ffl) else 'inf'}) mm, e.g. {int(bad[0])} mm")
    return out


def _fmt(p) -> str:
    return "(" + ", ".join(f"{float(v):.3f}" for v in p) + ")"


def stair_errors(grid: Grid, stairs: list, limit: int = 0) -> list[str]:
    """What breaks the rules of this module's docstring in the web JSON's ``stairs`` on ``grid``, one line per
    stair and rule (``limit``: stop after that many lines, 0 for all)."""
    index = {tag: i for i, tag in enumerate(grid.tags)}
    out = []
    for s in stairs:
        name, P = s.get("name") or "?", [list(map(float, p)) for p in s.get("path") or ()]
        if len(P) < 2:
            out.append(f"{name}: a path of {len(P)} point(s)")
            continue
        dz = [b[2] - a[2] for a, b in zip(P, P[1:])]
        if min(dz) < -1e-6 or max(dz) > grid.step + 1e-6:
            out.append(f"{name}: the path falls or rises more than a step between points ({min(dz):.3f} .. "
                       f"{max(dz):.3f} m)")
        far = [(i, p) for i, p in enumerate(P) if grid.nearest(*p, SNAP, layers=(grid.band(p[2]),)) is None]
        if far:
            i, p = far[0]
            out.append(f"{name}: {len(far)} point(s) with no walkable cell of their band within {SNAP} m, first "
                       f"path[{i}] {_fmt(p)} ({grid.tags[grid.band(p[2])]})")
        off = [(k, gap) for k, (a, b) in enumerate(zip(P, P[1:])) for gap in grid.gaps(a, b)]
        if off:
            k, (ix, iy, *at) = off[0]
            out.append(f"{name}: {len(off)} cell(s) crossed by the path with no floor within {grid.step:.2f} m of it "
                       f"all the way across, first ({ix}, {iy}) by path[{k}] -> path[{k + 1}], entered at {_fmt(at)}")
        for which, p, tag, ffl in (("first", P[0], s.get("storey"), s.get("from_ffl")),
                                   ("last", P[-1], s.get("to"), s.get("to_ffl"))):
            hit = grid.floor_at(*p)
            if tag not in index or ffl is None:
                out.append(f"{name}: the {which} point's storey {tag!r} is not a layer of the grid")
            elif hit is None or hit[1] != index[tag] or abs(hit[0] - float(ffl)) > END_DZ:
                got = "no floor" if hit is None else f"{hit[0]:.3f} on {grid.tags[hit[1]]}"
                out.append(f"{name}: the {which} point {_fmt(p)} stands on {got}, not {tag} at {float(ffl):.3f}")
        if limit and len(out) >= limit:
            return out[:limit]
    return out


def site_errors(walk_bytes: bytes, web: dict, limit: int = 0) -> list[str]:
    """band_errors and stair_errors of one building's walk grid and web JSON (its walk.band_pad when recorded)."""
    grid = Grid(walk_bytes, (web.get("walk") or {}).get("band_pad", BAND_PAD))
    out = band_errors(grid) + stair_errors(grid, web.get("stairs") or [], limit)
    return out[:limit] if limit else out
