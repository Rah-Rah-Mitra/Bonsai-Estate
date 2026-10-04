"""Building walk test: can a person-sized agent reach every room, the roof and (by lift) every front door?

Generalises legacy/toolkit/nav_check.py to any building of the estate. It reads world coordinates from the
shared mesh cache, needs no scipy, and works within a fixed memory budget. It tests the exported geometry,
not what the generator intended.

1. Voxelise at 0.1 m, Recast style, with cell centres on multiples of the voxel size as in the legacy test.
   Every solid surface is sampled into one dense occupancy grid, ``S[z, x, y]``. The inside of every closed
   solid is filled from vertical ray crossings, so a tank or a lift car is not hollow. The leaf of a passable
   door (Navigation.Passable) is left out because it opens, but its linings stay: a jamb box at each end of
   the opening, (NominalWidth - ClearWidth) / 2 wide and full door height, so the agent has to fit the clear
   width. Lift landing doors stay closed.
2. A cell is walkable when it has solid below it, a free feet zone up to the step height, and a free disc of
   the agent radius from the step height to the agent height. The radius is rounded up to whole voxels and
   the report records the effective radius. This is computed in z-bands with doubling run tests, so memory
   stays bounded and no full-height derived grid is held.
3. Walkable cells are spans in a multi-layer heightfield. Neighbouring spans connect when their floors
   differ by at most one step, so stairs are walkable. Components come from a vectorised union-find (hook
   plus pointer jumping) instead of 3D labelling.
4. Door bands. At 0.1 m a 0.70 m clear opening passes a 0.60 m agent or not depending on where its jambs fall
   on the grid (the disc test needs 2r + 0.1 to 2r + 0.2 m). Each passable door is therefore tested again
   in a small grid of 0.025 m voxels laid along the door (the same voxeliser and walkability test on the
   geometry within reach of the opening). When the agent fits through there, the coarse cells in front of
   the door on both sides are joined by a portal. Stacked doors with identical surroundings share one test.
5. The start is walkable ground on the margin of the bounding box, which is the street around the void
   deck. A building file has no terrain, so a virtual ground plane fills empty columns at ground level.
   Every IfcSpace (except lift shafts and AC ledges; stair enclosures and refuse rooms included) and the
   roof deck must hold a reachable cell.
6. Step-free means the same graph without level changes above a few millimetres, plus one portal per
   lift. The portal links the cells in front of the lift's landing doors on every level it serves. Every
   flat's main door must be crossed this way: a step-free cell must lie inside the flat's entry room
   (Navigation.FromSpace/ToSpace), not merely in the corridor outside the door.
"""
from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import math
import re
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely

import ifcopenshell.util.element as uel
import ifcopenshell.util.placement as uplace

from estate.export import meshcache

LANDING_RE = re.compile(r"^(?P<level>\S+)\s+lift(?:\s+(?P<k>\d+))?\s+landing door$", re.I)
LIFT_NO_RE = re.compile(r"lift\s*(\d+)", re.I)
EXCLUDED_SPACES = re.compile(r"lift shaft|ac ledge", re.I)
SPACE_KINDS = (("stair", re.compile(r"-STAIR\d+$|^stair\b", re.I)),     # checked like any room, counted apart
               ("refuse", re.compile(r"-REF\d+$|refuse|bin centre", re.I)),
               ("roof", re.compile(r"-DECK$|roof deck", re.I)))
COLOURS = {"wall": (64, 64, 64), "low": (165, 165, 165), "reach": (115, 199, 115), "cut": (230, 102, 89),
           "frag": (244, 196, 160), "bg": (255, 255, 255), "outline": (200, 30, 30)}
ROOF_WORD = {True: "yes", False: "NO", None: "n/a"}
FRAGMENT_M2 = 0.5            # cut-off components smaller than this (rail tops, sills) are drawn pale, not red
DEFAULT_JAMB = 0.05          # IfcOpenShell's default door lining, used when Navigation gives no clear width
DOOR_VOXEL = 0.025           # voxel of the door-band test
WALL_HALF = 0.2              # beyond this distance from the door axis the agent is clearly on one side


# ----------------------------------------------------------------------------- measurement
def peak_memory_mb() -> float | None:
    """Peak working set of this process in MB, read from the OS (Windows) or getrusage (POSIX)."""
    try:
        if sys.platform == "win32":
            class PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            fn = k32.K32GetProcessMemoryInfo
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            if fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return pmc.PeakWorkingSetSize / 2 ** 20
            return None
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------- grid
@dataclass
class Grid:
    lo: np.ndarray            # corner of cell (0, 0, 0); cell centres sit on multiples of P
    P: float
    nx: int
    ny: int
    nz: int

    @classmethod
    def around(cls, lo, hi, P):
        lo = (np.floor(np.asarray(lo, float) / P) - 0.5) * P
        n = np.ceil((np.asarray(hi, float) - lo) / P).astype(int) + 1
        return cls(lo, P, int(n[0]), int(n[1]), int(n[2]))

    @property
    def shape(self):
        return (self.nz, self.nx, self.ny)

    def floor_z(self, k):
        """Height of the floor under walkable cell k: the centre of voxel k - 1, where lattice floors lie."""
        return self.lo[2] + (np.asarray(k) - 0.5) * self.P

    def kz(self, z):
        """Index of the walkable cell standing on a floor at height z."""
        return int(math.floor((z - self.lo[2]) / self.P)) + 1

    def ix(self, x):
        return int(math.floor((x - self.lo[0]) / self.P))

    def iy(self, y):
        return int(math.floor((y - self.lo[1]) / self.P))

    def cx(self, i):
        return self.lo[0] + (np.asarray(i) + 0.5) * self.P

    def cy(self, j):
        return self.lo[1] + (np.asarray(j) + 0.5) * self.P


# ----------------------------------------------------------------------------- voxelisation
def _split_big(T, box_lo, box_hi, max_edge=4.0, ids=None):
    """Subdivide triangles that stick out of the box so that the parts outside are not sampled.

    With ``ids`` (one per triangle) the ids follow the pieces and (T, ids) is returned.
    """
    out, out_id = [], []
    want_ids = ids is not None
    ids = np.asarray(ids) if want_ids else np.zeros(len(T), np.int64)
    for _ in range(12):
        tmin, tmax = T.min(axis=1), T.max(axis=1)
        inside_box = np.all(tmax >= box_lo, axis=1) & np.all(tmin <= box_hi, axis=1)
        T, ids = T[inside_box], ids[inside_box]
        tmin, tmax = tmin[inside_box], tmax[inside_box]
        edge = np.max(tmax - tmin, axis=1)
        sticks_out = (np.any(tmin < box_lo, axis=1) | np.any(tmax > box_hi, axis=1)) & (edge > max_edge)
        out.append(T[~sticks_out])
        out_id.append(ids[~sticks_out])
        T, ids = T[sticks_out], ids[sticks_out]
        if not len(T):
            break
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        T = np.concatenate([np.stack(s, 1) for s in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
        ids = np.tile(ids, 4)
    if len(T):
        out.append(T)
        out_id.append(ids)
    T = np.concatenate(out) if out else np.zeros((0, 3, 3))
    if want_ids:
        return T, (np.concatenate(out_id) if out_id else np.zeros(0, np.int64))
    return T


def _unit_normals(T):
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    a = np.linalg.norm(n, axis=1)
    return n / np.maximum(a, 1e-300)[:, None], a


def _expand(start, count):
    """Concatenated ranges start[i] .. start[i] + count[i] - 1 and the owner of each element."""
    owner = np.repeat(np.arange(len(count)), count)
    return start[owner] + (np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)), owner


def _chunks(cost, budget):
    csum = np.cumsum(cost)
    start = 0
    while start < len(cost):
        base = csum[start - 1] if start else 0
        stop = max(int(np.searchsorted(csum, base + budget, side="right")), start + 1)
        yield slice(start, stop)
        start = stop


def _sample_chunks(T, h, budget=1_000_000):
    """Yield points that cover triangles T (n, 3, 3) at spacing <= h (used for steep, non-vertical faces).

    The grid spans the two shorter edges and the longest edge gets its own row. Each point is nudged 0.1 mm
    into the solid (against the face normal) and 0.02 mm towards the triangle centroid. A face lying exactly
    on a cell boundary then marks the cell on the solid side.
    """
    if not len(T):
        return
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    la, lb, lc = (np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1), np.linalg.norm(a - b, axis=1))
    opp = np.argmax(np.stack([la, lb, lc], 1), axis=1)           # vertex opposite the longest edge
    rows = np.arange(len(T))
    p0, p1, p2 = T[rows, opp], T[rows, (opp + 1) % 3], T[rows, (opp + 2) % 3]
    e1, e2, e3 = p1 - p0, p2 - p0, p2 - p1
    nrm = np.cross(e1, e2)
    area2 = np.linalg.norm(nrm, axis=1)
    ok = area2 > 1e-12
    p0, p1, e1, e2, e3, nrm, area2 = p0[ok], p1[ok], e1[ok], e2[ok], e3[ok], nrm[ok], area2[ok]
    nrm = nrm / area2[:, None]
    cen = p0 + (e1 + e2) / 3
    n1 = np.maximum(1, np.ceil(np.linalg.norm(e1, axis=1) / h)).astype(np.int64)
    n2 = np.maximum(1, np.ceil(np.linalg.norm(e2, axis=1) / h)).astype(np.int64)
    n3 = np.maximum(1, np.ceil(np.linalg.norm(e3, axis=1) / h)).astype(np.int64)
    for sl in _chunks((n1 + 1) * (n2 + 1) + n3 + 1, budget):
        g = (n1[sl] + 1) * (n2[sl] + 1)
        local, tri = _expand(np.zeros(len(g), np.int64), g)
        w = (n2[sl] + 1)[tri]
        s, t = (local // w) / n1[sl][tri], (local % w) / n2[sl][tri]
        keep = s + t <= 1.0 + 1e-9
        tri, s, t = tri[keep], s[keep], t[keep]
        pts = p0[sl][tri] + s[:, None] * e1[sl][tri] + t[:, None] * e2[sl][tri]
        u, tri3 = _expand(np.zeros(len(g), np.int64), n3[sl] + 1)
        pts = np.concatenate([pts, p1[sl][tri3] + (u / n3[sl][tri3])[:, None] * e3[sl][tri3]])
        tri = np.concatenate([tri, tri3])
        d = cen[sl][tri] - pts
        dl = np.linalg.norm(d, axis=1)
        pts += d * (np.minimum(dl, 2e-5) / np.maximum(dl, 1e-12))[:, None] - 1e-4 * nrm[sl][tri]
        yield pts


def _vertical_voxels(T, nrm, g: Grid, h, budget=4_000_000):
    """Yield linear voxel indices covered by vertical triangles (walls, jambs, risers, slab edges).

    A vertical triangle projects to a segment in plan. It is sampled along that segment at spacing <= h, and
    each sample fills the exact z-interval of the triangle above that point. Samples are nudged 0.1 mm into
    the solid and 0.02 mm inwards along the face, as for the 2D sampling. This costs a fraction of
    sampling the whole face area.
    """
    if not len(T):
        return
    d = np.stack([-nrm[:, 1], nrm[:, 0]], 1)
    d /= np.maximum(np.linalg.norm(d, axis=1), 1e-300)[:, None]
    xy0 = T[:, 0, :2]
    u = np.einsum("tvk,tk->tv", T[:, :, :2] - xy0[:, None, :], d)          # (t, 3) along-face coordinate
    z = T[:, :, 2]
    umin, umax = u.min(1), u.max(1)
    ns = np.maximum(1, np.ceil((umax - umin) / h)).astype(np.int64)
    zspan = (z.max(1) - z.min(1)) / g.P + 2
    for sl in _chunks((ns + 1) * zspan, budget):
        k, tri = _expand(np.zeros(sl.stop - sl.start, np.int64), ns[sl] + 1)
        uu = umin[sl][tri] + (umax - umin)[sl][tri] * k / ns[sl][tri]
        uc = ((umin + umax) / 2)[sl][tri]
        uu += np.sign(uc - uu) * np.minimum(2e-5, np.abs(uc - uu))
        U, Z = u[sl][tri], z[sl][tri]
        zlo = np.full(len(uu), np.inf)
        zhi = np.full(len(uu), -np.inf)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            du = U[:, j] - U[:, i]
            slanted = np.abs(du) > 1e-12
            t = np.where(slanted, (uu - U[:, i]) / np.where(slanted, du, 1.0), 0.0)
            ok = slanted & (t >= -1e-9) & (t <= 1 + 1e-9)
            zz = Z[:, i] + t * (Z[:, j] - Z[:, i])
            zlo = np.where(ok, np.minimum(zlo, zz), zlo)
            zhi = np.where(ok, np.maximum(zhi, zz), zhi)
        ok = zhi >= zlo
        zlo, zhi, uu, tri = zlo[ok], zhi[ok], uu[ok], tri[ok]
        mid = (zlo + zhi) / 2
        zlo, zhi = np.minimum(zlo + 2e-5, mid), np.maximum(zhi - 2e-5, mid)
        xy = xy0[sl][tri] + uu[:, None] * d[sl][tri] - 1e-4 * nrm[sl][tri][:, :2]
        ix = np.floor((xy[:, 0] - g.lo[0]) / g.P).astype(np.int64)
        iy = np.floor((xy[:, 1] - g.lo[1]) / g.P).astype(np.int64)
        ka = np.maximum(0, np.floor((zlo - g.lo[2]) / g.P)).astype(np.int64)
        kb = np.minimum(g.nz - 1, np.floor((zhi - g.lo[2]) / g.P)).astype(np.int64)
        ok = (ix >= 0) & (ix < g.nx) & (iy >= 0) & (iy < g.ny) & (kb >= ka)
        ix, iy, ka, kb = ix[ok], iy[ok], ka[ok], kb[ok]
        kk, owner = _expand(ka, kb - ka + 1)
        yield (kk * g.nx + ix[owner]) * g.ny + iy[owner]


def _ray_hits(T, elem, g: Grid, budget=3_000_000):
    """Vertical rays through every cell centre (plus a tiny generic offset) against non-vertical triangles.

    Returns (column index ix * ny + iy, z, up-facing, element id) per crossing.
    """
    P = g.P
    X0, Y0 = g.lo[0] + P / 2 + 1.37e-5, g.lo[1] + P / 2 + 2.71e-5
    x, y, z = T[:, :, 0], T[:, :, 1], T[:, :, 2]
    area2 = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    ok = np.abs(area2) > 1e-10
    i0 = np.maximum(0, np.ceil((x.min(1) - X0) / P)).astype(np.int64)
    i1 = np.minimum(g.nx - 1, np.floor((x.max(1) - X0) / P)).astype(np.int64)
    j0 = np.maximum(0, np.ceil((y.min(1) - Y0) / P)).astype(np.int64)
    j1 = np.minimum(g.ny - 1, np.floor((y.max(1) - Y0) / P)).astype(np.int64)
    ok &= (i1 >= i0) & (j1 >= j0)
    T, elem, area2, i0, i1, j0, j1 = T[ok], elem[ok], area2[ok], i0[ok], i1[ok], j0[ok], j1[ok]
    x, y, z = T[:, :, 0], T[:, :, 1], T[:, :, 2]
    cy = j1 - j0 + 1
    out = []
    for sl in _chunks((i1 - i0 + 1) * cy, budget):
        local, tri = _expand(np.zeros(sl.stop - sl.start, np.int64), ((i1 - i0 + 1) * cy)[sl])
        w = cy[sl][tri]
        ii = i0[sl][tri] + local // w
        jj = j0[sl][tri] + local % w
        qx, qy = X0 + ii * P, Y0 + jj * P
        xs, ys, zs, a2 = x[sl][tri], y[sl][tri], z[sl][tri], area2[sl][tri]
        w0 = ((xs[:, 1] - qx) * (ys[:, 2] - qy) - (xs[:, 2] - qx) * (ys[:, 1] - qy)) / a2
        w1 = ((xs[:, 2] - qx) * (ys[:, 0] - qy) - (xs[:, 0] - qx) * (ys[:, 2] - qy)) / a2
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        zz = w0 * zs[:, 0] + w1 * zs[:, 1] + w2 * zs[:, 2]
        out.append((ii[inside] * g.ny + jj[inside], zz[inside], (a2 > 0)[inside], elem[sl][tri][inside]))
    if not out:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0, bool), np.zeros(0, np.int64)
    return tuple(np.concatenate(v) for v in zip(*out))


def _hit_voxels(col, zz, up, g: Grid):
    """Voxels of the crossings themselves (the surface of flat faces), nudged 0.1 mm into the solid."""
    k = np.floor((zz - np.where(up, 1e-4, -1e-4) - g.lo[2]) / g.P).astype(np.int64)
    ok = (k >= 0) & (k < g.nz)
    return (k[ok] * g.nx + col[ok] // g.ny) * g.ny + col[ok] % g.ny


def _fill_intervals(col, zz, up, elem, g: Grid):
    """Voxel indices (linear) inside closed solids, from per-element ray crossings (winding count)."""
    if not len(col):
        return np.zeros(0, np.int64)
    order = np.lexsort((zz, col, elem))
    col, zz, up, elem = col[order], zz[order], up[order], elem[order]
    w = np.where(up, -1, 1)
    new = np.ones(len(col), bool)
    new[1:] = (col[1:] != col[:-1]) | (elem[1:] != elem[:-1])
    cs = np.cumsum(w)
    starts = np.nonzero(new)[0]
    inside = cs - (cs[starts] - w[starts])[np.cumsum(new) - 1]
    seg = np.nonzero((inside[:-1] > 0) & ~new[1:])[0]
    if not len(seg):
        return np.zeros(0, np.int64)
    za, zb, c = zz[seg], zz[seg + 1], col[seg]
    P, z0 = g.P, g.lo[2]
    ka = np.maximum(0, np.ceil((za - z0) / P - 0.5 - 1e-6)).astype(np.int64)
    kb = np.minimum(g.nz - 1, np.floor((zb - z0) / P - 0.5 + 1e-6)).astype(np.int64)
    k, owner = _expand(ka, np.maximum(0, kb - ka + 1))
    return (k * g.nx + c[owner] // g.ny) * g.ny + c[owner] % g.ny


def voxelise(T, elem, g: Grid, ground_z: float | None = None, h=None, batch=4_000_000):
    """Dense occupancy S[z, x, y] plus the up-facing crossings (column, z) used for exact floor heights.

    Vertical faces are sampled along their length, steep faces over their area, flat faces are marked
    where the cell-centre rays cross them, and closed solids are filled between their crossings. Ray work
    runs in batches of whole elements so memory stays bounded.
    """
    h = h or 0.45 * g.P
    S = np.zeros(g.shape, dtype=bool)
    flat = S.reshape(-1)
    nrm, area = _unit_normals(T)
    good = area > 1e-12
    vertical = good & (np.abs(nrm[:, 2]) < 0.02)
    steep = good & ~vertical & (np.abs(nrm[:, 2]) < 0.7)
    for idx in _vertical_voxels(T[vertical], nrm[vertical], g, h):
        flat[idx] = True
    box_lo = g.lo - g.P
    box_hi = g.lo + np.array([g.nx, g.ny, g.nz]) * g.P + g.P
    shape = np.array([g.nx, g.ny, g.nz])
    for pts in _sample_chunks(_split_big(T[steep], box_lo, box_hi), h):
        ii = np.floor((pts - g.lo) / g.P).astype(np.int64)
        ii = ii[np.all((ii >= 0) & (ii < shape), axis=1)]
        flat[(ii[:, 2] * g.nx + ii[:, 0]) * g.ny + ii[:, 1]] = True
    # rays, per batch of whole elements
    nv = good & ~vertical
    Tn, en = T[nv], elem[nv]
    ext = Tn.max(axis=1) - Tn.min(axis=1)
    est = (ext[:, 0] / g.P + 1) * (ext[:, 1] / g.P + 1)
    cut = np.nonzero(np.diff(en))[0] + 1                       # element boundaries
    bounds = np.concatenate([[0], cut, [len(en)]])
    per_el = np.add.reduceat(est, bounds[:-1]) if len(en) else np.zeros(0)
    upc, upz = [], []
    for sl in _chunks(per_el, batch):
        a, b = bounds[sl.start], bounds[sl.stop]
        col, zz, up, el = _ray_hits(Tn[a:b], en[a:b], g)
        flat[_hit_voxels(col, zz, up, g)] = True
        flat[_fill_intervals(col, zz, up, el, g)] = True
        upc.append(col[up].astype(np.int64))
        upz.append(zz[up])
    n_virtual = 0
    if ground_z is not None:
        # a virtual ground plane in columns with nothing near ground level (building files carry no terrain)
        ka = max(0, int(math.floor((ground_z - 1.0 - g.lo[2]) / g.P)))
        kb = min(g.nz, int(math.ceil((ground_z + 0.3 - g.lo[2]) / g.P)))
        empty = ~S[ka:kb].any(axis=0)
        S[int(math.floor((ground_z - 1e-4 - g.lo[2]) / g.P))][empty] = True
        n_virtual = int(empty.sum())
    up = (np.concatenate(upc), np.concatenate(upz)) if upc else (np.zeros(0, np.int64), np.zeros(0))
    return S, up, n_virtual


# ----------------------------------------------------------------------------- walkable cells
def _dilate_disk(A, r):
    """OR of A over a disc of radius r cells in the (x, y) axes (1, 2); out-of-grid counts as free."""
    if r <= 0:
        return A.copy()
    widths = {dy: math.isqrt(r * r - dy * dy) for dy in range(0, r + 1)}
    need = set(widths.values())
    X, cur = {0: A}, A
    for w in range(1, max(need) + 1):
        nxt = cur.copy()
        nxt[:, w:, :] |= A[:, :-w, :]
        nxt[:, :-w, :] |= A[:, w:, :]
        cur = nxt
        if w in need:
            X[w] = nxt
    out = X[widths[0]].copy()
    for dy in range(1, r + 1):
        Xw = X[widths[dy]]
        out[:, :, dy:] |= Xw[:, :, :-dy]
        out[:, :, :-dy] |= Xw[:, :, dy:]
    return out


def _run_all(F, n):
    """G[k] = all(F[k:k + n]) along axis 0, by doubling (len(F) - n + 1 rows)."""
    G, p = F, 1
    while 2 * p <= n:
        G = G[:-p] & G[p:]
        p *= 2
    if p < n:
        d = n - p
        G = G[:-d] & G[d:]
    return G


def voxel_radius(r, P) -> int:
    """Agent radius in whole voxels, rounded up so the test is never more lenient than the radius asked for."""
    return int(math.ceil(r / P - 1e-9))


def walkable_cells(S, g: Grid, r_vox, H, St, band=None):
    """Walkable cells (k, ix, iy) as int32 arrays, computed in z-bands of the occupancy grid."""
    nz = S.shape[0]
    band = band or int(max(32, min(128, 25e6 // max(1, g.nx * g.ny))))
    ks, xs, ys = [], [], []
    for k0 in range(1, nz - H, band):
        k1 = min(k0 + band, nz - H)
        L = k1 - k0
        Sb = S[k0 - 1:k1 + H]
        feet = _run_all(~Sb[1:], St)[:L]
        body = _run_all(~_dilate_disk(Sb[1 + St:], r_vox), H - St)[:L]
        Wb = Sb[:L] & feet & body
        kk, ix, iy = np.nonzero(Wb)
        ks.append((kk + k0).astype(np.int32))
        xs.append(ix.astype(np.int32))
        ys.append(iy.astype(np.int32))
    if not ks:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int32)
    return np.concatenate(ks), np.concatenate(xs), np.concatenate(ys)


def components(n, a, b):
    """Connected components of an undirected graph (n nodes, edges a-b): label = smallest node id.

    Vectorised union-find: every round hooks the larger root of each edge onto the smaller one
    (``np.minimum.at``) and then jumps pointers until every node points at its root. Labels never grow,
    so the forest stays acyclic. Rounds shrink the number of roots quickly on grid-like graphs.
    """
    lab = np.arange(n, dtype=np.int64)
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    while len(a):
        la, lb = lab[a], lab[b]
        keep = la != lb
        if not keep.any():
            break
        a, b, la, lb = a[keep], b[keep], la[keep], lb[keep]
        np.minimum.at(lab, np.maximum(la, lb), np.minimum(la, lb))
        while True:
            nxt = lab[lab]
            if np.array_equal(nxt, lab):
                break
            lab = nxt
    return lab


@dataclass
class Cells:
    """Walkable spans sorted by (column, k) and their connectivity."""
    ix: np.ndarray
    iy: np.ndarray
    k: np.ndarray
    fz: np.ndarray            # exact floor height (ray crossing), else the floor voxel centre
    key: np.ndarray
    ea: np.ndarray            # edges (indices into the sorted cells)
    eb: np.ndarray


def connect(k, ix, iy, g: Grid, St, up):
    key = (ix.astype(np.int64) * g.ny + iy) * g.nz + k
    order = np.argsort(key, kind="stable")
    key, k, ix, iy = key[order], k[order], ix[order], iy[order]
    n = len(key)
    ea, eb = [], []
    for dx, dy in ((1, 0), (0, 1)):
        jx, jy = ix.astype(np.int64) + dx, iy.astype(np.int64) + dy
        ok = (jx < g.nx) & (jy < g.ny)
        base = (jx * g.ny + jy) * g.nz
        pos = np.searchsorted(key, base + k - St)
        pc = np.minimum(pos, n - 1)
        hit = ok & (pos < n) & (key[pc] <= base + k + St)
        ea.append(np.nonzero(hit)[0])
        eb.append(pc[hit])
    # exact floor heights from the up-facing ray crossings in the floor voxel
    upc, upz = up
    zs = int(g.nz * g.P * 1e4) + 10
    zq = np.floor((upz - g.lo[2]) * 1e4).astype(np.int64)
    ukey = upc.astype(np.int64) * zs + zq
    o = np.argsort(ukey, kind="stable")
    ukey, upz = ukey[o], upz[o]
    col = ix.astype(np.int64) * g.ny + iy
    q_hi = col * zs + np.floor((k * g.P + 1e-4) * 1e4).astype(np.int64)
    q_lo = col * zs + np.floor(((k - 1) * g.P - 1e-4) * 1e4).astype(np.int64)
    pos = np.searchsorted(ukey, q_hi, side="right") - 1
    pc = np.maximum(pos, 0)
    valid = (pos >= 0) & (ukey[pc] >= q_lo) if len(ukey) else np.zeros(n, bool)
    fz = np.where(valid, upz[pc] if len(ukey) else 0.0, g.floor_z(k))
    return Cells(ix, iy, k, fz.astype(np.float64), key, np.concatenate(ea), np.concatenate(eb))


# ----------------------------------------------------------------------------- model queries
def _polys(tri_xy):
    if not len(tri_xy):
        return shapely.Polygon()
    p = shapely.polygons(tri_xy)
    p = p[shapely.area(p) > 1e-8]
    return shapely.union_all(p).buffer(0) if len(p) else shapely.Polygon()


def _flat_faces(d, which, tol=0.02):
    tri = d["verts"][d["faces"]]
    z = tri[:, :, 2]
    zref = z.min() if which == "bottom" else z.max()
    sel = np.all(np.abs(z - zref) < tol, axis=1)
    return _polys(tri[sel][:, :, :2]), float(zref)


def _storeys(f):
    out = {}
    for st in f.by_type("IfcBuildingStorey"):
        try:
            z = (float(uplace.get_local_placement(st.ObjectPlacement)[2][3]) if st.ObjectPlacement
                 else float(st.Elevation or 0))
        except Exception:  # noqa: BLE001
            z = float(st.Elevation or 0.0)
        out[st.Name] = z
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def _face_id(name: str) -> str:
    """Plate face id behind an IfcSpace name: '#02-101 LD' -> 'U101:LD', 'L1-AM3' -> 'AM3'."""
    if name.startswith("#") and " " in name:
        uid, room = name.split(" ", 1)
        return f"U{uid.split('-')[-1]}:{room}"
    return name.split("-", 1)[1] if "-" in name else name


def _space_kind(name: str, long_name: str) -> str:
    for kind, rx in SPACE_KINDS:
        if rx.search(name) or rx.search(long_name):
            return kind
    return "room"


def _box_tris(lo, hi):
    """12 outward-facing triangles of an axis-aligned box."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return v[np.array(f)]


def _to_world(T, Mw):
    """Triangles in a placement's local frame -> world, keeping them outward-facing under a mirror."""
    W = T @ Mw[:3, :3].T + Mw[:3, 3]
    return W[:, [0, 2, 1]] if np.linalg.det(Mw[:3, :3]) < 0 else W


def _to_local(T, Mw):
    L = (T - Mw[:3, 3]) @ Mw[:3, :3]
    return L[:, [0, 2, 1]] if np.linalg.det(Mw[:3, :3]) < 0 else L


def _placement(p, memo):
    """World matrix of an IfcLocalPlacement, memoising the shared storey and building placements."""
    if p is None:
        return np.eye(4)
    m = memo.get(p.id())
    if m is None:
        if p.is_a("IfcLocalPlacement"):
            m = _placement(p.PlacementRelTo, memo) @ uplace.get_axis2placement(p.RelativePlacement)
        else:
            m = np.array(uplace.get_local_placement(p), float)
        memo[p.id()] = m
    return m


def door_record(d, n, mesh, memo=None):
    """Opening of one passable door in its own frame (x along the opening 0..w, y across, z up from the sill).

    The jamb is (NominalWidth - ClearWidth) / 2 from the Navigation pset, else IfcOpenShell's default lining.
    """
    Mw = np.array(_placement(d.ObjectPlacement, {} if memo is None else memo), float)
    loc = (mesh["verts"] - Mw[:3, 3]) @ Mw[:3, :3]
    nominal = float(n.get("NominalWidth") or d.OverallWidth or (np.ptp(loc[:, 0]) - 2 * 0.025))
    clear = n.get("ClearWidth")
    if clear is not None and 0.0 < float(clear) < nominal - 1e-6:
        jamb, src = (nominal - float(clear)) / 2, "pset"
    else:
        jamb, src = DEFAULT_JAMB, "default"
    h = float(d.OverallHeight or loc[:, 2].max())
    y0, y1 = float(loc[:, 1].min()), float(loc[:, 1].max())
    return dict(guid=d.GlobalId, name=d.Name or "", kind=n.get("DoorKind", ""), M=Mw, w=nominal,
                clear=round(nominal - 2 * jamb, 3), jamb=jamb, jamb_src=src, y0=y0, y1=y1, yc=(y0 + y1) / 2, h=h,
                z=float(Mw[2, 3]), storey=mesh["storey"], bbox=(mesh["verts"].min(0), mesh["verts"].max(0)),
                sides=tuple(x for x in (n.get("FromSpace"), n.get("ToSpace")) if x))


def jamb_tris(door):
    """World triangles of the two jamb boxes that stay when the leaf of a passable door is left out."""
    j, w, h = door["jamb"], door["w"], door["h"]
    lo_y, hi_y = door["y0"], max(door["y1"], door["y0"] + 0.05)
    boxes = [_box_tris((0.0, lo_y, 0.0), (j, hi_y, h)), _box_tris((w - j, lo_y, 0.0), (w, hi_y, h))]
    return _to_world(np.concatenate(boxes), door["M"])


@dataclass
class Model:
    file: object
    meshes: dict
    storeys: dict             # name -> world elevation
    spaces: list              # dict(guid, name, long_name, storey, z, poly, kind, fid)
    zones: list               # dict(name, type, guids)
    lifts: list               # dict(k, name, served)
    landing: list             # dict(guid, name, level, k, bbox, z)
    main_doors: list          # dict(guid, name, unit, bbox, z, storey, sides)
    restricted: set           # space names / face ids only behind non-passable doors
    obstacles: list           # guids
    bounds: tuple             # (lo, hi) of the building's own elements
    building: str = ""
    doors: list = None        # passable doors with a mesh: door_record()


def read_model(ifc_path) -> Model:
    f, meshes = meshcache.load(ifc_path)
    storeys = _storeys(f)
    psets = {}

    def nav(el):
        if el.GlobalId not in psets:
            psets[el.GlobalId] = uel.get_psets(el).get("Navigation", {})
        return psets[el.GlobalId]

    open_doors, passable_ref, closed_ref = set(), set(), set()
    landing, main_doors, doors, memo = [], [], [], {}
    roof_access = False
    for d in sorted(f.by_type("IfcDoor"), key=lambda e: e.GlobalId):
        n = nav(d)
        passable = bool(n.get("Passable", True))
        refs = {n.get("FromSpace"), n.get("ToSpace")} - {None}
        (passable_ref if passable else closed_ref).update(refs)
        if passable:
            open_doors.add(d.GlobalId)
        m = meshes.get(d.GlobalId)
        if m is None:
            continue
        if passable and d.ObjectPlacement is not None:
            doors.append(door_record(d, n, m, memo))
        bb = (m["verts"].min(0), m["verts"].max(0))
        roof_access |= passable and (n.get("DoorKind") == "roof" or (d.Name or "").lower().endswith("roof door"))
        mm = LANDING_RE.match((d.Name or "").strip())
        if not passable and (mm or n.get("DoorKind") == "lift"):
            landing.append(dict(guid=d.GlobalId, name=d.Name, level=mm["level"] if mm else m["storey"],
                                k=int(mm["k"]) if mm and mm["k"] else 1, bbox=bb, z=float(bb[0][2])))
        elif n.get("DoorKind") == "main" or (d.Name or "").lower().endswith("main door"):
            nm = d.Name or ""
            main_doors.append(dict(guid=d.GlobalId, name=nm, unit=nm.split(" ")[0] if nm.startswith("#") else "",
                                   bbox=bb, z=float(bb[0][2]), storey=m["storey"],
                                   sides=tuple(x for x in (n.get("FromSpace"), n.get("ToSpace")) if x)))
    main_doors.sort(key=lambda x: (x["name"], x["guid"]))
    restricted = closed_ref - passable_ref
    lifts = []
    for t in f.by_type("IfcTransportElement"):
        n = nav(t)
        if not (n.get("Elevator") or (t.PredefinedType or "") == "ELEVATOR"):
            continue
        mm = LIFT_NO_RE.search(t.Name or "")
        served = [s.strip() for s in str(n.get("ServesLevels", "")).split(",") if s.strip()]
        lifts.append(dict(k=int(mm.group(1)) if mm else len(lifts) + 1, name=t.Name, served=served or list(storeys)))
    if not lifts:          # landing doors without a lift element: one lift per door group, serving its doors
        for k in sorted({d["k"] for d in landing}):
            lifts.append(dict(k=k, name=f"Lift {k}", served=sorted({d["level"] for d in landing if d["k"] == k})))

    spaces = []
    top_storey = next(reversed(storeys)) if storeys else None
    for sp in f.by_type("IfcSpace"):
        d = meshes.get(sp.GlobalId)
        if d is None:
            continue
        label = f"{sp.Name or ''} {sp.LongName or ''}"
        if EXCLUDED_SPACES.search(label):
            continue
        poly, z = _flat_faces(d, "bottom")
        spaces.append(dict(guid=sp.GlobalId, name=sp.Name or "", long_name=sp.LongName or sp.Name or "",
                           storey=d["storey"], z=z, poly=poly, kind=_space_kind(sp.Name or "", sp.LongName or ""),
                           fid=_face_id(sp.Name or "")))
    if roof_access and top_storey and not any(s["storey"] == top_storey for s in spaces):
        # no roof deck space but a roof door (the legacy block): the roof slab top is the target
        roofs = [s for s in f.by_type("IfcSlab") if (s.PredefinedType or "") == "ROOF" and s.GlobalId in meshes
                 and meshes[s.GlobalId]["storey"] == top_storey]
        best = None
        for s in roofs:
            poly, z = _flat_faces(meshes[s.GlobalId], "top")
            if abs(z - storeys[top_storey]) < 0.3 and (best is None or poly.area > best[0].area):
                best = (poly, z, s)
        if best is not None:
            spaces.append(dict(guid=best[2].GlobalId, name=f"{top_storey} roof", long_name="Roof deck",
                               storey=top_storey, z=best[1], poly=best[0], kind="roof", fid="ROOF"))
    zones = []
    for zn in f.by_type("IfcZone"):
        guids = [o.GlobalId for rel in zn.IsGroupedBy for o in rel.RelatedObjects if o.is_a("IfcSpace")]
        ps = uel.get_psets(zn)
        props = ps.get("SampleCity_Flat", {})
        zones.append(dict(name=zn.Name, type=props.get("FlatType", ""), long_name=zn.LongName or "",
                          storey=props.get("Storey", ""), guids=guids, flat="SampleCity_Flat" in ps))
    obstacles = [g for g, d in meshes.items() if d["cls"] != "IfcSpace" and g not in open_doors]
    own = [d["verts"] for g, d in meshes.items() if d["storey"] in storeys and d["cls"] != "IfcSpace"]
    allv = np.concatenate(own) if own else np.concatenate([d["verts"] for d in meshes.values()])
    return Model(f, meshes, storeys, spaces, zones, lifts, landing, main_doors, restricted, obstacles,
                 (allv.min(0), allv.max(0)), next((b.Name for b in f.by_type("IfcBuilding")), "") or "", doors)


# ----------------------------------------------------------------------------- rendering
def render_level(path, g: Grid, layers, title, lines=(), outlines=(), max_px=2400):
    """PNG plan of one level. layers: (colour key, 2D bool mask [x, y]) painted in order."""
    from PIL import Image, ImageDraw
    from estate.draw.raster import font
    img = np.empty((g.ny, g.nx, 3), np.uint8)
    img[:] = COLOURS["bg"]
    for key, mask in layers:
        img[mask.T] = COLOURS[key]
    img = img[::-1]
    s = int(max(1, min(4, max_px // max(g.nx, g.ny))))
    top, pad = 30 + 16 * len(lines), 24
    im = Image.fromarray(img).resize((g.nx * s, g.ny * s), Image.NEAREST)
    canvas = Image.new("RGB", (g.nx * s + 2 * pad, g.ny * s + top + 2 * pad), COLOURS["bg"])
    canvas.paste(im, (pad, top + pad))
    dr = ImageDraw.Draw(canvas)
    x0, y1 = g.lo[0], g.lo[1] + g.ny * g.P

    def P(x, y):
        return pad + (x - x0) / g.P * s, top + pad + (y1 - y) / g.P * s

    f9 = font(11)
    for gx in range(int(math.ceil(x0 / 5)) * 5, int(x0 + g.nx * g.P) + 1, 5):
        X, _ = P(gx, 0)
        dr.line([(X, top + pad - 4), (X, top + pad)], fill=(120, 120, 120))
        dr.text((X, top + pad - 6), f"{gx}", fill=(90, 90, 90), font=f9, anchor="mb")
    for gy in range(int(math.ceil(g.lo[1] / 5)) * 5, int(y1) + 1, 5):
        _, Y = P(0, gy)
        dr.line([(pad - 4, Y), (pad, Y)], fill=(120, 120, 120))
        dr.text((pad - 6, Y), f"{gy}", fill=(90, 90, 90), font=f9, anchor="rm")
    for poly in outlines:
        for part in getattr(poly, "geoms", [poly]):
            if part.geom_type == "Polygon" and not part.is_empty:
                dr.line([P(*c) for c in part.exterior.coords], fill=COLOURS["outline"], width=2)
    dr.text((pad, 8), title, fill=(30, 30, 30), font=font(14))
    for i, ln in enumerate(lines):
        dr.text((pad, 28 + 16 * i), ln, fill=(70, 70, 70), font=font(12))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def default_levels(storeys: dict) -> list:
    names = list(storeys)
    if not names:
        return []
    res = [n for n in names if n not in (names[0], names[-1])]
    typ = "L5" if "L5" in res else (res[len(res) // 2] if res else None)
    return [n for n in dict.fromkeys([names[0], typ, names[-1]]) if n]


# ----------------------------------------------------------------------------- the check
@dataclass
class CellIndex:
    """Walkable cells sorted by (k, ix), so the cells of a box are a few contiguous runs."""
    order: np.ndarray
    key: np.ndarray
    nx: int

    @classmethod
    def of(cls, cells: Cells, g: Grid):
        key = cells.k.astype(np.int64) * g.nx + cells.ix
        order = np.argsort(key, kind="stable")
        return cls(order, key[order], g.nx)

    def box(self, ka, kb, i0, i1):
        ks = np.arange(ka, kb + 1, dtype=np.int64) * self.nx
        lo = np.searchsorted(self.key, ks + i0)
        hi = np.searchsorted(self.key, ks + i1, side="right")
        idx, _ = _expand(lo, np.maximum(hi - lo, 0))
        return self.order[idx]


def _cells_near(cells: Cells, g: Grid, bbox, z, pad=0.8, below=0.25, above=0.45, index: CellIndex = None):
    """Indices of walkable cells within `pad` of an xy box whose floor is near height z."""
    (bx0, by0, _), (bx1, by1, _) = bbox
    idx = index.box(g.kz(z - below), g.kz(z + above), max(0, g.ix(bx0 - pad)), min(g.nx - 1, g.ix(bx1 + pad)))
    x, y = g.cx(cells.ix[idx]), g.cy(cells.iy[idx])
    m = (x >= bx0 - pad) & (x <= bx1 + pad) & (y >= by0 - pad) & (y <= by1 + pad)
    return idx[m]


# ----------------------------------------------------------------------------- door bands
@dataclass
class Tris:
    """Obstacle triangles, contiguous per element, with element bounds for local queries."""
    T: np.ndarray
    start: np.ndarray
    count: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def of(cls, T, elem):
        count = np.bincount(elem, minlength=int(elem.max()) + 1 if len(elem) else 0)
        start = np.cumsum(count) - count
        lo, hi = np.full((len(count), 3), np.inf), np.full((len(count), 3), -np.inf)
        has = count > 0
        if has.any():
            lo[has] = np.minimum.reduceat(T.min(axis=1), start[has])
            hi[has] = np.maximum.reduceat(T.max(axis=1), start[has])
        return cls(T, start, count, lo, hi)

    def near(self, lo, hi):
        sel = np.nonzero(np.all(self.hi >= lo, axis=1) & np.all(self.lo <= hi, axis=1))[0]
        idx, owner = _expand(self.start[sel], self.count[sel])
        return self.T[idx], sel[owner]


def door_band(door, tris: Tris, radii, height, step, level_tol, P=DOOR_VOXEL, cache=None, ground=None):
    """Can the agent pass through this door opening? Tested on P voxels in the door's own frame.

    The band spans the opening plus 0.15 m along the wall, and the wall plus the agent radius and a margin
    across it. Every obstacle within it (walls, jambs, slabs, furniture) is voxelised and the same walkability
    test runs. The door is passable when a walkable cell clearly on one side (more than WALL_HALF from the door
    axis) and one on the other side fall into one component, near the sill level. Returns ({radius key:
    dict(passable, step_free, zA, zB, zA_sf, zB_sf)}, shared) with the side floor heights relative to the sill.
    Doors with the same surroundings (stacked storeys) share one result through ``cache``. A door near street
    level (``ground``) gets the same virtual ground plane as the building grid, so an exit to the street counts.
    """
    Mw, w, yc = door["M"], door["w"], door["yc"]
    gz = ground - door["z"] if ground is not None and abs(ground - door["z"]) <= 0.3 else None
    dy = WALL_HALF + max(radii) + 0.15
    blo, bhi = np.array([-0.15, yc - dy, -0.45]), np.array([w + 0.15, yc + dy, height + 0.3])
    corners = np.stack(np.meshgrid(*zip(blo, bhi), indexing="ij"), -1).reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
    T, owner = tris.near(corners.min(0) - 1e-3, corners.max(0) + 1e-3)
    T = _to_local(T, Mw)
    key = None
    if cache is not None:      # keyed before clipping: subdivision midpoints would round differently per storey
        q = np.round(T.reshape(len(T), 9) * 1e4).astype(np.int64)
        q = q[np.lexsort(q.T[::-1])] if len(q) else q
        key = hashlib.sha1(q.tobytes() + np.round([w, yc, 99.0 if gz is None else gz], 4).tobytes()).hexdigest()
        if key in cache:
            return cache[key], True
    T, owner = _split_big(T, np.array([blo[0], blo[1], -np.inf]), np.array([bhi[0], bhi[1], np.inf]), 0.5, owner)
    order = np.argsort(owner, kind="stable")
    T, el = T[order], np.unique(owner[order], return_inverse=True)[1]
    g = Grid.around(blo, bhi, P)
    S, up, _ = voxelise(T, el, g, gz)
    H, St = int(round(height / P)), int(round(step / P))
    out = {}
    for r in radii:
        res = dict(passable=False, step_free=False, zA=None, zB=None, zA_sf=None, zB_sf=None)
        k, ix, iy = walkable_cells(S, g, voxel_radius(r, P), H, St)
        if len(k):
            c = connect(k, ix, iy, g, St, up)
            x, y = g.cx(c.ix), g.cy(c.iy)
            near = (x >= -0.05) & (x <= w + 0.05) & (np.abs(c.fz) <= 0.2)
            A, B = np.nonzero(near & (y < yc - WALL_HALF))[0], np.nonzero(near & (y > yc + WALL_HALF))[0]
            sf = np.abs(c.fz[c.ea] - c.fz[c.eb]) <= level_tol
            for tag, ea, eb in (("", c.ea, c.eb), ("_sf", c.ea[sf], c.eb[sf])):
                lab = components(len(c.key), ea, eb)
                common = np.intersect1d(lab[A], lab[B])
                if len(common):
                    res["step_free" if tag else "passable"] = True
                    for side, idx in (("zA", A), ("zB", B)):
                        res[side + tag] = float(np.median(c.fz[idx[np.isin(lab[idx], common)]]))
        out[f"{r:.2f}"] = res
    if key is not None:
        cache[key] = out
    return out, False


def _door_portal(cells: Cells, g: Grid, door, za, zb, tol, index: CellIndex):
    """Edges chaining the coarse cells in front of a door on both sides (floors at za and zb), or None."""
    c = _cells_near(cells, g, door["bbox"], door["z"], WALL_HALF + 0.8, index=index)
    if not len(c):
        return None
    Mw = door["M"]
    loc = (np.stack([g.cx(cells.ix[c]), g.cy(cells.iy[c])], 1) - Mw[:2, 3]) @ Mw[:2, :2]
    lx, ly = loc[:, 0], loc[:, 1] - door["yc"]
    inx = (lx >= -0.05) & (lx <= door["w"] + 0.05) & (np.abs(ly) <= WALL_HALF + 0.6)
    fz = cells.fz[c]
    a = c[inx & (ly < -0.02) & (np.abs(fz - za) <= tol)]
    b = c[inx & (ly > 0.02) & (np.abs(fz - zb) <= tol)]
    if not len(a) or not len(b):
        return None
    fc = np.concatenate([a, b])
    return fc[:-1], fc[1:]


def _flat_side_cells(cells, g, d, spaces_by_ref, zone_polys, index: CellIndex):
    """Walkable cells on the flat side of a main door: inside its entry room (Navigation.FromSpace/ToSpace),
    else inside any space of the flat on that storey. Returns (cells, how)."""
    near = _cells_near(cells, g, d["bbox"], d["z"], 1.0, index=index)
    polys, how = [], "near door"
    for ref in d["sides"]:
        for s in spaces_by_ref.get((d["storey"], ref), []):
            if d["unit"] and s["name"].startswith(d["unit"] + " "):
                polys.append(s["poly"])
    if polys:
        how = "entry room"
    elif zone_polys.get((d["storey"], d["unit"])):
        polys, how = zone_polys[(d["storey"], d["unit"])], "flat spaces"
    if not polys or not len(near):
        return near, how
    inner = shapely.union_all([p.buffer(-0.05) for p in polys])
    return near[shapely.contains_xy(inner, g.cx(cells.ix[near]), g.cy(cells.iy[near]))], how


def check_building(ifc_path, out_dir, radius=0.30, height=1.8, step=0.4, voxel=0.10, margin=2.5,
                   ground=None, levels=None, starts=None, level_tol=0.015, trace_memory=False, png=True,
                   log=None, door_voxel=DOOR_VOXEL):
    """Walk test of one building IFC. Writes <out_dir>/<stem>_nav.json and <out_dir>/nav/<level>_r<cm>.png.

    radius may be one value or a list (the first one decides ``ok``); each is rounded up to whole voxels.
    ground: street level (default: the lowest storey). starts: optional world (x, y) start points instead of
    the automatic margin ring. door_voxel: voxel of the door-band test (0 or None: coarse grid only).
    Returns the report dict.
    """
    t_all = time.time()
    if trace_memory:
        tracemalloc.start()
    say = log or (lambda *_: None)
    ifc_path, out_dir = Path(ifc_path), Path(out_dir)
    radii = [float(r) for r in (radius if isinstance(radius, (list, tuple)) else [radius])]
    timing = {}

    t = time.time()
    M = read_model(ifc_path)
    timing["read"] = time.time() - t
    ground = float(ground if ground is not None else (next(iter(M.storeys.values())) if M.storeys else 0.0))
    blo, bhi = M.bounds
    lo = np.array([blo[0] - margin, blo[1] - margin, min(blo[2], ground) - 1.0])
    hi = np.array([bhi[0] + margin, bhi[1] + margin, bhi[2] + height + 0.5])
    g = Grid.around(lo, hi, voxel)
    say(f"  grid {g.nx} x {g.ny} x {g.nz} = {g.nx * g.ny * g.nz / 1e6:.0f} M cells")

    t = time.time()
    obst = [M.meshes[gid]["verts"][M.meshes[gid]["faces"]] for gid in M.obstacles] + [jamb_tris(d) for d in M.doors]
    T = np.concatenate(obst) if obst else np.zeros((0, 3, 3))
    elem = np.repeat(np.arange(len(obst)), [len(x) for x in obst])
    n_obst = len(M.obstacles)
    del obst
    M.meshes = M.file = None              # everything needed is extracted; free the IFC and the mesh dict
    gc.collect()
    S, up, n_virtual = voxelise(T, elem, g, ground)
    timing["voxelise"] = time.time() - t
    say(f"  voxelised {n_obst} elements and {len(M.doors)} door linings in {timing['voxelise']:.1f} s, "
        f"solid {S.mean() * 100:.1f}%")

    t = time.time()
    band, n_shared = [], 0
    if door_voxel:
        tris, cache = Tris.of(T, elem), {}
        for d in M.doors:
            res, shared = door_band(d, tris, radii, height, step, level_tol, door_voxel, cache, ground)
            band.append(res)
            n_shared += shared
        del tris, cache
        timing["door_bands"] = time.time() - t
        say(f"  door bands: {len(M.doors)} passable doors at {door_voxel} m, {len(M.doors) - n_shared} tested "
            f"({n_shared} share a stacked result) in {timing['door_bands']:.1f} s")
    del T, elem
    gc.collect()

    H, St = int(round(height / voxel)), int(round(step / voxel))
    levels = levels or default_levels(M.storeys)

    def kslice(z):
        return min(g.nz - 1, max(0, int(math.floor((z - g.lo[2]) / voxel))))

    walls = {}
    for lv in levels:
        if lv in M.storeys:
            z = M.storeys[lv]
            hi_ = S[kslice(z + 1.2)].copy()                       # the legacy cut: walls at 1.2 m
            walls[lv] = (hi_, S[kslice(z + 0.45):kslice(z + 1.2)].any(axis=0) & ~hi_)   # parapets, sills, rails

    warnings = []
    eff = {}
    for r in radii:
        eff[f"{r:.2f}"] = round(voxel_radius(r, voxel) * voxel, 4)
        if abs(eff[f"{r:.2f}"] - r) > 1e-6:
            warnings.append(f"radius {r} m is not a multiple of the {voxel} m voxel: tested as {eff[f'{r:.2f}']} m "
                            f"(rounded up)")
    for w_ in warnings:
        say(f"  warning: {w_}")
    jamb_src = {}
    for d in M.doors:
        jamb_src[d["jamb_src"]] = jamb_src.get(d["jamb_src"], 0) + 1
    report = {"file": str(ifc_path), "stem": ifc_path.stem,
              "building": M.building,
              "agent": {"radius": radii, "effective_radius": eff, "height": height, "step": step, "voxel": voxel,
                        "level_tol": level_tol, "door_voxel": door_voxel or None,
                        "disc_test": "every voxel whose centre lies within the radius of the cell centre is free; "
                                     "radii are rounded up to whole voxels"},
              "doors": {"passable_with_mesh": len(M.doors), "jambs_from": jamb_src,
                        "narrowest_clear_m": min((d["clear"] for d in M.doors), default=None),
                        "band_tests": len(M.doors) - n_shared if door_voxel else 0, "band_shared": n_shared},
              "grid": {"lo": [round(v, 3) for v in g.lo], "shape": [g.nx, g.ny, g.nz], "margin": margin,
                       "ground": ground, "virtual_ground_columns": n_virtual},
              "storeys": M.storeys, "warnings": warnings, "results": {}}
    spaces_by_ref, zone_polys = {}, {}
    for s in M.spaces:
        for ref in {s["name"], s["fid"]}:
            spaces_by_ref.setdefault((s["storey"], ref), []).append(s)
    by_guid_space = {s["guid"]: s for s in M.spaces}
    for zn in M.zones:
        if zn["flat"]:
            for gid in zn["guids"]:
                s = by_guid_space.get(gid)
                if s is not None and not s["poly"].is_empty:
                    zone_polys.setdefault((s["storey"], zn["name"]), []).append(s["poly"])

    for r in radii:
        tr = time.time()
        rk = f"{r:.2f}"
        r_vox = voxel_radius(r, voxel)
        k, ix, iy = walkable_cells(S, g, r_vox, H, St)
        tw = time.time() - tr
        cells = connect(k, ix, iy, g, St, up)
        del k, ix, iy
        n = len(cells.key)
        dz = np.abs(cells.fz[cells.ea] - cells.fz[cells.eb])
        sf = dz <= level_tol
        index = CellIndex.of(cells, g)

        # door portals: doors the band test passes join the coarse cells in front of them on both sides
        da, db, da_sf, db_sf = [], [], [], []
        door_rep = dict(fit=0, fit_step_free=0, linked=0, linked_step_free=0, impassable=0, impassable_examples=[],
                        no_cells_beside=0, no_cells_examples=[])
        for d, res in zip(M.doors, band):
            b = res[rk]
            if not b["passable"]:
                door_rep["impassable"] += 1
                if len(door_rep["impassable_examples"]) < 12:
                    door_rep["impassable_examples"].append(f"{d['name']} (clear {d['clear']:.2f} m)")
                continue
            door_rep["fit"] += 1
            door_rep["fit_step_free"] += b["step_free"]
            p = _door_portal(cells, g, d, d["z"] + b["zA"], d["z"] + b["zB"], 0.05, index)
            if p is None:
                door_rep["no_cells_beside"] += 1
                if len(door_rep["no_cells_examples"]) < 12:
                    door_rep["no_cells_examples"].append(d["name"])
                continue
            door_rep["linked"] += 1
            da.append(p[0])
            db.append(p[1])
            if b["step_free"]:
                p = _door_portal(cells, g, d, d["z"] + b["zA_sf"], d["z"] + b["zB_sf"], level_tol, index)
                if p is not None:
                    door_rep["linked_step_free"] += 1
                    da_sf.append(p[0])
                    db_sf.append(p[1])
        lab = components(n, np.concatenate([cells.ea] + da), np.concatenate([cells.eb] + db))

        # start cells: the street on the margin ring, or explicit points
        if starts:
            st_idx = []
            for (sx, sy) in starts:
                m = (cells.ix == g.ix(sx)) & (cells.iy == g.iy(sy))
                c = np.nonzero(m)[0]
                if len(c):
                    st_idx.append(c[np.argmin(np.abs(cells.fz[c] - ground))])
            st_idx = np.array(st_idx, np.int64)
        else:
            ring = max(1, int(round(0.5 / voxel)))
            edge = ((cells.ix < ring) | (cells.ix >= g.nx - ring) | (cells.iy < ring) | (cells.iy >= g.ny - ring))
            st_idx = np.nonzero(edge & (np.abs(cells.fz - ground) <= 0.6))[0]

        # lift portals for the step-free graph
        pa, pb, lift_rep = [], [], []
        for lf in sorted(M.lifts, key=lambda d: d["k"]):
            doors = [d for d in M.landing if d["k"] == lf["k"] and d["level"] in lf["served"]]
            front, linked = [], []
            for d in sorted(doors, key=lambda d: d["z"]):
                c = _cells_near(cells, g, d["bbox"], d["z"], 0.9, index=index)
                if len(c):
                    front.append(c)
                    linked.append(d["level"])
            if front:
                fc = np.concatenate(front)
                pa.append(fc[:-1])
                pb.append(fc[1:])
            lift_rep.append(dict(name=lf["name"], served=lf["served"], landing_doors=len(doors), linked_levels=linked,
                                 missing_levels=[lv for lv in lf["served"] if lv not in linked],
                                 _front=np.concatenate(front) if front else np.zeros(0, np.int64)))
        lab_sf = components(n, np.concatenate([cells.ea[sf]] + pa + da_sf), np.concatenate([cells.eb[sf]] + pb + db_sf))
        del da, db, da_sf, db_sf
        street = np.zeros(n, bool)
        street_sf = np.zeros(n, bool)
        if len(st_idx):
            street = np.isin(lab, np.unique(lab[st_idx]))
            street_sf = np.isin(lab_sf, np.unique(lab_sf[st_idx]))
        for lr in lift_rep:
            fc = lr.pop("_front")
            lr["reachable_from_street"] = bool(street[fc].any()) if len(fc) else False
            lr["step_free_from_street"] = bool(street_sf[fc].any()) if len(fc) else False

        # spaces
        sp_res = []
        for s in M.spaces:
            c = np.zeros(0, np.int64)
            if not s["poly"].is_empty:
                bx = s["poly"].bounds
                c = _cells_near(cells, g, (np.array([bx[0], bx[1], 0]), np.array([bx[2], bx[3], 0])), s["z"], 0.0,
                                index=index)
            if len(c):
                inner = s["poly"].buffer(-0.05)
                c = c[shapely.contains_xy(inner, g.cx(cells.ix[c]), g.cy(cells.iy[c]))]
            restricted = s["name"] in M.restricted or s["fid"] in M.restricted
            sp_res.append(dict(name=s["name"], long_name=s["long_name"], storey=s["storey"], kind=s["kind"],
                               reachable=bool(street[c].any()) if len(c) else False,
                               step_free=bool(street_sf[c].any()) if len(c) else False,
                               walkable_cells=int(len(c)), reachable_cells=int(street[c].sum()) if len(c) else 0,
                               restricted=restricted, guid=s["guid"]))
        by_guid = {s["guid"]: r_ for s, r_ in zip(M.spaces, sp_res)}

        # main doors and flats: the door is crossed when the street component continues into the entry room
        door_res = {}
        for d in M.main_doors:
            c, how = _flat_side_cells(cells, g, d, spaces_by_ref, zone_polys, index)
            door_res.setdefault(d["unit"], []).append(dict(name=d["name"], reachable=bool(street[c].any()),
                                                           step_free=bool(street_sf[c].any()), cells=int(len(c)),
                                                           checked_in=how))
        flats, other_zones = [], []
        for zn in M.zones:
            rs = [by_guid[gid] for gid in zn["guids"] if gid in by_guid]
            if not zn["flat"]:
                other_zones.append(dict(name=zn["name"], long_name=zn["long_name"], spaces=len(rs),
                                        reachable=bool(rs) and all(x["reachable"] for x in rs if not x["restricted"]),
                                        step_free_spaces=sum(x["step_free"] for x in rs)))
                continue
            doors = door_res.get(zn["name"], [])
            flats.append(dict(unit=zn["name"], type=zn["type"], storey=zn["storey"] or (rs[0]["storey"] if rs else ""),
                              spaces=len(rs), reachable_spaces=sum(x["reachable"] for x in rs),
                              step_free_spaces=sum(x["step_free"] for x in rs),
                              reachable=bool(rs) and all(x["reachable"] for x in rs),
                              main_door=doors[0]["name"] if doors else None,
                              main_door_reachable=any(x["reachable"] for x in doors),
                              main_door_step_free=any(x["step_free"] for x in doors),
                              main_door_checked_in=doors[0]["checked_in"] if doors else None))
        checked = [x for x in sp_res if not x["restricted"]]
        bad = [x for x in checked if not x["reachable"]]
        roof = [x for x in sp_res if x["storey"] == (next(reversed(M.storeys)) if M.storeys else None)]
        by_room, by_kind = {}, {}
        for x in bad:
            by_room.setdefault(x["long_name"], []).append(x["name"])
        for x in checked:
            kk = by_kind.setdefault(x["kind"], dict(spaces=0, reachable=0, step_free=0))
            kk["spaces"] += 1
            kk["reachable"] += x["reachable"]
            kk["step_free"] += x["step_free"]
        main_checks = {}
        for v in door_res.values():
            for x in v:
                main_checks[x["checked_in"]] = main_checks.get(x["checked_in"], 0) + 1
        summary = dict(
            walkable_cells=int(n), components=int(len(np.unique(lab))), start_cells=int(len(st_idx)),
            spaces=len(checked), reachable=len(checked) - len(bad), unreachable=len(bad),
            unreachable_by_room={kk: dict(count=len(v), example=v[0]) for kk, v in sorted(by_room.items())},
            restricted=[x["name"] for x in sp_res if x["restricted"]],
            roof_reachable=any(x["reachable"] for x in roof) if roof else None,     # None: no roof access
            flats=len(flats), flats_reachable=sum(fl["reachable"] for fl in flats),
            flats_main_door_step_free=sum(fl["main_door_step_free"] for fl in flats),
            main_doors_checked_in=main_checks,
            zones=len(other_zones), zones_reachable=sum(z["reachable"] for z in other_zones),
            spaces_step_free=sum(x["step_free"] for x in checked), spaces_by_kind=by_kind,
            lifts=len(lift_rep), doors_impassable=door_rep["impassable"])
        summary["ok"] = bool(summary["start_cells"] and not bad and summary["roof_reachable"] is not False)
        # step-free: every flat's main door; for buildings without flats, every (unrestricted) space
        summary["step_free_ok"] = (summary["flats_main_door_step_free"] == len(flats) if flats
                                   else bool(checked) and summary["spaces_step_free"] == len(checked))

        # maps
        if png:
            order_k = np.argsort(cells.k, kind="stable")
            k_sorted = cells.k[order_k].astype(np.int64)
            small = np.bincount(lab, minlength=n)[lab] < FRAGMENT_M2 / voxel ** 2
            for lv, (wall, low) in walls.items():
                z = M.storeys[lv]
                ka, kb = g.kz(z - 0.25), g.kz(z + 0.45)
                sel = order_k[np.searchsorted(k_sorted, ka):np.searchsorted(k_sorted, kb, side="right")]
                masks = {key: np.zeros((g.nx, g.ny), bool) for key in ("reach", "cut", "frag")}
                on_floor = np.abs(cells.fz[sel] - z) <= 0.15        # raised tops (tables, sills, rails) are pale
                cut = ~street[sel] & ~small[sel] & on_floor
                for key, m in (("reach", street[sel]), ("frag", ~street[sel] & ~cut), ("cut", cut)):
                    masks[key][cells.ix[sel[m]], cells.iy[sel[m]]] = True
                masks["cut"] &= ~masks["reach"]
                masks["frag"] &= ~masks["reach"] & ~masks["cut"]
                lv_bad = [s["poly"] for s, x in zip(M.spaces, sp_res)
                          if x["storey"] == lv and not x["reachable"] and not x["restricted"]]
                n_lv = [x for x in checked if x["storey"] == lv]
                render_level(out_dir / "nav" / f"{lv}_r{int(round(r * 100))}.png", g,
                             [("low", low), ("wall", wall), ("reach", masks["reach"]), ("frag", masks["frag"]),
                              ("cut", masks["cut"])],
                             f"{report['building'] or ifc_path.stem} {lv} (z {z:.2f} m): reachable from the street "
                             f"(green), walkable but cut off (red)",
                             [f"agent r={r:.2f} m (tested {eff[rk]} m), h={height} m, step {step} m, voxel {voxel} m, "
                              f"doors at {door_voxel or voxel} m with linings; "
                              f"{sum(x['reachable'] for x in n_lv)}/{len(n_lv)} spaces on this level reachable",
                              f"dark: solid at +1.2 m; grey: low obstacles +0.45..1.2 m; pale: cut off but raised "
                              f"(furniture, sills, rail tops) or < {FRAGMENT_M2} m2; red outline: unreachable space"],
                             lv_bad)
        timing[f"r{int(round(r * 100))}"] = time.time() - tr
        say(f"  r={r:.2f}: {n} walkable cells (bands {tw:.1f} s), {summary['reachable']}/{summary['spaces']} spaces, "
            f"roof {ROOF_WORD[summary['roof_reachable']]}, flats {summary['flats_reachable']}/{len(flats)}, "
            f"main doors step-free {summary['flats_main_door_step_free']}/{len(flats)}, doors impassable "
            f"{door_rep['impassable']}/{len(M.doors)} ({time.time() - tr:.1f} s)")
        report["results"][rk] = dict(summary=summary, agent_effective_radius=eff[rk], doors=door_rep, lifts=lift_rep,
                                     spaces=[{kk: v for kk, v in x.items() if kk != "guid"} for x in sp_res],
                                     flats=flats, zones=other_zones)
        del cells, lab, lab_sf, street, street_sf

    del S
    timing["total"] = time.time() - t_all
    report["timing_s"] = {kk: round(v, 2) for kk, v in timing.items()}
    report["memory"] = {"peak_working_set_mb": round(peak_memory_mb() or 0.0, 1)}
    if trace_memory:
        report["memory"]["tracemalloc_peak_mb"] = round(tracemalloc.get_traced_memory()[1] / 2 ** 20, 1)
        tracemalloc.stop()
    first = report["results"][f"{radii[0]:.2f}"]["summary"]
    report["ok"] = first["ok"]
    report["step_free_ok"] = first["step_free_ok"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{ifc_path.stem}_nav.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    return report
