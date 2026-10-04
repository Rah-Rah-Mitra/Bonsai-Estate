"""Wall specifications, placement matrices and opening snapping (lifted from legacy/toolkit/hdb_block.py)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient


def frame(origin, u) -> np.ndarray:
    """4x4 placement: x along u (2D), z up, at origin (x, y, z)."""
    x = np.array([u[0], u[1], 0.0], dtype=float)
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, 1.0])
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, np.cross(z, x), z, origin
    return m


def translate(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def rotate_z(quarter_turns: int, about=(0.0, 0.0)) -> np.ndarray:
    """Exact rotation by k * 90 degrees about a point (keeps placements right-handed, det = +1)."""
    k = quarter_turns % 4
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]
    m = np.eye(4)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    ax, ay = about
    m[0, 3] = ax - c * ax + s * ay
    m[1, 3] = ay - s * ax - c * ay
    return m


@dataclass
class Opening:
    s0: float
    s1: float
    sill: float
    height: float
    kind: str           # "door" | "window"
    name: str
    door_kind: str = ""
    swing: int = 1      # +1: leaf opens to the wall's left (+n side), -1: to the right
    meta: dict = field(default_factory=dict)


@dataclass
class WallSpec:
    p0: np.ndarray
    u: np.ndarray
    length: float
    t: float
    lo: float           # local-y start of the thickness band (0 = left aligned, -t/2 = centred)
    z0: float
    h: float
    name: str
    type_key: str
    external: bool
    storey: object
    climbable: bool = False
    openings: list = field(default_factory=list)
    faces: tuple = ()   # (left face id, right face id) when derived from a floor-plate arrangement
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return np.array([-self.u[1], self.u[0]])

    def footprint(self) -> Polygon:
        a, b = self.p0 + self.n * self.lo, self.p0 + self.n * (self.lo + self.t)
        e = self.u * self.length
        return Polygon([a, a + e, b + e, b])

    def locate(self, c, width):
        """Return the along-wall position of point c if an opening of `width` centred there fits."""
        d = np.asarray(c, float) - self.p0
        a, b = float(d @ self.u), float(d @ self.n)
        if a - width / 2 < -1e-6 or a + width / 2 > self.length + 1e-6:
            return None
        if not (self.lo - 0.06 <= b <= self.lo + self.t + 0.06):
            return None
        return a, abs(b - (self.lo + self.t / 2))


def ring_walls(poly: Polygon, t, z0, h, name, type_key, external, storey, skip=None, climbable=False):
    """Walls along a polygon's exterior, thickness inside, butt-jointed corners (orthogonal rings)."""
    pts = [np.array(p) for p in list(orient(poly, 1.0).exterior.coords)[:-1]]
    pts = [p for i, p in enumerate(pts)
           if abs(np.cross(np.append(p - pts[i - 1], 0), np.append(pts[(i + 1) % len(pts)] - p, 0))[2]) > 1e-9]
    out, n = [], len(pts)
    for i in range(n):
        p0, p1, prev = pts[i], pts[(i + 1) % n], pts[i] - pts[i - 1]
        if skip is not None and skip(p0, p1):
            continue
        seg = p1 - p0
        L = float(np.linalg.norm(seg))
        u, up = seg / L, prev / np.linalg.norm(prev)
        adj = t if (up[0] * u[1] - up[1] * u[0]) > 0 else -t   # trim convex, extend reflex corners
        out.append(WallSpec(p0 + u * adj, u, L - adj, t, 0.0, z0, h, f"{name}-W{i + 1:02d}", type_key,
                            external, storey, climbable))
    return out


def attach(walls, centre, width, sill, height, kind, name, door_kind="", swing=1, meta=None):
    """Snap an opening to the wall that contains `centre` most centrally; returns that wall."""
    best = None
    for w in walls:
        r = w.locate(centre, width)
        if r is not None and (best is None or r[1] < best[1]):
            best = (w, r[0], r[1])
    if best is None:
        raise ValueError(f"no wall found for {name} at {tuple(np.round(centre, 3))}")
    w, a, _ = best
    w.openings.append(Opening(a - width / 2, a + width / 2, sill, height, kind, name, door_kind, swing, meta or {}))
    return w
