"""Derive walls from a floor plate: one centred wall per maximal boundary run between two faces.

All faces are rectilinear (blocks rotate by multiples of 90 degrees), so every boundary lies on an axis-aligned
line. Edges are split at every vertex that touches them, each elementary segment is labelled with the faces
on its left and right, and collinear neighbours with the same pair are merged into a run. A classifier maps
the face pair to a wall type (or no wall / a railing). Wall ends are extended into perpendicular walls so
corners close: a run with no collinear continuation at a node extends by half the thickest perpendicular
wall there (L corners overlap t x t, T stems reach the far face of the bar).
"""
from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import Point
from shapely.strtree import STRtree

from estate.blocks.plate import OUT

ND = 4                  # coordinates are snapped to 0.1 mm


def snap(v) -> float:
    return round(float(v) + 0.0, ND)


@dataclass
class Run:
    p0: tuple
    p1: tuple
    left: str
    right: str
    wall: str | None = None        # wall type key (rules.WALL_TYPES), "RAILING", or None (no element)
    t: float = 0.0
    height: str = "storey"         # storey | parapet | railing | tower
    external: bool = False
    climbable: bool = False
    ext0: float = 0.0
    ext1: float = 0.0
    clear0: float = 0.05           # minimum distance of an opening from each end (beyond ext)
    clear1: float = 0.05
    idx: int = -1
    meta: dict = field(default_factory=dict)

    @property
    def length(self):
        return float(np.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1]))

    @property
    def u(self):
        d = np.subtract(self.p1, self.p0)
        return d / np.linalg.norm(d)

    @property
    def horizontal(self):
        return abs(self.p0[1] - self.p1[1]) < 1e-9

    def faces(self):
        return frozenset((self.left, self.right))

    def contains(self, pt, tol=2e-3) -> float | None:
        """Along-run coordinate of pt if it lies on the run (else None)."""
        d = np.subtract(pt, self.p0)
        u = self.u
        s = float(d @ u)
        off = abs(float(d @ np.array([-u[1], u[0]])))
        if off > tol or s < -tol or s > self.length + tol:
            return None
        return s


def _edges(poly):
    rings = [poly.exterior] + list(poly.interiors)
    for ring in rings:
        c = [(snap(x), snap(y)) for x, y in ring.coords]
        for p, q in zip(c[:-1], c[1:]):
            if p != q:
                if p[0] != q[0] and p[1] != q[1]:
                    raise ValueError(f"non-rectilinear edge {p} -> {q}")
                yield p, q


def runs_from_faces(faces: dict) -> list[Run]:
    """Label every boundary segment of the plate with (left, right) faces and merge into runs."""
    ids = list(faces)
    polys = [faces[i] for i in ids]
    tree = STRtree(polys)
    verts_h = defaultdict(set)   # y -> xs of vertices on that horizontal line
    verts_v = defaultdict(set)   # x -> ys
    edges = []
    for poly in polys:
        for p, q in _edges(poly):
            edges.append((p, q))
            for v in (p, q):
                verts_h[v[1]].add(v[0])
                verts_v[v[0]].add(v[1])
    sh = {k: sorted(v) for k, v in verts_h.items()}
    sv = {k: sorted(v) for k, v in verts_v.items()}
    segs = set()
    for p, q in edges:
        if p[1] == q[1]:     # horizontal
            y, a, b = p[1], min(p[0], q[0]), max(p[0], q[0])
            xs = sh[y]
            pts = [a] + xs[bisect.bisect_right(xs, a):bisect.bisect_left(xs, b)] + [b]
            segs.update(((pts[i], y), (pts[i + 1], y)) for i in range(len(pts) - 1))
        else:
            x, a, b = p[0], min(p[1], q[1]), max(p[1], q[1])
            ys = sv[x]
            pts = [a] + ys[bisect.bisect_right(ys, a):bisect.bisect_left(ys, b)] + [b]
            segs.update(((x, pts[i]), (x, pts[i + 1])) for i in range(len(pts) - 1))

    def locate(pt):
        cand = tree.query(Point(pt))
        hit = [ids[i] for i in cand if polys[i].contains(Point(pt))]
        if len(hit) > 1:
            raise ValueError(f"faces overlap at {pt}: {hit}")
        return hit[0] if hit else OUT

    labelled = []
    for p, q in sorted(segs):
        m = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        if p[1] == q[1]:     # oriented +x: left = north
            L, R = locate((m[0], m[1] + 1e-3)), locate((m[0], m[1] - 1e-3))
        else:                # oriented +y: left = west
            L, R = locate((m[0] - 1e-3, m[1])), locate((m[0] + 1e-3, m[1]))
        if L != R:
            labelled.append((p, q, L, R))

    # merge collinear contiguous segments with the same pair
    lines = defaultdict(list)
    for p, q, L, R in labelled:
        key = ("h", p[1]) if p[1] == q[1] else ("v", p[0])
        lines[key].append((p, q, L, R))
    runs = []
    for key in sorted(lines):
        segs_ = sorted(lines[key], key=lambda s: (s[0][0], s[0][1]))
        cur = None
        for p, q, L, R in segs_:
            if cur is not None and cur.p1 == p and cur.left == L and cur.right == R:
                cur.p1 = q
            else:
                if cur is not None:
                    runs.append(cur)
                cur = Run(p, q, L, R)
        if cur is not None:
            runs.append(cur)
    for i, r in enumerate(runs):
        r.idx = i
    return runs


def finish_runs(runs: list[Run]) -> list[Run]:
    """Compute end extensions and opening clearances for the runs that carry a wall (r.wall set, t > 0)."""
    walls = [r for r in runs if r.wall and r.wall != "RAILING" and r.t > 0]
    at_node = defaultdict(list)
    for r in walls:
        at_node[r.p0].append((r, 0))
        at_node[r.p1].append((r, 1))
    for r in walls:
        for end, node in ((0, r.p0), (1, r.p1)):
            others = [(o, e) for o, e in at_node[node] if o is not r]
            collinear = [o for o, e in others if o.horizontal == r.horizontal]
            perp = [o for o, e in others if o.horizontal != r.horizontal]
            half_perp = max((o.t / 2 for o in perp), default=0.0)
            ext = 0.0 if collinear else half_perp
            # openings keep 50 mm clear of any perpendicular wall at the node (run coordinates, ext excluded)
            if end == 0:
                r.ext0, r.clear0 = ext, half_perp + 0.05
            else:
                r.ext1, r.clear1 = ext, half_perp + 0.05
    return runs


def merge_collinear(runs: list[Run], openings: list) -> tuple[list[Run], list]:
    """Merge contiguous collinear wall runs of the same type/height (like a facade wall that partitions butt
    into). Openings (objects with .run, .s0, .s1) are re-indexed onto the merged runs. Only nodes where
    exactly two such runs meet in line are merged through; the merged run keeps its parts in meta['parts']."""
    walls = [r for r in runs if r.wall and r.wall != "RAILING" and r.t > 0]
    key = lambda r: (r.wall, r.height, r.external, r.climbable, r.horizontal, r.meta.get("core_kind"))  # noqa: E731
    at_node = defaultdict(list)
    for r in walls:
        at_node[r.p0].append(r)
        at_node[r.p1].append(r)
    nxt = {}
    for node, rs in at_node.items():
        for a in rs:
            for b in rs:
                if a is not b and a.p1 == node and b.p0 == node and key(a) == key(b) and \
                        sum(1 for r in rs if r.horizontal == a.horizontal) == 2:
                    nxt[id(a)] = b
    has_prev = {id(b) for b in nxt.values()}
    merged, remap = [], {}
    for r in walls:
        if id(r) in has_prev:
            continue
        chain = [r]
        while id(chain[-1]) in nxt:
            chain.append(nxt[id(chain[-1])])
        m = Run(chain[0].p0, chain[-1].p1, chain[0].left, chain[0].right, chain[0].wall, chain[0].t, chain[0].height,
                chain[0].external, chain[0].climbable, chain[0].ext0, chain[-1].ext1, chain[0].clear0, chain[-1].clear1,
                meta=dict(chain[0].meta))
        off = 0.0
        parts = []
        for c in chain:
            remap[c.idx] = (m, off)
            parts.append((off, off + c.length, c.left, c.right))
            off += c.length
        m.meta["parts"] = parts
        merged.append(m)
    others = [r for r in runs if not (r.wall and r.wall != "RAILING" and r.t > 0)]
    out = merged + [r for r in others if r.wall]
    for i, r in enumerate(out):
        r.idx = i
    new_ops = []
    for o in openings:
        if o.run in remap:
            m, off = remap[o.run]
            o.run, o.s0, o.s1 = m.idx, o.s0 + off, o.s1 + off
        new_ops.append(o)
    return out, new_ops


def wall_footprint(r: Run):
    """Footprint polygon of a centred wall run including its end extensions."""
    u = r.u
    n = np.array([-u[1], u[0]])
    a = np.asarray(r.p0) - u * r.ext0
    b = np.asarray(r.p1) + u * r.ext1
    h = r.t / 2
    return shapely.Polygon([a - n * h, b - n * h, b + n * h, a + n * h])
