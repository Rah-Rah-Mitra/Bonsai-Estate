"""Grid routing for the site works: covered linkways, footpaths and driveways, plus the pedestrian graph.

Paths are routed on a 2 m raster of the estate. Every cell carries a cost (inf = blocked) and the axes a path may
take through it, so carriageways, verges and driveways are only crossed at right angles. The routing graph has two
layers (east-west and north-south moves) joined by a turn penalty, which keeps the routes orthogonal with few
bends. Networks are grown with the shortest-path Steiner heuristic (Takahashi-Matsuyama): starting from a root set,
the nearest unconnected terminal is attached through an incremental multi-source Dijkstra until all are joined.
A building's entrances are tied to a hub through the void deck, so a route may pass under a block as residents do.
A Steiner tree is shortest in total, not per trip, so the linkway network is then augmented with direct routes from
every bus stop to every building, existing linkway cells discounted (reused) and cells running alongside one made
dearer (so new routes join rather than duplicate).

Inside a building the 2 m raster is far too coarse: a void deck is a field of 0.6 m columns, cores and refuse
rooms, and a hawker centre is mostly tables. Legs through ground floors are routed by ``FreeSpace`` on the floor
minus every obstacle grown by the agent's clearance, so a pedestrian-graph edge never runs through a wall, a
column or a table.

Node ids are plain ints (never strings) so that iteration order, and with it the result, is reproducible under
Python's per-process string hash seed.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import shapely
from shapely.geometry import LineString, Point, box

H, V = 1, 2              # axis bit masks
INF = math.inf


class Grid:
    """Cost raster: cell (i, j) has centre (x0 + (i + .5) res, y0 + (j + .5) res); arrays are indexed [j, i]."""

    def __init__(self, bounds, res=2.0):
        self.x0, self.y0, x1, y1 = (float(v) for v in bounds)
        self.res = float(res)
        self.nx = int(math.ceil((x1 - self.x0) / res - 1e-9))
        self.ny = int(math.ceil((y1 - self.y0) / res - 1e-9))
        xs = self.x0 + (np.arange(self.nx) + 0.5) * res
        ys = self.y0 + (np.arange(self.ny) + 0.5) * res
        self.X, self.Y = np.meshgrid(xs, ys)
        self.cost = np.ones((self.ny, self.nx))
        self.axes = np.full((self.ny, self.nx), H | V, np.uint8)

    def inside(self, geom):
        if geom is None or geom.is_empty:
            return np.zeros(self.X.shape, bool)
        shapely.prepare(geom)
        return shapely.contains_xy(geom, self.X, self.Y)

    def set(self, geom=None, cost=None, axes=None, mul=None, mask=None, floor=None):
        m = self.inside(geom) if mask is None else mask
        if cost is not None:
            self.cost[m] = cost
        if mul is not None:
            self.cost[m] *= mul
        if floor is not None:
            self.cost[m] = np.minimum(self.cost[m], floor)
        if axes is not None:
            self.axes[m] = axes
        return m

    def ij(self, x, y):
        return (min(self.nx - 1, max(0, int((x - self.x0) // self.res))),
                min(self.ny - 1, max(0, int((y - self.y0) // self.res))))

    def xy(self, i, j):
        return (self.x0 + (i + 0.5) * self.res, self.y0 + (j + 0.5) * self.res)

    def free(self, i, j, axis):
        return 0 <= i < self.nx and 0 <= j < self.ny and math.isfinite(self.cost[j, i]) and bool(self.axes[j, i] & axis)


class Router:
    """Two-layer routing graph over a Grid (layer 0: moves along x, layer 1: moves along y)."""

    def __init__(self, grid: Grid, turn=6.0):
        self.g = grid
        nx_, ny_ = grid.nx, grid.ny
        n = nx_ * ny_ * 2
        self.adj = [[] for _ in range(n)]
        c, ax = grid.cost, grid.axes
        ok = np.isfinite(c)
        hm = ok & ((ax & H) > 0)
        vm = ok & ((ax & V) > 0)
        res = grid.res

        def nid(i, j, layer):
            return (j * nx_ + i) * 2 + layer
        js, is_ = np.nonzero(hm[:, :-1] & hm[:, 1:])
        w = res * (c[js, is_] + c[js, is_ + 1]) / 2
        for i, j, ww in zip(is_.tolist(), js.tolist(), w.tolist()):
            a, b = nid(i, j, 0), nid(i + 1, j, 0)
            self.adj[a].append((b, ww))
            self.adj[b].append((a, ww))
        js, is_ = np.nonzero(vm[:-1, :] & vm[1:, :])
        w = res * (c[js, is_] + c[js + 1, is_]) / 2
        for i, j, ww in zip(is_.tolist(), js.tolist(), w.tolist()):
            a, b = nid(i, j, 1), nid(i, j + 1, 1)
            self.adj[a].append((b, ww))
            self.adj[b].append((a, ww))
        js, is_ = np.nonzero(hm & vm)
        for i, j in zip(is_.tolist(), js.tolist()):
            a, b = nid(i, j, 0), nid(i, j, 1)
            self.adj[a].append((b, turn))
            self.adj[b].append((a, turn))
        self.n_grid = n
        self.virtual = {}         # virtual node id -> info dict

    def node(self, i, j, layer):
        return (j * self.g.nx + i) * 2 + layer

    def cell_of(self, v):
        if v >= self.n_grid:
            return None
        k = v // 2
        return k % self.g.nx, k // self.g.nx

    def add_virtual(self, **info) -> int:
        self.adj.append([])
        v = len(self.adj) - 1
        self.virtual[v] = info
        return v

    def link(self, a, b, w=0.0):
        self.adj[a].append((b, float(w)))
        self.adj[b].append((a, float(w)))

    def attach(self, v, x, y, axis, direction=None, max_steps=6, any_layer=False):
        """Link virtual node v to the first free cell at (x, y), stepping along `direction` (unit (dx, dy)) if blocked.
        With any_layer, a cell that may only be crossed the other way (a driveway in front of a porch) is accepted
        and the route turns there. Returns the cell (i, j) or None."""
        g = self.g
        other = V if axis == H else H
        for k in range(max_steps + 1):
            px = x + (direction[0] * k * g.res if direction is not None else 0.0)
            py = y + (direction[1] * k * g.res if direction is not None else 0.0)
            i, j = g.ij(px, py)
            for ax in ((axis, other) if any_layer else (axis,)):
                if g.free(i, j, ax):
                    self.link(v, self.node(i, j, 0 if ax == H else 1), 0.0)
                    return (i, j)
            if direction is None:
                break
        return None

    def steiner(self, roots, terminals):
        """Shortest-path heuristic Steiner tree. Returns (paths, unreachable): each path is a node list running from
        the tree grown so far to a newly attached terminal."""
        n = len(self.adj)
        dist = [INF] * n
        pred = [-1] * n
        adj = self.adj
        heap = []

        def grow(sources):
            for s in sources:
                dist[s] = 0.0
                pred[s] = -1
                heapq.heappush(heap, (0.0, s))
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue
                for v, w in adj[u]:
                    nd = d + w
                    if nd < dist[v] - 1e-9:
                        dist[v] = nd
                        pred[v] = u
                        heapq.heappush(heap, (nd, v))
        in_tree = set(roots)
        grow(sorted(in_tree))
        remaining = sorted(set(terminals) - in_tree)
        paths, unreachable = [], []
        while remaining:
            best = min(remaining, key=lambda t: (dist[t], t))
            remaining.remove(best)
            if not math.isfinite(dist[best]):
                unreachable.append(best)
                continue
            path = [best]
            u = best
            while u not in in_tree and pred[u] != -1:
                u = pred[u]
                path.append(u)
            path.reverse()
            paths.append(path)
            new = [p for p in path if p not in in_tree]
            in_tree.update(new)
            grow(new)
        return paths, unreachable

    def _ids(self, cells):
        nx_ = self.g.nx
        ids = set()
        for i, j in cells:
            ids.add((j * nx_ + i) * 2)
            ids.add((j * nx_ + i) * 2 + 1)
        return ids

    def discount(self, cells, factor):
        """Make moves between already-built cells cheaper (route reuse), in place."""
        ids = self._ids(cells)
        for u in ids:
            self.adj[u] = [(v, w * factor if v in ids else w) for v, w in self.adj[u]]

    def penalize_near(self, cells, reach, factor, done):
        """Make moves through cells within `reach` cells of `cells` (but not on them) dearer, so a new route joins
        an existing one instead of running beside it. `done` (a set) remembers cells already penalised."""
        g = self.g
        m = np.zeros((g.ny, g.nx), bool)
        for i, j in cells:
            m[j, i] = True
        d = m.copy()
        for _ in range(reach):
            e = d.copy()
            e[1:, :] |= d[:-1, :]
            e[:-1, :] |= d[1:, :]
            e[:, 1:] |= d[:, :-1]
            e[:, :-1] |= d[:, 1:]
            d = e
        band = {(int(i), int(j)) for j, i in zip(*np.nonzero(d & ~m))} - done
        built = set(cells)
        band -= built
        done |= band
        ids = self._ids(band)
        for u in ids:
            self.adj[u] = [(v, w * factor) for v, w in self.adj[u]]
        for u in ids:
            for v, _ in self.adj[u]:
                if v not in ids:
                    self.adj[v] = [(x, w * factor if x == u else w) for x, w in self.adj[v]]

    def tree_from(self, source):
        """Single-source Dijkstra predecessors (for backtracking several targets)."""
        n = len(self.adj)
        dist = [INF] * n
        pred = [-1] * n
        dist[source] = 0.0
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v] - 1e-9:
                    dist[v] = nd
                    pred[v] = u
                    heapq.heappush(heap, (nd, v))
        return dist, pred

    @staticmethod
    def backtrack(pred, target):
        path = [target]
        while pred[path[-1]] != -1:
            path.append(pred[path[-1]])
        return path[::-1]

    def distances(self, roots):
        """Plain multi-source Dijkstra distances (used to choose between candidate terminals)."""
        n = len(self.adj)
        dist = [INF] * n
        heap = [(0.0, s) for s in sorted(set(roots))]
        for _, s in heap:
            dist[s] = 0.0
        heapq.heapify(heap)
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v] - 1e-9:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist


@dataclass
class Route:
    """A routed network as straight orthogonal segments plus the terminal stubs and void-deck passes it used."""
    segments: list = field(default_factory=list)       # [(p0, p1)] coordinate tuples
    stubs: list = field(default_factory=list)          # [(entrance point, cell point, terminal info)]
    passes: list = field(default_factory=list)         # [(terminal info a, terminal info b)] void-deck hops
    cells: set = field(default_factory=set)            # grid cells used
    unreachable: list = field(default_factory=list)


def simplify(router: Router, paths) -> Route:
    """Grid paths -> straight segments between key cells (ends, junctions, bends, terminal cells); terminal runs are
    shifted onto the entrance line when that keeps the network orthogonal."""
    g = router.g
    cg = nx.Graph()
    term_cells = {}
    virt_edges = set()
    for path in paths:
        for a, b in zip(path, path[1:]):
            ca, cb = router.cell_of(a), router.cell_of(b)
            if ca is not None and cb is not None:
                if ca != cb:
                    cg.add_edge(ca, cb)
                else:
                    cg.add_node(ca)
            elif ca is None and cb is None:
                virt_edges.add((min(a, b), max(a, b)))
            else:
                v, c = (a, cb) if ca is None else (b, ca)
                cg.add_node(c)
                term_cells.setdefault(v, c)
    route = Route()
    route.cells = set(cg.nodes)
    # stub alignment: shift the first straight run from a terminal cell onto the entrance's line
    coord = {c: list(g.xy(*c)) for c in cg.nodes}
    shifted = set()
    for v, c in sorted(term_cells.items()):
        info = router.virtual.get(v, {})
        E = info.get("point")
        axis = info.get("axis")
        if E is None or axis is None:
            continue
        k = 1 if axis == H else 0          # coordinate to align (y for an east-west stub)
        if abs(coord[c][k] - E[k]) > g.res / 2 + 1e-6 or c in shifted:
            continue
        run = _straight_run(cg, c, axis)
        if run is None:
            continue
        cells, far = run
        if far is not None and (far in shifted or far in term_cells.values() or
                                any(_axis_of(far, nb) == axis for nb in cg.neighbors(far) if nb not in cells)):
            continue
        for cc in cells + ([far] if far is not None else []):
            coord[cc][k] = E[k]
            shifted.add(cc)
    # key cells
    def is_key(c):
        d = cg.degree(c)
        if d != 2 or c in term_cells.values():
            return True
        a, b = list(cg.neighbors(c))
        return _axis_of(c, a) != _axis_of(c, b)
    keys = sorted(c for c in cg.nodes if is_key(c))
    seen = set()
    for c in keys:
        for nb in sorted(cg.neighbors(c)):
            e = (min(c, nb), max(c, nb))
            if e in seen:
                continue
            prev, cur = c, nb
            seen.add(e)
            while not is_key(cur):
                nxt = [x for x in cg.neighbors(cur) if x != prev][0]
                seen.add((min(cur, nxt), max(cur, nxt)))
                prev, cur = cur, nxt
            p0, p1 = tuple(coord[c]), tuple(coord[cur])
            if p0 != p1:
                route.segments.append((p0, p1))
    for v, c in sorted(term_cells.items()):
        info = router.virtual.get(v, {})
        if info.get("point") is not None:
            route.stubs.append((tuple(info["point"]), tuple(coord[c]), info))
    for a, b in sorted(virt_edges):
        ia, ib = router.virtual.get(a, {}), router.virtual.get(b, {})
        if ia.get("hub") or ib.get("hub"):
            route.passes.append((ia, ib))
    return route


def _axis_of(a, b):
    return H if a[1] == b[1] else V


def _straight_run(cg, c, axis):
    """Cells of the straight run leaving terminal cell c along `axis` and the cell that ends it (or None)."""
    nbs = [nb for nb in cg.neighbors(c) if _axis_of(c, nb) == axis]
    if len(nbs) != 1:
        return None
    if any(_axis_of(c, nb) != axis for nb in cg.neighbors(c)):
        return None
    cells = [c]
    prev, cur = c, nbs[0]
    while True:
        nbrs = list(cg.neighbors(cur))
        straight = [x for x in nbrs if x != prev and _axis_of(cur, x) == axis]
        if cg.degree(cur) == 2 and straight:
            cells.append(cur)
            prev, cur = cur, straight[0]
            continue
        return cells, cur


# --------------------------------------------------------------------------- segment geometry helpers
def seg_rect(p0, p1, half, ext0=0.0, ext1=0.0):
    """Rectangle around an axis-aligned segment, optionally extended past either end."""
    (x0, y0), (x1, y1) = p0, p1
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1e-9:
        return box(x0 - half, y0 - half, x0 + half, y0 + half)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    nx_, ny_ = -uy, ux
    a = (x0 - ux * ext0, y0 - uy * ext0)
    b = (x1 + ux * ext1, y1 + uy * ext1)
    return shapely.Polygon([(a[0] + nx_ * half, a[1] + ny_ * half), (b[0] + nx_ * half, b[1] + ny_ * half),
                            (b[0] - nx_ * half, b[1] - ny_ * half), (a[0] - nx_ * half, a[1] - ny_ * half)])


def end_degrees(segments):
    """Number of segments meeting at each (rounded) end point."""
    deg = {}
    for p0, p1 in segments:
        for p in (p0, p1):
            k = (round(p[0], 2), round(p[1], 2))
            deg[k] = deg.get(k, 0) + 1
    return deg


def split_line(line: LineString, cutters):
    """Split a straight line by polygons: [(sub LineString, index of the cutter it lies in or -1)] in order."""
    pieces = [(line, -1)]
    for k, poly in enumerate(cutters):
        if poly is None or poly.is_empty or not line.intersects(poly):
            continue
        out = []
        for seg, tag in pieces:
            if tag != -1 or not seg.intersects(poly):
                out.append((seg, tag))
                continue
            inside = seg.intersection(poly)
            outside = seg.difference(poly)
            parts = [(g, k) for g in getattr(inside, "geoms", [inside]) if g.geom_type == "LineString" and g.length > 1e-6]
            parts += [(g, -1) for g in getattr(outside, "geoms", [outside]) if g.geom_type == "LineString" and g.length > 1e-6]
            s0 = Point(seg.coords[0])
            parts.sort(key=lambda t: s0.distance(Point(t[0].interpolate(0.5, normalized=True))))
            out.extend(parts)
        pieces = out
    return pieces


def columns_along(p0, p1, offset, spacing, size=0.2):
    """Column footprints on both sides of a segment at <= spacing centres (ends included)."""
    (x0, y0), (x1, y1) = p0, p1
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 0.5:
        return []
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    nx_, ny_ = -uy, ux
    n = max(1, int(math.ceil(L / spacing - 1e-9)))
    out = []
    for k in range(n + 1):
        s = L * k / n
        for side in (-1, 1):
            cx, cy = x0 + ux * s + nx_ * offset * side, y0 + uy * s + ny_ * offset * side
            out.append(box(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2))
    return out


# --------------------------------------------------------------------------- free-space routing (ground floors)
SQ2 = math.sqrt(2.0)
MOVES = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2))


class FreeSpace:
    """Where the centre of an agent may go inside a walkable region: the region minus every obstacle grown by the
    clearance.

    Routes are searched on a raster (8-connected Dijkstra, no corner cutting) and string-pulled against the exact
    polygon, so every leg keeps the clearance from every obstacle and the route stays close to the true shortest
    path. A terminal outside the free area (an entrance on a facade line, a lobby point close to a wall) is joined
    to the nearest free cell by a short leg that may pass inside the clearance band but never through an obstacle.

    With a ``cover`` polygon and ``penalty`` > 1, a cell outside the cover costs ``penalty`` times as much, so a
    route keeps under the slabs and canopies where it can; the string pull then straightens covered runs within the
    cover only, so it never trades covered ground for uncovered.
    """

    def __init__(self, walk, obstacles=None, clearance=0.35, res=0.2, cover=None, penalty=1.0):
        self.raw = obstacles if obstacles is not None and not obstacles.is_empty else None
        free = walk if self.raw is None else walk.difference(self.raw.buffer(clearance, quad_segs=2))
        self.free = shapely.make_valid(free) if not free.is_valid else free
        shapely.prepare(self.free)
        if self.raw is not None:
            shapely.prepare(self.raw)
        self.res = float(res)
        x0, y0, x1, y1 = self.free.bounds if not self.free.is_empty else (0.0, 0.0, res, res)
        self.x0, self.y0 = x0, y0
        self.nx = max(1, int(math.ceil((x1 - x0) / res)))
        self.ny = max(1, int(math.ceil((y1 - y0) / res)))
        xs = x0 + (np.arange(self.nx) + 0.5) * res
        ys = y0 + (np.arange(self.ny) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        mask = shapely.contains_xy(self.free, X, Y) if not self.free.is_empty else np.zeros(X.shape, bool)
        self.mask = mask
        self.ok = mask.ravel().tolist()
        self.cover = self.dry = self.cov = self.wt = None
        if cover is not None and not cover.is_empty and penalty > 1.0 and not self.free.is_empty:
            cover = shapely.make_valid(cover) if not cover.is_valid else cover
            shapely.prepare(cover)
            cm = shapely.contains_xy(cover, X, Y)
            self.cover = cover
            self.cov = cm.ravel().tolist()
            self.wt = np.where(cm, 1.0, float(penalty)).ravel().tolist()
            dry = self.free.intersection(cover)
            self.dry = shapely.make_valid(dry) if not dry.is_valid else dry
            shapely.prepare(self.dry)

    def xy(self, k):
        return (self.x0 + (k % self.nx + 0.5) * self.res, self.y0 + (k // self.nx + 0.5) * self.res)

    def clear_of(self, line) -> bool:
        return self.raw is None or line.intersection(self.raw).length < 1e-6

    def attach(self, p, reach=2.5):
        """(cell, p is free) joining point p to the raster: the nearest free cell p sees within `reach` (through
        free space when p is free, else without crossing an obstacle), or None."""
        P = Point(p)
        inside = self.free.covers(P)
        r = int(math.ceil(reach / self.res))
        i0, j0 = int((p[0] - self.x0) // self.res), int((p[1] - self.y0) // self.res)
        lo_i, hi_i = max(0, i0 - r), min(self.nx - 1, i0 + r)
        lo_j, hi_j = max(0, j0 - r), min(self.ny - 1, j0 + r)
        if lo_i > hi_i or lo_j > hi_j:
            return None
        sub = self.mask[lo_j:hi_j + 1, lo_i:hi_i + 1]
        jj, ii = np.nonzero(sub)
        if not len(ii):
            return None
        ks = (jj + lo_j) * self.nx + (ii + lo_i)
        cx = self.x0 + (ii + lo_i + 0.5) * self.res
        cy = self.y0 + (jj + lo_j + 0.5) * self.res
        d = np.hypot(cx - p[0], cy - p[1])
        order = np.lexsort((ks, d))
        order = order[d[order] <= reach][:64]
        if not len(order):
            return None
        lines = shapely.linestrings(np.stack([np.broadcast_to(np.asarray(p, float), (len(order), 2)),
                                              np.stack([cx[order], cy[order]], axis=1)], axis=1))
        ok = shapely.covers(self.free, lines) if inside else np.array([self.clear_of(ln) for ln in lines])
        hit = np.flatnonzero(ok)
        return (int(ks[order[hit[0]]]), inside) if len(hit) else None

    def _dijkstra(self, src, goals):
        nx_, ny_, ok, res, wt = self.nx, self.ny, self.ok, self.res, self.wt
        n = nx_ * ny_
        dist = [INF] * n
        pred = [-1] * n
        done = bytearray(n)
        dist[src] = 0.0
        heap = [(0.0, src)]
        left = set(goals)
        while heap and left:
            d, k = heapq.heappop(heap)
            if done[k]:
                continue
            done[k] = 1
            left.discard(k)
            i, j = k % nx_, k // nx_
            for di, dj, w in MOVES:
                ii, jj = i + di, j + dj
                if ii < 0 or jj < 0 or ii >= nx_ or jj >= ny_:
                    continue
                kk = jj * nx_ + ii
                if not ok[kk] or done[kk]:
                    continue
                if di and dj and not (ok[j * nx_ + ii] and ok[jj * nx_ + i]):
                    continue
                nd = d + w * res * (1.0 if wt is None else 0.5 * (wt[k] + wt[kk]))
                if nd < dist[kk] - 1e-12:
                    dist[kk] = nd
                    pred[kk] = k
                    heapq.heappush(heap, (nd, kk))
        return dist, pred, done

    def pull(self, pts, cov=None):
        """Greedy string pull over the dense cell path: from each anchor jump to the furthest later point it sees
        through free space. With per-point cover flags (cov), each run of covered points is pulled within the
        covered free space and each uncovered run within the free space, the runs joined end to end."""
        if cov is None or self.dry is None:
            return self._pull(self.free, pts)
        out = [pts[0]]
        s, n = 0, len(pts)
        while s < n - 1:
            e = s
            while e + 1 < n and cov[e + 1] == cov[s]:
                e += 1
            end = min(e + 1, n - 1)                    # the run and the first point of the next one
            out += self._pull(self.dry if cov[s] else self.free, pts[s:end + 1])[1:]
            s = end
        return out

    @staticmethod
    def _pull(region, pts):
        out = [pts[0]]
        i, n = 0, len(pts)
        P = np.asarray(pts, float)
        while i < n - 1:
            ends = P[i + 1:]
            lines = shapely.linestrings(np.stack([np.broadcast_to(P[i], ends.shape), ends], axis=1))
            vis = np.flatnonzero(shapely.covers(region, lines))
            j = i + 1 + int(vis[-1]) if len(vis) else i + 1
            out.append(pts[j])
            i = j
        return out

    def routes(self, src, targets, reach=2.5):
        """Shortest routes from src to each target point: {target index: LineString} for the reachable ones."""
        a = self.attach(src, reach)
        if a is None:
            return {}
        att = {}
        for t_i, t in enumerate(targets):
            at = self.attach(t, reach)
            if at is not None:
                att[t_i] = at
        if not att:
            return {}
        dist, pred, done = self._dijkstra(a[0], {c for c, _ in att.values()})
        out = {}
        for t_i, (c, t_free) in sorted(att.items()):
            if not done[c]:
                continue
            cells = [c]
            while cells[-1] != a[0]:
                cells.append(pred[cells[-1]])
            cells.reverse()
            src_p, dst_p = tuple(map(float, src)), tuple(map(float, targets[t_i]))
            seq = ([src_p] if a[1] else []) + [self.xy(k) for k in cells] + ([dst_p] if t_free else [])
            cov = None
            if self.cov is not None:
                cov = (([bool(self.cover.covers(Point(src_p)))] if a[1] else []) + [self.cov[k] for k in cells]
                       + ([bool(self.cover.covers(Point(dst_p)))] if t_free else []))
            core = self.pull(seq, cov) if len(seq) > 1 else seq
            pts = ([] if a[1] else [src_p]) + core + ([] if t_free else [dst_p])   # legs out of free space kept
            clean = [pts[0]]
            for q in pts[1:]:
                if math.dist(q, clean[-1]) > 1e-6:
                    clean.append(q)
            if len(clean) >= 2:
                out[t_i] = LineString(clean)
        return out


def split_cover(line, cover):
    """Pieces of a line under / not under a cover polygon (prepared, already grown by any tolerance):
    [(LineString, covered)], pieces shorter than 2 cm dropped."""
    if cover is None or cover.is_empty:
        return [(line, False)]
    if cover.covers(line):
        return [(line, True)]
    if not cover.intersects(line):
        return [(line, False)]
    out = []
    for g, cov in ((line.intersection(cover), True), (line.difference(cover), False)):
        if g.geom_type == "MultiLineString":
            g = shapely.line_merge(g)
        for part in shapely.get_parts(g):
            if part.geom_type == "LineString" and part.length > 0.02:
                out.append((part, cov))
    return out


# --------------------------------------------------------------------------- pedestrian graph
KIND_PRIORITY = {"crossing": 0, "linkway": 1, "footpath": 2, "park": 3, "sidewalk": 4}
INDOOR_VIA = ("void_deck", "indoor", "bus_shelter")      # routed legs: never snapped to or from


def pedestrian_graph(lines, points, prune_kinds=("sidewalk",), stub_max=4.0):
    """Node a set of attributed centrelines into a graph.

    lines: [(LineString, dict(kind, width, covered, slope, ...))]; points: [dict(xy, kind, ref, z)] special nodes
    (entrances, lift lobbies, bus stops, crossings) that must sit on line ends. Returns a networkx Graph whose nodes
    carry xy/kind/ref and edges kind/width/covered/slope/length.
    """
    lines = snap_dangling(lines)
    geoms = [shapely.set_precision(l, 0.01) for l, _ in lines]
    attrs = [a for _, a in lines]
    keep = [k for k, g in enumerate(geoms) if not g.is_empty and g.length > 0.01]
    noded = shapely.node(shapely.MultiLineString([geoms[k] for k in keep if g_ok(geoms[k])]))
    tree = shapely.STRtree([geoms[k] for k in keep])
    G = nx.Graph()

    def key(p):
        return (round(p[0], 2), round(p[1], 2))
    for piece in getattr(noded, "geoms", [noded]):
        if piece.length < 1e-6:
            continue
        mid = piece.interpolate(0.5, normalized=True)
        cand = tree.query(mid.buffer(0.02))
        best = None
        for ci in sorted(cand.tolist()):
            k = keep[ci]
            if geoms[k].distance(mid) > 0.02:
                continue
            a = attrs[k]
            rank = (KIND_PRIORITY.get(a["kind"], 9), -float(a.get("covered", False)), k)
            if best is None or rank < best[0]:
                best = (rank, a)
        if best is None:
            continue
        a = best[1]
        coords = list(piece.coords)
        for p, q in zip(coords, coords[1:]):
            u, v = key(p), key(q)
            if u == v:
                continue
            L = math.dist(p, q)
            if G.has_edge(u, v):
                old = G.edges[u, v]
                if (KIND_PRIORITY.get(a["kind"], 9), -a.get("covered", False)) >= \
                        (KIND_PRIORITY.get(old["kind"], 9), -old.get("covered", False)):
                    continue
            G.add_edge(u, v, kind=a["kind"], width=float(a.get("width", 1.8)), covered=bool(a.get("covered", False)),
                       slope=float(a.get("slope", 0.0)), length=L, **{k_: v_ for k_, v_ in a.items()
                                                                    if k_ not in ("kind", "width", "covered", "slope")})
    for u in G.nodes:
        G.nodes[u].update(xy=u, kind="junction", ref=None)
    for p in points:
        u = key(p["xy"])
        if u not in G:
            # snap to the nearest node within 5 cm
            near = min(G.nodes, key=lambda w: (math.dist(w, u), w)) if G.number_of_nodes() else None
            if near is None or math.dist(near, u) > 0.05:
                continue
            u = near
        G.nodes[u].update(kind=p["kind"], ref=p.get("ref"), z=p.get("z", 0.0))
    # prune short dangling stubs of the given kinds (sidewalk ends left inside other roads' verges)
    changed = True
    while changed:
        changed = False
        for u in sorted(G.nodes):
            if u not in G or G.degree(u) != 1 or G.nodes[u]["kind"] != "junction":
                continue
            v = next(iter(G.neighbors(u)))
            e = G.edges[u, v]
            if e["kind"] in prune_kinds and e["length"] < stub_max:
                G.remove_node(u)
                changed = True
    G.remove_nodes_from([u for u in list(G.nodes) if G.degree(u) == 0 and G.nodes[u]["kind"] == "junction"])
    return G


def snap_dangling(lines, tol=3.0, kinds=("footpath", "linkway", "park", "crossing")):
    """Join path ends that stop just short of another centreline (a footpath ending inside a sidewalk, a linkway
    meeting a crossing a little off its axis) with a short connector of the same kind. Legs routed through
    buildings and shelters (INDOOR_VIA) are never extended and never snapped onto: a straight connector to them
    could cut through a wall."""
    geoms = [l for l, _ in lines]
    tree = shapely.STRtree(geoms)
    out = list(lines)
    routed = {k for k, (_, a) in enumerate(lines) if a.get("via") in INDOOR_VIA}
    for k, (ln, a) in enumerate(lines):
        if a.get("kind") not in kinds or ln.length < 1e-6 or k in routed:
            continue
        for end in (ln.coords[0], ln.coords[-1]):
            P = Point(end)
            near = [int(c) for c in tree.query(P.buffer(tol)) if int(c) != k]
            if any(geoms[c].distance(P) < 0.02 for c in near):
                continue
            near = [c for c in near if c not in routed]
            best = min(near, key=lambda c: (geoms[c].distance(P), c), default=None)
            if best is None or geoms[best].distance(P) > tol:
                continue
            q = geoms[best].interpolate(geoms[best].project(P))
            out.append((LineString([end, (q.x, q.y)]), dict(a, via="connector")))
    return out


def g_ok(g):
    return g.geom_type == "LineString" and not g.is_empty


def graph_json(G, meta=None):
    ids = {u: f"n{k:05d}" for k, u in enumerate(sorted(G.nodes))}
    nodes = []
    for u in sorted(G.nodes):
        d = G.nodes[u]
        rec = {"id": ids[u], "xy": [round(u[0], 3), round(u[1], 3)], "z": round(float(d.get("z", 0.0)), 3),
               "kind": d["kind"]}
        if d.get("ref"):
            rec["ref"] = d["ref"]
        nodes.append(rec)
    edges = []
    for u, v in sorted(G.edges, key=lambda e: (min(e), max(e))):
        a, b = sorted((u, v))
        d = G.edges[u, v]
        rec = {"a": ids[a], "b": ids[b], "length": round(d["length"], 3), "width": round(d["width"], 2),
               "covered": d["covered"], "kind": d["kind"], "slope": round(d["slope"], 4)}
        if d.get("via"):
            rec["via"] = d["via"]
        edges.append(rec)
    return {"schema": "estate.site.pedestrian_graph/1", "units": "m", "frame": "estate local (x east, y north, z up)",
            "nodes": nodes, "edges": edges, "meta": meta or {}}


def graph_stats(G):
    """Reachability, walking distances and covered share from every bus stop to every residential lift lobby."""
    comps = list(nx.connected_components(G))
    stops = sorted(u for u in G.nodes if G.nodes[u]["kind"] == "bus_stop")
    lobbies = sorted(u for u in G.nodes if G.nodes[u]["kind"] == "lift_lobby")
    total = sum(d["length"] for _, _, d in G.edges(data=True))
    cov = sum(d["length"] for _, _, d in G.edges(data=True) if d["covered"])
    by_kind = {}
    for _, _, d in G.edges(data=True):
        by_kind[d["kind"]] = round(by_kind.get(d["kind"], 0.0) + d["length"], 1)
    unreachable, shares, nearest = [], [], {}
    for s in stops:
        lengths, paths = nx.single_source_dijkstra(G, s, weight="length")
        for lb in lobbies:
            if lb not in lengths:
                unreachable.append((G.nodes[s]["ref"], G.nodes[lb]["ref"]))
                continue
            p = paths[lb]
            L = lengths[lb]
            c = sum(G.edges[a, b]["length"] for a, b in zip(p, p[1:]) if G.edges[a, b]["covered"])
            shares.append(c / L if L > 0 else 1.0)
            ref = G.nodes[lb]["ref"]
            nearest[ref] = min(nearest.get(ref, INF), L)
    # covered-preferring routes (what a resident on a rainy day takes): uncovered metres weigh 3x
    for _, _, d in G.edges(data=True):
        d["_w"] = d["length"] * (1.0 if d["covered"] else 3.0)
    rain = []
    for s in stops:
        _, paths = nx.single_source_dijkstra(G, s, weight="_w")
        for lb in lobbies:
            if lb in paths:
                p = paths[lb]
                L = sum(G.edges[a, b]["length"] for a, b in zip(p, p[1:]))
                c = sum(G.edges[a, b]["length"] for a, b in zip(p, p[1:]) if G.edges[a, b]["covered"])
                rain.append(c / L if L > 0 else 1.0)
    best = {}
    for s in stops:
        dist_w, paths = nx.single_source_dijkstra(G, s, weight="_w")
        for lb in lobbies:
            if lb in paths and (lb not in best or dist_w[lb] < best[lb][0]):
                p = paths[lb]
                Lp = sum(G.edges[a, b]["length"] for a, b in zip(p, p[1:]))
                c = sum(G.edges[a, b]["length"] for a, b in zip(p, p[1:]) if G.edges[a, b]["covered"])
                best[lb] = (dist_w[lb], c / Lp if Lp > 0 else 1.0, Lp)
    near = [v[1] for v in best.values()]
    for _, _, d in G.edges(data=True):
        d.pop("_w", None)
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "components": len(comps),
            "largest_component": max((len(c) for c in comps), default=0),
            "length_m": round(total, 1), "covered_m": round(cov, 1), "length_by_kind": dict(sorted(by_kind.items())),
            "bus_stops": len(stops), "lift_lobbies": len(lobbies), "unreachable_pairs": unreachable[:20],
            "n_unreachable": len(unreachable),
            "covered_share_shortest": {"min": round(min(shares), 3) if shares else None,
                                       "mean": round(sum(shares) / len(shares), 3) if shares else None},
            "covered_share_sheltered_route": {"min": round(min(rain), 3) if rain else None,
                                              "mean": round(sum(rain) / len(rain), 3) if rain else None},
            "covered_share_from_nearest_stop": {"min": round(min(near), 3) if near else None,
                                                "mean": round(sum(near) / len(near), 3) if near else None,
                                                "max_walk_m": round(max(v[2] for v in best.values()), 1) if best else None},
            "walk_to_lobby_from_nearest_stop_max_m": round(max(nearest.values()), 1) if nearest else None}
