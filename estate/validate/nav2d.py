"""Site walk test: from every bus stop to every lift lobby: how far, how much of it covered, step-free?

It works on the ground plane of the whole estate, on a 0.25 m grid:
  walkable      footpaths, verges and sidewalks (road reserve minus carriageway), linkways, greens, the park
                connector, and void decks under the blocks (the open part of each ground plate)
  not walkable  enclosed ground-floor rooms (cores, amenity rooms, shops), and carriageways except at
                crossings
  covered       under a linkway canopy or a building footprint
  crossings     at bus stops, plus the junction mouths where a side road's carriageway meets the main road
                (kerb ramps assumed <= 1:10 unless the site model says otherwise)

Routes come from a bucketed Dijkstra (Dial's algorithm, vectorised per bucket, 8 neighbours, no corner
cutting) run once per bus stop. Two weightings are used: the shortest walk, and a covered-preferring walk
where uncovered cells cost double. Lengths are measured on the string-pulled polyline. A string pull may
not trade covered ground for uncovered ground. Covered share is the covered fraction of that polyline.

Walk length and covered share are judged from the nearest bus stop of each lobby. Reachability and
step-free access are judged from every stop.

Sources, best first:
  model/SITE_graph.json  the site works' pedestrian graph (schema estate.site.pedestrian_graph/1: nodes
                         {id, xy, kind, ref}, edges {a, b, length, covered, kind, slope}). Routes come from
                         networkx shortest paths, with the same two weightings. It is used only when its
                         meta.ifc_sha256 is the sha256 of the current SITE.ifc, so a graph left over from an
                         earlier layout is never judged. Its edges are then checked against SITE.ifc: a covered
                         edge keeps only the share that lies under a canopy or a building, and an edge that
                         runs over a carriageway or a zebra crossing ('Zebra crossing X01'; carriageways are cut
                         around their zebras) is a kerb drop (not step-free) unless it stays on the zebra and a
                         kerb ramp of 1:10 or less meets it at both kerbs
  model/SITE.ifc         grid with linkway canopies, zebra crossings, carriageways, kerbs and kerb ramps,
                         read from the element names and classes
  masterplan only        grid from masterplan.resolve() geometry; the report lists every assumption
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from estate import env

CELL = 0.25
MAX_WALK = 400.0
MIN_COVERED = 0.90
KERB_RAMP_MAX = 1 / 10
RAMP_MAX = 1 / 12
OPEN_GROUND_KINDS = {"void", "deck"}           # ground-plate face kinds that are open (walkable, covered)
OFFS = [(1, 0, 5), (-1, 0, 5), (0, 1, 5), (0, -1, 5), (1, 1, 7), (1, -1, 7), (-1, 1, 7), (-1, -1, 7)]


# ----------------------------------------------------------------------------- grid
class SiteGrid:
    def __init__(self, bounds, cell=CELL):
        x0, y0, x1, y1 = bounds
        self.cell = cell
        self.x0 = math.floor(x0 / cell) * cell
        self.y0 = math.floor(y0 / cell) * cell
        self.nx = int(math.ceil((x1 - self.x0) / cell)) + 1
        self.ny = int(math.ceil((y1 - self.y0) / cell)) + 1
        xs = self.x0 + (np.arange(self.nx) + 0.5) * cell
        ys = self.y0 + (np.arange(self.ny) + 0.5) * cell
        self.X, self.Y = np.meshgrid(xs, ys, indexing="ij")       # [ix, iy]

    def mask(self, geom):
        m = np.zeros((self.nx, self.ny), bool)
        if geom is None or geom.is_empty:
            return m
        for part in getattr(geom, "geoms", [geom]):
            if part.is_empty:
                continue
            x0, y0, x1, y1 = part.bounds
            i0, i1 = max(0, self.ix(x0)), min(self.nx, self.ix(x1) + 2)
            j0, j1 = max(0, self.iy(y0)), min(self.ny, self.iy(y1) + 2)
            if i1 <= i0 or j1 <= j0:
                continue
            m[i0:i1, j0:j1] |= shapely.contains_xy(part, self.X[i0:i1, j0:j1], self.Y[i0:i1, j0:j1])
        return m

    def ix(self, x):
        return int(math.floor((x - self.x0) / self.cell))

    def iy(self, y):
        return int(math.floor((y - self.y0) / self.cell))

    def xy(self, i, j):
        return self.x0 + (np.asarray(i) + 0.5) * self.cell, self.y0 + (np.asarray(j) + 0.5) * self.cell


def _snap(walk, g: SiteGrid, x, y, radius=6.0):
    """Nearest walkable cell to (x, y) within radius; None if there is none."""
    r = int(math.ceil(radius / g.cell))
    i, j = g.ix(x), g.iy(y)
    i0, i1, j0, j1 = max(0, i - r), min(g.nx, i + r + 1), max(0, j - r), min(g.ny, j + r + 1)
    sub = walk[i0:i1, j0:j1]
    if not sub.any():
        return None
    ii, jj = np.nonzero(sub)
    d = (g.X[i0:i1, j0:j1][ii, jj] - x) ** 2 + (g.Y[i0:i1, j0:j1][ii, jj] - y) ** 2
    k = int(np.argmin(d))
    return int(ii[k] + i0), int(jj[k] + j0)


# ----------------------------------------------------------------------------- shortest paths
def dial(walk, cost_mult, src, extra=None):
    """Integer-cost Dijkstra on an 8-connected grid (orthogonal 5, diagonal 7, times cost_mult of the target).

    walk: bool [nx, ny]; cost_mult: int [nx, ny] (>= 1); src: (i, j). Returns int32 distances (-1 = unreached).
    Diagonal steps need both orthogonal neighbours walkable.
    """
    nx, ny = walk.shape
    W = ny + 2
    pw = np.zeros((nx + 2, ny + 2), bool)
    pw[1:-1, 1:-1] = walk
    pw = pw.ravel()
    cm = np.ones((nx + 2, ny + 2), np.int32)
    cm[1:-1, 1:-1] = cost_mult
    cm = cm.ravel()
    INF = np.iinfo(np.int32).max
    dist = np.full(pw.size, INF, np.int32)
    s = (src[0] + 1) * W + src[1] + 1
    dist[s] = 0
    ring = 7 * int(cm.max()) + 1
    buckets = [[] for _ in range(ring)]
    buckets[0].append(np.array([s], np.int64))
    pending, d = 1, 0
    offs = [(dx * W + dy, c, dx * W, dy) for dx, dy, c in OFFS]
    while pending:
        b = buckets[d % ring]
        if b:
            cells = np.unique(np.concatenate(b)) if len(b) > 1 else b[0]
            pending -= len(b)
            b.clear()
            cells = cells[dist[cells] == d]
            for off, c, ox, oy in offs:
                nb = cells + off
                ok = pw[nb]
                if ox and oy:
                    ok &= pw[cells + ox] & pw[cells + oy]
                nb = nb[ok]
                nd = d + c * cm[nb]
                better = nd < dist[nb]
                nb, nd = nb[better], nd[better]
                if not len(nb):
                    continue
                np.minimum.at(dist, nb, nd)
                for v in np.unique(nd):
                    buckets[int(v) % ring].append(nb[nd == v])
                    pending += 1
        d += 1
    out = dist.reshape(nx + 2, ny + 2)[1:-1, 1:-1].copy()
    out[out == INF] = -1
    return out


def trace(dist, walk, cost_mult, dst):
    """Cells of one shortest path from the source (dist 0) to dst, following exact predecessors."""
    nx, ny = dist.shape
    path = [dst]
    i, j = dst
    guard = 0
    while dist[i, j] > 0 and guard < nx * ny:
        guard += 1
        here = dist[i, j]
        step = None
        for dx, dy, c in OFFS:
            a, b = i - dx, j - dy
            if not (0 <= a < nx and 0 <= b < ny) or dist[a, b] < 0 or not walk[a, b]:
                continue
            if dx and dy and not (walk[i - dx, j] and walk[i, j - dy]):
                continue
            if dist[a, b] + c * cost_mult[i, j] == here:
                step = (a, b)
                break
        if step is None:
            break
        i, j = step
        path.append(step)
    return path[::-1]


def _los(walk, cov, g: SiteGrid, p, qs, need_cover):
    """Line of sight from p to each q (walkable all along; covered all along where need_cover)."""
    q = np.asarray(qs, float)
    L = np.linalg.norm(q - p, axis=1)
    n = int(max(2, math.ceil(L.max() / (g.cell * 0.5)))) + 1
    t = np.linspace(0.0, 1.0, n)
    pts = p[None, None, :] + (q - p)[:, None, :] * t[None, :, None]
    ii = np.clip(np.floor((pts[..., 0] - g.x0) / g.cell).astype(int), 0, g.nx - 1)
    jj = np.clip(np.floor((pts[..., 1] - g.y0) / g.cell).astype(int), 0, g.ny - 1)
    ok = walk[ii, jj].all(axis=1)
    ok &= ~need_cover | cov[ii, jj].all(axis=1)
    return ok


def pull(path, walk, cov, g: SiteGrid, window=600):
    """String-pull a grid path into a polyline (list of xy). Covered stretches stay covered."""
    xy = np.stack(g.xy(np.array([p[0] for p in path]), np.array([p[1] for p in path])), 1)
    c = np.array([cov[p] for p in path], bool)
    # prefix count of uncovered cells, to know whether a sub-path is fully covered
    unc = np.concatenate([[0], np.cumsum(~c)])
    out = [xy[0]]
    i = 0
    n = len(xy)
    while i < n - 1:
        js = np.arange(i + 1, min(n, i + window))
        need = (unc[js + 1] - unc[i]) == 0
        ok = _los(walk, cov, g, xy[i], xy[js], need)
        ok[0] = True
        j = int(js[np.nonzero(ok)[0].max()])
        out.append(xy[j])
        i = j
    return np.array(out)


def _measure(poly_xy, cov, g: SiteGrid, step=0.1):
    seg = np.diff(poly_xy, axis=0)
    L = np.linalg.norm(seg, axis=1)
    total = float(L.sum())
    if total <= 0:
        return 0.0, 1.0
    covered = 0.0
    for p, s, l in zip(poly_xy[:-1], seg, L):
        n = max(1, int(math.ceil(l / step)))
        t = (np.arange(n) + 0.5) / n
        pts = p + s * t[:, None]
        ii = np.clip(np.floor((pts[:, 0] - g.x0) / g.cell).astype(int), 0, g.nx - 1)
        jj = np.clip(np.floor((pts[:, 1] - g.y0) / g.cell).astype(int), 0, g.ny - 1)
        covered += l * cov[ii, jj].mean()
    return total, covered / total


# ----------------------------------------------------------------------------- site model
def _xf_poly(M, poly):
    return shapely.affinity.affine_transform(poly, [M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[0, 3], M[1, 3]])


def _ground_plate_geometry(site):
    """(open ground, enclosed rooms) of a block's void deck in estate coordinates, or None if not planned."""
    from estate import config
    plan = getattr(site, "plan", None)
    if plan is None:
        return None
    M = config.placement_matrix(site.at, site.rot)
    open_, closed = [], []
    for f in plan.ground.faces.values():
        (open_ if f.kind in OPEN_GROUND_KINDS else closed).append(_xf_poly(M, f.poly))
    return unary_union(open_) if open_ else shapely.Polygon(), unary_union(closed) if closed else shapely.Polygon()


def _junction_mouths(roads):
    """Carriageway gaps where one road's carriageway ends at another road's reserve (treated as crossings)."""
    mouths = []
    for r in roads:
        line = LineString(r["centreline"])
        for end, other_end in ((0, 1), (-1, -2)):
            p = np.array(line.coords[end])
            q = np.array(line.coords[other_end])
            u = (p - q) / np.linalg.norm(p - q)
            for o in roads:
                if o is r or o["reserve_poly"].distance(Point(p)) > 0.5:
                    continue
                ext = LineString([p, p + u * (o["reserve"] / 2 + 0.5)])
                strip = ext.buffer(r["carriageway"] / 2, cap_style="flat")
                gap = strip.intersection(o["reserve_poly"]).difference(o["carriage_poly"])
                if not gap.is_empty and gap.area > 1.0:
                    tag = "a" if end == 0 else "b"
                    mouths.append(dict(id=f"X-{r['id']}-{o['id']}-{tag}", poly=gap, kind="junction"))
    return mouths


def _bus_stop_crossings(mp, width=4.0):
    roads = {r["id"]: r for r in mp["roads"]}
    out = []
    for bs in mp["bus_stops"]:
        r = roads.get(bs.get("road"))
        if r is None:
            continue
        line = LineString(r["centreline"])
        p = Point(bs["at"])
        s = line.project(p)
        c = np.array(line.interpolate(s).coords[0])
        a = np.array(line.interpolate(max(0.0, s - 1.0)).coords[0])
        b = np.array(line.interpolate(min(line.length, s + 1.0)).coords[0])
        u = (b - a) / np.linalg.norm(b - a)
        n = np.array([-u[1], u[0]])
        half = r["reserve"] / 2
        strip = LineString([c - n * half, c + n * half]).buffer(width / 2, cap_style="flat")
        out.append(dict(id=f"X-{bs['id']}", poly=strip.intersection(r["reserve_poly"]), kind="bus_stop"))
    return out


NAME_RX = {
    "covered": re.compile(r"linkway|canopy|shelter|covered walk|porch", re.I),
    # zebras only: the site works name them 'Zebra crossing X01'; 'Crossing approaches' (footpath) and
    # 'Crossing X01 kerb ramp 1' must not match
    "crossing": re.compile(r"^zebra crossing\b", re.I),
    "carriageway": re.compile(r"carriageway|traffic ?lane|road surface|asphalt|^junction j\d", re.I),
    "kerb_ramp": re.compile(r"kerb ramp|curb ramp|dropped kerb", re.I),
    "ramp": re.compile(r"ramp", re.I),
    "kerb": re.compile(r"kerb|curb", re.I),
}
ROAD_TOUCH = 0.05      # m of an edge on a carriageway or zebra before it counts as crossing the road
OFF_ZEBRA_MAX = 0.2    # m of a road-crossing edge allowed on the carriageway beside its zebra


def classify(label, cls, z_min, z_max):
    """Category of one SITE.ifc element for the walk test, from its label (name, object type, predefined type and
    class, in that order) and the height range of its geometry: 'covered', 'ramp', 'crossing', 'carriageway',
    'kerb', 'paving' or None."""
    if NAME_RX["covered"].search(label) and z_min > 1.8:
        return "covered"
    if cls in ("IfcRamp", "IfcRampFlight") or NAME_RX["ramp"].search(label):
        return "ramp"
    for k in ("crossing", "carriageway", "kerb"):
        if NAME_RX[k].search(label):
            return k
    if cls in ("IfcSlab", "IfcPavement") and abs(z_max) <= 0.05:
        return "paving"
    return None


def read_site_ifc(path, cache=True):
    """Covered canopies, zebra crossings, carriageways, kerbs, ramps and paving of SITE.ifc, as plan polygons.

    ``crossing_names`` runs parallel to ``crossing``. Paving is every slab or pavement whose top is at grade.
    ``cache`` is passed to meshcache.load (False for a throwaway copy).
    """
    from estate.export import meshcache
    f, meshes = meshcache.load(path, cache=cache)
    out = {k: [] for k in ("covered", "crossing", "crossing_names", "carriageway", "kerb", "ramps", "paving")}
    for gid, d in sorted(meshes.items(), key=lambda kv: (kv[1]["name"], kv[0])):
        el = f.by_guid(gid)
        label = " ".join(str(v) for v in (d["name"], getattr(el, "ObjectType", None),
                                          getattr(el, "PredefinedType", None), d["cls"]) if v)
        tri = d["verts"][d["faces"]]
        polys = shapely.polygons(tri[:, :, :2])
        polys = polys[shapely.area(polys) > 1e-6]
        if not len(polys):
            continue
        plan = shapely.union_all(polys).buffer(0)
        z = d["verts"][:, 2]
        what = classify(label, d["cls"], z.min(), z.max())
        if what == "ramp":
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            area = np.linalg.norm(n, axis=1) / 2
            walk_face = (n[:, 2] > 0.5 * np.linalg.norm(n, axis=1)) & (area > 0.005)    # the walking surface
            slope = 0.0
            if walk_face.any():
                slope = float((np.linalg.norm(n[walk_face, :2], axis=1) / n[walk_face, 2]).max())
            kind = "kerb_ramp" if NAME_RX["kerb_ramp"].search(label) else "ramp"
            out["ramps"].append(dict(name=d["name"], kind=kind, poly=plan, slope=slope))
        elif what == "crossing":
            out["crossing"].append(plan)
            out["crossing_names"].append(d["name"])
        elif what is not None:
            out[what].append(plan)
    return out


def _sha256(path):
    from estate.export.meshcache import sha256
    return sha256(path)


def usable_graph(site_graph_json, site_ifc, assumptions):
    """The pedestrian graph (parsed JSON) when its meta.ifc_sha256 matches SITE.ifc, else None with the reason
    added to ``assumptions``."""
    if not site_graph_json or not Path(site_graph_json).exists():
        return None
    name = env.rel(site_graph_json)
    try:
        data = json.loads(Path(site_graph_json).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        assumptions.append(f"{name} unreadable ({e}); grid walk test instead")
        return None
    want = (data.get("meta") or {}).get("ifc_sha256")
    if not site_ifc or not Path(site_ifc).exists():
        assumptions.append(f"{name} not used: there is no SITE.ifc to check it against")
        return None
    if not want:
        assumptions.append(f"{name} not used: its meta has no ifc_sha256, so it cannot be matched to "
                           f"{env.rel(site_ifc)}; grid walk test on SITE.ifc instead")
        return None
    have = _sha256(site_ifc)
    if want != have:
        assumptions.append(f"{name} not used: written for another SITE.ifc (sha256 {want[:12]}, current "
                           f"{have[:12]}); grid walk test on SITE.ifc instead")
        return None
    return data


def graph_vs_ifc(data, ifc, footprints, tol=0.3):
    """Check the pedestrian graph against SITE.ifc. Returns (fixes {(a, b): dict}, report).

    covered      the share of each covered edge that lies under a canopy (higher than 1.8 m) or a building
                 footprint, both grown by ``tol``; 'indoor' and 'void_deck' edges count footprints only. An edge
                 whose measured share is lower gets that share.
    road         every edge that runs over a carriageway or a zebra crossing (more than ROAD_TOUCH m; the site
                 works cut the carriageways around their zebras, so both count) must lie on a zebra apart from
                 less than OFF_ZEBRA_MAX m, and an acceptable kerb ramp (<= 1:10) must touch it at both kerbs.
                 Otherwise the edge has a kerb drop and is not step-free.
    zebras       crossings listed in the graph meta (bus stops and junctions) must exist as 'Zebra crossing
                 <id>' in SITE.ifc. ``zebras_crossed`` lists the zebras that graph edges run over.
    paving       metres of edge off any paved surface, grass or terrain (reported, not judged).
    """
    xy = {n["id"]: tuple(map(float, n.get("xy") or (n["x"], n["y"]))) for n in data["nodes"]}
    bld = unary_union(list(footprints)).buffer(tol) if footprints else shapely.Polygon()
    canopy = unary_union(ifc["covered"]).buffer(tol) if ifc["covered"] else shapely.Polygon()
    cover_all = unary_union([canopy, bld])
    carriage = unary_union(ifc["carriageway"]) if ifc["carriageway"] else shapely.Polygon()
    zebras = list(ifc["crossing"])
    zebra_names = list(ifc.get("crossing_names") or [])
    if len(zebra_names) != len(zebras):
        zebra_names = [f"crossing {k + 1}" for k in range(len(zebras))]
    zebra = unary_union(zebras).buffer(tol) if zebras else shapely.Polygon()
    road_area = unary_union([carriage] + zebras)
    off_zebra_area = road_area.difference(zebra)
    shapely.prepare(road_area)
    ramps = [r["poly"].buffer(0.5) for r in ifc["ramps"]
             if r["kind"] == "kerb_ramp" and r["slope"] <= KERB_RAMP_MAX + 1e-9]
    paved = unary_union(list(ifc.get("paving", [])) + zebras + [r["poly"] for r in ifc["ramps"]]
                        + [bld]).buffer(tol)
    fixes, short, drops = {}, [], []
    claimed = measured = off_paving = 0.0
    over_road = 0
    crossed = set()
    for e in data["edges"]:
        a, b = e["a"], e["b"]
        line = LineString([xy[a], xy[b]])
        L = line.length
        if L <= 1e-6:
            continue
        via, kind = str(e.get("via") or ""), str(e.get("kind", ""))
        cov = e.get("covered", False)
        cov = (1.0 if cov else 0.0) if isinstance(cov, bool) else float(cov)
        if cov > 0:
            share = line.intersection(bld if via in ("indoor", "void_deck") else cover_all).length / L
            claimed += cov * L
            measured += min(cov, share) * L
            if share < cov - 0.05:
                fixes.setdefault((a, b), {})["covered"] = round(share, 3)
                short.append((L * (cov - share), f"{a}-{b} {kind}{'/' + via if via else ''} {L:.1f} m: "
                                                 f"{share:.2f} under cover"))
        if road_area.intersects(line) and line.intersection(road_area).length > ROAD_TOUCH:
            over_road += 1
            on = [nm for nm, z in zip(zebra_names, zebras) if line.intersection(z).length > ROAD_TOUCH]
            crossed.update(on)
            off_zebra = line.intersection(off_zebra_area).length
            n_ramps = sum(1 for r in ramps if r.intersects(line))
            if off_zebra >= OFF_ZEBRA_MAX or n_ramps < 2:
                why = (f"{off_zebra:.1f} m over the carriageway off a zebra crossing" if off_zebra >= OFF_ZEBRA_MAX
                       else f"kerb ramps (<= 1:10) at {n_ramps} of 2 kerbs")
                fixes.setdefault((a, b), {})["step_free"] = False
                drops.append(f"{a}-{b} {kind} at ({line.centroid.x:.1f}, {line.centroid.y:.1f})"
                             f"{' over ' + ', '.join(on) if on else ''}: {why}")
        if via != "indoor":
            off_paving += L - line.intersection(paved).length
    names = set(zebra_names)
    listed = [c["id"] for c in (data.get("meta") or {}).get("crossings", []) if c.get("kind") in ("bus", "junction")]
    missing = [c for c in listed if not any(re.search(rf"\b{re.escape(c)}\b", nm) for nm in names)]
    short.sort(key=lambda t: -t[0])
    rep = dict(edges=len(data["edges"]), covered_claimed_m=round(claimed, 1), covered_measured_m=round(measured, 1),
               covered_short_edges=len(short), covered_short_m=round(sum(t[0] for t in short), 1),
               covered_short_examples=[t[1] for t in short[:12]], edges_over_carriageway=over_road,
               zebras_crossed=sorted(crossed), kerb_drops=drops, zebras_listed=len(listed), zebras_missing=missing,
               off_paving_m=round(off_paving, 1))
    rep["ok"] = not short and not drops and not missing
    return fixes, rep


def _graph_routes(path, mp, targets, penalty=2.0, edge_fix=None):
    """Routes on the site works' pedestrian graph (schema estate.site.pedestrian_graph/1, networkx).

    Nodes carry ``xy`` (or ``x``/``y``), ``kind`` and ``ref`` (a bus stop id or site id). Edges carry
    ``length``, ``covered``, ``kind`` and ``slope``. A crossing edge is step-free when its kerb ramps are
    <= 1:10, and any other edge when its slope is <= 1:12. ``path`` may be the parsed JSON. ``edge_fix``
    ({(a, b): {covered, step_free}}, from graph_vs_ifc) lowers an edge's cover or step-free flag. Raises
    ValueError when the file does not fit.
    """
    import networkx as nx
    data = path if isinstance(path, dict) else json.loads(Path(path).read_text(encoding="utf-8"))
    edge_fix = edge_fix or {}
    nodes, edges = data.get("nodes"), data.get("edges")
    if not nodes or not edges:
        raise ValueError("graph has no nodes/edges")
    G = nx.Graph()
    xy = {}
    for n in nodes:
        p = n.get("xy") or (n["x"], n["y"])
        xy[n["id"]] = (float(p[0]), float(p[1]))
        G.add_node(n["id"], kind=n.get("kind", ""), ref=n.get("ref"))
    for e in edges:
        a, b = e["a"], e["b"]
        L = float(e.get("length") or math.dist(xy[a], xy[b]))
        cov = e.get("covered", False)
        cov = (1.0 if cov else 0.0) if isinstance(cov, bool) else float(cov)
        kind = str(e.get("kind", ""))
        slope = abs(float(e.get("slope", 0.0) or 0.0))
        limit = KERB_RAMP_MAX if "crossing" in kind or "kerb" in kind else RAMP_MAX
        sf = bool(e.get("step_free", True)) and slope <= limit + 1e-9
        fix = edge_fix.get((a, b)) or edge_fix.get((b, a)) or {}
        cov = min(cov, fix.get("covered", cov))
        sf = sf and fix.get("step_free", True)
        G.add_edge(a, b, length=L, covered=cov, step_free=sf, w_cov=L * (cov + penalty * (1.0 - cov)),
                   crossing="crossing" in kind, kind=kind, slope=slope)
    Gsf = nx.Graph()
    Gsf.add_nodes_from(G.nodes)
    Gsf.add_edges_from((a, b, d) for a, b, d in G.edges(data=True) if d["step_free"])

    def nearest(x, y, cands):
        cands = sorted(cands or G.nodes)
        d = [math.dist(xy[c], (x, y)) for c in cands]
        k = int(np.argmin(d))
        return cands[k], d[k]

    bus_nodes = {}
    for bs in mp["bus_stops"]:
        refs = [i for i in G.nodes if G.nodes[i]["ref"] == bs["id"]]
        bus_nodes[bs["id"]] = nearest(*bs["at"], refs or [i for i in G.nodes if G.nodes[i]["kind"] == "bus_stop"])
    tgt_nodes = {}
    for t in targets:
        refs = [i for i in G.nodes if G.nodes[i]["ref"] == t["id"] and G.nodes[i]["kind"] != "entrance"]
        lob = [i for i in G.nodes if G.nodes[i]["kind"] in ("lift_lobby", "lobby")]
        tgt_nodes[t["id"]] = [nearest(*p, refs or lob) for p in t["lobbies"]]
        t["snapped"] = [list(xy[n]) for n, _ in tgt_nodes[t["id"]]]
    routes, polylines = [], []
    for bs in mp["bus_stops"]:
        s, _ = bus_nodes[bs["id"]]
        L_all = nx.single_source_dijkstra(G, s, weight="length")
        C_all = nx.single_source_dijkstra(G, s, weight="w_cov")
        sf_reach = nx.node_connected_component(Gsf, s)
        for t in targets:
            for li, (tn, t_off) in enumerate(tgt_nodes[t["id"]]):
                r = dict(bus_stop=bs["id"], target=t["id"], lobby=li, mode="graph", node=tn, snap_m=round(t_off, 1))
                if tn not in L_all[0]:
                    routes.append(dict(r, reachable=False, reason="no path in the pedestrian graph"))
                    continue
                sp, cp = L_all[1][tn], C_all[1][tn]
                Lc = sum(G.edges[a, b]["length"] for a, b in zip(cp, cp[1:]))
                cov = sum(G.edges[a, b]["length"] * G.edges[a, b]["covered"] for a, b in zip(cp, cp[1:]))
                share = cov / Lc if Lc else 1.0
                L = L_all[0][tn]
                routes.append(dict(r, reachable=True, length_m=round(L, 1), route_m=round(Lc, 1),
                                   covered_share=round(share, 3), step_free=tn in sf_reach,
                                   crossings=[f"{a}-{b}" for a, b in zip(sp, sp[1:]) if G.edges[a, b]["crossing"]],
                                   lobby_xy=list(xy[tn]), ok_length=bool(L <= MAX_WALK),
                                   ok_covered=bool(share >= MIN_COVERED)))
                polylines.append((bs["id"], np.array([xy[i] for i in cp])))
    return routes, polylines, G, xy


def render_graph(path, G, xy, polylines, bus, targets, title):
    """Plan of the pedestrian graph (linkways blue, crossings yellow) with the covered-preferring routes."""
    from PIL import Image, ImageDraw
    from estate.draw.raster import font
    P_ = np.array(list(xy.values()))
    x0, y0 = P_.min(0) - 10
    x1, y1 = P_.max(0) + 10
    s = 2.0                                              # px per m
    top = 40
    W, H = int((x1 - x0) * s) + 20, int((y1 - y0) * s) + top + 20
    im = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(im)

    def P(x, y):
        return 10 + (x - x0) * s, top + 10 + (y1 - y) * s

    for a, b, d in G.edges(data=True):
        if not d["step_free"]:
            col, w = (220, 60, 60), 4
        elif d["covered"] >= 0.5:
            col, w = (90, 140, 220), 4
        elif d["crossing"]:
            col, w = (240, 200, 60), 4
        else:
            col, w = (190, 190, 190), 2
        dr.line([P(*xy[a]), P(*xy[b])], fill=col, width=w)
    colour = {b["id"]: ROUTE_COLOURS[i % len(ROUTE_COLOURS)] for i, b in enumerate(bus)}
    for bid, pl in polylines:
        dr.line([P(*p) for p in pl], fill=colour[bid], width=1)
    for t in targets:
        for p in t.get("snapped") or []:
            X, Y = P(*p)
            dr.rectangle([X - 3, Y - 3, X + 3, Y + 3], fill=(0, 0, 0))
        if t.get("snapped"):
            X, Y = P(*t["snapped"][0])
            dr.text((X + 5, Y - 5), t["id"], fill=(0, 0, 0), font=font(10))
    for b in bus:
        X, Y = P(*b["at"])
        dr.ellipse([X - 6, Y - 6, X + 6, Y + 6], fill=colour[b["id"]], outline=(0, 0, 0))
        dr.text((X + 8, Y), b["id"], fill=colour[b["id"]], font=font(12))
    dr.text((10, 8), title, fill=(30, 30, 30), font=font(14))
    dr.text((10, 24), "grey path, blue covered, yellow crossing, red not step-free; thin lines: covered-preferring "
                      "routes", fill=(80, 80, 80), font=font(11))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


# ----------------------------------------------------------------------------- the check
def _targets(mp):
    out = []
    for s in mp["sites"]:
        out.append(dict(id=s.id, kind=s.kind, name=s.name, lobbies=[tuple(map(float, p)) for p in s.lift_lobbies],
                        approximate=bool(getattr(s, "approximate", False))))
    return out


def check_site(mp, site_graph_json=None, site_ifc=None, cell=CELL, out_dir=None, png=True, log=None):
    """Bus stop -> lift lobby reachability of the estate. Returns the result dict (and writes it to out_dir)."""
    import time
    t0 = time.time()
    say = log or (lambda *_: None)
    site_graph_json = site_graph_json if site_graph_json is not None else env.MODEL / "SITE_graph.json"
    site_ifc = site_ifc if site_ifc is not None else env.MODEL / "SITE.ifc"
    targets = _targets(mp)
    assumptions, mode = [], None
    res = {"cell": cell, "targets": targets, "bus_stops": mp["bus_stops"], "thresholds": {
        "max_walk_m": MAX_WALK, "min_covered_share": MIN_COVERED, "kerb_ramp_max": KERB_RAMP_MAX, "ramp_max": RAMP_MAX}}

    data = usable_graph(site_graph_json, site_ifc, assumptions)
    if data is not None:
        try:
            fixes, checks = graph_vs_ifc(data, read_site_ifc(site_ifc), [s.footprint for s in mp["sites"]])
            routes, polylines, G, xy = _graph_routes(data, mp, targets, edge_fix=fixes)
            mode = "graph"
            res["source"] = env.rel(site_graph_json)
            res["graph"] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                            "ifc_sha256": data["meta"]["ifc_sha256"],
                            "not_step_free_edges": [f"{a}-{b} {d['kind']} slope {d['slope']:.3f}"
                                                    for a, b, d in G.edges(data=True) if not d["step_free"]]}
            res["graph_checks"] = checks
            if checks["covered_short_edges"]:
                assumptions.append(f"{checks['covered_short_edges']} covered graph edges "
                                   f"({checks['covered_short_m']} m) are not under a SITE.ifc canopy or building: "
                                   f"cover reduced to the measured share")
            if checks["kerb_drops"]:
                assumptions.append(f"{len(checks['kerb_drops'])} graph edges cross a road off its zebra crossing or "
                                   f"without kerb ramps (<= 1:10) at both kerbs in SITE.ifc: counted as not step-free")
            if png and out_dir is not None:
                render_graph(Path(out_dir) / "nav_site.png", G, xy, polylines, mp["bus_stops"], targets,
                             f"Site walk test (graph {env.rel(site_graph_json)}, checked against SITE.ifc): "
                             f"bus stops to lift lobbies")
        except Exception as e:  # noqa: BLE001
            assumptions.append(f"{env.rel(site_graph_json)} unusable ({type(e).__name__}: {e}); fell back to the grid")
    if mode is None:
        routes, grid_info = _grid_routes(mp, targets, site_ifc, cell, assumptions, out_dir if png else None, say)
        mode = grid_info.pop("mode")
        res.update(grid_info)
    res["mode"] = mode
    res["assumptions"] = assumptions
    res["routes"] = routes
    res["summary"] = summarise(routes)
    res["timing_s"] = round(time.time() - t0, 2)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "nav_site.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    return res


def summarise(routes):
    """Every route must exist and be step-free; walk length and covered share are judged from the nearest
    bus stop of each lift lobby (a 400 m estate with stops on its edges cannot be within 400 m of every stop)."""
    ok_routes = [r for r in routes if r.get("reachable")]
    nearest = {}
    for r in ok_routes:
        k = (r["target"], r.get("lobby", 0))
        if k not in nearest or r["length_m"] < nearest[k]["length_m"]:
            nearest[k] = r
    near = list(nearest.values())
    s = dict(
        routes=len(routes), reachable=len(ok_routes),
        unreachable=[f"{r['bus_stop']}->{r['target']}" for r in routes if not r.get("reachable")],
        step_free=int(sum(bool(r.get("step_free")) for r in ok_routes)),
        lobbies=len(near),
        nearest_stop_max_walk_m=max((r["length_m"] for r in near), default=None),
        nearest_stop_over_400m=[f"{r['target']}/{r.get('lobby', 0)} {r['bus_stop']} {r['length_m']} m"
                                for r in near if r["length_m"] > MAX_WALK],
        nearest_stop_min_covered_share=min((r["covered_share"] for r in near), default=None),
        nearest_stop_covered_below_target=[f"{r['target']}/{r.get('lobby', 0)} {r['bus_stop']} {r['covered_share']}"
                                           for r in near if r["covered_share"] < MIN_COVERED],
        any_stop_max_walk_m=max((r["length_m"] for r in ok_routes), default=None),
        any_stop_over_400m=int(sum(r["length_m"] > MAX_WALK for r in ok_routes)),
        any_stop_mean_covered_share=(round(float(np.mean([r["covered_share"] for r in ok_routes])), 3)
                                     if ok_routes else None),
        any_stop_covered_below_target=int(sum(r["covered_share"] < MIN_COVERED for r in ok_routes)))
    s["reachability_ok"] = s["reachable"] == s["routes"] and s["routes"] > 0
    s["walk_ok"] = s["reachability_ok"] and not s["nearest_stop_over_400m"]
    s["covered_ok"] = s["reachability_ok"] and not s["nearest_stop_covered_below_target"]
    s["step_free_ok"] = s["reachability_ok"] and s["step_free"] == s["routes"]
    s["ok"] = s["walk_ok"] and s["covered_ok"] and s["step_free_ok"]
    return s


def _grid_routes(mp, targets, site_ifc, cell, assumptions, png_dir, say):
    sites = mp["sites"]
    ext = mp["extent"]
    allb = unary_union([ext] + [s.footprint for s in sites] + [Point(b["at"]).buffer(5) for b in mp["bus_stops"]])
    x0, y0, x1, y1 = allb.bounds
    g = SiteGrid((x0 - 5, y0 - 5, x1 + 5, y1 + 5), cell)
    say(f"  site grid {g.nx} x {g.ny} at {cell} m")

    walk = np.ones((g.nx, g.ny), bool)
    cov = np.zeros((g.nx, g.ny), bool)
    enclosed = np.zeros((g.nx, g.ny), bool)
    for s in sites:
        fp = g.mask(s.footprint)
        cov |= fp
        geo = _ground_plate_geometry(s)
        if geo is not None:
            enclosed |= g.mask(geo[1])
        elif s.kind == "block":
            assumptions.append(f"{s.id}: no ground plate yet (approximate footprint); void deck taken as fully open")
        else:
            assumptions.append(f"{s.id}: ground floor taken as open and covered (no interior model in nav2d)")
    walk &= ~enclosed

    ifc = None
    if site_ifc and Path(site_ifc).exists():
        try:
            ifc = read_site_ifc(site_ifc)
        except Exception as e:  # noqa: BLE001
            assumptions.append(f"{env.rel(site_ifc)} unreadable ({e}); masterplan geometry used")
    carriage = unary_union([r["carriage_poly"] for r in mp["roads"]])
    if ifc and ifc["carriageway"]:
        carriage = unary_union([carriage] + ifc["carriageway"])
    cw = g.mask(carriage)
    crossings = _bus_stop_crossings(mp) + _junction_mouths(mp["roads"])
    if ifc and ifc["crossing"]:
        crossings += [dict(id=f"X-ifc-{i}", poly=p, kind="site") for i, p in enumerate(ifc["crossing"])]
    xlab = np.zeros((g.nx, g.ny), np.int16)
    for k, c in enumerate(crossings, 1):
        m = g.mask(c["poly"])
        xlab[m & (xlab == 0)] = k
    walk &= ~cw | (xlab > 0)
    if ifc:
        mode = "grid-site-ifc"
        assumptions.append("grid on SITE.ifc: all ground outside carriageways and enclosed ground floors is walkable "
                           "(lawns and terrain included, so routes may cut across greens); covered = SITE.ifc "
                           "canopies above 1.8 m and building footprints")
        for p in ifc["covered"]:
            cov |= g.mask(p)
        bad_ramps = [r for r in ifc["ramps"]
                     if r["slope"] > (KERB_RAMP_MAX if r["kind"] == "kerb_ramp" else RAMP_MAX) + 1e-9]
        ramp_ok = np.zeros((g.nx, g.ny), bool)
        for r in ifc["ramps"]:
            if r not in bad_ramps:
                ramp_ok |= g.mask(r["poly"].buffer(0.3))          # the kerb is dropped where a ramp meets it
        steps = np.zeros((g.nx, g.ny), bool)
        for p in ifc["kerb"]:
            steps |= g.mask(p)
        steps &= (xlab == 0) & ~ramp_ok
        walk |= ramp_ok & ~enclosed
        for r in bad_ramps:
            steps |= g.mask(r["poly"])
        ramps = [dict(name=r["name"], kind=r["kind"], slope=round(r["slope"], 4),
                      ok=r not in bad_ramps) for r in ifc["ramps"]]
        if not ifc["covered"]:
            assumptions.append("SITE.ifc has no linkway canopies recognised by name (linkway/canopy/shelter)")
    else:
        mode = "grid-masterplan"
        steps = np.zeros((g.nx, g.ny), bool)
        ramps = []
        assumptions += [
            "SITE.ifc and SITE_graph.json not found: masterplan geometry only",
            "carriageways = masterplan carriage_poly; not walkable except at crossings",
            "crossings assumed at every bus stop (4 m strip across the road) and at junction mouths",
            "all crossings assumed to have kerb ramps <= 1:10; ground otherwise level (step-free)",
            "covered = building footprints only (no linkways modelled yet)"]

    walk_sf = walk & ~steps
    cm_short = np.ones((g.nx, g.ny), np.int32)
    cm_cov = np.where(cov, 1, 2).astype(np.int32)
    bus = []
    for b in mp["bus_stops"]:
        sc = _snap(walk, g, *b["at"])
        bus.append(dict(id=b["id"], at=list(b["at"]), cell=sc))
    tg_cells = []
    for t in targets:
        cells = [_snap(walk, g, *p) for p in t["lobbies"]]
        tg_cells.append(cells)
        t["snapped"] = [list(map(float, g.xy(*c))) if c else None for c in cells]

    routes, polylines = [], []
    for b in bus:
        if b["cell"] is None:
            for t in targets:
                routes.append(dict(bus_stop=b["id"], target=t["id"], reachable=False,
                                   reason="bus stop not on walkable ground"))
            continue
        d_short = dial(walk, cm_short, b["cell"])
        d_cov = dial(walk, cm_cov, b["cell"])
        d_sf = dial(walk_sf, cm_short, b["cell"]) if steps.any() else d_short
        for t, cells in zip(targets, tg_cells):
            for li, c in enumerate(cells):
                r = dict(bus_stop=b["id"], target=t["id"], lobby=li, mode=mode)
                if c is None or d_short[c] < 0:
                    routes.append(dict(r, reachable=False, reason="lobby not on walkable ground" if c is None
                                       else "no walkable route"))
                    continue
                ps = pull(trace(d_short, walk, cm_short, c), walk, np.zeros_like(cov), g)
                pc_cells = trace(d_cov, walk, cm_cov, c)
                pc = pull(pc_cells, walk, cov, g)
                Lc, share = _measure(pc, cov, g)
                L = min(_measure(ps, cov, g)[0], Lc)      # both are walkable; the pulled grid paths differ slightly
                xs = sorted({int(xlab[p]) for p in pc_cells if xlab[p]})
                r.update(reachable=True, length_m=round(L, 1), route_m=round(Lc, 1), covered_share=round(share, 3),
                         step_free=bool(d_sf[c] >= 0), crossings=[crossings[k - 1]["id"] for k in xs],
                         lobby_xy=t["snapped"][li],
                         ok_length=bool(L <= MAX_WALK), ok_covered=bool(share >= MIN_COVERED))
                routes.append(r)
                polylines.append((b["id"], pc))
        say(f"  {b['id']}: routes to {len(targets)} targets")
    info = {"mode": mode, "grid": [g.nx, g.ny], "ramps": ramps,
            "crossings": [dict(id=c["id"], kind=c["kind"], area_m2=round(c["poly"].area, 1)) for c in crossings]}
    for r in routes:
        r.pop("polyline", None)
    if png_dir is not None:
        render_site(Path(png_dir) / "nav_site.png", g, walk, cov, enclosed, cw & (xlab == 0), xlab > 0, polylines,
                    bus, targets, f"Site walk test ({mode}): routes from bus stops to lift lobbies, covered-preferring")
    return routes, info


ROUTE_COLOURS = [(200, 40, 40), (30, 90, 200), (20, 140, 60), (150, 60, 170), (220, 130, 0), (0, 150, 160)]


def render_site(path, g: SiteGrid, walk, cov, enclosed, carriage, crossing, polylines, bus, targets, title, px=0.5):
    from PIL import Image, ImageDraw
    from estate.draw.raster import font
    step = max(1, int(round(px / g.cell)))
    img = np.full((g.nx, g.ny, 3), 245, np.uint8)
    img[walk] = (255, 255, 255)
    img[cov & walk] = (205, 225, 245)
    img[enclosed] = (120, 120, 120)
    img[carriage] = (70, 70, 75)
    img[crossing] = (250, 220, 90)
    img = img[::step, ::step].transpose(1, 0, 2)[::-1]
    H = img.shape[0]
    top = 40
    canvas = Image.new("RGB", (img.shape[1] + 20, H + top + 20), (255, 255, 255))
    canvas.paste(Image.fromarray(np.ascontiguousarray(img)), (10, top + 10))
    dr = ImageDraw.Draw(canvas)
    s = g.cell * step

    def P(x, y):
        return 10 + (x - g.x0) / s, top + 10 + H - (y - g.y0) / s

    colour = {b["id"]: ROUTE_COLOURS[i % len(ROUTE_COLOURS)] for i, b in enumerate(bus)}
    for bid, pl in polylines:
        dr.line([P(*p) for p in pl], fill=colour[bid], width=2)
    for t in targets:
        for p in t["snapped"]:
            if p:
                X, Y = P(*p)
                dr.rectangle([X - 3, Y - 3, X + 3, Y + 3], fill=(0, 0, 0))
        if t["lobbies"]:
            X, Y = P(*t["lobbies"][0])
            dr.text((X + 5, Y - 5), t["id"], fill=(0, 0, 0), font=font(10))
    for b in bus:
        X, Y = P(*b["at"])
        dr.ellipse([X - 6, Y - 6, X + 6, Y + 6], fill=colour[b["id"]], outline=(0, 0, 0))
        dr.text((X + 8, Y), b["id"], fill=colour[b["id"]], font=font(12))
    dr.text((10, 8), title, fill=(30, 30, 30), font=font(14))
    dr.text((10, 24), "white walkable, blue covered, dark carriageway, yellow crossing, grey enclosed ground floor",
            fill=(80, 80, 80), font=font(11))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
