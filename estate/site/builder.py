"""SITE.ifc: all external works of the estate, planned from the masterplan and written in IFC4X3 (or IFC4).

``plan_site(mp)`` lays the works out as geometry: roads (roads.py), greens, porches and bus shelters
(amenities.py), then three routed networks on the same raster (linkways.py): driveways from the streets to a
drop-off porch at every block and to the car park and centre service bays; covered linkways joining every
entrance, the car park, the centre and the bus stops; and uncovered footpaths from every entrance and green to
the sidewalks. Where a route crosses a carriageway it becomes a zebra crossing with kerb ramps. Every stair discharge
the car park and centre publish (escape routes, not entrances) gets a paved path straight out to the nearest footway,
clear of the driveways (escape_paths), and an uncovered 'escape' edge in the pedestrian graph. Paved surfaces are
then cut against each other in a fixed priority so no two surfaces overlap; the grass terrain is what is left.

``write_site(layout, path, schema)`` writes the model. IFC4X3 uses IfcRoad facilities with IfcRoadPart parts
(CARRIAGEWAY, INTERSECTION, SIDEWALK, BUS_STOP, PEDESTRIAN_CROSSING) holding IfcPavement / IfcKerb / IfcSlab
(SIDEWALK, PAVING) / IfcRamp elements, and IfcElementAssembly SHELTER for linkways, shelters, porches and
pavilions. IFC4 has none of those, so the same pieces become IfcSlab / IfcCivilElement / USERDEFINED types
contained in the IfcSite. Both files share the project/site identity and map conversion of the building files and
reference every building IFC as a LINKED_MODEL.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from estate import config, env
from estate.site import Piece, amenities, landscape
from estate.site import roads as R
from estate.site.linkways import (H, INDOOR_VIA, INF, V, FreeSpace, Grid, Router, columns_along, end_degrees,
                                  graph_json, graph_stats, pedestrian_graph, seg_rect, simplify, split_cover, split_line)

FOOT_W = 1.8
DRIVE_W = 6.0
PCN_W = 4.0
PAVE_T = 0.15
CARRIAGE_T = 0.30
COL = 0.2
REUSE = 0.45             # cost factor on cells already carrying a linkway when adding direct bus-stop routes
# ground floors as the pedestrian router sees them
GROUND_CUTS = (0.1, 0.45, 0.73, 1.1, 1.7)   # plan sections: stool seats (0.42-0.46), table tops (0.71-0.75), rails
HEADROOM = 2.0           # slabs and roofs whose underside is higher than this give rain cover
BAND = (0.05, 1.8)       # an element reaching into this height band blocks walking
GROUND_VERSION = "ground-v4"
MARGIN = 0.05            # routes keep agent radius + MARGIN from obstacles; the checks allow HIT_TOL inside them
HIT_TOL = 0.05
APRON = 1.5              # outside ring of a building a ground-floor route may use (site ground, around facade columns)
TRUNK_R = 0.25           # tree trunk radius (estate.ifc.vegetation)
COVER_PREF = 2.0         # ground-floor legs: an uncovered metre costs this many covered ones (stay under the slab)
OPEN_MIN_AREA = 20.0     # smallest open-sky floor of a building (m2) the site treats as open ground (a forecourt)
OPEN_MIN_WIDTH = 2.0     # narrower open strips stay part of the building
ESCAPE_W = (1.8, 1.5)    # paved path from a stair discharge to a footway: preferred width, then the narrowest
ESCAPE_REACH = 30.0      # longest straight run from a stair discharge to a footway
ESCAPE_DOOR = 0.25       # the discharge point keeps at least this far inside the path's edge
DRIVE_CLEAR = 0.3        # an escape path keeps this far from any driveway


def _parts(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    return [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon" and not p.is_empty]


def _clean(g, min_area=0.02):
    """Valid polygons only, tiny slivers and holes dropped, near-collinear vertices removed."""
    out = []
    for p in _parts(shapely.make_valid(g) if g is not None and not g.is_empty else None):
        p = p.simplify(0.005)
        if p.is_empty or p.area < min_area:
            continue
        holes = [h for h in p.interiors if Polygon(h).area >= 0.01]
        p = Polygon(p.exterior, holes)
        if not p.is_valid:
            p = shapely.make_valid(p)
            out += [q for q in _parts(p) if q.area >= min_area]
        else:
            out.append(p)
    if not out:
        return Polygon()
    return out[0] if len(out) == 1 else MultiPolygon(out)


def _axis(v):
    return H if abs(v[0]) >= abs(v[1]) else V


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


@dataclass
class Entrance:
    site: object
    E: tuple
    out: tuple
    lobby: tuple
    k: int
    porch: dict | None = None
    planned: tuple | None = None     # the masterplan entrance on the footprint, before clearing it of columns

    @property
    def stub_pt(self):
        return self.porch["stub_pt"] if self.porch else self.E


@dataclass
class SiteLayout:
    mp: dict
    net: object
    sites: list
    entrances: list
    greens: list
    pcn: dict | None
    pieces: list = field(default_factory=list)
    parts: dict = field(default_factory=dict)        # key -> dict(road, predefined, name, usage)
    roads_meta: dict = field(default_factory=dict)   # road key -> name
    assemblies: dict = field(default_factory=dict)   # key -> dict(name, part, props)
    graph: object = None
    graph_stats: dict = field(default_factory=dict)
    draw: dict = field(default_factory=dict)         # geometry for the site plan
    trees: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    routes: dict = field(default_factory=dict)
    grounds: dict = field(default_factory=dict)      # site id -> Ground
    open_ground: dict = field(default_factory=dict)  # site id -> the building's own floor open to the sky (forecourt)
    cover: object = None                             # everything that keeps rain off: canopies + building cover
    obstacles: list = field(default_factory=list)    # [(label, polygon)] the graph is checked against
    graph_check: dict = field(default_factory=dict)
    escapes: list = field(default_factory=list)      # paths from published stair discharges (escape_paths)


# ============================================================================ planning
def _part_near(fp, p):
    """The polygon of fp whose outline is nearest p (the largest one on a tie)."""
    P = Point(p)
    return min(_parts(fp), key=lambda q: (round(q.exterior.distance(P), 6), -q.area))


def _snap_entrance(fp, p):
    poly = _part_near(fp, p)
    ring = poly.exterior
    d = ring.project(Point(p))
    q = ring.interpolate(d)
    a, b = ring.interpolate(max(d - 0.05, 0.0)), ring.interpolate(min(d + 0.05, ring.length))
    t = np.array([b.x - a.x, b.y - a.y])
    if np.linalg.norm(t) < 1e-9:
        t = np.array([1.0, 0.0])
    t = _unit(t)
    n = np.array([-t[1], t[0]])
    if poly.contains(Point(q.x + n[0] * 0.3, q.y + n[1] * 0.3)):
        n = -n
    n = np.array([1.0, 0.0]) * np.sign(n[0]) if abs(n[0]) >= abs(n[1]) else np.array([0.0, 1.0]) * np.sign(n[1])
    return (round(q.x, 3), round(q.y, 3)), (float(n[0]), float(n[1]))


def published(site, model_dir=None):
    """Entrances a building build published in model/<id>/<id>.build.json (car park, centre): pedestrian points,
    vehicle / service points and lift lobbies (used only where the masterplan gives none), and the points where an
    escape stair lets out at ground level ('stair_discharge'; escape_paths pave a way from each to a footway)."""
    p = Path(model_dir or env.MODEL) / site.id / f"{site.id}.build.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ped, veh = [], []
    ent = d.get("entrances") or {}
    if isinstance(ent, dict):
        for k, v in sorted(ent.items()):
            if not isinstance(v, list):
                continue
            pts = [(float(q[0]), float(q[1])) for q in v if isinstance(q, (list, tuple)) and len(q) >= 2]
            (veh if ("vehicle" in k or "service" in k) else ped).extend(pts)
    lob = [(float(q[0]), float(q[1])) for q in d.get("lift_lobbies", []) if isinstance(q, (list, tuple))]
    dis = [(float(q[0]), float(q[1])) for q in d.get("stair_discharge", []) or []
           if isinstance(q, (list, tuple)) and len(q) >= 2]
    return dict(ped=ped, veh=veh, lobbies=lob, discharge=dis)


# ============================================================================ building ground floors
@dataclass
class Ground:
    """A building's ground floor as the pedestrian router and the graph checks see it (estate coordinates), read
    from the building's own IFC so routes follow what was actually built, hand edits of frozen blocks included."""
    id: str
    source: str                      # the IFC read (project-relative) or "footprint" when there is none yet
    sha256: str | None = None
    obstacles: list = field(default_factory=list)     # [(label, Polygon)]: plan sections of what blocks walking
    floor: object = None             # slabs with their top at finished ground level (0.0)
    cover: object = None             # slabs and roofs above head height

    @property
    def blocked(self):
        return unary_union([p for _, p in self.obstacles] + [Polygon()])


def _sections(verts, faces, heights):
    """Union of the horizontal sections of a closed triangle mesh at the given heights (None when it misses).

    Each crossed triangle gives one segment; the crossing point of an edge is computed from its end points in a
    fixed (lexicographic) order, so the two triangles sharing an edge produce the very same point and the
    segments close into rings. Rings are filled (a hollow section counts as solid: conservative for walking)."""
    T = verts[faces]
    out = []
    for z in heights:
        dz = T[:, :, 2] - z
        sg = dz >= 0.0
        idx, pts = [], []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            m = sg[:, i] != sg[:, j]
            A, B = T[m, i], T[m, j]
            swap = (A[:, 0] > B[:, 0]) | ((A[:, 0] == B[:, 0]) & ((A[:, 1] > B[:, 1]) |
                                                                  ((A[:, 1] == B[:, 1]) & (A[:, 2] > B[:, 2]))))
            A, B = np.where(swap[:, None], B, A), np.where(swap[:, None], A, B)
            t = (z - A[:, 2]) / (B[:, 2] - A[:, 2])
            idx.append(np.flatnonzero(m))
            pts.append(np.round(A[:, :2] + t[:, None] * (B[:, :2] - A[:, :2]), 7))
        idx, pts = np.concatenate(idx), np.concatenate(pts)
        if not len(idx):
            continue
        order = np.argsort(idx, kind="stable")
        idx, pts = idx[order], pts[order]
        _, cnt = np.unique(idx, return_counts=True)
        pts = pts[np.repeat(cnt == 2, cnt)].reshape(-1, 2, 2)
        pts = pts[np.any(pts[:, 0] != pts[:, 1], axis=1)]
        if not len(pts):
            continue
        lines = shapely.node(shapely.multilinestrings(pts))      # solids of one element may overlap
        polys = shapely.get_parts(shapely.polygonize(shapely.get_parts(lines)))
        out.append(shapely.union_all(polys) if len(polys) else lines.buffer(0.01))
    return shapely.union_all(out).buffer(0) if out else None


def _ground_from_ifc(path):
    """(obstacles, floor, cover) of the lowest storey of a building IFC: every element of that storey (and its
    parts: stair flights, landings, railings) that reaches into BAND, except spaces and doors that open; slabs
    with their top at 0.0 as floor; slabs and roofs of the two lowest storeys above HEADROOM as cover."""
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.element as uel
    f = ifcopenshell.open(str(path))
    storeys = sorted(f.by_type("IfcBuildingStorey"), key=lambda s: (float(s.Elevation or 0.0), s.GlobalId))

    def members(st):
        out = []
        todo = [e for rel in (st.ContainsElements or []) for e in rel.RelatedElements]
        while todo:
            e = todo.pop()
            out.append(e)
            todo += [c for rel in (getattr(e, "IsDecomposedBy", None) or []) for c in rel.RelatedObjects]
        return out
    inc = {}
    for k, st in enumerate(storeys[:2]):
        for e in members(st):
            if not e.is_a("IfcElement") or e.is_a("IfcOpeningElement"):
                continue
            if k > 0 and not (e.is_a("IfcSlab") or e.is_a("IfcRoof") or e.is_a("IfcCovering")):
                continue
            if e.is_a("IfcDoor") and uel.get_psets(e).get("Navigation", {}).get("Passable", True):
                continue
            inc[e.id()] = e
    if not inc:
        return [], None, None
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    shapes = {}
    it = ifcopenshell.geom.iterator(s, f, 4, include=[inc[k] for k in sorted(inc)])
    if it.initialize():
        while True:
            sh = it.get()
            shapes[sh.guid] = (sh.type, np.array(sh.geometry.verts, float).reshape(-1, 3),
                               np.array(sh.geometry.faces, dtype=np.int64).reshape(-1, 3))
            if not it.next():
                break
    obstacles, floor, cover = [], [], []
    for guid in sorted(shapes):                 # threads finish in any order; the result must not depend on it
        cls, v, fc = shapes[guid]
        if not len(fc):
            continue
        z0, z1 = float(v[:, 2].min()), float(v[:, 2].max())
        if cls in ("IfcSlab", "IfcRoof", "IfcCovering", "IfcPlate") and z0 >= HEADROOM:
            g = _sections(v, fc, [(z0 + z1) / 2])
            if g is not None:
                cover.append(g)
        elif cls == "IfcSlab" and abs(z1) <= 0.06:
            g = _sections(v, fc, [(z0 + z1) / 2])
            if g is not None:
                floor.append(g)
        elif z0 < BAND[1] and z1 > BAND[0]:
            g = _sections(v, fc, [h for h in GROUND_CUTS if z0 < h < z1] or [(z0 + z1) / 2])
            if g is not None and not g.is_empty:
                obstacles.append((f"{cls} {f.by_guid(guid).Name or guid}", g))
    return obstacles, (unary_union(floor) if floor else None), (unary_union(cover) if cover else None)


def read_ground(site, model_dir=None) -> Ground:
    """The ground floor of a building from model/<id>/<id>.ifc (cached by the file's sha256 in build/cache), or
    its footprint, walkable, covered and empty, while that IFC does not exist yet."""
    import pickle

    from estate.export.meshcache import sha256
    p = Path(model_dir or env.MODEL) / site.id / f"{site.id}.ifc"
    if not p.exists():
        return Ground(site.id, "footprint", None, [], site.footprint, site.footprint)
    sha = sha256(p)
    cache = env.BUILD / "cache" / f"{p.stem}-{sha[:16]}-{GROUND_VERSION}.ground.pkl"
    data = None
    if cache.exists():
        try:
            data = pickle.loads(cache.read_bytes())
        except Exception:  # noqa: BLE001  (a stale or truncated cache is simply recomputed)
            data = None
    if data is None:
        obs, fl, cv = _ground_from_ifc(p)
        data = dict(obstacles=[(lb, shapely.to_wkb(g)) for lb, g in obs],
                    floor=shapely.to_wkb(fl) if fl is not None else None,
                    cover=shapely.to_wkb(cv) if cv is not None else None)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".{__import__('os').getpid()}.tmp")
            tmp.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
            tmp.replace(cache)
        except OSError:
            pass
    wkb = shapely.from_wkb
    return Ground(site.id, env.rel(p), sha, [(lb, wkb(g)) for lb, g in data["obstacles"]],
                  wkb(data["floor"]) if data["floor"] is not None else Polygon(),
                  wkb(data["cover"]) if data["cover"] is not None else Polygon())


OVERHEAD = ("IfcSlab", "IfcRoof", "IfcCovering", "IfcPlate", "IfcRamp", "IfcRampFlight", "IfcStair", "IfcStairFlight",
            "IfcBeam", "IfcMember")
OVERHEAD_VERSION = "overhead-v1"


def _overhead(path, sha):
    """Plan projection (union of triangles) of every slab, roof, ramp, stair, beam or member of a building IFC, any
    storey, that reaches above HEADROOM: what keeps the sky off its ground floor. Cached by the file's sha256."""
    import pickle
    cache = env.BUILD / "cache" / f"{Path(path).stem}-{sha[:16]}-{OVERHEAD_VERSION}.pkl"
    if cache.exists():
        try:
            return shapely.from_wkb(pickle.loads(cache.read_bytes()))
        except Exception:  # noqa: BLE001  (a stale or truncated cache is simply recomputed)
            pass
    import ifcopenshell
    import ifcopenshell.geom
    f = ifcopenshell.open(str(path))
    els = sorted({e.id(): e for c in OVERHEAD for e in f.by_type(c)}.values(), key=lambda e: e.id())
    polys = []
    if els:
        s = ifcopenshell.geom.settings()
        s.set("use-world-coords", True)
        it = ifcopenshell.geom.iterator(s, f, 1, include=els)
        if it.initialize():
            while True:
                sh = it.get()
                v = np.array(sh.geometry.verts, float).reshape(-1, 3)
                fc = np.array(sh.geometry.faces, dtype=np.int64).reshape(-1, 3)
                if len(fc) and float(v[:, 2].max()) > HEADROOM:
                    tris = shapely.polygons(v[fc][:, :, :2])
                    tris = tris[shapely.area(tris) > 1e-8]
                    if len(tris):
                        polys.append(shapely.union_all(tris))
                if not it.next():
                    break
    g = shapely.union_all(polys).buffer(0) if polys else Polygon()
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{__import__('os').getpid()}.tmp")
        tmp.write_bytes(pickle.dumps(shapely.to_wkb(g), protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(cache)
    except OSError:
        pass
    return g


def open_ground(site, gr, model_dir=None) -> Polygon:
    """The part of a building's footprint that is the building's own floor slab open to the sky (the centre's paved
    forecourt): nothing above head height over it at any storey (no slab, roof, ramp or stair) and nothing standing
    on it, strips narrower than OPEN_MIN_WIDTH and pieces under OPEN_MIN_AREA left out. The site treats it as open
    ground: linkways and footpaths may cross it and be roofed there (canopies and their columns are site elements
    standing on the slab), but the site never paves it, the slab being the building's. Empty while the building has
    no IFC or no such floor."""
    if gr.source == "footprint" or gr.floor is None or gr.floor.is_empty:
        return Polygon()
    r = OPEN_MIN_WIDTH / 2

    def opened(g):
        g = shapely.make_valid(g.buffer(-r, join_style="mitre").buffer(r, join_style="mitre"))
        parts = sorted((p for p in _parts(shapely.make_valid(g.intersection(site.footprint))) if p.area >= OPEN_MIN_AREA),
                       key=lambda q: (round(q.bounds[0], 3), round(q.bounds[1], 3)))
        return unary_union(parts).buffer(0) if parts else Polygon()
    g = site.footprint.intersection(gr.floor)
    if gr.cover is not None and not gr.cover.is_empty:           # the two lowest storeys' slabs and roofs (cheap)
        g = g.difference(gr.cover)
    if not gr.blocked.is_empty:
        g = g.difference(gr.blocked)
    g = opened(g)
    if g.is_empty:
        return g
    p = Path(model_dir or env.MODEL) / site.id / f"{site.id}.ifc"     # only then the storeys above (ramps, voids)
    over = _overhead(p, gr.sha256) if p.exists() and gr.sha256 else Polygon()
    return opened(g.difference(over)) if not over.is_empty else g


def _piece_obstacles(pieces):
    """[(name, plan polygon)] of the site pieces that reach into BAND: columns, furniture, fences, tree trunks."""
    out = []
    for p in pieces:
        sol = p.solid or {}
        if "tree" in sol:
            x, y, _ = sol["at"]
            out.append((p.name, Point(x, y).buffer(TRUNK_R, quad_segs=2)))
        elif "boxes" in sol:
            parts = [b["box"] for b in sol["boxes"] if b["z0"] < BAND[1] and b["z0"] + b["h"] > BAND[0]]
            if parts:
                out.append((p.name, unary_union(parts)))
        elif "type" in sol:
            _, size, hgt = sol["type"]
            if p.z0 < BAND[1] and p.z0 + hgt > BAND[0]:
                c = p.geom.centroid
                out.append((p.name, box(c.x - size / 2, c.y - size / 2, c.x + size / 2, c.y + size / 2)))
        elif "wedge" not in sol and p.kind in ("column", "furniture", "fence") and p.geom is not None \
                and p.z0 < BAND[1] and p.z0 + p.h > BAND[0]:
            out.append((p.name, p.geom))
    return out


def edge_hits(G, obstacles, tol=HIT_TOL):
    """Graph edges whose centreline runs more than tol through an obstacle:
    [dict(a, b, kind, via, obstacle, length)]."""
    if not obstacles:
        return []
    labels = [lb for lb, _ in obstacles]
    polys = [g for _, g in obstacles]
    tree = shapely.STRtree(polys)
    out = []
    for u, v in sorted((min(e), max(e)) for e in G.edges):
        ln = LineString([u, v])
        for i in sorted(int(k) for k in tree.query(ln)):
            L = ln.intersection(polys[i]).length
            if L > tol:
                d = G.edges[u, v]
                out.append(dict(a=u, b=v, kind=d["kind"], via=d.get("via"), obstacle=labels[i], length=round(L, 3)))
    return out


def _clear_entrance(fp, E, n, blocked, clearance, reach=4.0, step=0.1):
    """E slid along its facade (normal n kept) to the nearest point whose clearance disk misses every obstacle; a
    void deck entrance must not land on a facade column. Returns (E, moved by)."""
    if blocked.is_empty or not Point(E).buffer(clearance).intersects(blocked):
        return E, 0.0
    t = np.array([-n[1], n[0]])
    fp = _part_near(fp, E)
    ring = fp.exterior
    for k in range(1, int(reach / step) + 1):
        for sgn in (1, -1):
            q = np.asarray(E, float) + t * sgn * k * step
            Q = Point(q)
            if ring.distance(Q) > 0.02 or fp.contains(Point(q + np.asarray(n) * 0.3)):
                continue
            if not Q.buffer(clearance).intersects(blocked):
                return (round(float(q[0]), 3), round(float(q[1]), 3)), round(k * step, 2)
    return E, 0.0


def _entrances(sites, centre, pub, grounds=None, clearance=0.35, notes=None, shapes=None):
    """Entrances of every site, snapped onto the outline of what the site routes around: the footprint, or for a
    building with open ground (the centre's forecourt) the roofed parts of it (shapes[site id]), so linkways and
    footpaths cross the forecourt to the hawker hall and shop doors instead of stopping at its edge."""
    out = []
    for s in sites:
        shape = (shapes or {}).get(s.id, s.footprint)
        pts = list(s.entrances)
        pb = pub.get(s.id)
        if not pts and pb:
            ring = max(_parts(s.footprint), key=lambda q: q.area).exterior
            for q in pb["ped"]:
                if ring.distance(Point(q)) <= 1.0 and all(math.dist(q, r) > 6.0 for r in pts):
                    pts.append(q)
        if not pts:      # car park / centre without explicit entrances: the side facing the estate centre
            x0, y0, x1, y1 = s.footprint.bounds
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = centre[0] - cx, centre[1] - cy
            if abs(dx) * (y1 - y0) >= abs(dy) * (x1 - x0):
                pts = [(x1 if dx > 0 else x0, cy)]
            else:
                pts = [(cx, y1 if dy > 0 else y0)]
        lobbies = (pb["lobbies"] if pb and pb["lobbies"] and not s.entrances else list(s.lift_lobbies)) or \
            [tuple(s.footprint.centroid.coords[0])]
        blocked = grounds[s.id].blocked if grounds and s.id in grounds else Polygon()
        for k, p in enumerate(pts):
            E0, n = _snap_entrance(shape, p)
            E, moved = _clear_entrance(shape, E0, n, blocked, clearance)
            if moved and notes is not None:
                notes.append(f"{s.id} entrance {k}: moved {moved} m along the facade, clear of the ground floor")
            lob = min(lobbies, key=lambda q: (math.dist(q, E), q))
            out.append(Entrance(s, E, n, tuple(float(v) for v in lob), k, planned=E0))
    return out


def escape_paths(sites, pub, walk, blocked, drives, floors=None, extent=None):
    """A paved path from every stair discharge a building published (build.json 'stair_discharge': where an escape
    stair lets out at ground level) to the nearest footway. Escape routes are not entrances, so no linkway or
    footpath is routed to them. The path runs straight out from the facade until it meets `walk` (sidewalks,
    footpaths, linkway paths, platforms ...): ESCAPE_W[0] wide, or ESCAPE_W[-1] where that does not fit, shifted
    along the facade as little as possible while the discharge point stays at least ESCAPE_DOOR inside the path.
    It stays off `blocked` (carriageways, kerbs, ramps, other buildings, fenced greens) and DRIVE_CLEAR from any
    driveway. A discharge inside the footprint lets out onto the building's own floor (floors[site id]) and needs no
    site path.

    Returns (escapes, warnings). Each escape is dict(site, k, at, start, out, width, offset, length, poly, end) with
    poly None where the door opens straight onto a footway or the building's floor (on_floor), or where no path
    fits (failed, with a warning)."""
    shapely.prepare(walk)
    shapely.prepare(blocked)
    out, warnings = [], []
    for s in sorted(sites, key=lambda q: q.id):
        for k, q in enumerate((pub.get(s.id) or {}).get("discharge") or []):
            Q = Point(q)
            rec = dict(site=s, k=k, at=(round(q[0], 3), round(q[1], 3)), start=None, out=None, width=None, offset=0.0,
                       length=0.0, poly=None, end=None)
            if s.footprint.buffer(-0.05).contains(Q):
                fl = (floors or {}).get(s.id)
                rec["on_floor"] = True
                if fl is None or fl.is_empty or fl.distance(Q) > 0.05:
                    warnings.append(f"{s.id} stair discharge {k + 1} at {rec['at']}: inside the footprint, "
                                    f"off its floor")
                out.append(rec)
                continue
            E, n = _snap_entrance(s.footprint, q)
            P = np.asarray(q if s.footprint.exterior.distance(Q) > 0.05 else E, float)
            nv = np.asarray(n, float)
            tv = np.array([-nv[1], nv[0]])
            found = None
            for w in ESCAPE_W:
                steps = int(round((w / 2 - ESCAPE_DOOR) / 0.05))
                offs = sorted({0.0} | {round(sg * i * 0.05, 3) for i in range(1, steps + 1) for sg in (1, -1)},
                              key=lambda o: (abs(o), o))
                for o in offs:
                    c = P + tv * o
                    corridor = seg_rect(tuple(c), tuple(c + nv * ESCAPE_REACH), w / 2)
                    hits = [g for g in _parts(corridor.intersection(walk)) if g.area >= 0.05]
                    if not hits:
                        continue
                    d = max(0.0, min(float(np.dot(np.asarray(g.exterior.coords) - c, nv).min()) for g in hits))
                    if d <= 0.05:                                 # the door opens straight onto a footway
                        found = (w, o, d, None)
                        break
                    poly = seg_rect(tuple(c), tuple(c + nv * d), w / 2)
                    if poly.intersection(blocked).area > 1e-3 or (extent is not None and not extent.covers(poly)):
                        continue
                    if drives is not None and not drives.is_empty and poly.distance(drives) < DRIVE_CLEAR - 1e-6:
                        continue
                    found = (w, o, d, poly)
                    break
                if found:
                    break
            if found is None:
                rec["failed"] = True
                warnings.append(f"{s.id} stair discharge {k + 1} at {rec['at']}: no straight path of "
                                f"{ESCAPE_W[-1]} m or more to a footway within {ESCAPE_REACH} m, "
                                f"{DRIVE_CLEAR} m clear of driveways")
                out.append(rec)
                continue
            w, o, d, poly = found
            c = P + tv * o
            rec.update(start=(round(float(P[0]), 3), round(float(P[1]), 3)), out=(float(nv[0]), float(nv[1])), width=w,
                       offset=o, length=round(d, 3), poly=poly,
                       end=(round(float(P[0] + nv[0] * d), 3), round(float(P[1] + nv[1] * d), 3)))
            out.append(rec)
    return out, warnings


def _escape_lines(escapes, lines, reach=8.0):
    """Pedestrian-graph lines of the escape paths: from each stair discharge straight along its path to the first
    footway centreline it meets (else from the path's far end to the nearest one within reach), kind 'escape' (never
    counted as covered). Returns (lines, points)."""
    cands = [ln for ln, a in lines if a["kind"] in ("sidewalk", "footpath", "linkway", "park")
             and a.get("via") not in INDOOR_VIA and ln.length > 1e-6]
    tree = shapely.STRtree(cands) if cands else None
    out, points = [], []
    for e in escapes:
        if e["start"] is None or tree is None:
            continue
        P, n = np.asarray(e["start"], float), np.asarray(e["out"], float)
        ray = LineString([tuple(P), tuple(P + n * (e["length"] + reach))])
        best = None
        for i in sorted(int(x) for x in tree.query(ray)):
            for g in shapely.get_parts(ray.intersection(cands[i])):
                for c in g.coords:
                    s_ = float(np.dot(np.asarray(c[:2]) - P, n))
                    if s_ > 0.05 and (best is None or s_ < best[0]):
                        best = (s_, (float(c[0]), float(c[1])))
        if best is not None:
            pts = [tuple(P), best[1]]
        else:
            E = Point(e["end"])
            i = min(sorted(int(x) for x in tree.query(E.buffer(reach))), key=lambda j: (cands[j].distance(E), j),
                    default=None)
            if i is None or cands[i].distance(E) > reach:
                continue
            q = cands[i].interpolate(cands[i].project(E))
            pts = [tuple(P), tuple(e["end"]), (q.x, q.y)]
        e["graph_end"] = (round(pts[-1][0], 3), round(pts[-1][1], 3))
        out.append((LineString(pts), dict(kind="escape", width=e["width"], via="stair_discharge")))
        points.append(dict(xy=tuple(P), kind="stair_discharge", ref=e["site"].id))
    return out, points


def _pcn(cfg, net, extent):
    pc = cfg.get("park_connector")
    if not pc:
        return None
    x0, y0, x1, y1 = pc["rect"]
    rect = box(x0, y0, x1, y1)
    vertical = (y1 - y0) >= (x1 - x0)
    if vertical:
        cx = (x0 + x1) / 2
        a, b = (cx, y0 - 12.0), (cx, y1 + 12.0)
    else:
        cy = (y0 + y1) / 2
        a, b = (x0 - 12.0, cy), (x1 + 12.0, cy)
    full = LineString([a, b])
    walk = unary_union([ln for ln, *_ in net.sidewalk_lines])
    hits = full.intersection(walk)
    pts = [g for g in getattr(hits, "geoms", [hits]) if g.geom_type == "Point"]
    s_lo = max([full.project(p) for p in pts if full.project(p) <= 12.5] or [12.0])
    s_hi = min([full.project(p) for p in pts if full.project(p) >= full.length - 12.5] or [full.length - 12.0])
    line = LineString([full.interpolate(s_lo).coords[0], full.interpolate(s_hi).coords[0]])
    path = line.buffer(PCN_W / 2, cap_style="flat").difference(net.C.buffer(R.KERB_W))
    return dict(id=pc.get("id", "PCN"), name=pc.get("name", "Park connector"), rect=rect, line=line,
                path=path.intersection(extent), width=PCN_W, area=rect.buffer(-0.5))


def plan_site(mp: dict, log=None, model_dir=None) -> SiteLayout:
    """Lay out the external works and the pedestrian graph. The building IFCs under model_dir (default model/) are
    read for their ground floors: the graph's legs through void decks and the centre are routed around what was
    built there, and the site paves whatever part of a footprint the building leaves without a floor slab. A
    building's own floor open to the sky (the centre's forecourt, open_ground) is open ground: linkways and footpaths
    cross it to the doors of the roofed buildings and canopies with their columns roof it, but it is never paved."""
    t0 = time.time()
    say = log or (lambda *a: None)
    cfg = mp["cfg"]
    seed = int(cfg["estate"]["seed"])
    extent = mp["extent"]
    lk = dict(clear_width=2.4, canopy_width=3.0, soffit=2.7, column_spacing=6.0)
    lk.update(cfg.get("linkways", {}))
    lk = {k: v for k, v in lk.items() if k != "mode"}
    sites = list(mp["sites"])
    fps = unary_union([s.footprint for s in sites]).buffer(0).intersection(extent)
    clearance = float(cfg.get("agent", {}).get("radius", 0.3)) + MARGIN
    grounds = {s.id: read_ground(s, model_dir) for s in sites}
    t_grounds = round(time.time() - t0, 2)
    floors = unary_union([g.floor for g in grounds.values() if g.floor is not None] + [Polygon()]).buffer(0)
    # a building's floor open to the sky (the centre's forecourt) is open ground to the linkway and footpath routers:
    # they go round `masses` (the footprints without it) and land on the roofed buildings' own doors
    opens = {s.id: open_ground(s, grounds[s.id], model_dir) for s in sites}
    open_u = unary_union(list(opens.values()) + [Polygon()]).buffer(0)
    masses = fps if open_u.is_empty else shapely.make_valid(fps.difference(open_u)).buffer(0)
    shapes = {}
    for s in sites:
        if not opens[s.id].is_empty:
            rest = [p for p in _parts(shapely.make_valid(s.footprint.difference(opens[s.id]))) if p.area >= 1.0]
            shapes[s.id] = unary_union(rest) if rest else s.footprint
    net = R.layout_roads(mp)
    pub = {s.id: published(s, model_dir) for s in sites if s.kind != "block"}
    pub = {k: v for k, v in pub.items() if v}
    notes = [f"{sid}: {round(g.area)} m2 of its floor open to the sky (forecourt) carries site linkways and "
             f"footpaths" for sid, g in sorted(opens.items()) if not g.is_empty]
    ents = _entrances(sites, extent.centroid.coords[0], pub, grounds, clearance, notes, shapes)
    obstacles = unary_union([fps, net.reserve])
    greens = [amenities.layout_green(g, obstacles) for g in mp.get("greens", [])]
    pcn = _pcn(cfg, net, extent)
    L = SiteLayout(mp, net, sites, ents, greens, pcn)
    L.notes += notes
    L.grounds = grounds
    L.open_ground = {sid: g for sid, g in opens.items() if not g.is_empty}
    L.timings["grounds"] = t_grounds
    L.timings["roads+greens"] = round(time.time() - t0 - t_grounds, 2)
    green_block = unary_union([g["block"] for g in greens] + [Polygon()])
    green_rects = unary_union([g["rect"] for g in greens] + [Polygon()])
    green_paths = unary_union([g["paths"] for g in greens] + [Polygon()])
    lawns = unary_union([lw for g in greens for lw in g["lawns"]] + [Polygon()])
    bays = unary_union([b.bay for b in net.bus_stops] + [Polygon()])
    platforms = unary_union(net.platforms + [Polygon()])
    jboxes = unary_union([box(j["p"][0] - j["half"], j["p"][1] - j["half"], j["p"][0] + j["half"], j["p"][1] + j["half"])
                          for j in net.junctions] + [Polygon()])

    def base_grid():
        g = Grid(extent.bounds, 2.0)
        for r in net.roads:
            perp = V if r.axis == H else H
            g.set(r.band(-r.reserve / 2, r.reserve / 2), axes=perp)
        return g

    # ------------------------------------------------------------------ driveways and porches
    t = time.time()
    porch_cands = {}
    for e in ents:
        if e.site.kind != "block":
            continue
        pg = amenities.porch_geometry(e.E, e.out)
        others = unary_union([s.footprint for s in sites if s is not e.site] + [Polygon()])
        bad = (pg["area"].intersects(others) or pg["area"].intersects(net.reserve) or pg["area"].intersects(green_block)
               or pg["area"].buffer(-0.01).intersects(e.site.footprint) or not extent.contains(pg["area"]))
        if not bad:
            porch_cands.setdefault(e.site.id, []).append((e, pg))
    service = []
    for s in sites:
        if s.kind == "block":
            continue
        ped = [e for e in ents if e.site is s]
        best = None
        x0, y0, x1, y1 = s.footprint.bounds
        veh = pub.get(s.id, {}).get("veh", []) if s.id in pub else []
        if veh:
            P, n = _snap_entrance(s.footprint, veh[0])
            bg = amenities.bay_geometry(P, n)
            if extent.contains(bg["bay"]) and not bg["bay"].buffer(-0.05).intersects(net.reserve):
                service.append((s, bg))
                continue
        for P, n in (((x0 + 0.3 * (x1 - x0), y0), (0, -1)), ((x1, y0 + 0.3 * (y1 - y0)), (1, 0)),
                     ((x1 - 0.3 * (x1 - x0), y1), (0, 1)), ((x0, y1 - 0.3 * (y1 - y0)), (-1, 0))):
            if any(abs(e.out[0] - n[0]) < 1e-6 and abs(e.out[1] - n[1]) < 1e-6 for e in ped):
                continue
            bg = amenities.bay_geometry(P, n)
            if bg["bay"].intersects(net.reserve) or bg["bay"].intersects(fps.difference(s.footprint)) or \
                    bg["bay"].intersects(green_rects) or not extent.contains(bg["bay"]):
                continue
            d = net.C.distance(bg["bay"])
            if best is None or d < best[0]:
                best = (d, P, n, bg)
        if best:
            service.append((s, best[3]))
    g = base_grid()
    g.set(fps.buffer(DRIVE_W / 2 + 1.0), cost=INF)
    g.set(green_rects, cost=6.0)
    g.set(green_block.buffer(1.0), cost=INF)
    if pcn:
        g.set(pcn["rect"], cost=INF)
    g.set(net.C, cost=INF)
    res_mask = g.inside(net.reserve)
    for zone in (jboxes.buffer(15.0), bays.buffer(6.0), platforms.buffer(3.0),
                 unary_union([c.strip.buffer(6.0) for c in net.crossings] + [Polygon()])):
        g.set(mask=g.inside(zone) & res_mask, cost=INF)
    for cands in porch_cands.values():
        for e, pg in cands:
            g.set(pg["canopy"].difference(pg["lane"]), cost=INF)
    for cands in porch_cands.values():
        for e, pg in cands:
            g.set(pg["lane"], cost=1.0, axes=H | V)
    for s, bg in service:
        g.set(bg["bay"], cost=1.0, axes=H | V)
    rt = Router(g, turn=8.0)
    ROAD = rt.add_virtual(kind="road")
    attach = g.inside(net.C.buffer(2.2).difference(net.C).intersection(net.reserve)) & np.isfinite(g.cost)
    for j, i in zip(*np.nonzero(attach)):
        r = net.road_at(g.xy(i, j))
        if r is None:
            continue
        layer = 1 if r.axis == H else 0
        rt.link(ROAD, rt.node(int(i), int(j), layer), 0.0)
    cand_nodes = {}
    for sid, cands in sorted(porch_cands.items()):
        for e, pg in cands:
            ax = _axis(pg["along"])
            v = rt.add_virtual(point=pg["drive_pt"], axis=ax, site=sid, k=e.k)
            if rt.attach(v, *pg["drive_pt"], ax, direction=pg["out"], max_steps=1):
                cand_nodes.setdefault(sid, []).append((v, e, pg))
    dist = rt.distances([ROAD])
    terms, chosen = [], {}
    for sid, lst in sorted(cand_nodes.items()):
        v, e, pg = min(lst, key=lambda x: (dist[x[0]], x[1].k))
        if not math.isfinite(dist[v]):
            L.warnings.append(f"{sid}: no driveway route to a drop-off porch")
            continue
        terms.append(v)
        chosen[sid] = (e, pg)
        e.porch = pg
    for s in sites:
        if s.kind == "block" and s.id not in chosen:
            L.warnings.append(f"{s.id}: no feasible drop-off porch")
    for s, bg in service:
        v = rt.add_virtual(point=bg["drive_pt"], axis=_axis(bg["out"]), site=s.id, k=-1)
        if rt.attach(v, *bg["drive_pt"], _axis(bg["out"]), direction=bg["out"], max_steps=2):
            terms.append(v)
    paths, unreach = rt.steiner([ROAD], terms)
    if unreach:
        L.warnings.append(f"driveways: {len(unreach)} terminals unreachable")
    droute = simplify(rt, paths)
    L.routes["driveways"] = droute
    dsegs = list(droute.segments) + _stub_segments(droute)
    deg = end_degrees(dsegs)
    drive_polys, drive_lines = [], []
    for p0, p1 in dsegs:
        ends = []
        for p in (p0, p1):
            k = (round(p[0], 2), round(p[1], 2))
            ends.append(deg.get(k, 0))
        q0, q1 = p0, p1
        for idx, p in ((0, p0), (1, p1)):
            if deg.get((round(p[0], 2), round(p[1], 2)), 0) == 1 and net.reserve.contains(Point(p)):
                r = net.road_at(p)
                if r is None:
                    continue
                s_, d_ = r.sd(p)
                ko = R.kerb_offsets(net, r, s_)
                if ko is None:
                    continue
                dk = ko[1] if d_ > 0 else ko[0]
                kp = tuple(float(v) for v in r.pt(s_, dk))
                inner = p1 if idx == 0 else p0
                co = R.crossover_at(net, kp, inner, DRIVE_W, f"Driveway crossover {len(net.crossovers) + 1}")
                top = co["top"] if co else kp
                if idx == 0:
                    q0 = top
                else:
                    q1 = top
        e0 = DRIVE_W / 2 if ends[0] > 1 else 0.0
        e1 = DRIVE_W / 2 if ends[1] > 1 else 0.0
        drive_polys.append(seg_rect(q0, q1, DRIVE_W / 2, e0, e1))
        drive_lines.append((q0, q1))
    lanes = [pg["lane"] for e, pg in chosen.values()] + [bg["bay"] for s, bg in service]
    crossover_u = unary_union([c["poly"] for c in net.crossovers] + [Polygon()])
    drives = unary_union(drive_polys + lanes + [Polygon()]).difference(fps).difference(net.C.buffer(R.KERB_W, join_style="mitre"))
    drives = drives.difference(crossover_u).intersection(extent)
    L.timings["driveways"] = round(time.time() - t, 2)
    say(f"  driveways: {len(dsegs)} segments, {len(net.crossovers)} crossovers, {len(chosen)} porches "
        f"({L.timings['driveways']} s)")

    # ------------------------------------------------------------------ covered linkways
    t = time.time()
    porch_canopies = unary_union([pg["canopy"] for e, pg in chosen.values()] + [Polygon()])
    aprons = unary_union([pg["apron"] for e, pg in chosen.values()] + [Polygon()])
    lanes_u = unary_union([pg["lane"] for e, pg in chosen.values()] + [Polygon()])
    g = base_grid()
    g.set(net.reserve, cost=2.5)
    g.set(net.C, cost=10.0)
    for zone in (jboxes.buffer(6.0), bays.buffer(4.0), crossover_u.buffer(3.0)):
        g.set(mask=g.inside(zone) & g.inside(net.C.buffer(R.RAMP_LEN + 0.5)), cost=INF)
    for c in net.crossings:
        g.set(unary_union([c.strip] + [rp["poly"] for rp in c.ramps]), cost=1.0)
    g.set(green_rects, cost=1.5)
    g.set(lawns, cost=3.0)
    g.set(green_paths, floor=0.8)
    if pcn:
        g.set(pcn["rect"], cost=3.0)
    g.set(platforms, cost=1.0)
    for (p0, p1), poly in zip(drive_lines, drive_polys):
        ax = _axis((p1[0] - p0[0], p1[1] - p0[1]))
        g.set(poly.difference(net.reserve), cost=4.0, axes=V if ax == H else H)
    for e, pg in chosen.values():
        g.set(pg["lane"], cost=2.0, axes=V if _axis(pg["along"]) == H else H)
        g.set(pg["apron"], cost=1.0, axes=H | V)
    g.set(green_block.buffer(lk["canopy_width"] / 2 + 0.2), cost=INF)
    g.set(masses.buffer(lk["canopy_width"] / 2 + 0.2), cost=INF)
    rt = Router(g, turn=6.0)
    tnodes = []
    hubs = {}
    by_site = {}
    for e in ents:
        by_site.setdefault(e.site.id, []).append(e)
    for e in ents:
        P = e.stub_pt
        ax = _axis(e.out)
        v = rt.add_virtual(point=P, axis=ax, site=e.site.id, k=e.k, kind="entrance")
        if rt.attach(v, P[0] + e.out[0] * 2.0, P[1] + e.out[1] * 2.0, ax, direction=e.out, max_steps=5,
                     any_layer=True) is None:
            L.warnings.append(f"{e.site.id} entrance {e.k}: no free cell for a linkway")
            continue
        tnodes.append((v, e))
    for sid, lst in by_site.items():
        vs = [(v, e) for v, e in tnodes if e.site.id == sid]
        if len(vs) < 2:
            continue
        cx = sum(e.E[0] for _, e in vs) / len(vs)
        cy = sum(e.E[1] for _, e in vs) / len(vs)
        hub = rt.add_virtual(hub=True, site=sid)
        hubs[sid] = hub
        for v, e in vs:
            rt.link(hub, v, 0.5 * math.dist(e.E, (cx, cy)))
    bs_nodes = []
    for bs in net.bus_stops:
        ax = _axis(bs.out)
        v = rt.add_virtual(point=bs.stub, axis=ax, kind="bus_stop", ref=bs.id)
        if rt.attach(v, bs.stub[0] + bs.out[0] * 2.0, bs.stub[1] + bs.out[1] * 2.0, ax, direction=bs.out,
                     max_steps=6) is None:
            L.warnings.append(f"{bs.id}: no free cell for a linkway")
            continue
        bs_nodes.append(v)
    nc = [v for v, e in tnodes if e.site.kind == "nc"]
    root = nc[0] if nc else (bs_nodes[0] if bs_nodes else tnodes[0][0])
    terms = [v for v, _ in tnodes] + bs_nodes
    paths, unreach = rt.steiner([root], [x for x in terms if x != root])
    if unreach:
        L.warnings.append(f"linkways: {len(unreach)} terminals unreachable: "
                          f"{[rt.virtual[u].get('site') or rt.virtual[u].get('ref') for u in unreach]}")
    # direct covered routes from every bus stop to every building where the tree detours (existing cells reused)
    built = {rt.cell_of(u) for pth in paths for u in pth if rt.cell_of(u) is not None}
    rt.discount(built, REUSE)
    near_done = set()
    rt.penalize_near(built, 4, 1.6, near_done)
    targets = []
    for s in sites:
        if s.id in hubs:
            targets.append(hubs[s.id])
        else:
            targets += [v for v, e in tnodes if e.site is s][:1]
    n_extra = 0
    for b in bs_nodes:
        dist_b, pred_b = rt.tree_from(b)
        for tv in targets:
            if math.isfinite(dist_b[tv]):
                pth = Router.backtrack(pred_b, tv)
                new = [u for u in pth if rt.cell_of(u) is not None and rt.cell_of(u) not in built]
                if new:
                    paths.append(pth)
                    n_extra += 1
                    fresh = {rt.cell_of(u) for u in new}
                    built |= fresh
                    rt.discount(fresh, REUSE)
                    rt.penalize_near(fresh, 4, 1.6, near_done)
    lroute = simplify(rt, paths)
    L.routes["linkways"] = lroute
    lsegs = list(lroute.segments) + _stub_segments(lroute)
    L.timings["linkways"] = round(time.time() - t, 2)

    # crossings where linkways cross carriageways
    def cross_parts(segs, kind):
        out = []
        for p0, p1 in segs:
            ln = LineString([p0, p1])
            for sub, tag in split_line(ln, [net.C, drives]):
                if tag == 0:
                    mid = sub.interpolate(0.5, normalized=True)
                    r = net.road_at((mid.x, mid.y))
                    if r is None:
                        continue
                    s_ = r.sd((mid.x, mid.y))[0]
                    cr = next((c for c in net.crossings if c.road is r and abs(c.s - s_) < 2.5), None)
                    if cr is None:
                        cr = R.add_crossing(net, r, s_, kind)
                    out.append(("crossing", sub, cr))
                elif tag == 1:
                    out.append(("drive", sub, None))
                else:
                    out.append(("path", sub, None))
        return out
    lparts = cross_parts(lsegs, "linkway")
    say(f"  linkways: {len(lsegs)} segments, {sum(1 for p in lparts if p[0] == 'crossing')} road crossings "
        f"({L.timings['linkways']} s)")

    # ------------------------------------------------------------------ footpaths
    t = time.time()
    link_paths = unary_union([seg_rect(sub.coords[0], sub.coords[-1], lk["clear_width"] / 2)
                              for kind, sub, _ in lparts if kind == "path"] + [Polygon()])
    g = base_grid()
    g.set(net.reserve, cost=2.0)
    walk_sw = unary_union([net.sw_all, platforms])
    g.set(walk_sw, cost=1.0, axes=H | V)
    g.set(net.C, cost=15.0)
    for zone in (jboxes.buffer(6.0), bays.buffer(4.0), crossover_u.buffer(3.0)):
        g.set(mask=g.inside(zone) & g.inside(net.C.buffer(R.RAMP_LEN + 0.5)), cost=INF)
    for c in net.crossings:
        g.set(unary_union([c.strip] + [rp["poly"] for rp in c.ramps]), cost=1.0)
    g.set(green_rects, cost=1.5)
    g.set(lawns, cost=3.0)
    for (p0, p1), poly in zip(drive_lines, drive_polys):
        ax = _axis((p1[0] - p0[0], p1[1] - p0[1]))
        g.set(poly.difference(net.reserve), cost=3.0, axes=V if ax == H else H)
    for e, pg in chosen.values():
        g.set(pg["lane"], cost=2.0, axes=V if _axis(pg["along"]) == H else H)
        g.set(pg["apron"], cost=1.0, axes=H | V)
    g.set(green_paths, floor=0.3)
    g.set(link_paths, floor=0.3)
    if pcn:
        g.set(pcn["rect"], cost=3.0)
        g.set(pcn["path"], floor=0.3)
    g.set(green_block.buffer(FOOT_W / 2 + 0.2), cost=INF)
    g.set(masses.buffer(FOOT_W / 2 + 0.2), cost=INF)
    rt = Router(g, turn=4.0)
    SW = rt.add_virtual(kind="sidewalk")
    wm = g.inside(walk_sw) & np.isfinite(g.cost)
    for j, i in zip(*np.nonzero(wm)):
        rt.link(SW, rt.node(int(i), int(j), 0), 0.0)
        rt.link(SW, rt.node(int(i), int(j), 1), 0.0)
    fterms = []
    for e in ents:
        P = e.stub_pt
        ax = _axis(e.out)
        v = rt.add_virtual(point=P, axis=ax, site=e.site.id, k=e.k, kind="entrance")
        if rt.attach(v, P[0] + e.out[0] * 1.2, P[1] + e.out[1] * 1.2, ax, direction=e.out, max_steps=5,
                     any_layer=True):
            fterms.append(v)
    for gr in greens:
        for P, out in gr["gates"]:
            ax = _axis(out)
            v = rt.add_virtual(point=P, axis=ax, kind="gate", ref=gr["id"])
            if rt.attach(v, P[0] + out[0] * 1.0, P[1] + out[1] * 1.0, ax, direction=out, max_steps=5):
                fterms.append(v)
    if pcn:
        x0, y0, x1, y1 = pcn["rect"].bounds
        if (y1 - y0) >= (x1 - x0):
            P, out = (x0, (y0 + y1) / 2), (-1.0, 0.0)
        else:
            P, out = ((x0 + x1) / 2, y1), (0.0, 1.0)
        cpt = pcn["line"].interpolate(pcn["line"].project(Point(P)))
        v = rt.add_virtual(point=(cpt.x, cpt.y), axis=_axis(out), kind="pcn")
        if rt.attach(v, P[0] + out[0] * 1.0, P[1] + out[1] * 1.0, _axis(out), direction=out, max_steps=5):
            fterms.append(v)
    paths, unreach = rt.steiner([SW], fterms)
    if unreach:
        L.warnings.append(f"footpaths: {len(unreach)} terminals unreachable")
    froute = simplify(rt, paths)
    L.routes["footpaths"] = froute
    fsegs = list(froute.segments) + _stub_segments(froute)
    existing = unary_union([link_paths, green_paths, walk_sw, aprons, pcn["path"] if pcn else Polygon()]).buffer(0.25)
    fparts = []
    for kind, sub, cr in cross_parts(fsegs, "footpath"):
        if kind != "path":
            fparts.append((kind, sub, cr))
            continue
        for sub2, tag in split_line(sub, [existing]):
            if tag == -1 and sub2.length > 0.3:
                fparts.append(("path", sub2, None))
    L.timings["footpaths"] = round(time.time() - t, 2)
    say(f"  footpaths: {len(fsegs)} segments, {sum(1 for p in fparts if p[0] == 'path')} new runs "
        f"({L.timings['footpaths']} s)")

    # ------------------------------------------------------------------ surfaces in priority order
    t = time.time()
    ramps = unary_union([rp["poly"] for c in net.crossings for rp in c.ramps] + [crossover_u, Polygon()])
    hard_road = unary_union([net.C, net.kerb_ring])
    taken = unary_union([ramps, hard_road, fps, floors])
    pieces = []

    def take(geom):
        nonlocal taken
        g_ = _clean(geom.difference(taken).intersection(extent))
        if not g_.is_empty:
            taken = unary_union([taken, g_])
        return g_
    sidewalk_geoms = {r.id: take(net.sidewalks[r.id]) for r in net.roads}
    platform_geoms = [take(b.platform) for b in net.bus_stops]
    approach_geom = take(unary_union([a for c in net.crossings for a in c.approaches] + [Polygon()]))
    drive_geom = take(drives)
    lw_pieces, lw_lines = [], []
    lw_ext_deg = end_degrees(lsegs)
    half_c, half_r = lk["clear_width"] / 2, lk["canopy_width"] / 2
    roof_block = unary_union([net.C.buffer(0.3), drives.buffer(0.3), porch_canopies,
                              unary_union([b.shelter for b in net.bus_stops] + [Polygon()]), masses]).buffer(0)
    roofs_taken = Polygon()
    k_lw = 0
    lw_columns_src = []
    lw_path_rects = []           # linkway paths as routed, also where a building's open floor carries them unpaved
    for kind, sub, cr in lparts:
        if kind != "path":
            continue
        p0, p1 = sub.coords[0], sub.coords[-1]
        e = []
        for p in (p0, p1):
            dg = lw_ext_deg.get((round(p[0], 2), round(p[1], 2)), 0)
            e.append(dg > 1)
        prect = seg_rect(p0, p1, half_c, half_c if e[0] else 0.0, half_c if e[1] else 0.0)
        lw_path_rects.append(prect)
        path = take(prect)
        roof = _clean(seg_rect(p0, p1, half_r, half_r if e[0] else 0.0, half_r if e[1] else 0.0)
                      .difference(roof_block).difference(roofs_taken))
        if roof.is_empty and path.is_empty:
            continue
        k_lw += 1
        key = f"lw:{k_lw}"
        L.assemblies[key] = dict(name=f"Covered linkway {k_lw}", part=None, kind="linkway",
                                 props={"ClearWidth": lk["clear_width"], "CanopyWidth": lk["canopy_width"],
                                        "Soffit": lk["soffit"], "Covered": True, "Length": round(sub.length, 2)})
        if not path.is_empty:
            pieces.append(Piece("paving", f"Covered linkway {k_lw} path", path, -PAVE_T, PAVE_T, "paving",
                                assembly=key, object_type="Linkway path", props={"Kind": "linkway"}))
        if not roof.is_empty:
            roofs_taken = unary_union([roofs_taken, roof])
            pieces.append(Piece("roof", f"Covered linkway {k_lw} canopy", roof, lk["soffit"], 0.1, "metal_roof",
                                assembly=key))
        lw_columns_src.append((key, k_lw, p0, p1, roof))
        lw_pieces.append(key)
    apron_geom = take(aprons)
    foot_polys = []
    fdeg = end_degrees([(sub.coords[0], sub.coords[-1]) for kind, sub, _ in fparts if kind == "path"])
    for kind, sub, _ in fparts:
        if kind != "path":
            continue
        p0, p1 = sub.coords[0], sub.coords[-1]
        e0 = FOOT_W / 2 if fdeg.get((round(p0[0], 2), round(p0[1], 2)), 0) > 1 else FOOT_W / 2
        e1 = FOOT_W / 2 if fdeg.get((round(p1[0], 2), round(p1[1], 2)), 0) > 1 else FOOT_W / 2
        foot_polys.append(seg_rect(p0, p1, FOOT_W / 2, e0, e1))
    foot_geom = take(unary_union(foot_polys + [Polygon()]))
    # escape routes: a path from every stair discharge the car park and centre publish to the nearest footway
    lw_paved = unary_union([p.geom for p in pieces if p.object_type == "Linkway path"] + [Polygon()])
    walk_u = unary_union(list(sidewalk_geoms.values()) + platform_geoms +
                         [approach_geom, foot_geom, apron_geom, lw_paved, green_paths, open_u,
                          pcn["path"] if pcn else Polygon()]).buffer(0)
    escapes, esc_warn = escape_paths(sites, pub, walk_u, unary_union([hard_road, ramps, fps, green_block]).buffer(0),
                                     unary_union([drive_geom, crossover_u]),
                                     floors={sid: g_.floor for sid, g_ in grounds.items()}, extent=extent)
    L.warnings += esc_warn
    for e in escapes:
        e["geom"] = take(e["poly"]) if e["poly"] is not None else Polygon()
    L.escapes = escapes
    escape_geom = unary_union([e["geom"] for e in escapes] + [Polygon()])
    green_geoms = [take(gr["paths"]) for gr in greens]
    pcn_geom = take(pcn["path"]) if pcn else Polygon()
    # ---- pieces: roads
    for r in net.roads:
        rk = f"road:{r.id}"
        L.roads_meta[rk] = dict(name=r.name, id=r.id, cls=r.cls)
        cp = f"{rk}:CARRIAGEWAY"
        L.parts[cp] = dict(road=rk, predefined="CARRIAGEWAY", name=f"{r.name} carriageway", usage="LONGITUDINAL")
        cg = net.carriage_parts[r.id]
        strips = unary_union([c.strip for c in net.crossings if c.road is r] + [Polygon()])
        cg = _clean(cg.difference(strips))
        if not cg.is_empty:
            pieces.append(Piece("carriageway", f"{r.name} carriageway", cg, R.CARRIAGE_TOP - CARRIAGE_T, CARRIAGE_T,
                                "asphalt", part=cp, solid={"markings": _centre_marks(r, net)},
                                props={"Kind": "carriageway", "Road": r.id}))
        sp = f"{rk}:SIDEWALK"
        L.parts[sp] = dict(road=rk, predefined="SIDEWALK", name=f"{r.name} sidewalks", usage="LONGITUDINAL")
        if not sidewalk_geoms[r.id].is_empty:
            pieces.append(Piece("sidewalk", f"{r.name} sidewalk", sidewalk_geoms[r.id], -PAVE_T, PAVE_T, "paving", part=sp,
                                props={"Kind": "sidewalk", "Road": r.id, "Width": r.sw}))
    kerb_geoms = R.kerbs(net)
    for r in net.roads:
        kg = _clean(kerb_geoms[r.id])
        if not kg.is_empty:
            pieces.append(Piece("kerb", f"{r.name} kerbs", kg, -0.45, 0.45, "kerb", part=f"road:{r.id}:CARRIAGEWAY",
                                props={"Kind": "kerb", "Upstand": 0.15}))
    for j in net.junctions:
        rk = f"road:{j['through']}"
        jp = f"{rk}:INTERSECTION:{j['id']}"
        L.parts[jp] = dict(road=rk, predefined="INTERSECTION", name=f"Junction {j['id']} ({' / '.join(j['roads'])})",
                           usage="REGION")
        jg = _clean(j["poly"].difference(unary_union([c.strip for c in net.crossings] + [Polygon()])))
        if not jg.is_empty:
            pieces.append(Piece("carriageway", f"Junction {j['id']}", jg, R.CARRIAGE_TOP - CARRIAGE_T, CARRIAGE_T, "asphalt",
                                part=jp, props={"Kind": "junction", "Roads": " / ".join(j["roads"])}))
    for b, pg in zip(net.bus_stops, platform_geoms):
        rk = f"road:{b.road.id}"
        bp = f"{rk}:BUS_STOP:{b.id}"
        L.parts[bp] = dict(road=rk, predefined="BUS_STOP", name=f"Bus stop {b.id}", usage="REGION")
        if not pg.is_empty:
            pieces.append(Piece("paving", f"Bus stop {b.id} platform", pg, -PAVE_T, PAVE_T, "paving", part=bp,
                                object_type="Bus platform", props={"Kind": "bus_platform"}))
        L.assemblies[f"bs:{b.id}"] = dict(name=f"Bus stop {b.id} shelter", part=bp, kind="bus_shelter",
                                          props={"BusStop": b.id, "Road": b.road.id, "Covered": True})
        pieces += amenities.bus_shelter_pieces(b, bp)
    for c in net.crossings:
        rk = f"road:{c.road.id}"
        xp = f"{rk}:PEDESTRIAN_CROSSING:{c.id}"
        L.parts[xp] = dict(road=rk, predefined="PEDESTRIAN_CROSSING", name=f"Crossing {c.id} ({c.kind})", usage="LATERAL")
        sg = _clean(c.strip)
        if not sg.is_empty:
            pieces.append(Piece("crossing", f"Zebra crossing {c.id}", sg, R.CARRIAGE_TOP - CARRIAGE_T, CARRIAGE_T,
                                "asphalt", part=xp, solid={"markings": _zebra(c)},
                                props={"Kind": "crossing", "Road": c.road.id, "Width": c.width}))
        for k, rp in enumerate(c.ramps):
            pieces.append(Piece("ramp", f"Crossing {c.id} kerb ramp {k + 1}", rp["poly"], part=xp, mat="paving",
                                solid={"wedge": dict(top=rp["top"], dir=rp["dir"], length=rp["length"], width=rp["width"],
                                                     z_hi=0.0, z_lo=R.CARRIAGE_TOP, t=PAVE_T)},
                                props={"Kind": "kerb_ramp", "Slope": round(0.15 / rp["length"], 4)}))
    if not approach_geom.is_empty:
        pieces.append(Piece("paving", "Crossing approaches", approach_geom, -PAVE_T, PAVE_T, "paving",
                            object_type="Footpath", props={"Kind": "footpath"}))
    for k, comp in enumerate(sorted(_parts(foot_geom), key=lambda q: (round(q.bounds[0], 1), round(q.bounds[1], 1)))):
        pieces.append(Piece("paving", f"Footpath {k + 1}", comp, -PAVE_T, PAVE_T, "paving", object_type="Footpath",
                            props={"Kind": "footpath", "Width": FOOT_W}))
    for e in escapes:
        if not e["geom"].is_empty:
            pieces.append(Piece("paving", f"{e['site'].name} escape path {e['k'] + 1}", e["geom"], -PAVE_T, PAVE_T,
                                "paving", object_type="Escape path",
                                props={"Kind": "escape_path", "Width": e["width"], "Building": e["site"].id}))
    # ---- driveways
    dk = "drive:ESTATE"
    L.roads_meta[dk] = dict(name="Estate access roads", id="ESTATE", cls="driveway")
    for k, comp in enumerate(sorted(_parts(drive_geom), key=lambda p: (round(p.bounds[0], 1), round(p.bounds[1], 1)))):
        dp = f"{dk}:CARRIAGEWAY:{k + 1}"
        L.parts[dp] = dict(road=dk, predefined="CARRIAGEWAY", name=f"Driveway {k + 1}", usage="LONGITUDINAL")
        pieces.append(Piece("driveway", f"Driveway {k + 1}", comp, -CARRIAGE_T, CARRIAGE_T, "asphalt", part=dp,
                            props={"Kind": "driveway"}))
    for k, co in enumerate(net.crossovers):
        rk = f"road:{co['road']}"
        pieces.append(Piece("crossover", co["name"], co["poly"], part=f"{rk}:CARRIAGEWAY", mat="asphalt",
                            solid={"wedge": dict(top=co["top"], dir=co["dir"], length=co["length"], width=co["width"],
                                                 z_hi=0.0, z_lo=R.CARRIAGE_TOP, t=CARRIAGE_T)},
                            props={"Kind": "driveway_crossover"}))
    # ---- porches and service bays
    for sid, (e, pg) in sorted(chosen.items()):
        name = e.site.name
        L.assemblies[f"porch:{sid}"] = dict(name=f"{name} drop-off porch", part=None, kind="porch",
                                            props={"Covered": True, "Soffit": amenities.PORCH_SOFFIT, "Block": sid})
        for p in amenities.porch_pieces(pg, name, sid):
            if p.object_type == "Drop-off apron":
                p.geom = _clean(p.geom.intersection(apron_geom))
                if p.geom.is_empty:
                    continue
            pieces.append(p)
    # ---- linkway columns
    # (on a building's open floor no site paving marks the walking lines, so the routed path widths stand in for it)
    open_paths = unary_union(lw_path_rects + foot_polys + [Polygon()]).intersection(open_u) if not open_u.is_empty \
        else Polygon()
    no_col = unary_union([hard_road, ramps, masses.buffer(0.15), green_block, drive_geom,
                          unary_union(list(sidewalk_geoms.values()) + platform_geoms + [Polygon()]), approach_geom,
                          apron_geom, foot_geom, escape_geom, unary_union(green_geoms + [Polygon()]), pcn_geom,
                          open_paths,
                          unary_union([p.geom for p in pieces if p.object_type == "Linkway path"] + [Polygon()])]).buffer(0)
    shapely.prepare(no_col)
    col_hash = {}
    n_cols = 0
    for key, k_lw_, p0, p1, roof in lw_columns_src:
        if roof.is_empty:
            continue
        rbuf = roof.buffer(0.01)
        shapely.prepare(rbuf)
        kk = 0
        for c in columns_along(p0, p1, half_r - 0.15, lk["column_spacing"], COL):
            if no_col.intersects(c) or not rbuf.contains(c):
                continue
            cc = c.centroid
            hk = (int(cc.x // 2), int(cc.y // 2))
            near = [q for di in (-1, 0, 1) for dj in (-1, 0, 1) for q in col_hash.get((hk[0] + di, hk[1] + dj), ())]
            if any(math.hypot(cc.x - x, cc.y - y) < 1.0 for x, y in near):
                continue
            col_hash.setdefault(hk, []).append((cc.x, cc.y))
            kk += 1
            n_cols += 1
            pieces.append(Piece("column", f"Covered linkway {k_lw_} column {kk}", c, 0.0, lk["soffit"], "steel",
                                assembly=key, solid={"type": ("Linkway column", COL, lk["soffit"])}))
    # ---- greens and park connector
    for gr, gg in zip(greens, green_geoms):
        for p in gr["pieces"]:
            if p.object_type == "Park path":
                p.geom = gg
                if gg.is_empty:
                    continue
            elif p.geom is not None and p.kind == "paving":
                p.geom = take(p.geom)
                if p.geom.is_empty:
                    continue
            if p.assembly and p.assembly not in L.assemblies:
                L.assemblies[p.assembly] = dict(name=f"{gr['name']} pavilion", part=None, kind="pavilion",
                                                props={"Covered": True, "Green": gr["id"]})
            pieces.append(p)
    if pcn and not pcn_geom.is_empty:
        pieces.append(Piece("paving", pcn["name"], pcn_geom, -PAVE_T, PAVE_T, "paving", object_type="Park connector",
                            props={"Kind": "park", "Width": PCN_W}))
    # ---- ground a building leaves without a floor slab inside its footprint (under an overhang, an open forecourt):
    # paved level with the slabs, so the site meets every building's floor with neither a gap nor an overlap
    ground_geoms = []
    for s in sites:
        if grounds[s.id].source == "footprint":
            continue
        gap = _clean(s.footprint.intersection(extent).difference(floors).difference(unary_union([hard_road, ramps])),
                     min_area=0.05)
        if not gap.is_empty:
            ground_geoms.append(gap)
            pieces.append(Piece("paving", f"{s.name} ground paving", gap, -PAVE_T, PAVE_T, "paving",
                                object_type="Ground paving", props={"Kind": "ground_paving", "Building": s.id}))
    L.timings["surfaces"] = round(time.time() - t, 2)
    # ---- trees and terrain
    t = time.time()
    canopy_all = unary_union([roofs_taken, porch_canopies, unary_union([b.shelter for b in net.bus_stops] + [Polygon()])])
    sw_u = unary_union(list(sidewalk_geoms.values()) + [Polygon()])
    soft_paved = taken.difference(unary_union([fps, hard_road, sw_u, ramps])).buffer(1.0)
    keep_out = unary_union([fps.buffer(4.0), soft_paved, hard_road.buffer(0.5), sw_u.buffer(0.4), ramps.buffer(1.5),
                            canopy_all.buffer(1.0), green_block.buffer(2.5), lawns,
                            extent.exterior.buffer(1.5)]).buffer(0)
    open_area = extent.difference(unary_union([net.reserve, green_rects, fps.buffer(7.0), drive_geom.buffer(2.0),
                                               pcn["rect"] if pcn else Polygon()]))
    verge_keep = unary_union([jboxes.buffer(4.0), unary_union([c.strip for c in net.crossings] + [Polygon()]).buffer(3.0),
                              crossover_u.buffer(2.5), platforms.buffer(2.0), bays.buffer(1.0)])
    L.trees = landscape.plant(seed, net, greens, pcn, keep_out, verge_keep, open_area)
    pieces += landscape.tree_pieces(L.trees)
    hard = unary_union([taken, net.C, net.kerb_ring, fps]).buffer(0)
    pieces = landscape.terrain_pieces(extent, hard) + pieces
    L.timings["landscape"] = round(time.time() - t, 2)
    L.pieces = pieces
    # ---- pedestrian graph
    t = time.time()
    lines, points = [], []
    for ln, rid, side, w in net.sidewalk_lines:
        lines.append((ln, dict(kind="sidewalk", width=w, road=rid)))
    for c in net.crossings:
        a, b = c.tops
        lines.append((LineString([a, b]), dict(kind="crossing", width=c.width, slope=round(0.15 / R.RAMP_LEN, 4))))
        for cn in c.connectors:
            lines.append((cn, dict(kind="footpath", width=c.width)))
        for tp in c.tops:
            points.append(dict(xy=tp, kind="crossing", ref=c.id))
    for parts_, kind, w in ((lparts, "linkway", lk["clear_width"]), (fparts, "footpath", FOOT_W)):
        for pk, sub, cr in parts_:
            if pk == "crossing":
                continue
            seg = _trim_at_kerbs(sub, net)
            if seg is None:
                continue
            if pk == "drive":
                lines.append((seg, dict(kind="crossing", width=w, via="driveway")))
            else:
                lines.append((seg, dict(kind=kind, width=w)))
    for gr in greens:
        for ln, w in gr["lines"]:
            lines.append((ln, dict(kind="park", width=w)))
    if pcn:
        lines.append((pcn["line"], dict(kind="park", width=PCN_W)))
    for e in ents:
        if e.porch:
            lines.append((LineString([e.E, e.porch["stub_pt"]]), dict(kind="footpath", width=3.0, via="porch")))
        points.append(dict(xy=e.E, kind={"block": "entrance", "mscp": "mscp", "nc": "nc"}[e.site.kind], ref=e.site.id))
    site_obs = _piece_obstacles(pieces)
    # covered = under a canopy, shelter, porch or pavilion roof, or under a building's overhead slabs, measured
    L.cover = unary_union([p.geom for p in pieces if p.kind == "roof" and p.geom is not None]
                          + [g.cover for g in grounds.values() if g.cover is not None]).buffer(0)
    g_lines, g_points = _ground_routes(L, clearance, site_obs)
    lines += g_lines
    points += g_points
    for b in net.bus_stops:
        lines.append((LineString([b.node, b.stub]), dict(kind="footpath", width=2.0, via="bus_shelter")))
        far = (b.stub[0] + b.out[0] * (R.PLATFORM_DEPTH - R.SHELTER_REAR), b.stub[1] + b.out[1] * (R.PLATFORM_DEPTH - R.SHELTER_REAR))
        lines.append((LineString([b.stub, far]), dict(kind="footpath", width=2.0, via="bus_platform")))
        points.append(dict(xy=b.node, kind="bus_stop", ref=b.id))
    e_lines, e_points = _escape_lines(L.escapes, lines)
    lines += e_lines
    points += e_points
    lines = _with_cover(lines, L.cover)
    G = pedestrian_graph(lines, points)
    L.graph = G
    L.graph_stats = graph_stats(G)
    L.obstacles = site_obs + [(f"{sid} {lb}", g_) for sid, gr in sorted(grounds.items()) for lb, g_ in gr.obstacles]
    hits = edge_hits(G, L.obstacles)
    L.graph_check = dict(obstacle_hits=len(hits), obstacle_hit_examples=[
        dict(h, a=list(h["a"]), b=list(h["b"])) for h in hits[:20]],
        buildings={sid: dict(source=gr.source, sha256=gr.sha256) for sid, gr in sorted(grounds.items())})
    L.timings["graph"] = round(time.time() - t, 2)
    # ---- drawing geometry
    L.draw = dict(footprints=[(s, s.footprint) for s in sites], drives=drive_geom, drive_lines=drive_lines,
                  linkway_roofs=roofs_taken, linkway_lines=[sub for k_, sub, _ in lparts if k_ == "path"],
                  foot_lines=[sub for k_, sub, _ in fparts if k_ == "path"], foot=foot_geom, sidewalks=sidewalk_geoms,
                  escape=escape_geom, escape_lines=[ln for ln, _ in e_lines],
                  discharges=[(e["at"], e.get("out")) for e in L.escapes],
                  platforms=platform_geoms, approaches=approach_geom, aprons=apron_geom, porch_canopies=porch_canopies,
                  green_paths=green_geoms, pcn=pcn_geom, kerbs=kerb_geoms,
                  ramps=unary_union([rp["poly"] for c in net.crossings for rp in c.ramps] + [Polygon()]),
                  linkway_paths=unary_union([p.geom for p in pieces if p.object_type == "Linkway path"] + [Polygon()]),
                  ground_paving=unary_union(ground_geoms + [open_u, Polygon()]))   # open floors: the building's paving
    L.timings["total_plan"] = round(time.time() - t0, 2)
    return L


def _ground_routes(L, clearance, site_obs):
    """Legs through every building's ground floor: each entrance to the lift lobby it reaches soonest on foot, and
    the lobbies of a building joined to each other (a minimum spanning tree of walking distance). They are routed
    on the floor slabs (plus an APRON ring of site ground outside the footprint, to step round facade columns)
    around the walls, columns, cores, stairs and furniture of the building's IFC and the site's own columns,
    shelters and trees. Where the floor is partly open to the sky (round the outside of a void deck, across the
    centre's forecourt) the legs keep under the building's slabs and the site's canopies (L.cover) as a resident
    does in the rain: an uncovered metre counts COVER_PREF times. Returns (lines, points)."""
    lines, points = [], []
    tree = shapely.STRtree([g for _, g in site_obs]) if site_obs else None
    dry = L.cover.buffer(0.05) if L.cover is not None and not L.cover.is_empty else None   # as _with_cover splits
    for s in L.sites:
        gr = L.grounds[s.id]
        ents = [e for e in L.entrances if e.site is s]
        kind = {"block": "lift_lobby", "mscp": "mscp", "nc": "nc"}[s.kind]
        via = "void_deck" if s.kind == "block" else "indoor"
        lobs = sorted({e.lobby for e in ents} |
                      ({tuple(float(v) for v in q) for q in s.lift_lobbies} if s.kind == "block" else set()))
        others = unary_union([o.footprint for o in L.sites if o is not s] + [Polygon()])
        floor = gr.floor if gr.floor is not None else Polygon()
        walk = unary_union([s.footprint.buffer(APRON, join_style="mitre"), floor])
        walk = walk.difference(others).difference(L.net.C.buffer(R.KERB_W + clearance))
        near = [site_obs[int(i)][1] for i in sorted(tree.query(walk))] if tree is not None else []
        fs = FreeSpace(walk, unary_union([gr.blocked] + near), clearance,
                       cover=dry.intersection(walk.envelope) if dry is not None else None, penalty=COVER_PREF)
        targets = [e.E for e in ents] + lobs
        routes = [fs.routes(lob, targets) for lob in lobs]
        for k, e in enumerate(ents):
            best = min(((routes[li][k].length, li) for li in range(len(lobs)) if k in routes[li]), default=None)
            if best is None:
                L.warnings.append(f"{s.id} entrance {e.k} at {e.E}: no ground-floor route to a lobby")
                continue
            e.lobby = lobs[best[1]]
            lines.append((routes[best[1]][k], dict(kind="footpath", width=2.0, via=via)))
        n_e, done, rest = len(ents), [0], list(range(1, len(lobs)))
        while rest:
            cand = sorted((routes[a][n_e + b].length, a, b) for a in done for b in rest if n_e + b in routes[a])
            if not cand:
                L.warnings.append(f"{s.id}: lobbies {[lobs[b] for b in rest]} have no ground-floor route to "
                                  f"{lobs[done[0]]}")
                break
            _, a, b = cand[0]
            lines.append((routes[a][n_e + b], dict(kind="footpath", width=2.0, via=via)))
            done.append(b)
            rest.remove(b)
        points += [dict(xy=q, kind=kind, ref=s.id) for q in lobs]
    return lines, points


def _with_cover(lines, cover, tol=0.05):
    """Split every path line where it passes in and out of cover and set `covered` from the geometry; crossings,
    sidewalks and escape paths are never counted as covered."""
    cb = cover.buffer(tol) if cover is not None and not cover.is_empty else None
    if cb is not None:
        shapely.prepare(cb)
    out = []
    for ln, a in lines:
        if a["kind"] in ("crossing", "sidewalk", "escape") or cb is None:
            out.append((ln, dict(a, covered=False)))
            continue
        out += [(piece, dict(a, covered=cov)) for piece, cov in split_cover(ln, cb)]
    return out


def _stub_segments(route):
    out = []
    for E, c, info in route.stubs:
        ax = info.get("axis")
        P = (c[0], E[1]) if ax == H else (E[0], c[1])
        if math.dist(E, P) > 0.01:
            out.append((tuple(E), tuple(P)))
        if math.dist(P, c) > 0.01:
            out.append((tuple(P), tuple(c)))
    return out


def _trim_at_kerbs(sub, net):
    """Path sub-line that ends at a carriageway edge: cut back by the kerb-ramp length (the crossing edge starts at
    the ramp top)."""
    a, b = list(sub.coords[0]), list(sub.coords[-1])
    Lg = sub.length
    Cb = net.C.boundary
    for idx in (0, 1):
        p = a if idx == 0 else b
        if Cb.distance(Point(p)) < 0.05:
            q = b if idx == 0 else a
            if Lg <= R.RAMP_LEN + 0.05:
                return None
            f = R.RAMP_LEN / Lg
            p[0], p[1] = p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f
            Lg -= R.RAMP_LEN
    return LineString([tuple(a), tuple(b)])


def _zebra(c):
    """Zebra bars (0.5 m wide, 1.0 m pitch across the road, as long as the crossing is wide)."""
    r = c.road
    bars = []
    d = c.da + 0.5
    while d + 0.5 <= c.db - 0.3:
        bars.append(shapely.normalize(r.band(d, d + 0.5, c.s - c.width / 2 + 0.2, c.s + c.width / 2 - 0.2)))
        d += 1.0
    return bars


def _centre_marks(r, net):
    """Dashed centre line (3 m dash, 6 m gap), skipping junctions and crossings."""
    skip = unary_union([j["poly"].buffer(3.0) for j in net.junctions] +
                       [c.strip.buffer(3.0) for c in net.crossings if c.road is r] + [Polygon()])
    out = []
    s = 3.0
    while s + 3.0 < r.length:
        d = r.band(-0.06, 0.06, s, s + 3.0)
        if not d.intersects(skip) and net.extent.contains(d):
            out.append(shapely.normalize(d))
        s += 9.0
    return out


# ============================================================================ IFC writing
KINDS = {  # kind -> ((IFC4X3 class, predefined), (IFC4 class, predefined, object type))
    "carriageway": (("IfcPavement", "FLEXIBLE"), ("IfcSlab", "USERDEFINED", "Carriageway")),
    "crossing": (("IfcPavement", "FLEXIBLE"), ("IfcSlab", "USERDEFINED", "Pedestrian crossing")),
    "driveway": (("IfcPavement", "FLEXIBLE"), ("IfcSlab", "USERDEFINED", "Driveway")),
    "crossover": (("IfcPavement", "FLEXIBLE"), ("IfcSlab", "USERDEFINED", "Driveway crossover")),
    "kerb": (("IfcKerb", "NOTDEFINED"), ("IfcCivilElement", None, "Kerb")),
    "sidewalk": (("IfcSlab", "SIDEWALK"), ("IfcSlab", "USERDEFINED", "Sidewalk")),
    "paving": (("IfcSlab", "PAVING"), ("IfcSlab", "USERDEFINED", "Paving")),
    "ramp": (("IfcRamp", "STRAIGHT_RUN_RAMP"), ("IfcRamp", "STRAIGHT_RUN_RAMP", None)),
    "roof": (("IfcSlab", "ROOF"), ("IfcSlab", "ROOF", None)),
    "column": (("IfcColumn", "COLUMN"), ("IfcColumn", "COLUMN", None)),
    "terrain": (("IfcGeographicElement", "TERRAIN"), ("IfcGeographicElement", "TERRAIN", None)),
    "furniture": (("IfcFurniture", "USERDEFINED"), ("IfcFurniture", "USERDEFINED", None)),
    "fence": (("IfcRailing", "FENCE"), ("IfcRailing", "USERDEFINED", "Fence")),
    "tree": (("IfcGeographicElement", "VEGETATION"), ("IfcGeographicElement", "USERDEFINED", "Vegetation")),
}


def _prism_rep(W, origin, items):
    """Body representation: one extruded solid per polygon of geom (z0/h relative to the element), plus extra
    (polygon, z0, h, palette key) items carrying their own surface style."""
    m = W.m
    solids = []
    ox, oy = origin
    up = m.createIfcDirection((0.0, 0.0, 1.0))
    for poly, z0, h, mat in items:
        for p in _parts(poly):
            local = shapely.affinity.translate(p, -ox, -oy)
            s = m.createIfcExtrudedAreaSolid(W._profile(local), W._place3d((0.0, 0.0, z0)), up, float(h))
            if mat:
                m.createIfcStyledItem(s, [W.mat[mat][1]], None)
            solids.append(s)
    return m.createIfcShapeRepresentation(W.body, "Body", "SweptSolid", solids) if solids else None


def _wedge_rep(W, w):
    """Ramp solid in the element frame (x downhill from the top edge centre, z up): a vertical profile extruded
    across the width."""
    m = W.m
    Lg, wd, zh, zl, t = w["length"], w["width"], w["z_hi"], w["z_lo"], w["t"]
    pts = [(0.0, zh), (Lg, zl), (Lg, zl - t), (0.0, zl - t)]
    prof = m.createIfcArbitraryClosedProfileDef("AREA", None, W._curve(pts))
    pos = W._place3d((0.0, wd / 2, 0.0), axis=(0.0, -1.0, 0.0), ref=(1.0, 0.0, 0.0))
    s = m.createIfcExtrudedAreaSolid(prof, pos, m.createIfcDirection((0.0, 0.0, 1.0)), float(wd))
    return m.createIfcShapeRepresentation(W.body, "Body", "SweptSolid", [s])


def write_site(L: SiteLayout, path, schema="IFC4X3", log=None):
    import ifcopenshell.api.aggregate as aggregate
    import ifcopenshell.api.geometry as geometry
    import ifcopenshell.api.material as material
    import ifcopenshell.api.root as root

    from estate.geom.walls import frame, translate
    from estate.ifc.vegetation import ESTATE_SPECIES, tree_types
    from estate.ifc.writer import IfcWriter
    say = log or (lambda *a: None)
    t0 = time.time()
    cfg = L.mp["cfg"]
    ident = config.identity(cfg)
    code = cfg["estate"]["code"]
    W = IfcWriter(schema=schema, project_name=ident["project_name"], project_guid=ident["project_guid"],
                  site_name=ident["site_name"], site_guid=ident["site_guid"], georef=ident["georef"],
                  guid_key=f"{code}/SITE/{schema}", guid_base=f"{code}/SITE")
    try:
        m = W.m
        x4 = schema.upper().startswith("IFC4X3")
        W.site.LongName = f"{cfg['estate']['name']} external works"
        # ---- spatial structure: roads and road parts (IFC4X3 only)
        containers = {}
        if x4:
            road_el = {}
            for rk, meta in L.roads_meta.items():
                rd = root.create_entity(m, "IfcRoad", name=meta["name"], predefined_type="NOTDEFINED")
                rd.LongName = meta["id"]
                geometry.edit_object_placement(m, product=rd)
                road_el[rk] = rd
            aggregate.assign_object(m, products=list(road_el.values()), relating_object=W.site)
            by_road = {}
            for pk, meta in L.parts.items():
                rp = root.create_entity(m, "IfcRoadPart", name=meta["name"], predefined_type=meta["predefined"])
                rp.UsageType = meta["usage"]
                geometry.edit_object_placement(m, product=rp)
                containers[pk] = rp
                by_road.setdefault(meta["road"], []).append(rp)
            for rk, lst in by_road.items():
                aggregate.assign_object(m, products=lst, relating_object=road_el[rk])
        # ---- assemblies
        asm_el = {}
        for ak, meta in L.assemblies.items():
            pre, ot = ("SHELTER", None) if x4 else ("USERDEFINED", "Shelter")
            a = root.create_entity(m, "IfcElementAssembly", name=meta["name"], predefined_type=pre)
            if ot:
                a.ObjectType = ot
            a.AssemblyPlace = "SITE"
            W.place(a, np.eye(4))
            W.contain(containers.get(meta.get("part"), W.site), a)
            W._n("IfcElementAssembly")
            props = {"Kind": meta.get("kind", "")}
            props.update(meta.get("props", {}))
            W.props(a, "SampleCity_SiteWorks", props)
            asm_el[ak] = a
        asm_parts = {}
        # ---- types (columns, trees)
        col_types = {}
        tree_t = tree_types(W, ESTATE_SPECIES)
        # ---- pieces
        for p in L.pieces:
            spec4x3, spec4 = KINDS[p.kind]
            cls, pre = spec4x3 if x4 else spec4[:2]
            ot = p.object_type if x4 else (spec4[2] if p.object_type is None else p.object_type)
            if p.kind == "furniture" and ot is None:
                ot = "Street furniture"
            if pre == "USERDEFINED" and not ot:
                ot = p.kind.title()
            container = containers.get(p.part, W.site) if p.assembly is None else None
            sol = p.solid or {}
            if "tree" in sol:
                x, y, z = sol["at"]
                el = W.product(cls, p.name, None, translate(x, y, z), W.site, pre, None, ot if not x4 else None)
                W.batch_type.setdefault(tree_t[sol["tree"]], []).append(el)
            elif "type" in sol:
                tname, size, hgt = sol["type"]
                key = (tname, size, hgt)
                if key not in col_types:
                    ct = root.create_entity(m, "IfcColumnType", name=f"{tname} {int(size * 1000)}x{int(size * 1000)}",
                                            predefined_type="COLUMN")
                    rep = W.extrusion(box(-size / 2, -size / 2, size / 2, size / 2), hgt, (0.0, 0.0))
                    geometry.assign_representation(m, product=ct, representation=rep)
                    if W.mat[p.mat][0] is not None:
                        material.assign_material(m, products=[ct], type="IfcMaterial", material=W.mat[p.mat][0])
                    col_types[key] = ct
                c = p.geom.centroid
                el = W.product(cls, p.name, None, translate(c.x, c.y, p.z0), container, pre, None, ot)
                W.batch_type.setdefault(col_types[key], []).append(el)
            elif "wedge" in sol:
                w = sol["wedge"]
                M = frame((w["top"][0], w["top"][1], 0.0), w["dir"])
                el = W.product(cls, p.name, _wedge_rep(W, w), M, container, pre, p.mat, ot)
            elif "boxes" in sol:
                allg = unary_union([b["box"] for b in sol["boxes"]])
                c = allg.centroid
                rep = _prism_rep(W, (c.x, c.y), [(b["box"], b["z0"], b["h"], b["mat"]) for b in sol["boxes"]])
                el = W.product(cls, p.name, rep, translate(c.x, c.y, 0.0), container, pre, p.mat, ot)
            else:
                g = p.geom
                if g is None or g.is_empty:
                    continue
                c = g.centroid
                items = [(g, 0.0, p.h, None)]
                for mk in sol.get("markings", []):
                    items.append((mk, p.h, 0.005, "marking"))
                rep = _prism_rep(W, (c.x, c.y), items)
                if rep is None:
                    continue
                el = W.product(cls, p.name, rep, translate(c.x, c.y, p.z0), container, pre, p.mat, ot)
            props = {"Kind": p.props.get("Kind", p.kind)}
            props.update({k: v for k, v in p.props.items() if k != "Kind"})
            if not x4 and p.part:
                props["RoadPart"] = L.parts.get(p.part, {}).get("name", p.part)
            W.props(el, "SampleCity_SiteWorks", props)
            if p.kind in ("paving", "sidewalk", "ramp", "crossing") or (p.kind == "driveway"):
                nav = {"Walkable": True}
                if p.kind == "ramp":
                    nav["Slope"] = p.props.get("Slope", 0.0)
                W.props(el, "Navigation", nav)
            if p.assembly:
                asm_parts.setdefault(p.assembly, []).append(el)
        W.flush()
        for ak, els in asm_parts.items():
            aggregate.assign_object(m, products=els, relating_object=asm_el[ak])
        # (the wall types the shared writer sets up, which the site never uses, are purged by W.write)
        # ---- federation
        linked = []
        try:
            from estate.ifc.links import add_linked_models
            suffix = "_ifc4" if schema == "IFC4" else ""      # the IFC4 federation links the IFC4 copies
            entries = [dict(path=f"{s.id}/{s.id}{suffix}.ifc", name=s.name) for s in L.sites]
            linked = add_linked_models(W, entries)
            for ref in linked:        # WR1: Name xor ReferencedDocument (Bonsai's own links carry no Name either)
                if getattr(ref, "ReferencedDocument", None) is not None and ref.Name:
                    ref.Name = None
        except ImportError as e:
            L.warnings.append(f"estate.ifc.links not available ({e}); no LINKED_MODEL references written")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        W.write(path)
        counts = {}
        for el in m.by_type("IfcElement"):
            counts[el.is_a()] = counts.get(el.is_a(), 0) + 1
        info = dict(path=str(path), schema=schema, elements=sum(counts.values()), by_class=dict(sorted(counts.items())),
                    linked_models=len(linked), seconds=round(time.time() - t0, 1))
        say(f"  wrote {env.rel(path)}: {info['elements']} IfcElement in {info['seconds']} s")
        return info
    finally:
        W.close()


class GraphCheckError(RuntimeError):
    """The pedestrian graph does not match the model it was written with (see .issues)."""

    def __init__(self, issues, info=None):
        super().__init__(f"pedestrian graph self-check failed: {len(issues)} issue(s); first: {issues[:3]}")
        self.issues, self.info = issues, info


def _plans_from_ifc(path, keep):
    """{element name: plan polygon} of the IfcElements of a written IFC for which keep(element) is true (the
    union of the plan projections of their triangles), with each element's lowest z."""
    import ifcopenshell
    import ifcopenshell.geom
    f = ifcopenshell.open(str(path))
    els = sorted((e for e in f.by_type("IfcElement") if keep(e)), key=lambda e: e.id())
    out = {}
    if not els:
        return out
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    it = ifcopenshell.geom.iterator(s, f, 1, include=els)
    if it.initialize():
        while True:
            sh = it.get()
            v = np.array(sh.geometry.verts, float).reshape(-1, 3)
            fc = np.array(sh.geometry.faces, dtype=np.int64).reshape(-1, 3)
            tris = shapely.polygons(v[fc][:, :, :2])
            tris = tris[shapely.area(tris) > 1e-8]
            if len(tris):
                el = f.by_guid(sh.guid)
                out[el.Name or sh.guid] = (shapely.union_all(tris).buffer(0.001), float(v[:, 2].min()))
            if not it.next():
                break
    return out


def check_graph(L, ifc_path) -> list[str]:
    """The pedestrian graph against the SITE.ifc it was written with and the building IFCs it was routed on:
    - no edge runs more than HIT_TOL through a wall, column, stair, table or other obstacle of a building's
      ground floor, or through a site column, shelter, bench, fence or tree trunk (L.graph_check, plan time);
    - every road-crossing edge runs over its 'Zebra crossing <id>' and the kerb ramps of that crossing, and every
      crossing of the layout has its zebra and both ramps in the IFC;
    - every covered edge lies under a roof above head height in the IFC (linkway canopies, shelters, porches,
      pavilions) or under a building's overhead slabs."""
    issues = [f"edge {h['a']}-{h['b']} ({h['kind']}{', ' + h['via'] if h.get('via') else ''}) runs {h['length']} m "
              f"through {h['obstacle']}" for h in L.graph_check.get("obstacle_hit_examples", [])]
    n_hits = L.graph_check.get("obstacle_hits", 0)
    if n_hits > len(issues):
        issues.append(f"... {n_hits - len(issues)} more obstacle hits")
    plans = _plans_from_ifc(ifc_path, lambda e: (e.Name or "").startswith(("Zebra crossing ", "Crossing "))
                            or (e.is_a("IfcSlab") and e.PredefinedType == "ROOF"))
    zebra, ramps, roofs = {}, {}, []
    for nm, (g, z0) in plans.items():
        if nm.startswith("Zebra crossing "):
            zebra[nm.split()[-1]] = g
        elif nm.startswith("Crossing ") and " kerb ramp " in nm:
            ramps.setdefault(nm.split()[1], []).append(g)
        elif z0 > BAND[1]:
            roofs.append(g)
    for c in L.net.crossings:
        if c.id not in zebra or len(ramps.get(c.id, [])) < 2:
            issues.append(f"crossing {c.id}: zebra {'found' if c.id in zebra else 'missing'}, "
                          f"{len(ramps.get(c.id, []))} kerb ramp(s) in {Path(ifc_path).name}")
    walk_x = {k: unary_union([zebra[k]] + ramps.get(k, [])).buffer(0.2) for k in zebra}
    cover = unary_union(roofs + [g.cover for g in L.grounds.values() if g.cover is not None] + [Polygon()]).buffer(0.1)
    shapely.prepare(cover)
    G = L.graph
    for u, v in sorted((min(e), max(e)) for e in G.edges):
        d = G.edges[u, v]
        ln = LineString([u, v])
        if d["kind"] == "crossing" and d.get("via") != "driveway":
            ids = [k for k, g in sorted(walk_x.items()) if g.intersects(ln)]
            if not any(ln.difference(walk_x[k]).length <= HIT_TOL and len(ramps.get(k, [])) >= 2 for k in ids):
                issues.append(f"crossing edge {u}-{v}: not over a zebra crossing with kerb ramps "
                              f"(near {ids or 'none'})")
        if d["covered"]:
            out = ln.difference(cover).length
            if out > HIT_TOL:
                issues.append(f"covered edge {u}-{v} ({d['kind']}{', ' + d['via'] if d.get('via') else ''}): "
                              f"{out:.2f} m not under any roof")
    return issues


def build_site(mp: dict, out_path, schema="IFC4X3", graph_path=None, plan_png=None, layout=None, log=None,
               model_dir=None, check=True):
    """Plan (unless a layout is given) and write the site model; optionally the pedestrian graph and site plan.

    The graph is checked against the IFC just written (check_graph) and records that file's sha256, so a reader
    can tell a graph that no longer belongs to SITE.ifc. A failed check raises GraphCheckError after everything
    is written (the files are kept for inspection)."""
    from estate.export.meshcache import sha256
    L = layout or plan_site(mp, log, model_dir)
    info = write_site(L, out_path, schema, log)
    issues = check_graph(L, out_path)
    escapes = [dict(site=e["site"].id, at=list(e["at"]), width=e["width"], length=e["length"], offset=e["offset"],
                    graph_end=list(e["graph_end"]) if e.get("graph_end") else None,
                    status="failed" if e.get("failed") else "on_floor" if e.get("on_floor") else
                    "paved" if e["poly"] is not None else "on_footway") for e in L.escapes]
    if graph_path:
        Path(graph_path).parent.mkdir(parents=True, exist_ok=True)
        meta = dict(source=Path(out_path).name, path=env.rel(out_path), ifc_sha256=sha256(out_path), schema=schema,
                    buildings=L.graph_check.get("buildings", {}), checks=dict(issues=len(issues), first=issues[:10]),
                    entrances=[dict(site=e.site.id, k=e.k, node=list(e.E), planned=list(e.planned or e.E))
                               for e in L.entrances],
                    stats=L.graph_stats, agent=mp["cfg"].get("agent", {}),
                    crossings=[dict(id=c.id, kind=c.kind, road=c.road.id, tops=c.tops) for c in L.net.crossings],
                    bus_stops=[dict(id=b.id, road=b.road.id, node=b.node) for b in L.net.bus_stops],
                    stair_discharges=escapes)
        Path(graph_path).write_text(json.dumps(graph_json(L.graph, meta), indent=1), encoding="utf-8")
        info["graph"] = str(graph_path)
    if plan_png:
        from estate.site.siteplan import render
        render(L, plan_png)
        info["site_plan"] = str(plan_png)
    info.update(graph_stats=L.graph_stats, warnings=L.warnings, notes=L.notes, timings=L.timings, trees=len(L.trees),
                crossings=len(L.net.crossings), crossovers=len(L.net.crossovers), assemblies=len(L.assemblies),
                escape_paths=escapes,
                graph_check=dict(issues=issues, buildings=L.graph_check.get("buildings", {})))
    if issues and check:
        raise GraphCheckError(issues, info)
    return info
