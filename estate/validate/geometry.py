"""Geometric validation of a building IFC, re-measured from its tessellation (estate/export/meshcache.py).

The rules are measured on the solids instead of read back from the parameters the generator wrote, so neither a
generator defect nor a hand edit in Bonsai can hide behind its own property sets:

- stairs: risers (<= 175 mm, variation <= 5 mm), goings (>= 275 mm), risers per flight (<= 18), clear width
  (>= 1.0 m between walls and balustrades), headroom (>= 2.0 m above the pitch line), flights landing on a
  surface, each IfcStair climbing exactly one storey, open flight sides guarded by a balustrade >= 1.0 m above
  the pitch line;
- barriers: parapet walls and guard rails >= 1.0 m; falling edges (slab, ledge, landing, corridor and roof edges
  with a drop > 0.5 m) covered by a wall / parapet / railing >= 1.0 m, approximated by sampling the slab
  perimeters and casting vertical rays through the barrier solids (window sills and door openings included;
  closed doors count as barriers, and so does a window declaring Navigation.BarrierHeight for a guarded low sill);
- clashes: wall-wall overlap per storey within the junction tolerance t1 * t2, doors / windows clear of every
  wall except their host, no slab inside a lift shaft, IfcSpace footprints overlapping <= 0.05 m2;
- wet stacks: per stack, the wet rooms keep their footprint from storey to storey (IoU >= 0.99) and no wet room
  sits over a habitable room (<= 0.10 m2 per storey pair).

Everything works on world-coordinate triangle meshes with shapely footprints and vectorised vertical ray casts
(numpy + shapely STRtree; no scipy / trimesh). Each failure also becomes an ifcqa item: the GlobalIds of the
elements involved (a flight and its balustrade, both walls of an overlap, the slab of a falling edge) and a world
point on the defect (the middle of an unguarded run at the slab top, the centre of an overlap) for a BCF viewpoint.
"""
from __future__ import annotations

import time
import tomllib
from collections import defaultdict
from functools import cached_property

import numpy as np
import shapely
from shapely.geometry import LineString, box
from shapely.geometry.polygon import orient

import ifcopenshell
import ifcopenshell.util.element as uel
import ifcopenshell.util.placement as upl

from estate import env
from estate.validate.ifcqa import Hits, Report, summarize, where

LIMITS = dict(riser_max=0.175, riser_var=0.005, going_min=0.275, risers_max=18, stair_width=1.0, headroom=2.0,
              barrier=1.0, wall_overlap_tol=1e-4, clash_area=1e-4, shaft_slab=0.01, space_overlap=0.05,
              wet_iou=0.99, wet_over_dry=0.10, drop=0.5, gap_narrow=0.40)
TOL = 1e-4

SURFACE = ("IfcSlab", "IfcStairFlight", "IfcRamp", "IfcRampFlight")
OVERHEAD = SURFACE + ("IfcWall", "IfcBeam", "IfcRoof", "IfcCovering", "IfcPlate", "IfcMember",
                      "IfcBuildingElementProxy")
BARRIER = ("IfcWall", "IfcRailing", "IfcPlate", "IfcMember", "IfcCurtainWall")
OBSTACLE = ("IfcWall", "IfcRailing", "IfcColumn", "IfcMember", "IfcPlate")
WET, DRY = {"wet"}, {"habitable"}


def limits() -> dict:
    """LIMITS, overridden by an optional [geometry] table in config/rules.toml."""
    out = dict(LIMITS)
    try:
        out.update(tomllib.loads((env.CONFIG / "rules.toml").read_text(encoding="utf-8")).get("geometry", {}))
    except (OSError, tomllib.TOMLDecodeError):
        pass
    return out


# ----------------------------------------------------------------------------- elements and rays
class El:
    """One tessellated element: triangles (n, 3, 3) in world coordinates plus lazily derived footprints."""

    def __init__(self, guid, ent, mesh, storey):
        self.guid, self.ent, self.storey = guid, ent, storey
        self.cls, self.name = mesh["cls"], mesh.get("name", "")
        V = np.asarray(mesh["verts"], float).reshape(-1, 3)
        F = np.asarray(mesh["faces"], np.int64).reshape(-1, 3)
        self.T = V[F] if len(F) else np.zeros((0, 3, 3))

    def __repr__(self):
        return f"{self.cls} {self.name}"

    @cached_property
    def N(self):
        n = np.cross(self.T[:, 1] - self.T[:, 0], self.T[:, 2] - self.T[:, 0])
        ln = np.linalg.norm(n, axis=1)
        return n / np.where(ln > 0, ln, 1.0)[:, None]

    @cached_property
    def zmin(self):
        return float(self.T[:, :, 2].min()) if len(self.T) else 0.0

    @cached_property
    def zmax(self):
        return float(self.T[:, :, 2].max()) if len(self.T) else 0.0

    @cached_property
    def footprint(self):
        """Plan projection: the union of the upward-facing faces (every vertical line through a closed solid
        leaves it through one), exact for prisms, extrusions, stepped flights and walls with openings."""
        return _union(self.T[self.N[:, 2] > 1e-3])

    def top(self, tol=0.005):
        """(polygon, z) of the highest upward-facing faces."""
        up = self.N[:, 2] > 0.99
        if not up.any():
            return None, None
        z = self.T[up][:, :, 2].mean(1)
        zt = float(z.max())
        return _union(self.T[up][z > zt - tol]), zt


def solid_box(e: El, zmin=None, zmax=None) -> El:
    """A closed prism over an element's convex footprint (closed doors, guarded low windows as barriers)."""
    zmin = e.zmin if zmin is None else zmin
    zmax = e.zmax if zmax is None else zmax
    b = El.__new__(El)
    b.guid, b.ent, b.storey, b.cls, b.name = e.guid, e.ent, e.storey, e.cls, e.name
    c = np.asarray(e.footprint.convex_hull.exterior.coords)[:-1] if not e.footprint.is_empty else np.zeros((0, 2))
    tris = []
    for i in range(1, len(c) - 1):
        for z, order in ((zmin, (0, i + 1, i)), (zmax, (0, i, i + 1))):
            tris.append([(*c[k], z) for k in order])
    b.T = np.array(tris, float).reshape(-1, 3, 3)
    return b


def _union(T):
    if not len(T):
        return shapely.Polygon()
    polys = shapely.polygons(T[:, :, :2])
    polys = polys[shapely.area(polys) > 1e-10]
    u = shapely.union_all(polys)
    return u if u.is_valid else u.buffer(0)


class Rays:
    """Vertical ray casts against the non-vertical triangles of a set of elements."""

    def __init__(self, els):
        self.els = list(els)
        Ts, own = [], []
        for i, e in enumerate(self.els):
            sel = np.abs(e.N[:, 2]) > 1e-3
            if sel.any():
                Ts.append(e.T[sel])
                own.append(np.full(int(sel.sum()), i))
        self.T = np.concatenate(Ts) if Ts else np.zeros((0, 3, 3))
        self.own = np.concatenate(own) if own else np.zeros(0, int)
        self.n = np.cross(self.T[:, 1] - self.T[:, 0], self.T[:, 2] - self.T[:, 0])
        self.tree = shapely.STRtree(shapely.polygons(self.T[:, :, :2])) if len(self.T) else None
        self.index = {id(e): i for i, e in enumerate(self.els)}

    def owner(self, e) -> int:
        return self.index.get(id(e), -1)

    def cast(self, pts):
        """Hits of vertical lines through pts (P, 2): (point index, owner index, z, upward-facing)."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        if self.tree is None or not len(pts):
            e = np.zeros(0, int)
            return e, e, np.zeros(0), np.zeros(0, bool)
        pi, ti = self.tree.query(shapely.points(pts), predicate="intersects")
        T, n, p = self.T[ti], self.n[ti], pts[pi]
        z = T[:, 0, 2] - (n[:, 0] * (p[:, 0] - T[:, 0, 0]) + n[:, 1] * (p[:, 1] - T[:, 0, 1])) / n[:, 2]
        return pi, self.own[ti], z, n[:, 2] > 0


def top_of(rays: Rays, pts, owner):
    """Highest crossing of one element's solid along vertical lines through pts (NaN where missed)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    out = np.full(len(pts), -np.inf)
    pi, own, z, _ = rays.cast(pts)
    m = own == owner
    np.maximum.at(out, pi[m], z[m])
    return np.where(np.isfinite(out), out, np.nan)


def barrier_heights(rays: Rays, pts, floors, start_tol=0.10, gap=0.02):
    """Height above floors[i] of the barrier standing at each point: the solid intervals of every barrier element
    along a vertical line (pairs of crossings), merged upwards from the one that starts at the floor."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    floors = np.asarray(floors, float)
    pi, own, z, _ = rays.cast(pts)
    if not len(pi):
        return np.zeros(len(pts))
    z = np.round(z, 5)
    order = np.lexsort((z, own, pi))
    pi, own, z = pi[order], own[order], z[order]
    new_grp = np.concatenate(([True], (np.diff(pi) != 0) | (np.diff(own) != 0)))
    keep = new_grp | np.concatenate(([True], np.diff(z) > 1e-5))      # shared edges report a crossing twice
    pi, own, z, new_grp = pi[keep], own[keep], z[keep], new_grp[keep]
    gid = np.cumsum(new_grp) - 1
    start = np.flatnonzero(new_grp)
    size = np.diff(np.concatenate((start, [len(z)])))
    rank = np.arange(len(z)) - start[gid]
    even = size[gid] % 2 == 0
    lo = np.flatnonzero(even & (rank % 2 == 0))                         # closed solid: (in, out) pairs
    a, b, ip = z[lo], z[lo + 1], pi[lo]
    odd = np.flatnonzero(size % 2 == 1)                                 # grazing hits: whole span
    a = np.concatenate((a, z[start[odd]]))
    b = np.concatenate((b, z[start[odd] + size[odd] - 1]))
    ip = np.concatenate((ip, pi[start[odd]]))
    top = np.full(len(pts), -np.inf)
    fl = floors[ip]
    m = (a <= fl + start_tol) & (b > fl + start_tol)
    np.maximum.at(top, ip[m], b[m])
    for _ in range(16):                                                 # grow through touching intervals
        t = top[ip]
        m = np.isfinite(t) & (a <= t + gap) & (b > t + 1e-9)
        if not m.any():
            break
        np.maximum.at(top, ip[m], b[m])
    return np.where(np.isfinite(top), top - floors, 0.0)


# ----------------------------------------------------------------------------- model
class Model:
    def __init__(self, f, meshes):
        self.f = f
        self._storey_memo = {}
        self.els = []
        for guid in sorted(meshes):
            try:
                ent = f.by_guid(guid)
            except RuntimeError:
                continue
            self.els.append(El(guid, ent, meshes[guid], self.storey_of(ent)))
        self.elev = {}
        self.building_storeys = defaultdict(list)
        for s in f.by_type("IfcBuildingStorey"):
            z = float(upl.get_local_placement(s.ObjectPlacement)[2, 3]) if s.ObjectPlacement else float(s.Elevation or 0)
            self.elev[s.id()] = z
            b = s.Decomposes[0].RelatingObject if s.Decomposes else None
            self.building_storeys[b.id() if b else 0].append(s)
        for v in self.building_storeys.values():
            v.sort(key=lambda s: self.elev[s.id()])
        self.ground = min(self.elev.values()) if self.elev else 0.0
        self._of, self._rays = {}, {}

    def storey_of(self, e, depth=0):
        if e is None or depth > 12:
            return None
        if e.is_a("IfcBuildingStorey"):
            return e
        i = e.id()
        if i not in self._storey_memo:
            try:
                parent = uel.get_container(e)
            except AttributeError:
                parent = None
            parent = parent or uel.get_aggregate(e)
            self._storey_memo[i] = self.storey_of(parent, depth + 1)
        return self._storey_memo[i]

    def next_storey(self, st):
        if st is None:
            return None
        for v in self.building_storeys.values():
            if st in v:
                k = v.index(st)
                return v[k + 1] if k + 1 < len(v) else None
        return None

    def of(self, *classes):
        if classes not in self._of:
            self._of[classes] = [e for e in self.els if any(e.ent.is_a(c) for c in classes)]
        return self._of[classes]

    def rays(self, key):
        if key not in self._rays:
            if key == "surface":
                els = self.of(*SURFACE)
            elif key == "overhead":
                els = self.of(*OVERHEAD)
            elif key == "barrier":   # plus closed doors (lift landing doors, locked plant rooms) as solid boxes
                els = self.of(*BARRIER) + [solid_box(e) for e in self.of("IfcDoor") if not _passable(e.ent)]
                for w in self.of("IfcWindow"):       # low sills declared as guarded (fixed lower pane, grille)
                    bh = (uel.get_pset(w.ent, "Navigation") or {}).get("BarrierHeight")
                    if isinstance(bh, (int, float)) and w.storey is not None:
                        els.append(solid_box(w, w.zmin, max(w.zmin, self.elev[w.storey.id()] + float(bh))))
            else:
                raise KeyError(key)
            self._rays[key] = Rays(els)
        return self._rays[key]


def _passable(door) -> bool:
    nav = uel.get_pset(door, "Navigation") or {}
    return bool(nav.get("Passable", True))


def _tag(e):
    st = e.storey.Name if getattr(e, "storey", None) is not None else "-"
    return f"{st} {e.cls} {e.name}".strip()


def _xy(p):
    return f"({p[0]:.2f}, {p[1]:.2f})"


def _centre(e):
    """Centre of an element's bounding box (world, m), the point a viewpoint looks at; None for an empty mesh."""
    if not len(e.T):
        return None
    V = e.T.reshape(-1, 3)
    return ((V.min(0) + V.max(0)) / 2).tolist()


def _on(poly, z):
    """A point on a (clash) polygon at height z: inside it when it has area, else its centroid."""
    if poly is None or poly.is_empty:
        return None
    c = poly.representative_point() if poly.area > 0 else poly.centroid
    return (c.x, c.y, z)


# ----------------------------------------------------------------------------- stairs
def _clusters(z, gap=0.005):
    order = np.argsort(z)
    zs = z[order]
    cut = np.flatnonzero(np.diff(zs) > gap) + 1
    return [order[g] for g in np.split(np.arange(len(zs)), cut)]


def measure_flight(M: Model, e: El, ctx: dict, lim=LIMITS) -> dict:
    """Treads, risers, goings, width, headroom and balustrades of one IfcStairFlight from its solid."""
    out = dict(flight=_tag(e), guid=e.guid, xyz=_centre(e), errors=[])
    up = e.N[:, 2] > 0.99
    Tu = e.T[up]
    if not len(Tu):
        out["errors"].append("no treads")
        return out
    zc = Tu[:, :, 2].mean(1)
    treads = sorted(((float(zc[g].mean()), Tu[g][:, :, :2].reshape(-1, 2)) for g in _clusters(zc)),
                    key=lambda t: t[0])
    if len(treads) >= 2:
        c = [(xy.min(0) + xy.max(0)) / 2 for _, xy in treads]
        d = c[-1] - c[0]
    else:                                # one tread: from the lowest to the highest vertices
        V = e.T.reshape(-1, 3)
        d = V[V[:, 2] > e.zmax - 0.01, :2].mean(0) - V[V[:, 2] < e.zmin + 0.01, :2].mean(0)
    if np.linalg.norm(d) < 1e-9:
        out["errors"].append("cannot find the run direction")
        return out
    u = d / np.linalg.norm(d)
    v = np.array([-u[1], u[0]])
    s0 = np.array([(xy @ u).min() for _, xy in treads])
    s1 = np.array([(xy @ u).max() for _, xy in treads])
    l0 = np.array([(xy @ v).min() for _, xy in treads])
    l1 = np.array([(xy @ v).max() for _, xy in treads])
    zt = np.array([t[0] for t in treads])
    lmid = (l0.max() + l1.min()) / 2

    def W(s, l):
        return s * u + l * v

    # walking surfaces at the foot and the head of the flight give the first and the last riser
    surf = M.rays("surface")
    me = surf.owner(e)
    pi, own, z, upf = surf.cast([W(s0[0] - 0.12, lmid), W(s1[-1] + 0.12, lmid)])
    ok = upf & (own != me)
    foot = z[ok & (pi == 0) & (z >= zt[0] - 0.5) & (z <= zt[0] - 0.01)]
    head = z[ok & (pi == 1) & (z >= zt[-1] + 0.01) & (z <= zt[-1] + 0.5)]
    step = float(np.median(np.diff(zt))) if len(zt) > 1 else lim["riser_max"]
    if len(foot):
        z_bot = float(foot.max())
    else:
        z_bot = zt[0] - step
        out["errors"].append(f"foot at {_xy(W(s0[0] - 0.12, lmid))} is not on a floor or landing")
    if len(head):
        z_top = float(head.min())
    else:
        z_top = zt[-1] + step
        out["errors"].append(f"head at {_xy(W(s1[-1] + 0.12, lmid))} does not arrive on a landing")
    risers = np.diff(np.concatenate(([z_bot], zt, [z_top])))
    out.update(z_bot=z_bot, z_top=z_top, risers=risers.round(4).tolist(), n_risers=len(risers),
               going_min=float((s1 - s0).min()))
    # pitch line through the nosings (front edge of each tread, then the landing edge)
    b, a = np.polyfit(np.concatenate((s0, [s1[-1]])), np.concatenate((zt, [z_top])), 1)

    def pitch(s):
        return a + b * s

    # clear width between obstacles (walls, balustrades, columns) across each tread
    obst, tree = ctx["obst"], ctx["obst_tree"]
    widths, free = [], []
    for i in range(len(zt)):
        sm = (s0[i] + s1[i]) / 2
        seg = LineString([W(sm, l0[i]), W(sm, l1[i])])
        blocked = []
        for k in tree.query(seg, predicate="intersects"):
            o = obst[k]
            if o is e or o.zmax < zt[i] + 0.05 or o.zmin > zt[i] + 1.5:
                continue
            inter = o.footprint.intersection(seg)
            if inter.is_empty:
                continue
            ls = np.asarray([cc for g in getattr(inter, "geoms", [inter]) for cc in g.coords]) @ v
            blocked.append((ls.min(), ls.max()))
        gaps, cur = [], l0[i]
        for bl0, bl1 in sorted(blocked):
            if bl0 > cur:
                gaps.append((cur, min(bl0, l1[i])))
            cur = max(cur, bl1)
        if cur < l1[i]:
            gaps.append((cur, l1[i]))
        best = max(gaps, key=lambda g: g[1] - g[0]) if gaps else (lmid, lmid)
        free.append(best)
        widths.append(best[1] - best[0])
    out["width"] = float(min(widths))
    # headroom: vertical clearance above the pitch line to anything overhead
    over = M.rays("overhead")
    pts, zp = [], []
    for i in range(len(zt)):
        f0, f1 = free[i]
        s = s0[i] + 0.03
        for l in (f0 + 0.1, (f0 + f1) / 2, f1 - 0.1):
            pts.append(W(s, l))
            zp.append(pitch(s))
    zp = np.array(zp)
    pi, own, z, _ = over.cast(pts)
    m = (own != over.owner(e)) & (z > zp[pi] + 0.05)
    clear = np.full(len(pts), np.inf)
    np.minimum.at(clear, pi[m], z[m] - zp[pi[m]])
    out["headroom"] = float(clear.min())
    # flight sides: bounded by a wall, or guarded by a balustrade >= 1.0 m above the pitch line
    bar = M.rays("barrier")
    sides = []
    run = s1[-1] - s0[0]
    for side, lside, sgn in (("left", float(l0.min()), -1.0), ("right", float(l1.max()), 1.0)):
        line = LineString([W(s0[0], lside + sgn * 0.05), W(s1[-1], lside + sgn * 0.05)])
        near = [obst[k] for k in tree.query(line, predicate="intersects")]
        walls = [w for w in near if w.ent.is_a("IfcWall") and w.zmax > zt.min() + 0.5 and w.zmin < zt.max() + 0.5]
        if walls and shapely.union_all([w.footprint for w in walls]).intersection(line).length >= 0.9 * line.length:
            sides.append(dict(side=side, bound="wall"))
            continue
        strip = shapely.Polygon([W(s0[0], lside - 0.25), W(s1[-1], lside - 0.25), W(s1[-1], lside + 0.25),
                                 W(s0[0], lside + 0.25)])
        cand = []
        for k in tree.query(strip, predicate="intersects"):
            r = obst[k]          # this flight's balustrade, not the one of the flight a storey above
            if not r.ent.is_a("IfcRailing") or r.zmax < zt.max() or r.zmin > zt.min() + 0.5:
                continue
            ov = r.footprint.intersection(strip)
            cs = np.asarray([cc for g in getattr(ov, "geoms", [ov]) if hasattr(g, "exterior")
                             for cc in g.exterior.coords]).reshape(-1, 2) @ u
            if len(cs):
                cand.append((cs.max() - cs.min(), r))
        if not cand or max(cc[0] for cc in cand) < 0.8 * run:
            sides.append(dict(side=side, bound="open", height=0.0))
            continue
        r = max(cand, key=lambda cc: cc[0])[1]
        lr = float((r.T.reshape(-1, 3)[:, :2] @ v).mean())
        st = s0 + 0.03
        hp = top_of(bar, [W(s, lr) for s in st], bar.owner(r)) - pitch(st)
        hp = hp[np.isfinite(hp)]
        sides.append(dict(side=side, bound="balustrade", rail=_tag(r), rail_guid=r.guid,
                          height=float(hp.min()) if len(hp) else 0.0))
    out["sides"] = sides
    return out


def check_stairs(M: Model, rep: Report, lim=LIMITS):
    flights = M.of("IfcStairFlight")
    obst = [o for o in M.of(*OBSTACLE) if not o.footprint.is_empty]
    ctx = dict(obst=obst, obst_tree=shapely.STRtree([o.footprint for o in obst]))
    meas = [measure_flight(M, e, ctx, lim) for e in flights]
    rows = {k: Hits() for k in ("riser", "var", "going", "count", "width", "head", "land", "open", "bal")}
    for m in meas:
        nm = m["flight"]

        def hit(k, text, *more):
            rows[k].add(text, m["guid"], *more, xyz=m["xyz"])

        for err in m["errors"]:
            hit("land", f"{nm}: {err}")
        if "risers" not in m:
            continue
        r = np.array(m["risers"])
        if r.max() > lim["riser_max"] + TOL:
            hit("riser", f"{nm}: riser {r.max():.4f} m")
        if r.max() - r.min() > lim["riser_var"] + TOL:
            hit("var", f"{nm}: risers {r.min():.4f}..{r.max():.4f} m")
        if m["going_min"] < lim["going_min"] - TOL:
            hit("going", f"{nm}: going {m['going_min']:.4f} m")
        if m["n_risers"] > lim["risers_max"]:
            hit("count", f"{nm}: {m['n_risers']} risers")
        if "width" in m and m["width"] < lim["stair_width"] - TOL:
            hit("width", f"{nm}: clear width {m['width']:.3f} m")
        if "headroom" in m and m["headroom"] < lim["headroom"] - TOL:
            hit("head", f"{nm}: headroom {m['headroom']:.3f} m")
        for s in m.get("sides", []):
            if s["bound"] == "open":
                hit("open", f"{nm}: {s['side']} side open (no wall, no balustrade)")
            elif s["bound"] == "balustrade" and s["height"] < lim["barrier"] - 1e-3:
                hit("bal", f"{nm}: balustrade {s['height']:.3f} m above the pitch line", s.get("rail_guid"))
    vals = [m for m in meas if "risers" in m]
    stat = lambda k, fn: round(float(fn([x[k] for x in vals])), 4) if vals else None  # noqa: E731
    rep.add("stair_riser", "error", rows["riser"], value=dict(max=stat("risers", lambda v: max(max(r) for r in v)),
                                                           limit=lim["riser_max"], flights=len(flights)))
    rep.add("stair_riser_variation", "error", rows["var"], value=lim["riser_var"])
    rep.add("stair_going", "error", rows["going"], value=dict(min=stat("going_min", min), limit=lim["going_min"]))
    rep.add("stair_risers_per_flight", "error", rows["count"], value=dict(max=stat("n_risers", max),
                                                                         limit=lim["risers_max"]))
    rep.add("stair_width", "error", rows["width"], value=dict(min=stat("width", min), limit=lim["stair_width"]))
    rep.add("stair_headroom", "error", rows["head"], value=dict(min=stat("headroom", min), limit=lim["headroom"]))
    rep.add("stair_landings", "error", rows["land"], note="each flight starts on a floor / landing and arrives on one")
    rep.add("stair_open_sides", "error", rows["open"], note="flight sides bounded by a wall or a balustrade")
    bal = [s["height"] for m in vals for s in m.get("sides", []) if s["bound"] == "balustrade"]
    rep.add("stair_balustrade_height", "error", rows["bal"],
            value=dict(min=round(min(bal), 4) if bal else None, limit=lim["barrier"]),
            note="measured vertically above the pitch line through the nosings")
    # every IfcStair climbs exactly from its storey to the next one
    by_stair = defaultdict(list)
    for e, m in zip(flights, meas):
        st = uel.get_aggregate(e.ent)
        if st is not None and "z_bot" in m:
            by_stair[st.id()].append(m)
    bad = Hits()
    for st in M.f.by_type("IfcStair"):
        ms = sorted(by_stair.get(st.id(), []), key=lambda m: m["z_bot"])
        storey = M.storey_of(st)
        nxt = M.next_storey(storey)
        at = next((m["xyz"] for m in ms if m.get("xyz")), None) or where(st)
        if not ms:
            bad.add(f"{st.Name}: no flights", st, xyz=at)
            continue
        if storey is None or nxt is None:
            continue
        z0, z1 = M.elev[storey.id()], M.elev[nxt.id()]
        if abs(ms[0]["z_bot"] - z0) > 0.01 or abs(ms[-1]["z_top"] - z1) > 0.01:
            bad.add(f"{st.Name}: climbs {ms[0]['z_bot']:.3f}..{ms[-1]['z_top']:.3f}, storeys {z0:.3f}..{z1:.3f}",
                    st, *(m["guid"] for m in ms), xyz=at)
            continue
        for p, q in zip(ms[:-1], ms[1:]):
            if abs(q["z_bot"] - p["z_top"]) > 0.01:
                bad.add(f"{st.Name}: {p['flight']} ends {p['z_top']:.3f}, {q['flight']} starts {q['z_bot']:.3f}",
                        st, p["guid"], q["guid"], xyz=q["xyz"])
    rep.add("stair_rise", "error", bad, value=len(M.f.by_type("IfcStair")),
            note="flights of each IfcStair chain from the storey level to the next storey level")
    return meas


# ----------------------------------------------------------------------------- barriers
def check_barrier_elements(M: Model, rep: Report, lim=LIMITS):
    low = Hits()
    for e in M.of("IfcRailing"):
        if uel.get_predefined_type(e.ent) == "BALUSTRADE" or (uel.get_aggregate(e.ent) is not None and
                                                                 uel.get_aggregate(e.ent).is_a("IfcStair")):
            continue                     # measured above the pitch line in check_stairs
        if e.zmax - e.zmin < lim["barrier"] - 1e-3:
            low.add(f"{_tag(e)}: {e.zmax - e.zmin:.3f} m", e.guid, xyz=_centre(e))
    for e in M.of("IfcWall"):
        if uel.get_predefined_type(e.ent) == "PARAPET" and e.zmax - e.zmin < lim["barrier"] - 1e-3:
            low.add(f"{_tag(e)}: parapet {e.zmax - e.zmin:.3f} m", e.guid, xyz=_centre(e))
    rep.add("barrier_heights", "error", low, value=lim["barrier"], note="guard rails and parapet walls")


def _ring_samples(poly, step=0.1, corner=0.06):
    out = []
    poly = orient(poly, 1.0)
    for rid, ring in enumerate([poly.exterior, *poly.interiors]):
        c = np.asarray(ring.coords)
        seq = 0
        for a, b in zip(c[:-1], c[1:]):
            d = b - a
            ln = float(np.hypot(*d))
            if ln < 2 * corner + 1e-6:
                continue
            u = d / ln
            k = max(1, int(round((ln - 2 * corner) / step)))
            ts = corner + (np.arange(k) + 0.5) * (ln - 2 * corner) / k
            for t in ts:
                out.append((a + u * t, (u[1], -u[0]), rid, seq, (ln - 2 * corner) / k))
                seq += 1
    return out


def check_falling_edges(M: Model, rep: Report, lim=LIMITS):
    surf, bar = M.rays("surface"), M.rays("barrier")
    P, NO, TOP, SID, RID, SEQ, STEP, names, slab_guid = [], [], [], [], [], [], [], [], []
    rings = 0
    for e in M.of("IfcSlab"):
        if e.storey is None:
            continue
        poly, top = e.top()
        if poly is None or poly.is_empty:
            continue
        if abs(top - M.elev[e.storey.id()]) > 0.6 and uel.get_predefined_type(e.ent) != "LANDING":
            continue                     # tower / motor-room roofs are not accessible
        sid = len(names)
        names.append(_tag(e))
        slab_guid.append(e.guid)
        for g in getattr(poly, "geoms", [poly]):
            base = rings
            for p, n, rid, seq, step in _ring_samples(g):
                P.append(p); NO.append(n); TOP.append(top); SID.append(sid); RID.append(base + rid)
                SEQ.append(seq); STEP.append(step)
                rings = max(rings, base + rid + 1)
    if not P:
        rep.add("falling_edges", "error", [], value=0.0)
        rep.add("falling_edge_gaps", "warn", [], value=0.0)
        return
    P, NO, TOP = np.array(P), np.array(NO), np.array(TOP)
    SID, RID, SEQ, STEP = np.array(SID), np.array(RID), np.array(SEQ), np.array(STEP)
    # drop just outside the edge: highest walking surface no more than a step above the slab
    below = np.full(len(P), M.ground)
    pi, own, z, up = surf.cast(P + NO * 0.15)
    m = up & (z <= TOP[pi] + lim["drop"])
    np.maximum.at(below, pi[m], z[m])
    need = (TOP - below) > lim["drop"]
    h = np.zeros(len(P))
    idx = np.flatnonzero(need)
    for d in (0.08, 0.14, 0.03, -0.03, -0.08, 0.21, 0.30):   # inside the edge, then walls just outside it
        if not len(idx):
            break
        h[idx] = np.maximum(h[idx], barrier_heights(bar, P[idx] - NO[idx] * d, TOP[idx]))
        idx = idx[h[idx] < lim["barrier"] - 1e-3]
    open_ = np.zeros(len(P), bool)
    open_[idx] = True
    # narrow voids (stair wells): a walking surface or barrier within 0.4 m across the void
    narrow = np.zeros(len(P), bool)
    if len(idx):
        for d in (0.25, 0.38):
            Q = P[idx] + NO[idx] * d
            pi, own, z, up = surf.cast(Q)
            m = up & (np.abs(z - TOP[idx][pi]) <= 0.6)
            narrow[idx[np.unique(pi[m])]] = True
            pi, own, z, up = bar.cast(Q)
            m = (z >= TOP[idx][pi] - 0.6) & (z <= TOP[idx][pi] + 1.2)
            narrow[idx[np.unique(pi[m])]] = True
    errors, gaps = Hits(), Hits()
    total_err = total_gap = 0.0
    for key in sorted(set(zip(SID[open_].tolist(), RID[open_].tolist()))):
        sel = np.flatnonzero(open_ & (SID == key[0]) & (RID == key[1]))
        sel = sel[np.argsort(SEQ[sel])]
        runs, cur = [], [sel[0]]
        for a, b in zip(sel[:-1], sel[1:]):
            if SEQ[b] == SEQ[a] + 1:
                cur.append(b)
            else:
                runs.append(cur)
                cur = [b]
        runs.append(cur)
        ring_n = int(((SID == key[0]) & (RID == key[1])).sum())
        if len(runs) > 1 and SEQ[runs[0][0]] == 0 and SEQ[runs[-1][-1]] == ring_n - 1:
            runs[0] = runs.pop() + runs[0]   # wrap around the ring start
        for r in runs:
            r = np.array(r)
            length = float(STEP[r].sum())
            wide = ~narrow[r]
            item = (f"{names[key[0]]}: {length:.2f} m unguarded from {_xy(P[r[0]])} to {_xy(P[r[-1]])}, "
                    f"drop {float((TOP - below)[r].max()):.2f} m, barrier {float(h[r].max()):.2f} m")
            mid = r[len(r) // 2]
            at = (P[mid][0], P[mid][1], TOP[mid])
            if wide.any() and float(STEP[r[wide]].sum()) >= lim["gap_narrow"] - 1e-6:
                errors.add(item, slab_guid[key[0]], xyz=at)
                total_err += length
            else:
                gaps.add(item, slab_guid[key[0]], xyz=at)
                total_gap += length
    rep.add("falling_edges", "error", errors, value=dict(unguarded_m=round(total_err, 2), samples=len(P),
                                                         with_drop=int(need.sum())),
            note=f"edges with a drop > {lim['drop']} m need a barrier >= {lim['barrier']} m")
    rep.add("falling_edge_gaps", "warn", gaps, value=dict(unguarded_m=round(total_gap, 2)),
            note=f"unguarded gaps narrower than {lim['gap_narrow']} m (stair wells at landings): not passable "
                 f"for the 0.30 m agent, but open to the 100 mm sphere rule")


# ----------------------------------------------------------------------------- clashes
def _thickness(poly):
    r = poly.minimum_rotated_rectangle
    c = np.asarray(r.exterior.coords)
    if len(c) < 4:
        return 0.0
    return float(min(np.hypot(*(c[1] - c[0])), np.hypot(*(c[2] - c[1]))))


def _by_storey(els):
    out = defaultdict(list)
    for e in els:
        out[e.storey.id() if e.storey is not None else 0].append(e)
    return out


def check_clashes(M: Model, rep: Report, lim=LIMITS):
    walls = [w for w in M.of("IfcWall") if not w.footprint.is_empty]
    over = Hits()
    worst = 0.0
    for sid, ws in sorted(_by_storey(walls).items()):
        tree = shapely.STRtree([w.footprint for w in ws])
        a, b = tree.query([w.footprint for w in ws], predicate="intersects")
        for i, j in zip(a, b):
            if i >= j:
                continue
            w1, w2 = ws[i], ws[j]
            if min(w1.zmax, w2.zmax) - max(w1.zmin, w2.zmin) <= 0.01:
                continue
            inter = w1.footprint.intersection(w2.footprint)
            ar = inter.area
            allow = _thickness(w1.footprint) * _thickness(w2.footprint) + lim["wall_overlap_tol"]
            worst = max(worst, ar - allow)
            if ar > allow:
                over.add(f"{_tag(w1)} / {w2.name}: {ar:.4f} m2 > {allow:.4f} m2", w1.guid, w2.guid,
                         xyz=_on(inter, (max(w1.zmin, w2.zmin) + min(w1.zmax, w2.zmax)) / 2))
    rep.add("wall_overlap", "error", over, value=dict(worst_excess_m2=round(worst, 5)),
            note="per storey, overlap <= t1 * t2 + 1e-4 m2 (junction tolerance)")

    # doors / windows against walls that do not host them
    host = {}
    for o in M.f.by_type("IfcOpeningElement"):
        w = o.VoidsElements[0].RelatingBuildingElement if o.VoidsElements else None
        for rel in o.HasFillings or ():
            host[rel.RelatedBuildingElement.id()] = w.id() if w is not None else None
    clash = Hits()
    wtree = shapely.STRtree([w.footprint for w in walls])
    for d in M.of("IfcDoor", "IfcWindow"):
        fp = d.footprint
        if fp.is_empty:
            continue
        for k in wtree.query(fp, predicate="intersects"):
            w = walls[k]
            if w.ent.id() == host.get(d.ent.id()):
                continue
            if min(d.zmax, w.zmax) - max(d.zmin, w.zmin) <= 0.01:
                continue
            inter = fp.intersection(w.footprint)
            ar = inter.area
            if ar > lim["clash_area"]:
                clash.add(f"{_tag(d)} x {w.name}: {ar:.4f} m2", d.guid, w.guid,
                          xyz=_on(inter, (max(d.zmin, w.zmin) + min(d.zmax, w.zmax)) / 2))
    rep.add("opening_clash", "error", clash, note="doors / windows clear of every wall except their host")

    # slabs inside lift shafts
    bad, shafts = Hits(), []
    for car in M.of("IfcTransportElement"):
        nav = uel.get_pset(car.ent, "Navigation") or {}
        if uel.get_predefined_type(car.ent) not in ("ELEVATOR", "LIFT") and not nav.get("Elevator"):
            continue
        fp = car.footprint
        if fp.is_empty:
            continue
        x0, y0, x1, y1 = fp.bounds
        frame = box(x0 - 1.5, y0 - 1.5, x1 + 1.5, y1 + 1.5)
        ws = [w for w in walls if w.zmin <= car.zmax and w.zmax >= car.zmin and w.footprint.intersects(frame)]
        free = frame.difference(shapely.union_all([w.footprint for w in ws])) if ws else frame
        shaft = None
        for g in getattr(free, "geoms", [free]):
            if g.contains(fp.centroid) and not g.exterior.intersects(frame.exterior):
                shaft = g
        if shaft is None:
            shaft = fp.buffer(0.25, join_style="mitre")
        ring = shaft.buffer(0.05, join_style="mitre")
        z0 = car.zmin - 0.04
        z1 = max([w.zmax for w in walls if w.footprint.intersects(ring) and w.zmin >= z0 - 1.0] + [car.zmax])
        # the shaft runs from the pit to the head of its top landing door; a slab above that is the shaft cap
        landing = [d.zmax for d in M.of("IfcDoor")
                   if (uel.get_pset(d.ent, "Navigation") or {}).get("DoorKind") == "lift"
                   and d.footprint.intersects(ring.buffer(0.3))]
        if landing:
            z1 = min(z1, max(landing) + 0.05)
        shafts.append(dict(car=car.name, area=round(shaft.area, 3), z=(round(z0, 3), round(z1, 3))))
        for s in M.of("IfcSlab"):
            if s.zmax <= z0 + 0.01 or s.zmin >= z1 - 0.01:
                continue
            inter = s.footprint.intersection(shaft)
            ar = inter.area
            if ar > lim["shaft_slab"]:
                bad.add(f"{_tag(s)}: {ar:.3f} m2 inside the {car.name} shaft", s.guid, car.guid, xyz=_on(inter, s.zmax))
    rep.add("lift_shaft_slab", "error", bad, value=shafts, note=f"slab area inside a lift shaft <= {lim['shaft_slab']} m2")

    # space overlaps per storey
    spaces = [s for s in M.of("IfcSpace") if not s.footprint.is_empty]
    sov = Hits()
    for sid, ss in sorted(_by_storey(spaces).items()):
        fps = [s.top()[0] for s in ss]
        tree = shapely.STRtree(fps)
        a, b = tree.query(fps, predicate="intersects")
        for i, j in zip(a, b):
            if i >= j or min(ss[i].zmax, ss[j].zmax) - max(ss[i].zmin, ss[j].zmin) <= 0.01:
                continue
            inter = fps[i].intersection(fps[j])
            ar = inter.area
            if ar > lim["space_overlap"]:
                sov.add(f"{_tag(ss[i])} / {ss[j].name}: {ar:.3f} m2", ss[i].guid, ss[j].guid,
                        xyz=_on(inter, max(ss[i].zmin, ss[j].zmin)))
    rep.add("space_overlap", "error", sov, note=f"IfcSpace overlap per storey <= {lim['space_overlap']} m2")


# ----------------------------------------------------------------------------- wet stacks
def check_wet_stack(M: Model, rep: Report, lim=LIMITS):
    from estate.flats.template import ROOM_CODES
    rooms = []
    for s in M.of("IfcSpace"):
        p = uel.get_pset(s.ent, "SampleCity_Room") or {}
        code = p.get("RoomCode")
        if not code or s.storey is None:
            continue
        cat = p.get("Category") or (ROOM_CODES[code][1] if code in ROOM_CODES else "")
        unit = str(p.get("UnitNumber") or "")
        stack = unit.rsplit("-", 1)[-1] if "-" in unit else unit
        b = s.storey.Decomposes[0].RelatingObject.id() if s.storey.Decomposes else 0
        rooms.append((b, stack, s.storey, cat, s.top()[0], s.guid))
    if not rooms:
        rep.add("wet_stack", "error", [], value="no residential spaces")
        rep.add("wet_over_dry", "error", [])
        return
    wet = defaultdict(list)
    dry = defaultdict(list)
    wet_g, dry_g = defaultdict(list), defaultdict(list)     # GlobalIds, same order
    for b, stack, st, cat, fp, g in rooms:
        if cat in WET:
            wet[(b, stack, st.id())].append(fp)
            wet_g[(b, stack, st.id())].append(g)
        elif cat in DRY:
            dry[(b, st.id())].append(fp)
            dry_g[(b, st.id())].append(g)
    bad, ious = Hits(), []
    stacks = sorted({(b, stack) for b, stack, *_ in rooms})
    for b, stack in stacks:
        sts = [s for s in M.building_storeys[b] if (b, stack, s.id()) in wet]
        for lo, hi in zip(sts[:-1], sts[1:]):
            A = shapely.union_all(wet[(b, stack, lo.id())])
            B = shapely.union_all(wet[(b, stack, hi.id())])
            iou = A.intersection(B).area / max(A.union(B).area, 1e-9)
            ious.append(iou)
            if iou < lim["wet_iou"]:
                diff = A.symmetric_difference(B)
                bad.add(f"stack {stack} {lo.Name}->{hi.Name}: wet IoU {iou:.3f}",
                        *wet_g[(b, stack, hi.id())], *wet_g[(b, stack, lo.id())],
                        xyz=_on(diff if not diff.is_empty else B, M.elev[hi.id()]))
    rep.add("wet_stack", "error", bad, value=dict(min_iou=round(min(ious), 4) if ious else None, stacks=len(stacks)),
            note=f"wet rooms per stack keep their footprint storey to storey (IoU >= {lim['wet_iou']})")
    wod = Hits()
    worst = 0.0
    for b, sts in M.building_storeys.items():
        res = [s for s in sts if any(k[0] == b and k[2] == s.id() for k in wet) or (b, s.id()) in dry]
        for lo, hi in zip(res[:-1], res[1:]):
            W_ = [(fp, g) for k, v in wet.items() if k[0] == b and k[2] == hi.id() for fp, g in zip(v, wet_g[k])]
            D_ = list(zip(dry.get((b, lo.id()), []), dry_g.get((b, lo.id()), [])))
            if not W_ or not D_:
                continue
            inter = shapely.union_all([fp for fp, _ in W_]).intersection(shapely.union_all([fp for fp, _ in D_]))
            ar = inter.area
            worst = max(worst, ar)
            if ar > lim["wet_over_dry"]:
                hit = [g for fp, g in W_ + D_ if fp.intersection(inter).area > 1e-6]
                wod.add(f"{hi.Name} wet over {lo.Name} habitable: {ar:.3f} m2", *hit, xyz=_on(inter, M.elev[hi.id()]))
    rep.add("wet_over_dry", "error", wod, value=dict(worst_m2=round(worst, 4)),
            note=f"wet room over a habitable room <= {lim['wet_over_dry']} m2 per storey pair")


# ----------------------------------------------------------------------------- entry points
def check(f, meshes, lim=None) -> list:
    lim = lim or limits()
    M = Model(f, meshes)
    rep = Report()
    check_stairs(M, rep, lim)
    check_barrier_elements(M, rep, lim)
    check_falling_edges(M, rep, lim)
    check_clashes(M, rep, lim)
    check_wet_stack(M, rep, lim)
    return rep.results


def check_file(path, meshes=None) -> dict:
    from estate.export import meshcache
    t = time.time()
    if meshes is None:
        f, meshes = meshcache.load(path)
    else:
        f = ifcopenshell.open(str(path))
    t_mesh = time.time() - t
    res = check(f, meshes)
    return {"file": env.rel(path), "runtime_s": round(time.time() - t, 2), "mesh_s": round(t_mesh, 2),
            "summary": summarize(res), "results": res}
