"""Road works of the estate (geometry only): carriageways, junctions, kerbs, verges, sidewalks, bus bays and crossings.

Every road in config/estate.toml is a straight centreline with a reserve and a carriageway width. A road that
ends at another road's reserve is extended to that road's centreline so the carriageways meet in a junction, and
junction corners are filleted with a kerb radius. Across the reserve, from the centreline out: carriageway (top
-0.15), 150 mm kerb, planting verge (trees), sidewalk (top 0.0) and a grass margin to the reserve line. Bus stops
cut a bay into the verge; the platform behind the bay carries the shelter and diverts the sidewalk behind it.
Crossings are zebra strips across the carriageway with a kerb ramp (<= 1:10) on each side; side roads get one at
the junction mouth, bus stops one beyond the bay, and the linkway and footpath routers add more where they cross.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from estate.site.linkways import H, V

CARRIAGE_TOP = -0.15
PAVE_TOP = 0.0
KERB_W = 0.15
FILLET = {"collector": 7.5, "local": 4.5, "access": 4.5}
RAMP_LEN = 1.9                     # 150 mm over 1.9 m = 1:12.7 (<= 1:10 required, 1:12 for the nav agent)
CROSSING_W = 4.0
CROSSOVER_LEN = 1.5                # driveway crossover: kerb dropped, wedge up to the verge
BAY_DEPTH, BAY_LEN, BAY_TAPER = 3.0, 24.0, 15.0
PLATFORM_DEPTH, PLATFORM_LEN = 5.5, 30.0
SHELTER_LEN, SHELTER_FRONT, SHELTER_REAR, SHELTER_SOFFIT = 10.0, 0.4, 3.0, 2.6
DETOUR = 4.3                       # sidewalk line behind the shelter, from the bay kerb


def _t(p, nd=3):
    return tuple(round(float(v), nd) for v in p)


@dataclass
class Road:
    id: str
    name: str
    cls: str
    p0: np.ndarray
    p1: np.ndarray
    cw: float
    reserve: float
    meets: dict = field(default_factory=dict)       # end (0/1) -> id of the road this end was extended to

    @property
    def length(self):
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def u(self):
        return (self.p1 - self.p0) / self.length

    @property
    def n(self):
        u = self.u
        return np.array([-u[1], u[0]])

    @property
    def axis(self):
        return H if abs(self.u[0]) >= abs(self.u[1]) else V

    @property
    def side_w(self):
        return (self.reserve - self.cw) / 2

    @property
    def sw(self):
        return round(min(2.5, max(1.5, 0.45 * self.side_w)), 1)

    @property
    def verge(self):
        return round(min(3.0, max(1.0, self.side_w - self.sw - 0.6)), 1)

    @property
    def fillet(self):
        return FILLET.get(self.cls, 4.5)

    def pt(self, s, d):
        return self.p0 + self.u * s + self.n * d

    def sd(self, p):
        q = np.asarray(p, float)[:2] - self.p0
        return float(q @ self.u), float(q @ self.n)

    def band(self, d0, d1, s0=0.0, s1=None):
        s1 = self.length if s1 is None else s1
        return Polygon([tuple(self.pt(s0, d0)), tuple(self.pt(s1, d0)), tuple(self.pt(s1, d1)), tuple(self.pt(s0, d1))])

    @property
    def line(self):
        return LineString([tuple(self.p0), tuple(self.p1)])


@dataclass
class Crossing:
    id: str
    kind: str                 # junction | bus | linkway | footpath
    road: Road
    s: float
    width: float
    da: float                 # signed kerb offsets (carriageway edges) on either side
    db: float
    strip: object             # crossing pavement polygon (inside the carriageway)
    ramps: list               # [dict(top, dir, length, width, poly)]
    approaches: list = field(default_factory=list)   # paving patches from the ramp top to the sidewalk
    connectors: list = field(default_factory=list)   # graph lines ramp top -> sidewalk line

    @property
    def tops(self):
        return [tuple(r["top"]) for r in self.ramps]

    @property
    def opening(self):
        return unary_union([r["poly"].buffer(0.05, join_style="mitre") for r in self.ramps])


@dataclass
class BusStop:
    id: str
    road: Road
    s: float
    side: int
    bay: object
    platform: object
    shelter: object           # shelter roof footprint
    kerb_d: float             # signed offset of the bay kerb line
    node: tuple               # shelter centre (graph node)
    stub: tuple               # rear edge of the roof, where the linkway lands
    out: tuple                # unit vector away from the road


@dataclass
class RoadNet:
    roads: list
    extent: object
    C: object                 # all carriageways incl. junction fillets and bus bays
    junctions: list           # dict(id, p, roads, poly, through, side)
    reserve: object
    verges: dict              # road id -> verge polygon (both sides)
    sidewalks: dict           # road id -> sidewalk polygon
    sidewalk_lines: list      # [(LineString, road id, side, width)]
    bus_stops: list
    crossings: list = field(default_factory=list)
    crossovers: list = field(default_factory=list)   # dict(road, s, width, poly, top, dir, name)
    carriage_parts: dict = field(default_factory=dict)  # road id -> carriageway polygon (junctions excluded)

    def road(self, rid):
        return next(r for r in self.roads if r.id == rid)

    def road_at(self, p, margin=0.5):
        """Road whose reserve contains point p (first in config order)."""
        P = Point(p)
        for r in self.roads:
            if r.band(-r.reserve / 2 - margin, r.reserve / 2 + margin).contains(P):
                return r
        return None

    def near_junction(self, p, dist):
        return any(math.dist(j["p"], p) < dist for j in self.junctions)

    @property
    def kerb_ring(self):
        return (self.C.buffer(KERB_W, join_style="mitre").difference(self.C)).intersection(self.extent)


def _extend(roads):
    """Extend road ends that stop at another road's reserve to that road's centreline."""
    for r in roads:
        for end in (0, 1):
            p = r.p0 if end == 0 else r.p1
            out = -r.u if end == 0 else r.u
            for o in roads:
                if o is r:
                    continue
                s, d = o.sd(p)
                if not (-1.0 <= s <= o.length + 1.0) or abs(d) > o.reserve / 2 + 1.0:
                    continue
                den = float(out @ o.n)
                if abs(den) < 0.5:              # parallel-ish: not a T into o
                    continue
                t = -d / den
                if 0.0 < t <= o.reserve / 2 + 1.0:
                    q = p + out * t
                    if end == 0:
                        r.p0 = q
                    else:
                        r.p1 = q
                    r.meets[end] = o.id
                break


def layout_roads(mp: dict) -> RoadNet:
    cfg = mp["cfg"]
    extent = mp["extent"]
    roads = []
    for rd in cfg.get("road", []):
        pts = np.array(rd["centreline"], float)
        roads.append(Road(rd["id"], rd.get("name", rd["id"]), rd.get("class", "local"), pts[0].copy(), pts[-1].copy(),
                          float(rd["carriageway"]), float(rd["reserve"])))
    _extend(roads)
    # ---- carriageways and junctions
    C = unary_union([r.band(-r.cw / 2, r.cw / 2) for r in roads])
    junctions = []
    for i, a in enumerate(roads):
        for b in roads[i + 1:]:
            if not a.band(-a.cw / 2, a.cw / 2).intersects(b.band(-b.cw / 2, b.cw / 2)):
                continue
            ip = a.line.intersection(b.line)
            if ip.is_empty or ip.geom_type != "Point":
                continue
            P = (ip.x, ip.y)
            rf = max(a.fillet, b.fillet)
            half = max(a.cw, b.cw) / 2 + rf + 2.0
            sq = box(P[0] - half, P[1] - half, P[0] + half, P[1] + half)
            closed = C.intersection(sq).buffer(rf, quad_segs=6).buffer(-rf, quad_segs=6).intersection(sq)
            C = unary_union([C, closed])
            through, side = (a, b) if a.id in b.meets.values() else (b, a) if b.id in a.meets.values() else (a, None)
            hj = max(a.cw, b.cw) / 2 + rf
            junctions.append(dict(id=f"J{len(junctions) + 1}", p=P, roads=(a.id, b.id), fillet=rf, half=hj,
                                  through=through.id, side=side.id if side else None))
    # ---- bus bays
    stops = []
    for k, bs in enumerate(mp.get("bus_stops", [])):
        r = next((x for x in roads if x.id == bs.get("road")), None)
        if r is None:
            continue
        s, d = r.sd(bs["at"])
        side = 1 if d >= 0 else -1
        e = side * (r.cw / 2 - 0.01)
        kd = side * (r.cw / 2 + BAY_DEPTH)
        bay = Polygon([tuple(r.pt(s - BAY_LEN / 2 - BAY_TAPER, e)), tuple(r.pt(s - BAY_LEN / 2, kd)),
                       tuple(r.pt(s + BAY_LEN / 2, kd)), tuple(r.pt(s + BAY_LEN / 2 + BAY_TAPER, e))])
        C = unary_union([C, bay])
        stops.append(dict(id=bs.get("id", f"BS{k + 1}"), road=r, s=s, side=side, bay=bay, kd=kd))
    C = C.intersection(extent).buffer(0)
    for j in junctions:
        P, hj = j["p"], j["half"]
        j["poly"] = C.intersection(box(P[0] - hj, P[1] - hj, P[0] + hj, P[1] + hj))
    jun_u = unary_union([j["poly"] for j in junctions]) if junctions else Polygon()
    kerb_zone = C.buffer(KERB_W, join_style="mitre")
    # ---- verges and sidewalks
    verge_bands, sw_bands = {}, {}
    for r in roads:
        a, v, w = r.cw / 2, r.verge, r.sw
        verge_bands[r.id] = unary_union([r.band(a, a + v), r.band(-a - v, -a)])
        sw_bands[r.id] = unary_union([r.band(a + v, a + v + w), r.band(-a - v - w, -a - v)])
    all_verge = unary_union(list(verge_bands.values()))
    S_all = unary_union(list(sw_bands.values())).difference(all_verge).difference(kerb_zone).intersection(extent)
    platforms = []
    for st in stops:
        r, s, side, kd = st["road"], st["s"], st["side"], st["kd"]
        plat = r.band(kd, kd + side * PLATFORM_DEPTH, s - PLATFORM_LEN / 2, s + PLATFORM_LEN / 2)
        plat = shapely.normalize(plat).difference(kerb_zone).intersection(extent)
        platforms.append(plat)
        roof = r.band(kd + side * SHELTER_FRONT, kd + side * SHELTER_REAR, s - SHELTER_LEN / 2, s + SHELTER_LEN / 2)
        node = _t(r.pt(s, kd + side * (SHELTER_FRONT + SHELTER_REAR) / 2), 3)
        stub = _t(r.pt(s, kd + side * SHELTER_REAR), 3)
        out = _t(r.n * side, 6)
        st["obj"] = BusStop(st["id"], r, s, side, st["bay"], plat, shapely.normalize(roof), kd, node, stub, out)
    plat_u = unary_union(platforms) if platforms else Polygon()
    S_all = S_all.difference(plat_u)
    sidewalks, verges, assigned = {}, {}, Polygon()
    for r in roads:
        mine = S_all.intersection(sw_bands[r.id].buffer(0.01, join_style="mitre")).difference(assigned)
        sidewalks[r.id] = mine
        assigned = unary_union([assigned, mine])
        verges[r.id] = verge_bands[r.id].difference(kerb_zone).difference(plat_u).intersection(extent)
    reserve = unary_union([r.band(-r.reserve / 2, r.reserve / 2) for r in roads]).intersection(extent)
    # ---- sidewalk centrelines (graph), detouring behind bus shelters
    walk = unary_union([S_all, plat_u]).buffer(0.3, join_style="mitre")
    lines = []
    for r in roads:
        for side in (-1, 1):
            dm = side * (r.cw / 2 + r.verge + r.sw / 2)
            pts = [(0.0, dm)]
            for st in sorted((x for x in stops if x["road"] is r and x["side"] == side), key=lambda x: x["s"]):
                db = st["kd"] + side * DETOUR
                s = st["s"]
                pts += [(s - PLATFORM_LEN / 2, dm), (s - PLATFORM_LEN / 2 + 3, db), (s + PLATFORM_LEN / 2 - 3, db),
                        (s + PLATFORM_LEN / 2, dm)]
            pts.append((r.length, dm))
            ln = LineString([tuple(r.pt(s, d)) for s, d in pts]).intersection(walk)
            for g in getattr(ln, "geoms", [ln]):
                if g.geom_type == "LineString" and g.length > 0.5:
                    lines.append((g, r.id, side, r.sw))
    net = RoadNet(roads, extent, C, junctions, reserve, verges, sidewalks, lines, [st["obj"] for st in stops])
    net.platforms = platforms
    net.sw_all = S_all
    # carriageway parts per road (junctions separate)
    rest = C.difference(jun_u)
    done = Polygon()
    for r in roads:
        mine = rest.intersection(r.band(-r.cw / 2 - BAY_DEPTH - 0.5, r.cw / 2 + BAY_DEPTH + 0.5)).difference(done)
        net.carriage_parts[r.id] = mine
        done = unary_union([done, mine])
    # ---- crossings at junction mouths and next to bus stops
    for j in junctions:
        if not j["side"]:
            continue
        sd, th = net.road(j["side"]), net.road(j["through"])
        off = th.cw / 2 + j["fillet"] + 1.0 + CROSSING_W / 2
        for end, other in sd.meets.items():
            if other != th.id:
                continue
            s = off if end == 0 else sd.length - off
            add_crossing(net, sd, s, "junction", approach=True)
    for st in net.bus_stops:
        r = st.road
        for direction in (1, -1):
            s = st.s + direction * (BAY_LEN / 2 + BAY_TAPER + CROSSING_W / 2 + 3.0)
            p = r.pt(s, 0.0)
            if 20.0 < s < r.length - 20.0 and not net.near_junction(p, 30.0) and extent.contains(Point(p)):
                add_crossing(net, r, s, "bus", approach=True)
                break
    return net


def kerb_offsets(net: RoadNet, r: Road, s: float):
    """Signed offsets of the carriageway edges across road r at chainage s (follows bays and fillets)."""
    a, b = r.pt(s, -r.reserve / 2 - 2.0), r.pt(s, r.reserve / 2 + 2.0)
    cut = LineString([tuple(a), tuple(b)]).intersection(net.C)
    c = Point(tuple(r.pt(s, 0.0)))
    best = None
    for g in getattr(cut, "geoms", [cut]):
        if g.geom_type != "LineString":
            continue
        if best is None or g.distance(c) < best.distance(c):
            best = g
    if best is None:
        return None
    ds = sorted(r.sd(p)[1] for p in best.coords)
    return ds[0], ds[-1]


def add_crossing(net: RoadNet, r: Road, s: float, kind: str, width=CROSSING_W, approach=False):
    """Zebra crossing across road r at chainage s with a kerb ramp on both sides; returns the Crossing."""
    ko = kerb_offsets(net, r, s)
    if ko is None:
        return None
    da, db = ko
    strip = r.band(da, db, s - width / 2, s + width / 2).intersection(net.C)
    ramps = []
    for d0, sign in ((da, -1), (db, 1)):
        top_d = d0 + sign * RAMP_LEN
        poly = shapely.normalize(r.band(d0, top_d, s - width / 2, s + width / 2))
        ramps.append(dict(top=_t(r.pt(s, top_d), 3), low=_t(r.pt(s, d0), 3),
                          dir=_t(-r.n * sign, 6), length=RAMP_LEN, width=width, poly=poly))
    cr = Crossing(f"X{len(net.crossings) + 1:02d}", kind, r, s, width, da, db, strip, ramps)
    if approach:
        for d0, sign in ((da, -1), (db, 1)):
            top_d = d0 + sign * RAMP_LEN
            dm = sign * (r.cw / 2 + r.verge + r.sw / 2)
            dout = sign * (r.cw / 2 + r.verge + r.sw)
            if abs(dout) > abs(top_d) + 0.05:
                cr.approaches.append(shapely.normalize(r.band(top_d, dout, s - width / 2, s + width / 2)))
            cr.connectors.append(LineString([tuple(r.pt(s, top_d)), tuple(r.pt(s, dm))]))
    net.crossings.append(cr)
    return cr


def crossover_at(net: RoadNet, p_in, p_out, width, name):
    """Crossover where a driveway centreline from p_out (inside the estate) reaches the carriageway at p_in."""
    r = net.road_at(p_in)
    if r is None:
        return None
    s, d = r.sd(p_in)
    ko = kerb_offsets(net, r, s)
    if ko is None:
        return None
    _, dout = r.sd(p_out)
    sign = 1 if dout > 0 else -1
    d0 = ko[1] if sign > 0 else ko[0]
    poly = shapely.normalize(r.band(d0, d0 + sign * CROSSOVER_LEN, s - width / 2, s + width / 2))
    co = dict(road=r.id, s=s, width=width, poly=poly, name=name, top=_t(r.pt(s, d0 + sign * CROSSOVER_LEN), 3),
              low=_t(r.pt(s, d0), 3), dir=_t(-r.n * sign, 6), length=CROSSOVER_LEN,
              edge=_t(r.pt(s, d0), 3))
    net.crossovers.append(co)
    return co


def kerbs(net: RoadNet):
    """Kerb ring around all carriageways with openings at crossing ramps and driveway crossovers, per road."""
    ring = net.kerb_ring
    holes = [c.opening for c in net.crossings] + [co["poly"].buffer(0.05, join_style="mitre") for co in net.crossovers]
    if holes:
        ring = ring.difference(unary_union(holes))
    out, done = {}, Polygon()
    for r in net.roads:
        mine = ring.intersection(r.band(-r.cw / 2 - BAY_DEPTH - 1.0, r.cw / 2 + BAY_DEPTH + 1.0)).difference(done)
        out[r.id] = mine
        done = unary_union([done, mine])
    left = ring.difference(done)
    if not left.is_empty and net.roads:
        out[net.roads[0].id] = unary_union([out[net.roads[0].id], left])
    return out
