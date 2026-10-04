"""Programme checks of the flats, read back from one IFC file alone (no plan, no template).

estate/flats/check.py checks the same HDB / SCDF programme on the derived plates before anything is written. A
block frozen for hand edits in Bonsai never passes through the planner again, and a builder bug that only shows in
the written file would slip past the plan check, so this module re-derives the programme from what the file says:

- a flat is an IfcZone with SampleCity_Flat (FlatType, legacy labels such as '4-Room' mapped to 4R; the IFA is
  InternalFloorArea, else GrossArea); its rooms are the zone's IfcSpaces, coded by SampleCity_Room.RoomCode, with
  areas from Qto_SpaceBaseQuantities.NetFloorArea;
- footprints come from the solids (extruded profiles are read directly, anything else is tessellated), so a space
  reshaped in Bonsai is measured as drawn;
- a window or door serves the spaces met by a short probe across its host wall, perpendicular to it from the
  centre of the leaf; a door's Navigation.FromSpace / ToSpace are used when they name existing spaces (and are
  compared with the probe);
- rooms with no wall between them (an open kitchen, a foyer open to the living room) are connected.

The rules are config/rules.toml: required rooms per flat type, minimum net areas and clear widths, living room
area, household shelter area by IFA band, glazing of habitable rooms (>= 10 % of the net floor area), a vent window
in every bathroom unless SampleCity_Flat.Variant lists it under mechanical_vent, a kitchen window or opening to
the living room / service yard, door widths, a main door from a common space, every room reachable from the
entrance, the shelter not entered from a bedroom or bathroom and the master bath entered only from the master
bedroom. Two warnings compare what the file declares with what it draws: door FromSpace / ToSpace against the
probe, and NetFloorArea against the footprint (a hand edit that moved one but not the other). Spaces outside flat
zones (lobbies, corridors, void deck, stairs, refuse rooms, shops), grouped in a common-property zone or in none,
are only used as the common side of entrances and are never checked themselves. Results use the {check, level,
severity, count, examples, items} format of ifcqa; each item carries the GlobalIds of the space (or the flat zone,
or the door and the spaces on either side) and a point inside the room at floor level for a BCF viewpoint.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

import ifcopenshell
import ifcopenshell.util.element as uel
import ifcopenshell.util.placement as upl
import ifcopenshell.util.unit as uunit

from estate import env
from estate.flats.check import BATHS, BEDROOMS
from estate.flats.template import ROOM_CODES
from estate.validate.ifcqa import Hits, Report, canonical_flat_type, flat_pset, rules_config, summarize, where

HABITABLE = {c for c, (_, cat) in ROOM_CODES.items() if cat == "habitable"}
LONG_NAMES = {v[0].lower(): c for c, v in ROOM_CODES.items()}
PROBE = 0.6            # m across the host wall on either side of a window / door centre
OPEN_TOUCH = 0.03      # rooms closer than this share an open boundary (walls are >= 0.1 m thick)
OPEN_MIN_LEN = 0.5     # m of shared boundary that makes two rooms open to each other
OUTSIDE = "outside"    # Navigation side that is not a space (writer.space_name)


# ----------------------------------------------------------------------------- model read-back
@dataclass(eq=False)
class Space:
    el: object
    name: str
    code: str | None
    category: str | None
    area: float               # Qto NetFloorArea (footprint area when missing)
    poly: object              # world footprint
    z0: float
    z1: float
    flat: object = None       # Flat or None (common space)

    def __repr__(self):
        return self.name


@dataclass(eq=False)
class Flat:
    zone: object
    name: str
    label: str | None
    ftype: str | None
    ifa: float | None
    ifa_source: str
    mech_vent: set
    rooms: dict = field(default_factory=dict)       # code -> [Space]

    def room(self, code):
        v = self.rooms.get(code)
        return v[0] if v else None


@dataclass(eq=False)
class Opening:
    el: object
    kind: str                  # window | door
    name: str
    width: float
    height: float
    sides: list                # Space | OUTSIDE | None, one per side
    door_kind: str | None = None
    declared: tuple | None = None    # Navigation (FromSpace, ToSpace) as resolved sides
    probed: list | None = None


def _profile_xy(p):
    if p.is_a("IfcArbitraryClosedProfileDef"):
        c = p.OuterCurve
        if c.is_a("IfcPolyline"):
            pts = [q.Coordinates[:2] for q in c.Points]
        elif c.is_a("IfcIndexedPolyCurve"):
            pts = [xy[:2] for xy in c.Points.CoordList]
        else:
            return None
    elif p.is_a("IfcRectangleProfileDef"):
        hx, hy = p.XDim / 2, p.YDim / 2
        pts = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    else:
        return None
    xy = np.asarray(pts, float)
    pos = getattr(p, "Position", None)
    if pos is not None:
        M = upl.get_axis2placement(pos)
        xy = xy @ M[:2, :2].T + M[:2, 3]
    return xy


def _body(el):
    reps = el.Representation.Representations if el.Representation else ()
    return next((r for r in reps if r.RepresentationIdentifier == "Body"), reps[0] if reps else None)


def solid_footprint(el, scale=1.0):
    """(polygon, z0, z1) in world coordinates from vertical extrusions, or None when the body is anything else."""
    rep = _body(el)
    if rep is None or not rep.Items:
        return None
    M = upl.get_local_placement(el.ObjectPlacement) if el.ObjectPlacement else np.eye(4)
    polys, zs = [], []
    for it in rep.Items:
        if not it.is_a("IfcExtrudedAreaSolid"):
            return None
        xy = _profile_xy(it.SweptArea)
        if xy is None or len(xy) < 3:
            return None
        holes = []
        if it.SweptArea.is_a("IfcArbitraryProfileDefWithVoids"):
            for c in it.SweptArea.InnerCurves:
                if c.is_a("IfcPolyline"):
                    holes.append(np.asarray([q.Coordinates[:2] for q in c.Points], float))
                elif c.is_a("IfcIndexedPolyCurve"):
                    holes.append(np.asarray([xy_[:2] for xy_ in c.Points.CoordList], float))
        P = M @ (upl.get_axis2placement(it.Position) if it.Position else np.eye(4))
        d = P[:3, :3] @ np.asarray(it.ExtrudedDirection.DirectionRatios, float)
        d /= np.linalg.norm(d)
        if abs(d[2]) < 0.999 or abs(P[2, 2]) < 0.999:
            return None
        w = np.c_[xy, np.zeros(len(xy)), np.ones(len(xy))] @ P.T
        hw = [(np.c_[h, np.zeros(len(h)), np.ones(len(h))] @ P.T)[:, :2] * scale for h in holes if len(h) >= 3]
        polys.append(shapely.make_valid(Polygon(w[:, :2] * scale, hw)))
        z0 = float(P[2, 3]) * scale
        zs += [z0, z0 + float(d[2] * it.Depth) * scale]
    return shapely.union_all(polys), min(zs), max(zs)


def mesh_footprint(f, el):
    from estate.export import meshcache
    from estate.validate import geometry as geom
    mesh = meshcache.tessellate(f, num_threads=1, include=[el]).get(el.GlobalId)
    if mesh is None:
        return None
    e = geom.El(el.GlobalId, el, mesh, None)
    return e.footprint, e.zmin, e.zmax


def _room_code(sp, pset):
    code = pset.get("RoomCode")
    if code:
        return code
    tail = (sp.Name or "").rsplit(" ", 1)[-1]
    if tail in ROOM_CODES:
        return tail
    return LONG_NAMES.get((sp.LongName or "").strip().lower())


def _mech_vent(variant):
    if isinstance(variant, str):
        try:
            variant = json.loads(variant)
        except ValueError:
            return set()
    v = (variant or {}).get("mechanical_vent", []) if isinstance(variant, dict) else []
    return {v} if isinstance(v, str) else set(v)


class Index:
    """Flats, spaces, windows and doors of one file with the spaces each window / door serves."""

    def __init__(self, f):
        self.f = f
        self.scale = uunit.calculate_unit_scale(f)
        self.area_scale = uunit.calculate_unit_scale(f, "AREAUNIT")
        self.flats, self.spaces, self.by_name = [], [], {}
        zone_of = {}
        for z in sorted(f.by_type("IfcZone"), key=lambda z: (z.Name or "", z.id())):
            p = flat_pset(z)
            if not p:
                continue
            label = p.get("FlatType")
            ifa, src = p.get("InternalFloorArea"), "InternalFloorArea"
            if not isinstance(ifa, (int, float)):
                ifa, src = p.get("GrossArea"), "GrossArea"
            fl = Flat(z, z.Name or f"#{z.id()}", label, canonical_flat_type(label),
                      float(ifa) if isinstance(ifa, (int, float)) else None, src, _mech_vent(p.get("Variant")))
            self.flats.append(fl)
            for rel in z.IsGroupedBy or ():
                for o in rel.RelatedObjects:
                    if o.is_a("IfcSpace"):
                        zone_of.setdefault(o.id(), fl)
        for sp in sorted(f.by_type("IfcSpace"), key=lambda s: (s.Name or "", s.id())):
            fp = solid_footprint(sp, self.scale) if sp.Representation else None
            if fp is None and sp.Representation:
                fp = mesh_footprint(f, sp)
            if fp is None or fp[0].is_empty:        # kept for the room list; reported by space_geometry
                fp = (Polygon(), float("nan"), float("nan"))
            poly, z0, z1 = fp
            rp = uel.get_pset(sp, "SampleCity_Room") or {}
            q = uel.get_pset(sp, "Qto_SpaceBaseQuantities") or {}
            area = q.get("NetFloorArea")
            area = float(area) * self.area_scale if isinstance(area, (int, float)) else float(poly.area)
            code = _room_code(sp, rp)
            cat = ROOM_CODES[code][1] if code in ROOM_CODES else rp.get("Category")
            s = Space(sp, sp.Name or f"#{sp.id()}", code, cat, area, poly, z0, z1, zone_of.get(sp.id()))
            self.spaces.append(s)
            self.by_name.setdefault(s.name, s)
            if s.flat is not None and code:
                s.flat.rooms.setdefault(code, []).append(s)
        self.tree = shapely.STRtree([s.poly for s in self.spaces])
        self.windows = [self._opening(w, "window") for w in sorted(f.by_type("IfcWindow"), key=lambda e: (e.Name or "", e.id()))]
        self.doors = [self._opening(d, "door") for d in sorted(f.by_type("IfcDoor"), key=lambda e: (e.Name or "", e.id()))]
        self.served = {}               # Space -> [Opening]
        self.flat_doors = {}           # Flat -> [door Opening] with a side in the flat
        for o in self.windows + self.doors:
            for s in dict.fromkeys(x for x in o.sides if isinstance(x, Space)):
                self.served.setdefault(s, []).append(o)
            if o.kind == "door":
                for fl in dict.fromkeys(x.flat for x in o.sides if isinstance(x, Space) and x.flat is not None):
                    self.flat_doors.setdefault(fl, []).append(o)

    # -------------------------------------------------------------- probing
    def probe(self, el, width, height):
        """The space on each side of a window / door leaf, probing across the wall from the leaf centre."""
        if el.ObjectPlacement is None:
            return [None, None]
        M = upl.get_local_placement(el.ObjectPlacement)
        o, ex, ey = M[:3, 3] * self.scale, M[:3, 0], M[:3, 1]
        m = o + ex * width / 2
        z = m[2] + min(max(height / 2, 0.3), 1.0)
        out = []
        for sgn in (1.0, -1.0):
            seg = LineString([m[:2], m[:2] + sgn * ey[:2] * PROBE])
            best = None
            for k in self.tree.query(seg, predicate="intersects"):
                s = self.spaces[k]
                if not (s.z0 - 0.05 <= z <= s.z1 + 0.05):
                    continue
                part = seg.intersection(s.poly)
                if part.length < 0.02:          # only touches the probe (a leaf set flush with this face)
                    continue
                d = part.distance(Point(m[:2]))
                if best is None or d < best[0]:
                    best = (d, s)
            out.append(best[1] if best else None)
        return out

    def _opening(self, el, kind):
        w = float(el.OverallWidth or 0.0) * self.scale
        h = float(el.OverallHeight or 0.0) * self.scale
        nav = uel.get_pset(el, "Navigation") or {}
        probed = self.probe(el, w, h)
        op = Opening(el, kind, el.Name or f"#{el.id()}", w, h, probed, nav.get("DoorKind"), None, probed)
        if kind == "door":
            if isinstance(nav.get("NominalWidth"), (int, float)):
                op.width = float(nav["NominalWidth"]) * self.scale
            names = (nav.get("FromSpace"), nav.get("ToSpace"))
            if all(isinstance(n, str) and n for n in names):
                sides = [OUTSIDE if n == OUTSIDE else self.by_name.get(n) for n in names]
                if all(s is not None for s in sides):
                    op.declared, op.sides = tuple(sides), sides
        return op

    # -------------------------------------------------------------- flats
    def adjacency(self, fl):
        """Graph of a flat's rooms: door edges (kind) and open boundaries; '@entry' for doors to common spaces."""
        G = nx.Graph()
        rooms = [s for v in fl.rooms.values() for s in v if not s.poly.is_empty]
        G.add_nodes_from(rooms)
        for d in self.flat_doors.get(fl, ()):
            a, b = d.sides
            ina = isinstance(a, Space) and a.flat is fl
            inb = isinstance(b, Space) and b.flat is fl
            if ina and inb:
                if a is not b:
                    G.add_edge(a, b, kind=d.door_kind or "door")
            elif ina or inb:
                other = b if ina else a
                if other is OUTSIDE or (isinstance(other, Space) and other.flat is None):
                    G.add_edge("@entry", a if ina else b, kind=d.door_kind or "door")
        for i, a in enumerate(rooms):
            near = a.poly.buffer(OPEN_TOUCH)
            for b in rooms[i + 1:]:
                if G.has_edge(a, b) or abs(a.z0 - b.z0) > 0.5 or not near.intersects(b.poly):
                    continue
                if near.intersection(b.poly.buffer(OPEN_TOUCH)).area >= 2 * OPEN_TOUCH * OPEN_MIN_LEN:
                    G.add_edge(a, b, kind="open")
        return G


# ----------------------------------------------------------------------------- checks
def _fmt(s, msg):
    return f"{s.name}: {msg}"


def _at(s):
    """A world point inside a space at floor level (its placement when it has no footprint) for a viewpoint."""
    if s.poly.is_empty or not np.isfinite(s.z0):
        return where(s.el)
    p = s.poly.representative_point()
    return (p.x, p.y, s.z0)


def _flat_at(fl):
    """A point in the first room of a flat that has a footprint, else the zone's first placed space."""
    return next((_at(s) for v in fl.rooms.values() for s in v if not s.poly.is_empty), None) or where(fl.zone)


def _door_at(X, d):
    """The centre of a door / window leaf (its placement origin is at a jamb), halfway up to 1 m."""
    if d.el.ObjectPlacement is None:
        return None
    M = upl.get_local_placement(d.el.ObjectPlacement)
    m = M[:3, 3] * X.scale + M[:3, 0] * d.width / 2
    return (m[0], m[1], m[2] + min(max(d.height / 2, 0.3), 1.0))


def _room(hits, s, msg):
    """Record a failure of one room: the example text '<space>: <msg>', the space's GlobalId and a point in it."""
    return hits.add(_fmt(s, msg), s.el, xyz=_at(s))


def check(f, rules=None, index=None) -> list:
    """All programme checks of one opened file; returns the list of result dicts."""
    R = rules_config() if rules is None else rules
    X = index or Index(f)
    rep = Report()
    req_rules = R.get("flat", {}).get("required", {})
    min_area = R.get("room", {}).get("min_area", {})
    min_width = R.get("room", {}).get("min_width", {})
    living_min = R.get("flat", {}).get("living_min", {})
    bands = R.get("shelter", {}).get("bands", [])
    gz = R.get("glazing", {})
    ratio_min, vent_min = float(gz.get("habitable_ratio", 0.10)), float(gz.get("vent_min_area", 0.20))
    doors = R.get("doors", {})

    unknown, missing, small, narrow, living, shelter, nogeom = (Hits() for _ in range(7))
    glazing, vent, kitchen, main, reach, hs_access, ensuite, door_w = (Hits() for _ in range(8))
    worst_ratio, flats_checked = None, 0
    for fl in X.flats:
        if fl.ftype is None:
            unknown.add(f"{fl.name}: flat type {fl.label!r}", fl.zone, xyz=_flat_at(fl))
            continue
        flats_checked += 1
        codes = set(fl.rooms)
        for req in req_rules.get(fl.ftype, []):
            if not codes & set(req.split("|")):
                missing.add(f"{fl.name}: {fl.ftype} needs {req}", fl.zone, xyz=_flat_at(fl))
        G = X.adjacency(fl)
        glaze = {}
        for code, sps in sorted(fl.rooms.items()):
            for s in sps:
                amin = min_area.get(code)
                if amin and s.area < amin - 1e-6:
                    _room(small, s, f"{s.area:.2f} m2 < {amin}")
                glaze[s] = sum(o.width * o.height for o in X.served.get(s, ()) if o.kind == "window")
                if s.poly.is_empty:
                    _room(nogeom, s, "no footprint (no Body representation that can be read or tessellated)")
                    continue
                wmin = min_width.get(code)
                if wmin and s.poly.buffer(-(wmin / 2 - 0.01), join_style="mitre").is_empty:
                    _room(narrow, s, f"narrower than {wmin} m")
        liv = fl.room("LD") or fl.room("LDK")
        lmin = living_min.get(fl.ftype)
        if liv is not None and lmin and liv.area < lmin - 1e-6:
            _room(living, liv, f"{liv.area:.1f} m2 < {lmin}")
        hs = fl.room("HS")
        if hs is not None and bands and fl.ifa is not None:
            need = next((a for m, a in bands if fl.ifa <= m), bands[-1][1])
            if hs.area < need - 1e-6:
                _room(shelter, hs, f"{hs.area:.2f} m2 < {need} m2 for IFA {fl.ifa:.1f} m2 ({fl.ifa_source})")
        for code, sps in sorted(fl.rooms.items()):
            for s in sps:
                if s.poly.is_empty:
                    continue
                if code in HABITABLE and s.area > 0:
                    r = glaze[s] / s.area
                    worst_ratio = r if worst_ratio is None else min(worst_ratio, r)
                    if r < ratio_min - 1e-6:
                        _room(glazing, s, f"glazing {glaze[s]:.2f} m2 = {r * 100:.1f}% of {s.area:.2f} m2")
                if code in BATHS and glaze[s] < vent_min and code not in fl.mech_vent:
                    _room(vent, s, f"vent windows {glaze[s]:.2f} m2 < {vent_min} m2 and no mechanical_vent")
                if code == "K" and glaze[s] <= 0:
                    nb = {n.code for n in G.neighbors(s) if isinstance(n, Space)} if s in G else set()
                    if not nb & {"SY", "LD", "LDK"}:
                        _room(kitchen, s, "no window, service yard or opening to the living room")
        # entrance, reachability, door graph
        mains = [d for d in X.flat_doors.get(fl, ()) if d.door_kind == "main"]
        ok_main = [d for d in mains if any(x is OUTSIDE or (isinstance(x, Space) and x.flat is None) for x in d.sides)]
        if not ok_main:
            main.add(f"{fl.name}: " + (f"main door '{mains[0].name}' does not open to a common space "
                                       f"({' / '.join(str(x) for x in mains[0].sides)})" if mains else "no main door"),
                     fl.zone, *(m.el for m in mains[:1]), xyz=_door_at(X, mains[0]) if mains else _flat_at(fl))
        if "@entry" in G:
            comp = nx.node_connected_component(G, "@entry")
            for s in sorted((n for n in G if n != "@entry" and n not in comp), key=lambda s: s.name):
                _room(reach, s, "not reachable from the entrance")
        hs_nb = {n.code for n in G.neighbors(hs) if isinstance(n, Space)} if hs in G else set()
        if hs_nb & (BEDROOMS | BATHS):
            _room(hs_access, hs, f"entered from {sorted(hs_nb & (BEDROOMS | BATHS))}")
        mba = fl.room("MBa")
        if mba is not None and mba in G:
            other = {n.code if isinstance(n, Space) else "common space" for n in G.neighbors(mba)} - {"MB"}
            if other:
                _room(ensuite, mba, f"also opens to {sorted(other, key=str)}")
        for d in X.flat_doors.get(fl, ()):
            if d.door_kind == "lift":
                continue
            mn = float(doors.get("main_min", 0.9) if d.door_kind == "main" else doors.get("internal_min", 0.8))
            if d.width < mn - 1e-6:
                door_w.add(f"{d.name}: {d.width:.2f} m < {mn}", d.el, xyz=_door_at(X, d))

    # declared door spaces against the probe; quantities against the drawn footprints
    mism = Hits()
    for d in X.doors:
        if d.declared and all(isinstance(x, Space) for x in d.probed):
            if {x.name for x in d.declared if isinstance(x, Space)} != {x.name for x in d.probed}:
                mism.add(f"{d.name}: Navigation {d.declared[0]} / {d.declared[1]}, "
                         f"geometry {d.probed[0]} / {d.probed[1]}",
                         d.el, *(x.el for x in (*d.declared, *d.probed) if isinstance(x, Space)), xyz=_door_at(X, d))
    qty = Hits()
    for s in X.spaces:
        if s.flat is not None and not s.poly.is_empty and abs(s.area - s.poly.area) > max(0.05, 0.01 * s.poly.area):
            _room(qty, s, f"NetFloorArea {s.area:.2f} m2, footprint {s.poly.area:.2f} m2")

    note_rules = "config/rules.toml"
    rep.add("flat_type", "error", unknown, value=flats_checked,
            note="flat zones whose FlatType is not an HDB type (legacy labels such as '4-Room' are accepted)")
    rep.add("required_rooms", "error", missing, note=f"{note_rules} [flat.required], '|' = any one of")
    rep.add("room_area", "error", small, note=f"Qto NetFloorArea >= {note_rules} [room.min_area]")
    rep.add("room_width", "error", narrow, note=f"space footprint clear width >= {note_rules} [room.min_width]")
    rep.add("space_geometry", "error", nogeom, note="flat rooms with a footprint the checks can measure")
    rep.add("living_area", "error", living, note=f"{note_rules} [flat.living_min]")
    rep.add("shelter_area", "error", shelter, value=bands, note="household shelter area by flat IFA, [shelter].bands")
    rep.add("glazing", "error", glazing, value={"min_ratio": round(worst_ratio, 4) if worst_ratio is not None else None},
            note=f"windows of habitable rooms >= {ratio_min:.0%} of the net floor area")
    rep.add("bath_vent", "error", vent,
            note=f"bathroom vent window >= {vent_min} m2 unless SampleCity_Flat.Variant declares mechanical_vent")
    rep.add("kitchen_vent", "error", kitchen, note="kitchen window, or a door / opening to the service yard or living room")
    rep.add("door_width", "error", door_w, note=f"flat doors: main >= {doors.get('main_min')}, others >= {doors.get('internal_min')} m")
    rep.add("main_door", "error", main, note="every flat has a main door opening to a common space")
    rep.add("reachable", "error", reach, note="every room reachable from the entrance through doors and open boundaries")
    rep.add("shelter_access", "error", hs_access, note="household shelter not entered from a bedroom or bathroom")
    rep.add("ensuite_access", "error", ensuite, note="master bathroom entered only from the master bedroom")
    rep.add("door_spaces", "warn", mism, note="Navigation.FromSpace / ToSpace match the spaces on either side of the door")
    rep.add("space_quantities", "warn", qty, note="flat room NetFloorArea within 1 % (0.05 m2) of the footprint area")
    return rep.results


def check_file(path, f=None, rules=None) -> dict:
    t = time.time()
    f = f if f is not None else ifcopenshell.open(str(path))
    res = check(f, rules)
    return {"file": env.rel(path), "runtime_s": round(time.time() - t, 2), "summary": summarize(res), "results": res}
