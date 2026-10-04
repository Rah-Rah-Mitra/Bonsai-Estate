"""Eye-level street cameras for the report's estate renders. Plain python (no bpy), so the build hooks can import
it and put the camera into the render stage key.

It lives in estate/report/, not estate/blender/, on purpose: the blend and render stages hash every
estate/blender/*.py and the ifc stage every estate/site/*.py (estate/pipeline/state.py STAGE_DEPS), so a camera
tweak there would rebuild all 16 .blend files and every render (or every IFC). Here it changes only the camera
dict in the ESTATE render key, and that re-renders ESTATE alone. The dict therefore carries everything of this
module that reaches the image: eye, target, lens, clip, shift, res and the sky colour.

``camera("BS1")`` stands a person at bus stop BS1 and looks into the estate. The stop comes from
model/SITE_graph.json (the ``bus_stop`` node at the shelter centre, its paving z) or, without a graph, is rebuilt
from the masterplan the way estate/site/roads.py lays the bay out. ``out`` is the unit normal of the stop's road
pointing from the carriageway towards the shelter, i.e. into the estate.

Eye: 1.6 m above the platform at the shelter's estate-side edge (the roof's rear edge, where the linkway lands),
moved CLEAR m further from the road so the rear posts and glass are behind the lens. Standing exactly there puts
a roof a metre over the eye (the shelter's, and the covered linkway's landing), which in a 24 mm frame blacks out
the top of the picture and cuts the tower tops off, so the eye then steps along the kerb, towards the target side
first, to the first spot with open sky: outside the shelter roof and every covered graph edge (linkways,
shelters, porches: their clear width plus the canopy overhang, square-ended), outside every building footprint,
inside the estate extent. For BS1 that is 3.5 m east of the shelter centre line, just past the linkway landing:
the avenue's kerb and tree-lined verge on the left, the neighbourhood centre on the right, the towers ahead.

Target: the nearest residential block in front of the stop (centroid in the ``out`` half-plane, nearest by
footprint distance), aimed at its footprint centroid. The camera stays level (target at eye height) so verticals
stay vertical, and a lens shift raises the frame instead, as an architectural shift lens does: every building
inside the horizontal field of view gets its top at or below TOP of the frame height (masterplan heights; MSCP
and NC record 0 and are low), with the horizon kept inside HORIZON. Everything is rounded so the dict is stable
inside the render stage key.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from estate import env

EYE = 1.6                  # eye height above the paving (m)
LENS = 24.0                # mm on the 36 mm sensor width: 74 degrees across the frame
CLIP = (0.1, 2000.0)
RES = (1600, 900)          # 16:9: an eye-level view is wide, not tall
CLEAR = 0.6                # eye clearance from roof edges, canopies and footprints (m)
CANOPY_OVERHANG = 0.3      # canopy edge beyond a covered edge's clear width, each side (canopy 3.0 m vs 2.4 m clear)
STEP = 0.5                 # along-kerb search step (m)
MAX_SLIDE = 40.0           # give up looking for open sky this far along the kerb
TOP = 0.88                 # highest building top in view sits at this fraction of the frame height ...
HORIZON = (0.15, 0.5)      # ... as long as the horizon stays between these fractions (no downward shift)
SENSOR = 36.0
SKY = (0.45, 0.62, 0.86)   # world colour behind an eye-level view (linear; sRGB 179 206 239 in the Standard transform)


def _load(obj, default: Path):
    """A JSON dict as given, read from a path, or from ``default`` (None when the file does not exist)."""
    if obj is None:
        obj = default
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return obj


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _unit(v):
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n)


def _r(v, nd=3):
    return [round(float(x), nd) + 0.0 for x in v]       # + 0.0: no -0.0 in the stage key


def stop_frame(stop_id: str, mp: dict, graph: dict | None = None) -> dict:
    """The stop's shelter centre ``node`` (xy), paving ``z``, unit ``out`` (away from the road), unit ``along``
    (the road direction) and the road record."""
    from estate.site.roads import BAY_DEPTH, SHELTER_FRONT, SHELTER_REAR
    bs = next((b for b in mp.get("bus_stops", []) if b.get("id") == stop_id), None)
    if bs is None:
        raise KeyError(f"no bus stop {stop_id!r} in the masterplan "
                       f"(stops: {[b.get('id') for b in mp.get('bus_stops', [])]})")
    road = next((r for r in mp.get("roads", []) if r["id"] == bs.get("road")), None)
    if road is None:
        raise KeyError(f"bus stop {stop_id!r}: no road {bs.get('road')!r} in the masterplan")
    p0, p1 = road["centreline"][0], road["centreline"][-1]
    u = _unit(_sub(p1, p0))
    n = (-u[1], u[0])                                     # estate.site.roads.Road.n
    node, z = None, 0.0
    for nd in (graph or {}).get("nodes", []):
        if nd.get("kind") == "bus_stop" and nd.get("ref") == stop_id:
            node, z = tuple(nd["xy"]), float(nd.get("z", 0.0))
            break
    if node is None:     # no graph: the shelter centre as roads.py places it behind the bay kerb
        rel = _sub(bs["at"], p0)
        s, side = _dot(rel, u), (1 if _dot(rel, n) >= 0 else -1)
        kd = side * (float(road["carriageway"]) / 2 + BAY_DEPTH)
        d = kd + side * (SHELTER_FRONT + SHELTER_REAR) / 2
        node = (p0[0] + u[0] * s + n[0] * d, p0[1] + u[1] * s + n[1] * d)
    side = 1 if _dot(_sub(node, p0), n) >= 0 else -1
    return dict(node=node, z=z, out=(n[0] * side, n[1] * side), along=u, road=road)


def _sites(mp):
    from shapely.geometry import shape
    return [(s, shape(s["footprint"])) for s in mp.get("sites", []) if s.get("footprint")]


def _canopies(graph, fr):
    """Roofs a person could stand under: the stop's shelter roof and every covered graph edge, as polygons."""
    from shapely.geometry import LineString, Polygon
    from estate.site.roads import SHELTER_FRONT, SHELTER_LEN, SHELTER_REAR
    (nx, ny), o, a = fr["node"], fr["out"], fr["along"]
    hl, hd = SHELTER_LEN / 2, (SHELTER_REAR - SHELTER_FRONT) / 2
    roof = Polygon([(nx + a[0] * sa + o[0] * so, ny + a[1] * sa + o[1] * so)
                    for sa, so in ((-hl, -hd), (hl, -hd), (hl, hd), (-hl, hd))])
    out = [roof]
    if graph:
        xy = {nd["id"]: tuple(nd["xy"]) for nd in graph.get("nodes", [])}
        for e in graph.get("edges", []):
            if e.get("covered") and e["a"] in xy and e["b"] in xy and xy[e["a"]] != xy[e["b"]]:
                half = float(e.get("width", 2.4)) / 2 + CANOPY_OVERHANG
                out.append(LineString([xy[e["a"]], xy[e["b"]]]).buffer(half, cap_style="square", join_style="mitre"))
    return out


def _pick_block(sites, stub, out):
    """The nearest residential block whose centroid lies on the estate side of the stop (None if there is none)."""
    from shapely.geometry import Point
    P = Point(stub)
    best = None
    for s, poly in sites:
        if s.get("kind") != "block":
            continue
        c = poly.centroid
        if _dot(_sub((c.x, c.y), stub), out) <= 0:
            continue
        d = poly.distance(P)
        if best is None or (d, s["id"]) < best[0]:
            best = ((d, s["id"]), s, poly)
    return best


def _shift_y(eye, heading, sites, res, lens):
    """Lens shift (units of the frame width) that puts the highest building top in view at TOP of the frame."""
    t_h = SENSOR / 2 / lens                               # tan(half the horizontal field of view)
    t_v = t_h * res[1] / res[0]
    right = (heading[1], -heading[0])
    top = 0.0
    for s, poly in sites:
        h = float(s.get("height") or 0.0)
        if h <= 0:
            continue
        for v in poly.exterior.coords:
            rel = _sub(v, eye)
            depth = _dot(rel, heading)
            if depth > 1.0 and abs(_dot(rel, right)) / depth <= t_h:
                top = max(top, (h - EYE) / depth)         # buildings stand on the same paving as the eye
    c = top + t_v - 2 * TOP * t_v                         # frame centre (as a tan) that puts ``top`` at TOP
    lo, hi = HORIZON                                      # horizon (tan 0) at (t_v - c) / (2 t_v) of the height
    c = min(max(c, t_v * (1 - 2 * hi)), t_v * (1 - 2 * lo))
    return c / (2 * t_h)


def camera(stop_id: str = "BS1", masterplan=None, graph=None) -> dict:
    """Eye-level camera at bus stop ``stop_id``: {eye, target, lens, clip, shift, res, sky, stop, look_at}.

    ``masterplan`` / ``graph``: dicts, JSON paths, or None for model/masterplan.json and model/SITE_graph.json
    (the graph is optional; the masterplan is not).
    """
    from shapely.geometry import Point, box
    from estate.site.roads import SHELTER_FRONT, SHELTER_REAR
    mp = _load(masterplan, env.MODEL / "masterplan.json")
    if mp is None:
        raise FileNotFoundError("model/masterplan.json missing: run ./estate.sh plan")
    g = _load(graph, env.MODEL / "SITE_graph.json")
    fr = stop_frame(stop_id, mp, g)
    o, a, z = fr["out"], fr["along"], fr["z"]
    hd = (SHELTER_REAR - SHELTER_FRONT) / 2
    stub = (fr["node"][0] + o[0] * hd, fr["node"][1] + o[1] * hd)      # rear edge of the shelter roof
    sites = _sites(mp)
    pick = _pick_block(sites, stub, o)
    if pick is not None:
        c = pick[2].centroid
        aim, look_at = (c.x, c.y), pick[1]["id"]
    else:                                                 # no block in front: look straight into the estate
        aim, look_at = (stub[0] + o[0] * 100.0, stub[1] + o[1] * 100.0), None
    x0, y0, x1, y1 = mp["estate"]["extent"]
    inside = box(x0, y0, x1, y1).buffer(-1.0)
    blocked = _canopies(g, fr) + [poly for _, poly in sites]
    base = (stub[0] + o[0] * CLEAR, stub[1] + o[1] * CLEAR)
    first = 1 if _dot(_sub(aim, stub), a) >= 0 else -1
    eye = None
    for k in range(int(MAX_SLIDE / STEP) + 1):
        for sgn in ((first, -first) if k else (1,)):
            p = (base[0] + a[0] * sgn * k * STEP, base[1] + a[1] * sgn * k * STEP)
            P = Point(p)
            if inside.contains(P) and all(b.distance(P) >= CLEAR for b in blocked):
                eye = p
                break
        if eye is not None:
            break
    if eye is None:
        raise ValueError(f"no open-sky eye position within {MAX_SLIDE} m of bus stop {stop_id}")
    heading = _unit(_sub(aim, eye))
    eye3 = (eye[0], eye[1], z + EYE)
    sy = _shift_y(eye3, heading, sites, RES, LENS)
    return {"stop": stop_id, "look_at": look_at, "eye": _r(eye3), "target": _r((aim[0], aim[1], z + EYE)),
            "lens": LENS, "clip": list(CLIP), "shift": [0.0, round(sy, 4)], "res": list(RES), "sky": list(SKY)}
