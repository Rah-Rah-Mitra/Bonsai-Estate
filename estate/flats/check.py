"""Programme checks for flats (plan level, before any IFC is written) and the flat-in-a-box test harness.

``check_flat`` works on a derived plate, so the same checks run on catalogue harnesses and on real blocks:
required rooms, net areas and clear widths, IFA band, household shelter size, door graph (every room reachable
from the main door; shelter not entered through a bedroom or bathroom; master bath only from the master
bedroom), glazing of habitable rooms, bathroom ventilation, door widths and any wall/opening fit errors.
"""
from __future__ import annotations

import itertools
import tomllib
from dataclasses import dataclass
from functools import lru_cache

import networkx as nx
import math

import numpy as np
import shapely
from shapely.geometry import box
from shapely.ops import unary_union

from estate.blocks.plate import CORRIDOR, LIFT, LOBBY, ROOM, STAIR, Face, Plate
from estate.env import CONFIG
from estate.flats.template import ROOM_CODES, Affine2, Template, category, place_flat

HABITABLE = {c for c, (_, cat) in ROOM_CODES.items() if cat == "habitable"}
BATHS = {"MBa", "CB", "BA", "B4a", "B2a"}
BEDROOMS = {"MB", "BED", "B2", "B3", "B4"}


def fi_face_id(stack, code):
    return f"U{stack}:{code}"


# ----------------------------------------------------------------------------- door swings
# Each hinged door sweeps a quarter disc in the room it opens into. Two sweeps must not overlap (both doors could
# not be opened at once), and the open household shelter door must leave a 0.8 m passage between the door
# approaches it previously connected. Sliding, rolling and lift doors are skipped.
PASSAGE = 0.8


def door_sweeps(dp, stack: str) -> list[dict]:
    """Swing sectors of the hinged doors of one flat (door openings of the derived plate that touch its rooms)."""
    from estate.rules import DOOR_KINDS
    runs = {r.idx: r for r in dp.runs}
    own = f"U{stack}:"
    out = []
    for o in dp.openings:
        if o.kind != "door":
            continue
        op = DOOR_KINDS.get(o.door_kind, ("",))[0]
        if not op.startswith("SINGLE_SWING") and op != "DOUBLE_DOOR_SINGLE_SWING":
            continue
        r = runs[o.run]
        sm = (o.s0 + o.s1) / 2
        parts = r.meta.get("parts") or [(0.0, r.length, r.left, r.right)]
        lf, rt = next(((a, b) for a0, a1, a, b in parts if a0 - 1e-6 <= sm <= a1 + 1e-6), (r.left, r.right))
        into, other = (lf, rt) if o.swing == 1 else (rt, lf)
        if not (into.startswith(own) or other.startswith(own)):
            continue
        u = np.asarray(r.u)
        n = np.array([-u[1], u[0]]) * o.swing
        a, b = np.asarray(r.p0) + u * o.s0, np.asarray(r.p0) + u * o.s1
        w = o.s1 - o.s0
        hinge, closed = (a, u) if o.hinge == "start" else (b, -u)
        a0 = math.atan2(closed[1], closed[0])
        da = (math.atan2(n[1], n[0]) - a0 + math.pi) % (2 * math.pi) - math.pi
        arc = [tuple(hinge + w * np.array([math.cos(a0 + da * t), math.sin(a0 + da * t)])) for t in np.linspace(0, 1, 13)]
        out.append(dict(name=o.name, kind=o.door_kind, into=into, other=other, mid=(a + b) / 2, n=n,
                        sector=shapely.Polygon([tuple(hinge)] + arc),
                        leaf=shapely.LineString([tuple(hinge), tuple(hinge + n * w)]).buffer(0.025, cap_style="flat")))
    return out


def swing_issues(dp, stack: str) -> list[tuple[str, str, str]]:
    """(code, room code, message) for overlapping door swings and a shelter door that blocks the passage."""
    sw = door_sweeps(dp, stack)
    out = []
    for i in range(len(sw)):
        for j in range(i + 1, len(sw)):
            ar = sw[i]["sector"].intersection(sw[j]["sector"]).area
            if ar > 0.005:
                out.append(("door-swing", sw[i]["into"].split(":")[-1],
                            f"'{sw[i]['name']}' and '{sw[j]['name']}' swing into each other ({ar:.2f} m²)"))
    group = {}
    for fid in dp.plate.faces:
        group[fid] = {fid}
    for pr in dp.plate.open_pairs:
        x, y = tuple(pr)
        g = group[x] | group[y]
        for f in g:
            group[f] = g
    for s in sw:
        if s["kind"] != "shelter" or s["into"] not in group:
            continue
        g = group[s["into"]]
        region = unary_union([dp.spaces[f] for f in g if dp.spaces.get(f) is not None])
        pts = [(t["name"], shapely.Point(t["mid"] + t["n"] * (0.45 if t["into"] in g else -0.45)))
               for t in sw if t is not s and (t["into"] in g or t["other"] in g)]

        def comp(geom, p):
            parts = [(gg.distance(p), k) for k, gg in enumerate(getattr(geom, "geoms", [geom])) if not gg.is_empty]
            d, k = min(parts, default=(9.0, None))
            return k if d < 0.6 else None
        before = region.buffer(-PASSAGE / 2 + 0.01, join_style="mitre")
        after = region.difference(s["leaf"]).buffer(-PASSAGE / 2 + 0.01, join_style="mitre")
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                (na, pa), (nb, pb) = pts[i], pts[j]
                if comp(before, pa) is not None and comp(before, pa) == comp(before, pb) and \
                        (comp(after, pa) is None or comp(after, pa) != comp(after, pb)):
                    out.append(("shelter-door-blocks", "HS", f"the open '{s['name']}' cuts '{na}' off '{nb}'"))
    return out


@lru_cache(maxsize=1)
def rules() -> dict:
    return tomllib.loads((CONFIG / "rules.toml").read_text(encoding="utf-8"))


@dataclass
class Issue:
    level: str          # error | warn
    code: str
    msg: str
    flat: str = ""
    room: str = ""

    def __str__(self):
        return f"{self.level.upper():5} {self.code:<14} {self.flat} {self.room} {self.msg}".replace("  ", " ")


# ----------------------------------------------------------------------------- harness
def harness(lf, fit: str, case: str) -> Plate:
    """Place one realized flat in a minimal context. fit: PT | SL | LC | free; case: N/S (PT), mid/end (SL)."""
    W, D = lf.width, lf.depth
    p = Plate()
    if fit == "PT":
        C = 2.8
        if case == "N":
            xf = Affine2(1, 0, 0, 1, C, 1.5)
            p.add(Face("LIFT1", box(0, 1.5, C, 4.0), LIFT, name="Lift"))
        else:
            xf = Affine2(1, 0, 0, -1, C, -1.5)
            p.add(Face("STAIR1", box(0, -7.1, C, -1.5), STAIR, name="Stair"))
        p.add(Face("LOBBY", box(0, -1.5, C + W + 1.0, 1.5), LOBBY, name="Lift lobby"))
        entry = "LOBBY"
    else:
        xf = Affine2(1, 0, 0, 1, 0.0, 0.0)
        p.add(Face("CORR", box(-1.0, -2.0, W + 1.0, 0.0), CORRIDOR, name="Corridor"))
        entry = "CORR"
        if fit == "SL":
            p.add(Face("NB:L", box(-6.0, 0.0, 0.0, D), ROOM, flat="NB1", room="LD", meta={"category": "habitable"}))
            if case == "mid":
                p.add(Face("NB:R", box(W, 0.0, W + 6.0, D), ROOM, flat="NB2", room="LD", meta={"category": "habitable"}))
    place_flat(p, lf, "T", xf, entry)
    return p


def harness_cases(t: Template):
    fits = t.fits or ["free"]
    for fit in fits:
        if fit == "PT":
            yield fit, "N"
            yield fit, "S"
        elif fit == "SL":
            yield fit, "mid"
            yield fit, "end"
        elif fit == "LC":
            yield "SL", "end"
        else:
            yield fit, "free"


# ----------------------------------------------------------------------------- checks
def check_flat(dp, stack: str, known: set | None = None) -> list[Issue]:
    R = rules()
    plate = dp.plate
    fi = plate.flats[stack]
    ftype = fi.flat_type
    issues: list[Issue] = []

    def add(level, code, msg, room=""):
        if known and code in known:
            level = "warn"
        issues.append(Issue(level, code, msg, stack, room))

    faces = {f.room: f for f in plate.faces.values() if f.flat == stack and f.kind in (ROOM, "balcony")}
    net = {code: dp.spaces.get(f.id) for code, f in faces.items()}
    # required rooms
    for req in R["flat"]["required"].get(ftype, []):
        if not any(c in faces for c in req.split("|")):
            add("error", "missing-room", f"{ftype} needs {req}")
    # areas and widths
    for code, g in net.items():
        if g is None or g.is_empty:
            add("error", "no-space", "room has no net area", code)
            continue
        amin = R["room"]["min_area"].get(code)
        if amin and g.area < amin - 1e-6:
            add("error", "room-area", f"{g.area:.2f} m² < {amin}", code)
        wmin = R["room"]["min_width"].get(code)
        if wmin and g.buffer(-(wmin / 2 - 0.01), join_style="mitre").is_empty:
            add("error", "room-width", f"narrower than {wmin} m", code)
    liv = net.get("LD") or net.get("LDK")
    lmin = R["flat"]["living_min"].get(ftype)
    if liv is not None and lmin and liv.area < lmin:
        add("error", "living-area", f"{liv.area:.1f} m² < {lmin}", "LD")
    # IFA band
    from estate.flats.template import flat_ifa
    ifa = flat_ifa(fi, plate)
    lo, hi = R["flat"]["ifa_band"][ftype]
    if not lo <= ifa <= hi:
        add("error", "ifa-band", f"IFA {ifa:.1f} m² outside {lo}-{hi}")
    # household shelter
    hs = net.get("HS")
    if hs is not None:
        need = next(a for m, a in R["shelter"]["bands"] if ifa <= m)
        if hs.area < need - 1e-6:
            add("error", "shelter-area", f"{hs.area:.2f} m² < {need} m² for IFA {ifa:.0f} m²", "HS")
    # door graph
    G = nx.Graph()
    ids = {f.id: code for code, f in faces.items()}
    for code in faces:
        G.add_node(code)
    entry_nodes = set()
    for d in plate.doors:
        a, b = ids.get(d.a), ids.get(d.b)
        if a and b:
            G.add_edge(a, b, kind=d.kind)
        elif a or b:
            entry_nodes.add(a or b)
            G.add_edge("@entry", a or b, kind=d.kind)
    for pr in plate.open_pairs:
        x, y = tuple(pr)
        if x in ids and y in ids:
            G.add_edge(ids[x], ids[y], kind="open")
    if "@entry" not in G:
        add("error", "no-entry", "no door to the lobby / corridor")
    else:
        reach = nx.node_connected_component(G, "@entry")
        for code in faces:
            if code not in reach:
                add("error", "unreachable", "not reachable from the main door", code)
        if "HS" in G:
            via = {n for n in G.neighbors("HS")}
            if via & (BEDROOMS | BATHS):
                add("error", "shelter-access", f"shelter entered from {sorted(via & (BEDROOMS | BATHS))}", "HS")
        if "MBa" in G and set(G.neighbors("MBa")) - {"MB"}:
            add("error", "ensuite-access", f"master bath also opens to {sorted(set(G.neighbors('MBa')) - {'MB'})}", "MBa")
    # doors
    for d in plate.doors:
        if ids.get(d.a) or ids.get(d.b):
            mn = R["doors"]["main_min"] if d.kind == "main" else R["doors"]["internal_min"]
            if d.kind not in ("lift",) and d.width < mn - 1e-6:
                add("error", "door-width", f"'{d.name}' {d.width} m < {mn}")
    for code, room, msg in swing_issues(dp, stack):
        add("error", code, msg, room)
    mb_low = [o for o in dp.openings if o.kind == "window" and ids.get(o.meta.get("face")) == "MB" and o.sill <= 1.0]
    if "MB" in faces and not mb_low:
        add("warn", "mb-outlook", "master bedroom has no window with a sill at or below 1.0 m", "MB")
    # household shelter blast door opens outwards (into the flat), service yards face outside, not the corridor
    for d in plate.doors:
        if d.kind == "shelter" and ids.get(d.into) == "HS":
            add("error", "shelter-swing", f"'{d.name}' swings into the shelter (blast doors open outwards)", "HS")
    for o in dp.openings:
        if o.kind == "window" and o.meta.get("kind") == "opening" and ids.get(o.meta.get("face")) == "SY":
            r = next(x for x in dp.runs if x.idx == o.run)
            parts = r.meta.get("parts") or [(0.0, r.length, r.left, r.right)]
            mid = (o.s0 + o.s1) / 2
            pair = next(((lf, rt) for a0, a1, lf, rt in parts if a0 - 1e-6 <= mid <= a1 + 1e-6), (r.left, r.right))
            other = pair[1] if pair[0] == fi_face_id(stack, "SY") else pair[0]
            if plate.kind(other) in (CORRIDOR, LOBBY):
                add("error", "yard-on-corridor", "the service yard opens onto the common corridor / lobby", "SY")
    # glazing and ventilation
    glaze = {}
    for o in dp.openings:
        if o.kind == "window":
            code = ids.get(o.meta.get("face"))
            if code:
                glaze[code] = glaze.get(code, 0.0) + (o.s1 - o.s0) * o.height
    mech = set(plate.flats[stack].variant.get("mechanical_vent", []))
    for code, g in net.items():
        if g is None or g.is_empty:
            continue
        if code in HABITABLE:
            ratio = glaze.get(code, 0.0) / g.area
            if ratio < R["glazing"]["habitable_ratio"] - 1e-6:
                add("error", "glazing", f"glazing {ratio * 100:.1f}% of floor area", code)
        if code in BATHS and glaze.get(code, 0.0) < R["glazing"]["vent_min_area"] and code not in mech:
            add("error", "bath-vent", "no vent window (declare mechanical_vent to accept)", code)
        if code == "K":
            nbrs = set(G.neighbors("K")) if "K" in G else set()
            if glaze.get("K", 0.0) <= 0 and not (nbrs & {"SY", "LD", "LDK"}):
                add("error", "kitchen-vent", "kitchen has no window, service yard or opening to living")
    # wall / opening errors from the derivation that mention this flat
    for e in dp.errors:
        if f"U{stack}:" in e or f"'{stack}" in e:
            add("error", "plan", e)
    for w in dp.dropped_windows:
        if f"U{stack}:" in w:
            issues.append(Issue("warn", "window-dropped", w, stack))
    return issues


def check_template(t: Template, stretch=True):
    """Check every variant combination of a template in each harness case. Returns list of result dicts."""
    from estate.blocks.builder import derive
    space = t.variant_space()
    results = []
    widths = [None]
    lims = t.variants.get("stretch", {})
    if stretch and lims.get("x"):
        lo = sum(a for a, b in lims["x"])
        hi = sum(b for a, b in lims["x"])
        widths = sorted({None, round(lo, 3), round(hi, 3)} - {round(t.width, 3)}, key=lambda v: (v is not None, v or 0))
    for mirror, kitchen, outdoor, width in itertools.product(space["mirror"], space["kitchen"], space["outdoor"], widths):
        v = {"mirror": mirror, "kitchen": kitchen, "outdoor": outdoor}
        if width:
            v["width"] = width
        if t.variants.get("mechanical_vent"):
            v["mechanical_vent"] = list(t.variants["mechanical_vent"])
        try:
            lf = t.realize(v)
        except Exception as e:  # noqa: BLE001
            results.append(dict(template=t.id, variant=v, case="-", issues=[Issue("error", "realize", str(e))], lf=None))
            continue
        for fit, case in harness_cases(t):
            plate = harness(lf, fit, case)
            dp = derive(plate, "typical")
            issues = check_flat(dp, "T", set(t.known_issues))
            results.append(dict(template=t.id, variant=v, case=f"{fit}-{case}", issues=issues, lf=lf, dp=dp))
    return results
