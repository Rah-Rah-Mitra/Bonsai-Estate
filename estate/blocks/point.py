"""PT4 point block: four flats around a central core (generalises the legacy Blk 123).

Block-local frame, centred on the core: a 3.0 m lift lobby runs east-west (y in [-1.5, 1.5]); two lifts and a
refuse chute room sit north of it and two dog-leg stairs south of it in a 5.6 m wide core strip. Each quadrant
holds one flat from its own template (stacks 101 NW, 103 NE, 105 SE, 107 SW); template x runs away from the core,
template y away from the lobby, so the main door always opens onto the lobby and the flats only move if C does.
The lift strip is two 2.2 m shafts (2.0 m clear) and a 1.2 m refuse chute room at the east end with a service
door to the lobby; at the void deck the same enclosure is the bin centre under the chute (door to the void deck),
and it stops below the roof.

The void deck uses of the config are handled here for every typology (slab.py and lblock.py import them):
amenity uses become rooms under a flat envelope (on a point block the pinned quadrants first, then the free ones
in QUADRANT_ORDER), 'bin' is the bin centre under each refuse chute and 'letterbox' gives every flat a letterbox
(place_letterboxes). Anything else is an error rather than silently dropped.

Letterbox banks (IfcFurniture via plan.meta['furniture']) are built of 0.3 m columns of boxes; config/rules.toml
[void_deck] sets the boxes in a standard 2.4 m bank. A block needs one box per flat ((storeys - 1) x flats per
floor), shared out between its lift lobbies. A bank stands against a core or amenity room wall on the open void
deck near its lift lobby, with clear_front kept free in front of it, clear of door openings and void deck
columns and off the walk path (the lift lobby / corridor band). Where the wall is too short the bank runs on
along the void deck edge, with at least half of it still against the wall; next comes a bank on the open edge of
the void deck (between the columns), then a bank against a wall on the walk path (still with clear_front in front
of it), and only then a wall farther from the lift lobbies. A block whose capacity cannot be met raises.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from estate.blocks import plate_ops
from estate.blocks.builder import BuildingPlan, StairSpec, derive
from estate.blocks.plate import AMENITY, COMMON_KINDS, DECK, LIFT, LOBBY, REFUSE, STAIR, VOID, DoorSpec, Face, Plate
from estate.flats.template import Affine2, catalogue, derive_seed, place_flat
from estate.rules import DOOR_H

C = 2.8              # half width of the core strip (two 2.8 m stair enclosures)
LOBBY_HALF = 1.5     # half depth of the lift lobby
LIFT_W = 2.2         # lift shaft (centreline) width: 2.0 m clear, a 1.5 x 1.8 m car
LIFT_DEPTH = 2.5     # lift shaft (centreline) depth north of the lobby
REF_W = round(2 * C - 2 * LIFT_W, 4)   # refuse chute room / bin centre at the east end of the lift strip (1.0 m clear)
STAIR_DEPTH = 5.6    # stair enclosure (centreline) depth south of the lobby
REF_DOOR_W, BIN_DOOR_W = 0.8, 0.9      # the bin centre door takes all of the 1.2 m side (0.15 m to each corner wall)

QUADRANTS = {        # stack: Affine2 from template frame to block frame
    "101": Affine2(-1, 0, 0, 1, -C, LOBBY_HALF),
    "103": Affine2(1, 0, 0, 1, C, LOBBY_HALF),
    "105": Affine2(1, 0, 0, -1, C, -LOBBY_HALF),
    "107": Affine2(-1, 0, 0, -1, -C, -LOBBY_HALF),
}

# ----------------------------------------------------------------------------- void deck uses (all typologies)
AMENITY_USES = ("rc_centre", "kindergarten", "senior_centre", "substation", "eldercare", "study_corner")
FIXTURE_USES = ("letterbox", "bin")    # not rooms: the letterbox banks and the bin centre under each refuse chute
QUADRANT_ORDER = ("105", "107", "101", "103")   # point block amenity rooms that are not pinned take these in turn
WIDE_DOOR_USES = {"kindergarten", "senior_centre", "eldercare", "rc_centre"}
LETTERBOX_D, LETTERBOX_H = 0.45, 1.8
CORE_GAP = 0.15      # letterbox back to the wall centreline (200 mm wall + 50 mm), or in from an open void deck edge
END_GAP = 0.05       # letterbox end to the end of its void deck edge
DOOR_CLEAR = 0.6     # letterbox to the nearest ground-floor door opening
DOOR_APPROACH = 1.5  # no letterbox straight in front of or behind a door opening, this deep
WALL_HALF = 0.1      # half a 200 mm core / amenity room wall
COLUMN_CLEAR = 0.02  # letterbox to a void deck column (blocks/plate_ops.py column_candidates)


def void_deck_uses(spec: dict) -> list[str]:
    """The configured void deck uses, checked: amenity rooms or fixtures; anything else raises ValueError."""
    uses = [str(u) for u in spec.get("void_deck", [])]
    unknown = sorted({u for u in uses if u not in AMENITY_USES and u not in FIXTURE_USES})
    if unknown:
        raise ValueError(f"Blk {spec.get('blk')}: unknown void deck use(s) {unknown} "
                         f"(amenity rooms {list(AMENITY_USES)}, fixtures {list(FIXTURE_USES)})")
    return uses


def assign_quadrants(spec: dict, uses: list[str]) -> list[tuple[str, str, str]]:
    """[(face id, use, quadrant)] for the amenity rooms of a point block. spec['void_deck_at'] pins a use to a
    quadrant; the other amenities take the free quadrants in QUADRANT_ORDER. Fixture uses (letterbox, bin) are not
    rooms and take no quadrant. Face ids AM<k> number the uses in config order. Two amenities pinned to one
    quadrant, a pin to a non-quadrant or more than four amenities raise ValueError."""
    blk = spec.get("blk")
    at = {str(u): str(q) for u, q in spec.get("void_deck_at", {}).items()}
    amen = [(k, u) for k, u in enumerate(uses) if u not in FIXTURE_USES]
    taken = {}
    for k, use in amen:
        if use in at:
            q = at[use]
            if q not in QUADRANTS or q in taken:
                why = f"taken by {taken[q]}" if q in taken else "not a quadrant"
                raise ValueError(f"Blk {blk}: void deck {use} -> quadrant {q} ({why})")
            taken[q] = use
    free = [q for q in QUADRANT_ORDER if q not in taken]
    auto = [(k, u) for k, u in amen if u not in at]
    if len(auto) > len(free):
        raise ValueError(f"Blk {blk}: {len(amen)} void deck amenity rooms but a point block has 4 quadrants")
    quad = {k: at[u] for k, u in amen if u in at} | {k: free[j] for j, (k, _) in enumerate(auto)}
    return [(f"AM{k + 1}", u, quad[k]) for k, u in amen]


# ----------------------------------------------------------------------------- letterbox banks (all typologies)
def letterbox_rules() -> dict:
    """config/rules.toml [void_deck] plus the box column it implies: col_w (m), per_col (boxes) and min_cols."""
    from estate.flats.check import rules
    r = {"letterbox_per_bank": 48, "bank_length": 2.4, "bank_columns": 8, "bank_min_length": 1.5,
         "clear_front": 1.2, "near_lobby": 10.0}
    r.update(rules().get("void_deck", {}))
    per_bank, cols = int(r["letterbox_per_bank"]), int(r["bank_columns"])
    if per_bank % cols:
        raise ValueError(f"rules.toml [void_deck]: letterbox_per_bank {per_bank} does not fill {cols} columns evenly")
    r["col_w"] = float(r["bank_length"]) / cols
    r["per_col"] = per_bank // cols
    r["min_cols"] = math.ceil(float(r["bank_min_length"]) / r["col_w"] - 1e-6)
    return r


@dataclass
class _Edge:
    """One straight edge of the open void deck: s in [0, length] from p0 along u, n points into the void deck.
    walled: the [s0, s1] stretches against a core or amenity room wall."""
    p0: np.ndarray
    u: np.ndarray
    n: np.ndarray
    length: float
    walled: list

    def rect(self, s0, s1, o0, o1):
        p, u, n = self.p0, self.u, self.n
        return Polygon([p + u * s0 + n * o0, p + u * s1 + n * o0, p + u * s1 + n * o1, p + u * s0 + n * o1])

    def on_wall(self, a, b) -> float:
        return sum(max(0.0, min(b, s1) - max(a, s0)) for s0, s1 in self.walled)


def _void_edges(gnd: Plate) -> list[_Edge]:
    """The straight edges of the open void deck in ring order, with the stretches that run along a core or amenity
    room wall (the rest is the open edge of the void deck under the facade)."""
    void = orient(gnd.faces["VOID"].poly.simplify(0), 1.0)      # every ring has the void deck on its left
    walls = [f.poly.boundary for f in sorted(gnd.faces.values(), key=lambda f: f.id) if f.kind != VOID]
    out = []
    for ring in (void.exterior, *void.interiors):
        c = np.asarray(ring.coords, float)
        for a, b in zip(c[:-1], c[1:]):
            length = float(np.hypot(*(b - a)))
            if length < 1e-6:
                continue
            u = (b - a) / length
            edge = LineString([a, b])
            walled = []
            for w in walls:
                g = edge.intersection(w)
                for piece in getattr(g, "geoms", [g]):
                    if piece.geom_type == "LineString" and piece.length > 1e-6:
                        s = (np.asarray(piece.coords, float) - a) @ u
                        walled.append((round(float(s.min()), 6), round(float(s.max()), 6)))
            out.append(_Edge(a, u, np.array([-u[1], u[0]]), length, sorted(walled)))
    return out


def _no_go(gnd: Plate, typ: Plate, clear: float) -> dict:
    """What a bank, and the clear space in front of it, must keep out of (polygons), and the depths of both."""
    solid = unary_union([f.poly for f in gnd.faces.values() if f.kind != VOID]).buffer(WALL_HALF, join_style="mitre")
    h = plate_ops.COLUMN / 2 + COLUMN_CLEAR
    pts = plate_ops.column_candidates(derive(typ, "typical"), gnd, plate_ops.COLUMN_KEEP_OFF - COLUMN_CLEAR)
    cols = unary_union([box(x - h, y - h, x + h, y + h) for x, y in pts])
    segs = [LineString([d.hinge, (2 * d.at[0] - d.hinge[0], 2 * d.at[1] - d.hinge[1])]) for d in gnd.doors]
    doors = unary_union([g.buffer(DOOR_CLEAR + 0.002) for g in segs]       # + 2 mm: the buffer is a polygon
                        + [g.buffer(DOOR_APPROACH, cap_style="flat") for g in segs])
    walk = unary_union([f.poly for f in typ.faces.values() if f.kind in COMMON_KINDS])
    depth = (CORE_GAP, CORE_GAP + LETTERBOX_D, CORE_GAP + LETTERBOX_D + clear)    # bank back, bank front, clear space
    return {"void": gnd.faces["VOID"].poly, "bank": unary_union([solid, cols, doors]), "walk": walk,
            "clear": unary_union([solid, cols]), "depth": depth}


def _forbid(z: dict, others: list, path_ok: bool) -> tuple:
    """(no bank, no clear space) given the other banks [(edge, a, b)]: a bank keeps out of walls, columns, doors,
    the walk path (unless path_ok) and the other banks and their clear space; the clear space in front of it keeps
    out of walls, columns and the other banks."""
    o0, o1, o2 = z["depth"]
    taken = [e.rect(a, b, o0, o1) for e, a, b in others]
    fronts = [e.rect(a, b, o1, o2) for e, a, b in others]
    return (unary_union([z["bank"], *taken, *fronts] + ([] if path_ok else [z["walk"]])),
            unary_union([z["clear"], *taken]))


def _free(edge: _Edge, z: dict, forbid: tuple) -> list[tuple]:
    """Free [s0, s1] intervals of a void deck edge for a bank (forbid from _forbid); bank and clear space both stay
    on the open void deck."""
    o0, o1, o2 = z["depth"]
    blocked = []
    for strip, no in zip((edge.rect(0, edge.length, o0, o1), edge.rect(0, edge.length, o1, o2)), forbid):
        for g in (strip.intersection(no), strip.difference(z["void"])):
            for piece in getattr(g, "geoms", [g]):
                if piece.geom_type == "Polygon" and piece.area > 1e-7:
                    s = (shapely.get_coordinates(piece) - edge.p0) @ edge.u
                    blocked.append((float(s.min()), float(s.max())))
    free, cur, end = [], END_GAP, edge.length - END_GAP
    for s0, s1 in sorted(blocked) + [(end, end)]:
        if s0 > cur + 1e-9:
            free.append((cur, min(s0, end)))
        cur = max(cur, s1)
    return [(round(a, 6), round(b, 6)) for a, b in free if b > a]


def place_letterboxes(blk: str, gnd: Plate, typ: Plate, flats: int, lobbies: list) -> tuple[list[dict], dict]:
    """Letterbox banks holding at least `flats` boxes on the void deck of gnd (typ: the typical floor, for the walk
    path and the columns), shared out between the lift lobbies: lobby k gets its share of what is still missing.
    For each lobby, within near_lobby of it: (1) banks wholly against a wall and off the walk path, the nearest
    that takes the whole share, else the one that takes most; (2) those banks lengthened a column at a time along
    their void deck edge (at least half of a bank stays against the wall); (3) as (1) on the open edge of the void
    deck; (4) as (1) on the walk path. Whatever is still missing goes anywhere on the void deck in the same four
    steps. Returns the plan.meta['furniture'] entries and a summary {flats, boxes, banks, per_bank}; raises
    ValueError when there is no room left."""
    R = letterbox_rules()
    col_w, per_col, min_cols = R["col_w"], R["per_col"], R["min_cols"]
    edges = _void_edges(gnd)
    z = _no_go(gnd, typ, float(R["clear_front"]))
    o0, o1, _ = z["depth"]
    lob = [shapely.Point(p) for p in lobbies]
    banks = []          # [edge index, a, b, lobby index]

    def others(skip=None):
        return [(edges[i], a, b) for j, (i, a, b, _) in enumerate(banks) if j != skip]

    def nearest(i, a, b):
        g = edges[i].rect(a, b, o0, o1)
        d = [g.distance(p) for p in lob]
        k = int(np.argmin(d))
        return round(d[k], 6), k

    def add_banks(want, lobby, near, path_ok, walled=True):
        got = 0
        while got < want:
            cands, forbid = [], _forbid(z, others(), path_ok)
            for i, e in enumerate(edges):
                if near and lobby is not None and lob[lobby].distance(e.rect(0, e.length, o0, o1)) > R["near_lobby"]:
                    continue
                for f0, f1 in _free(e, z, forbid):
                    for w0, w1 in e.walled if walled else [(0.0, e.length)]:
                        a0, a1 = max(f0, w0), min(f1, w1)
                        fit = int((a1 - a0) / col_w + 1e-6)
                        if fit < min_cols:
                            continue
                        cols = min(fit, max(min_cols, math.ceil((want - got) / per_col)))
                        short = cols * per_col < want - got
                        L = round(cols * col_w, 6)
                        for a in (a0, round(a1 - L, 6)):
                            d, k = nearest(i, a, a + L)
                            if (lobby is None or k == lobby) and (not near or d <= R["near_lobby"]):
                                cands.append((short, -cols if short else 0, d, i, a, round(a + L, 6), k))
            if not cands:
                break
            _, _, _, i, a, b, k = min(cands)
            banks.append([i, a, b, k])
            got += round((b - a) / col_w) * per_col
        return got

    def lengthen(want, lobby):
        got = 0
        for j, bk in enumerate(banks):
            if got >= want:
                break
            i, a, b, k = bk
            if lobby is not None and k != lobby:
                continue
            e = edges[i]
            room = next(((f0, f1) for f0, f1 in _free(e, z, _forbid(z, others(skip=j), False))
                         if f0 <= a + 1e-6 and b <= f1 + 1e-6), None)
            if room is None:
                continue
            lo_near = lob[k].distance(e.rect(a, a + col_w, o0, o1)) <= lob[k].distance(e.rect(b - col_w, b, o0, o1))
            while got < want:
                for da, db in ((0.0, col_w), (-col_w, 0.0)) if lo_near else ((-col_w, 0.0), (0.0, col_w)):
                    na, nb = round(a + da, 6), round(b + db, 6)
                    if na >= room[0] - 1e-6 and nb <= room[1] + 1e-6 and e.on_wall(na, nb) >= (nb - na) / 2 - 1e-6:
                        a, b = na, nb
                        got += per_col
                        break
                else:
                    break
            bk[1], bk[2] = a, b
        return got

    def steps(lobby, near):
        return (lambda w: add_banks(w, lobby, near, False), lambda w: lengthen(w, lobby),
                lambda w: add_banks(w, lobby, near, False, walled=False), lambda w: add_banks(w, lobby, near, True))

    total = 0
    for k in range(len(lob)):
        want, got = math.ceil((flats - total) / (len(lob) - k)), 0
        for step in steps(k, True):
            if got < want:
                got += step(want - got)
        total += got
    for step in steps(None, False):
        if total < flats:
            total += step(flats - total)
    if total < flats:
        raise ValueError(f"Blk {blk}: letterboxes for {total} of {flats} flats: no core or amenity room wall left on "
                         f"the void deck for another bank")
    items = []
    for n, (i, a, b, k) in enumerate(banks, 1):
        items.append(dict(name=f"Letterbox bank {n}", poly=plate_ops.snap_geom(edges[i].rect(a, b, o0, o1)),
                          height=LETTERBOX_H, object_type="Letterbox bank", level="ground", mat="steel",
                          boxes=round((b - a) / col_w) * per_col))
    return items, dict(flats=flats, boxes=total, banks=len(items), per_bank=int(R["letterbox_per_bank"]))


def use_door_kind(use: str) -> str:
    """Door kind for a void-deck amenity room (rules.DOOR_KINDS)."""
    if use == "substation":
        return "locked"
    if use in WIDE_DOOR_USES or use == "study_corner":
        return "shop"
    return "service"


USE_NAMES = {"rc_centre": "Residents' Committee centre", "kindergarten": "Kindergarten",
             "senior_centre": "Senior activity centre", "substation": "Electrical substation",
             "letterbox": "Letterbox bank", "bin": "Bin point", "eldercare": "Eldercare centre",
             "study_corner": "Study corner"}


# ----------------------------------------------------------------------------- core
def _core_faces(plate: Plate, access: str, level: str) -> list:
    """Lifts, refuse chute room / bin centre (not on the roof) and stairs with their doors; returns the polygons."""
    y0, y1 = LOBBY_HALF, LOBBY_HALF + LIFT_DEPTH
    lifts = [box(round(-C + k * LIFT_W, 4), y0, round(-C + (k + 1) * LIFT_W, 4), y1) for k in range(2)]
    stairs = [box(-C, -LOBBY_HALF - STAIR_DEPTH, 0, -LOBBY_HALF), box(0, -LOBBY_HALF - STAIR_DEPTH, C, -LOBBY_HALF)]
    out = lifts + stairs
    for k, p in enumerate(lifts, 1):
        plate.add(Face(f"LIFT{k}", p, LIFT, name=f"Lift {k} shaft"))
        if level != "roof":
            cx = round((p.bounds[0] + p.bounds[2]) / 2, 4)
            plate.doors.append(DoorSpec(f"LIFT{k}", access, (cx, LOBBY_HALF), 1.0, "lift", f"LIFT{k}", (cx - 0.5, LOBBY_HALF),
                                        f"lift {k} landing door", DOOR_H))
    if level != "roof":
        ref = box(round(C - REF_W, 4), y0, C, y1)
        cx = round(C - REF_W / 2, 4)
        if level == "ground":
            plate.add(Face("REF1", ref, REFUSE, name="Bin centre 1", meta={"use": "bin"}))
            w, nm = BIN_DOOR_W, "bin centre 1 door"
        else:
            plate.add(Face("REF1", ref, REFUSE, name="Refuse chute room 1"))
            w, nm = REF_DOOR_W, "refuse chute room 1 door"
        plate.doors.append(DoorSpec("REF1", access, (cx, LOBBY_HALF), w, "service", "REF1",
                                    (round(cx - w / 2, 4), LOBBY_HALF), nm, DOOR_H))
        out.append(ref)
    for k, p in enumerate(stairs, 1):
        plate.add(Face(f"STAIR{k}", p, STAIR, name=f"Stair {k}"))
        cx = (p.bounds[0] + p.bounds[2]) / 2
        kind = "roof" if level == "roof" else "stair"
        nm = f"stair {k} {'roof door' if level == 'roof' else 'fire door'}"
        plate.doors.append(DoorSpec(f"STAIR{k}", access, (cx, -LOBBY_HALF), 1.0, kind, f"STAIR{k}", (cx - 0.5, -LOBBY_HALF), nm, DOOR_H))
    return out


def plan_point(spec: dict, estate_seed: int = 0, cat=None) -> BuildingPlan:
    cat = cat or catalogue()
    blk = str(spec["blk"])
    stacks = spec["stacks"]
    uses = void_deck_uses(spec)
    # ---- typical residential floor
    typ = Plate()
    flats = {}
    for stack in ("101", "103", "105", "107"):
        s = stacks[stack]
        t = cat[s["t"]]
        v = t.pick_variant(derive_seed(estate_seed, blk, stack), s.get("v"))
        flats[stack] = t.realize(v)
    west = max(flats["101"].width, flats["107"].width)
    east = max(flats["103"].width, flats["105"].width)
    lobby = box(-C - west, -LOBBY_HALF, C + east, LOBBY_HALF)
    typ.add(Face("LOBBY", lobby, LOBBY, name="Lift lobby / common corridor"))
    _core_faces(typ, "LOBBY", "typical")
    for stack, lf in flats.items():
        place_flat(typ, lf, stack, QUADRANTS[stack], "LOBBY")

    residential = unary_union([f.poly for f in typ.faces.values()])

    # ---- void deck: amenity rooms under flat envelopes, the bin centre under the chute, letterbox banks
    gnd = Plate()
    core_g = _core_faces(gnd, "VOID", "ground")
    amen = []
    for fid, use, q in assign_quadrants(spec, uses):    # the bin centre is REF1, letterboxes go on the void deck
        x0, y0, x1, y1 = typ.flats[q].envelope.bounds
        rect = box(x0, y0, x1, y1)
        gnd.add(Face(fid, rect, AMENITY, name=USE_NAMES[use], meta={"use": use}))
        # door on the lobby line, centred in x
        cx = round((x0 + x1) / 2, 2)
        cy = y0 if y0 >= 0 else y1
        gnd.doors.append(DoorSpec(fid, "VOID", (cx, cy), 1.8 if use in WIDE_DOOR_USES else 1.0, use_door_kind(use),
                                  fid, (cx - 0.5, cy), f"{USE_NAMES[use]} door", DOOR_H))
        amen.append(rect)
    gnd.add(Face("VOID", residential.difference(unary_union(core_g + amen)), VOID, name="Void deck"))
    lift_x = [(typ.faces[f].poly.bounds[0] + typ.faces[f].poly.bounds[2]) / 2 for f in ("LIFT1", "LIFT2")]
    lobbies = [(round(sum(lift_x) / 2, 4), 0.0)]
    furniture, letterboxes = [], None
    if "letterbox" in uses:
        furniture, letterboxes = place_letterboxes(blk, gnd, typ, len(flats) * (int(spec["storeys"]) - 1), lobbies)

    # ---- roof (the refuse chute room stops below it)
    roof = Plate()
    core_r = _core_faces(roof, "DECK", "roof")
    roof.add(Face("DECK", residential.difference(unary_union(core_r)), DECK, name="Roof deck"))
    ne = typ.flats["103"].envelope.bounds
    tank = box(ne[0] + 2.0, ne[1] + 2.0, min(ne[0] + 6.0, ne[2] - 1.0), min(ne[1] + 5.0, ne[3] - 1.0))

    return BuildingPlan(
        blk=blk, name=f"Blk {blk}", long_name=spec.get("long_name", "HDB point block"), typology="PT4",
        storeys=int(spec["storeys"]), ground=gnd, typical=typ, roof=roof,
        stairs=[StairSpec("STAIR1", "N", "stair 1"), StairSpec("STAIR2", "N", "stair 2")],
        lifts=["LIFT1", "LIFT2"],
        address=dict(AddressLines=[f"Blk {blk} {spec.get('street', 'Sample Town N5')}"], Town="Singapore",
                     PostalCode=f"560{blk[-3:]}", Country="Singapore"),
        void_deck_uses=uses, roof_items=[("tank", tank, 2.2)],
        meta={"lift_lobbies": lobbies, "entrances": [(-C - west - 0.1, 0.0), (C + east + 0.1, 0.0)],
              "furniture": furniture, "letterboxes": letterboxes})
