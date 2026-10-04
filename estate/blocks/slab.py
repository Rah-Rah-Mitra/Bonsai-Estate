"""SL corridor slab block: one single-loaded open-air corridor with a row of flats, lift cores and end stairs.

Block-local strip frame: the common corridor runs along x on the south side (y in [-2.2, 0]: 2.0 m clear between
the facade wall and the parapet); every module sits north of it in y >= 0 and is entered from the corridor at
y = 0. In front of the lifts of each core the corridor widens to a 3.0 m deep lift lobby bay, so people waiting
for a lift (or turning a pram or a stretcher) do not block the corridor. Modules run west to east in the order
of spec["sequence"]:
  "stair"        end stair enclosure 2.8 (x) x 5.6 (y), entered from the corridor (entry S, the 5.6 m side
                 runs along the entry axis as the dog-leg flights need)
  "core"         9.8 m module: two lift shafts 2.8 x 2.5, a stair enclosure 2.8 x 5.6 and a refuse chute room
                 1.4 x 1.5 (the bin centre at the void deck); landing doors open onto the corridor
  template id    a flat placed with its template frame unchanged (x along the corridor, y = 0 on the corridor),
                 seeded variant per stack (mirror variants are handled by Template.realize)
The corridor face spans the whole length, so corridor|outside edges become 1.1 m parapets and every main door
opens onto it. Behind the lifts and the refuse room the module is open (a recess in the rear facade).
The void deck keeps the cores and puts amenity rooms under chosen flat envelopes (void deck uses are checked by
blocks/point.py: 'bin' is the bin centre, 'letterbox' letterbox banks for every flat at the lift lobbies);
the roof keeps the stair and lift towers on a deck with a water tank (the refuse chute room stops below the
roof). The block is centred on x = 0. spec["corridor"] other than "S" rotates the finished plan by quarter turns
(normally the estate placement does the rotating). The strip builder is shared with the L-block (blocks/lblock.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import box
from shapely.ops import unary_union

from estate.blocks import plate_ops
from estate.blocks.builder import BuildingPlan, StairSpec
from estate.blocks.plate import AMENITY, CORRIDOR, DECK, LIFT, REFUSE, STAIR, VOID, DoorSpec, Face, Plate
from estate.blocks.point import (AMENITY_USES, USE_NAMES, WIDE_DOOR_USES, place_letterboxes, use_door_kind,
                                 void_deck_uses)
from estate.flats.template import Affine2, catalogue, derive_seed, place_flat
from estate.rules import DOOR_H

CORR_D = 2.2                     # corridor band (centreline): 2.0 m clear between facade wall and parapet
LOBBY_D = 3.0                    # lift lobby bay in front of each core's lifts (centreline depth from the facade)
STAIR_W, STAIR_D = 2.8, 5.6      # stair enclosure (centreline) -> 2.6 x 5.4 clear
LIFT_W, LIFT_D = 2.8, 2.5
REF_W, REF_D = 1.4, 1.5
CORE_ORDER = ("lift", "lift", "stair", "refuse")
MODULE_W = {"stair": STAIR_W, "core": 2 * LIFT_W + STAIR_W + REF_W}
SIDE_TURNS = {"S": 0, "E": 1, "N": 2, "W": 3}
TANK_H = 2.2


@dataclass
class Module:
    kind: str                     # stair | core | flat
    width: float
    stack: str | None = None
    lf: object = None             # realized LocalFlat
    flip: bool = False            # place the template mirrored (template x = 0 at the module's far end)
    order: tuple = CORE_ORDER     # core parts west -> east (strip frame)


@dataclass
class Strip:
    typ: Plate
    gnd: Plate                    # without the VOID face (added once the whole building is assembled)
    roof: Plate                   # without the DECK face
    stairs: list = field(default_factory=list)
    lifts: list = field(default_factory=list)
    lobbies: list = field(default_factory=list)   # corridor points in front of each core's lifts
    fronts: list = field(default_factory=list)    # just outside the corridor edge in front of each core
    ends: list = field(default_factory=list)      # just outside both corridor ends (start, end)
    x0: float = 0.0
    x1: float = 0.0


# ----------------------------------------------------------------------------- flats
def item_of(item):
    """A sequence entry: 'stair' | 'core' | template id | {t = id, v = {variant pins}}."""
    if isinstance(item, dict):
        return str(item["t"]), dict(item.get("v", {}))
    return str(item), {}


def realize_flat(item, stack: str, cat: dict, estate_seed: int, blk: str, pins: dict | None = None):
    tid, v = item_of(item)
    if tid not in cat:
        raise KeyError(f"Blk {blk}: flat template {tid} is not in the catalogue")
    t = cat[tid]
    v = {**(pins or {}), **v}
    return t.realize(t.pick_variant(derive_seed(estate_seed, blk, stack), v or None))


def stack_id(n: int) -> str:
    return f"1{n:02d}"


def assign_amenities(spec: dict, stacks: list) -> dict:
    """Void-deck amenity rooms under flat envelopes: {stack: (face id, use)}. spec['void_deck_at'] pins a use to
    a stack; the others are spread evenly along the block."""
    uses = [u for u in void_deck_uses(spec) if u in AMENITY_USES]
    pinned = {u: str(s) for u, s in spec.get("void_deck_at", {}).items() if u in uses}
    out = {}
    for k, use in enumerate(uses):
        if use in pinned:
            if pinned[use] not in stacks or pinned[use] in out:
                raise ValueError(f"void_deck_at: {use} -> {pinned[use]} is not a free flat stack")
            out[pinned[use]] = (f"AM{k + 1}", use)
    free = [s for s in stacks if s not in out]
    auto = [(k, u) for k, u in enumerate(uses) if u not in pinned]
    if len(auto) > len(free):
        raise ValueError(f"{len(auto)} void deck amenities but only {len(free)} flat envelopes to put them under")
    for j, (k, use) in enumerate(auto):
        out[free[int((j + 0.5) * len(free) / len(auto))]] = (f"AM{k + 1}", use)
    return out


# ----------------------------------------------------------------------------- strip
def _door(plate, face, access, cx, width, kind, name):
    plate.doors.append(DoorSpec(face, access, (round(cx, 4), 0.0), width, kind, face, (round(cx - width / 2, 4), 0.0),
                                name, DOOR_H))


def build_strip(modules: list, x0: float, corridor_id: str, n: dict, amenities: dict | None = None,
                corridor_name: str = "Common corridor", lead: float = 0.0) -> Strip:
    """Author one corridor strip in the strip frame (corridor y in [-CORR_D, 0] plus a lift lobby bay down to
    -LOBBY_D in front of each core's lifts, modules from x0 eastwards). lead extends the corridor that far west
    of x0 with no modules beside it (the L-block's corridor bridge across the light well).
    n: running counters {'stair', 'lift', 'refuse'} shared by all strips of a building (unique face ids)."""
    amenities = amenities or {}
    typ, gnd, roof = Plate(), Plate(), Plate()
    s = Strip(typ, gnd, roof, x0=x0)

    def stair(x):
        n["stair"] += 1
        k = n["stair"]
        fid = f"STAIR{k}"
        p = box(x, 0.0, x + STAIR_W, STAIR_D)
        cx = x + STAIR_W / 2
        for pl, access, kind, nm in ((typ, corridor_id, "stair", "fire door"), (gnd, "VOID", "stair", "fire door"),
                                     (roof, "DECK", "roof", "roof door")):
            pl.add(Face(fid, p, STAIR, name=f"Stair {k}"))
            _door(pl, fid, access, cx, 1.0, kind, f"stair {k} {nm}")
        s.stairs.append(StairSpec(fid, "S", f"stair {k}"))
        return p

    def lift(x):
        n["lift"] += 1
        k = n["lift"]
        fid = f"LIFT{k}"
        p = box(x, 0.0, x + LIFT_W, LIFT_D)
        for pl, access in ((typ, corridor_id), (gnd, "VOID"), (roof, None)):
            pl.add(Face(fid, p, LIFT, name=f"Lift {k} shaft"))
            if access:
                _door(pl, fid, access, x + LIFT_W / 2, 1.0, "lift", f"lift {k} landing door")
        s.lifts.append(fid)
        return p

    def refuse(x):
        n["refuse"] += 1
        k = n["refuse"]
        fid = f"REF{k}"
        p = box(x, 0.0, x + REF_W, REF_D)
        typ.add(Face(fid, p, REFUSE, name=f"Refuse chute room {k}"))
        _door(typ, fid, corridor_id, x + REF_W / 2, 0.8, "service", f"refuse chute room {k} door")
        gnd.add(Face(fid, p, REFUSE, name=f"Bin centre {k}", meta={"use": "bin"}))
        _door(gnd, fid, "VOID", x + REF_W / 2, 1.0, "service", f"bin centre {k} door")
        return p

    parts_of = {"lift": (lift, LIFT_W), "stair": (stair, STAIR_W), "refuse": (refuse, REF_W)}
    x = round(x0, 4)
    bays = []
    for m in modules:
        if m.kind == "flat":
            xf = Affine2(-1, 0, 0, 1, x + m.width, 0.0) if m.flip else Affine2(1, 0, 0, 1, x, 0.0)
            fi = place_flat(typ, m.lf, m.stack, xf, corridor_id)
            if m.stack in amenities:
                fid, use = amenities[m.stack]
                env = plate_ops.snap_geom(fi.envelope)
                bb = box(*env.bounds)
                g = bb if abs(bb.area - env.area) < 1e-6 else env
                nm = USE_NAMES.get(use, use.replace("_", " ").title())
                gnd.add(Face(fid, g, AMENITY, name=nm, meta={"use": use, "under": m.stack}))
                _door(gnd, fid, "VOID", (env.bounds[0] + env.bounds[2]) / 2, 1.8 if use in WIDE_DOOR_USES else 1.0,
                      use_door_kind(use), f"{nm} door")
        elif m.kind == "stair":
            stair(x)
        elif m.kind == "core":
            cx, parts = x, []
            for part in m.order:
                if part not in parts_of:
                    raise ValueError(f"unknown core part {part}")
                build, w = parts_of[part]
                parts.append((part, build(cx)))
                cx += w
            lifts = [p for kind, p in parts if kind == "lift"]
            if lifts:
                lx0, lx1 = min(p.bounds[0] for p in lifts), max(p.bounds[2] for p in lifts)
                bays.append(box(lx0, -LOBBY_D, lx1, -CORR_D))
                s.lobbies.append((round((lx0 + lx1) / 2, 4), -LOBBY_D / 2))
                s.fronts.append((round((lx0 + lx1) / 2, 4), -LOBBY_D - 0.2))
            else:
                s.fronts.append((round(x + m.width / 2, 4), -CORR_D - 0.2))
        else:
            raise ValueError(f"unknown module kind {m.kind}")
        x = round(x + m.width, 4)
    band = box(round(x0 - lead, 4), -CORR_D, x, 0.0)
    corr = plate_ops.single_polygon(unary_union([band] + bays), f"corridor {corridor_id}")
    typ.add(Face(corridor_id, corr, CORRIDOR, name=corridor_name))
    s.x1 = x
    s.ends = [(round(x0 - lead - 0.2, 4), -CORR_D / 2), (round(x + 0.2, 4), -CORR_D / 2)]
    for pl in (typ, gnd, roof):
        plate_ops.snap(pl)
    return s


def module_list(seq: list, stacks: dict, cat: dict, estate_seed: int, blk: str, pins: dict,
                core_order=CORE_ORDER) -> list:
    """Modules for a sequence; stacks maps the sequence index of every flat to its stack id."""
    out = []
    for i, item in enumerate(seq):
        tid, _ = item_of(item)
        if tid in MODULE_W:
            out.append(Module(tid, MODULE_W[tid], order=tuple(core_order)))
        else:
            st = stacks[i]
            lf = realize_flat(item, st, cat, estate_seed, blk, pins.get(st, {}).get("v"))
            out.append(Module("flat", round(lf.width, 4), st, lf))
    return out


def close_plates(typ: Plate, gnd: Plate, roof: Plate):
    """Add the void deck (typical outline minus cores and amenity rooms) and the roof deck (minus towers)."""
    residential = plate_ops.union(typ)
    gnd.add(Face("VOID", plate_ops.single_polygon(residential.difference(plate_ops.union(gnd)), "void deck"), VOID,
                 name="Void deck"))
    roof.add(Face("DECK", plate_ops.single_polygon(residential.difference(plate_ops.union(roof)), "roof deck"), DECK,
                  name="Roof deck"))
    return residential


def void_deck_fixtures(spec: dict, gnd: Plate, typ: Plate, lobbies: list) -> tuple[list, dict | None]:
    """The void deck uses that are not rooms: 'bin' needs a bin centre (the ground floor of every refuse chute
    room); 'letterbox' puts letterbox banks for every flat of the block at the lift lobbies (point.place_letterboxes,
    on the closed plates). Returns the plan.meta['furniture'] entries and the letterbox summary (None without)."""
    uses = void_deck_uses(spec)
    if "bin" in uses and not any(f.kind == REFUSE for f in gnd.faces.values()):
        raise ValueError(f"Blk {spec['blk']}: void deck 'bin' but the block has no refuse chute / bin centre")
    if "letterbox" not in uses:
        return [], None
    flats = len(typ.flats) * (int(spec["storeys"]) - 1)
    return place_letterboxes(str(spec["blk"]), gnd, typ, flats, lobbies)


def roof_tank(typ: Plate, stack: str):
    """A 4 x 3 m water tank on the roof deck over a flat (clear of the stair roof doors on the corridor side)."""
    x0, y0, x1, y1 = typ.flats[stack].envelope.bounds
    w, d = (4.0, 3.0) if (x1 - x0) >= (y1 - y0) else (3.0, 4.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return plate_ops.snap_geom(box(cx - w / 2, cy - d / 2, cx + w / 2, cy + d / 2))


def address(spec: dict, blk: str) -> dict:
    return dict(AddressLines=[f"Blk {blk} {spec.get('street', 'Sample Town N5')}"], Town="Singapore",
                PostalCode=f"560{blk[-3:]}", Country="Singapore")


# ----------------------------------------------------------------------------- planner
def plan_slab(spec: dict, estate_seed: int = 0, cat=None) -> BuildingPlan:
    cat = cat or catalogue()
    blk = str(spec["blk"])
    side = str(spec.get("corridor", "S")).upper()
    if side not in SIDE_TURNS:
        raise ValueError(f"Blk {blk}: corridor side {side} (expected one of {sorted(SIDE_TURNS)})")
    seq = list(spec["sequence"])
    stacks, nf = {}, 0
    for i, item in enumerate(seq):
        if item_of(item)[0] not in MODULE_W:
            nf += 1
            stacks[i] = stack_id(nf)
    modules = module_list(seq, stacks, cat, estate_seed, blk, spec.get("stacks", {}), spec.get("core_order", CORE_ORDER))
    length = round(sum(m.width for m in modules), 4)
    order = [m.stack for m in modules if m.kind == "flat"]
    amen = assign_amenities(spec, order)
    s = build_strip(modules, -length / 2, "CORR", {"stair": 0, "lift": 0, "refuse": 0}, amen)
    close_plates(s.typ, s.gnd, s.roof)
    furniture, letterboxes = void_deck_fixtures(spec, s.gnd, s.typ, s.lobbies)
    mid = min(order, key=lambda st: (abs(s.typ.flats[st].envelope.centroid.x), st))
    plan = BuildingPlan(
        blk=blk, name=f"Blk {blk}", long_name=spec.get("long_name", f"HDB corridor slab block ({nf} flats per floor)"),
        typology="SL", storeys=int(spec["storeys"]), ground=s.gnd, typical=s.typ, roof=s.roof,
        stairs=s.stairs, lifts=s.lifts, address=address(spec, blk), void_deck_uses=void_deck_uses(spec),
        roof_items=[("tank", roof_tank(s.typ, mid), TANK_H)],
        meta={"lift_lobbies": s.lobbies, "entrances": s.fronts + s.ends, "corridor_side": "S", "length": length,
              "stacks": order, "amenities": {st: list(v) for st, v in sorted(amen.items())}, "furniture": furniture,
              "letterboxes": letterboxes})
    k = SIDE_TURNS[side]
    return plate_ops.transform_plan(plan, plate_ops.rotation(k)) if k else plan
