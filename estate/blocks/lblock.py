"""LB L-block: two corridor-slab wings joined at the outer corner by a corner module (corner unit + core).

Both wings are SL strips (blocks/slab.py) with the corridor on the inner side of the L, facing the courtyard,
and the flats on the outer side. Block-local frame with the origin at the outer corner of the L:
  wing A runs along +x from the corner, flats north of its corridor (y up to 0), corridor south of them;
  wing B runs along -y from the corner below wing A's corridor, flats west of its corridor, corridor east.
Wing A's strip starts with the corner module: the corner unit (spec["corner"], a 3Gen corner template (3G-C or 3G-C2), fits LC) is
placed mirrored so its template x = 0 side is against the corner core and its x = W side and rear face the
outside; its y = 0 side opens onto wing A's corridor. The corner core (stair next to the corner unit, two
lifts, refuse chute) follows, then spec["wing_a"]. Wing B's strip (spec["wing_b"], listed from the corner
outwards) is mapped by (s, t) -> (xb - t, -CORR_D - LIGHTWELL - s): it is set back from wing A's corridor by a
6 m light well, crossed by its own corridor (an open bridge with parapets on both sides) that meets wing A's
corridor edge to edge (corridor|corridor boundaries carry no wall). Butted straight against corridor A, the
first wing B flat put its blank flank 2 m in front of the corner unit's corridor windows (the grandparents'
bedroom and the kitchen); across the light well they look out over 8 m of open air. Wing B's outer facade lines
up with the corner unit's west facade, and its stair enclosures have their 5.6 m side along x, entered from the
east. Letterbox banks for every flat stand at the two lift lobbies (void deck uses: blocks/point.py).
spec["mirror"] = true mirrors the finished plan in x (wing A then runs along -x, stair entries E <-> W).
Stacks are numbered 101.. continuously along the corridor: wing B from its far end, the corner unit, wing A.
"""
from __future__ import annotations

from estate.blocks import plate_ops
from estate.blocks.builder import BuildingPlan, StairSpec
from estate.blocks.point import void_deck_uses
from estate.blocks.slab import (CORE_ORDER, CORR_D, MODULE_W, TANK_H, Module, address, assign_amenities, build_strip,
                                close_plates, item_of, module_list, realize_flat, roof_tank, stack_id,
                                void_deck_fixtures)
from estate.flats.template import Affine2, catalogue

CORNER_CORE_ORDER = ("stair", "lift", "lift", "refuse")
LIGHTWELL = 6.0      # wing B set back from wing A's corridor (spec["lightwell"] overrides)


def plan_lblock(spec: dict, estate_seed: int = 0, cat=None) -> BuildingPlan:
    cat = cat or catalogue()
    blk = str(spec["blk"])
    pins = spec.get("stacks", {})
    seq_a, seq_b = list(spec["wing_a"]), list(spec["wing_b"])
    flats_a = [i for i, it in enumerate(seq_a) if item_of(it)[0] not in MODULE_W]
    flats_b = [i for i, it in enumerate(seq_b) if item_of(it)[0] not in MODULE_W]
    stacks_b = {i: stack_id(k + 1) for k, i in enumerate(reversed(flats_b))}
    corner_stack = stack_id(len(flats_b) + 1)
    stacks_a = {i: stack_id(len(flats_b) + 2 + k) for k, i in enumerate(flats_a)}

    # the corner unit's sides are fixed by the site (x = 0 against the core, x = W and rear outside): no seeded
    # mirror; a pinned mirror is undone by placing the realized flat unflipped
    corner_lf = realize_flat(spec["corner"], corner_stack, cat, estate_seed, blk,
                             {"mirror": False, **pins.get(corner_stack, {}).get("v", {})})
    core_order = spec.get("core_order", CORE_ORDER)
    mods_a = [Module("flat", round(corner_lf.width, 4), corner_stack, corner_lf,
                     flip=not corner_lf.variant.get("mirror", False)),
              Module("core", MODULE_W["core"], order=tuple(spec.get("corner_core_order", CORNER_CORE_ORDER)))]
    mods_a += module_list(seq_a, stacks_a, cat, estate_seed, blk, pins, core_order)
    mods_b = module_list(seq_b, stacks_b, cat, estate_seed, blk, pins, core_order)
    order = [stacks_b[i] for i in reversed(flats_b)] + [corner_stack] + [stacks_a[i] for i in flats_a]
    amen = assign_amenities(spec, order)

    gap = float(spec.get("lightwell", LIGHTWELL))
    n = {"stair": 0, "lift": 0, "refuse": 0}
    sa = build_strip(mods_a, 0.0, "CORR_A", n, amen, "Common corridor (wing A)")
    sb = build_strip(mods_b, 0.0, "CORR_B", n, amen, "Common corridor (wing B)", lead=gap)
    xb = plate_ops.bounds(sb.typ)[3]                   # deepest wing B module -> its outer facade at x = 0
    xf_b = Affine2(0, -1, -1, 0, xb, -CORR_D - gap)   # strip (s, t) -> (xb - t, -CORR_D - gap - s)
    typ = plate_ops.merge(sa.typ, plate_ops.transform_plate(sb.typ, xf_b))
    gnd = plate_ops.merge(sa.gnd, plate_ops.transform_plate(sb.gnd, xf_b))
    roof = plate_ops.merge(sa.roof, plate_ops.transform_plate(sb.roof, xf_b))
    close_plates(typ, gnd, roof)

    def tb(pts):
        return [plate_ops.snap_pt(xf_b(p)) for p in pts]
    lobbies = sa.lobbies + tb(sb.lobbies)
    furniture, letterboxes = void_deck_fixtures(spec, gnd, typ, lobbies)
    stairs = sa.stairs + [StairSpec(s.face, plate_ops.map_entry(s.entry, xf_b), s.name) for s in sb.stairs]
    nf = len(order)
    plan = BuildingPlan(
        blk=blk, name=f"Blk {blk}", long_name=spec.get("long_name", f"HDB L-block ({nf} flats per floor, corner core)"),
        typology="LB", storeys=int(spec["storeys"]), ground=gnd, typical=typ, roof=roof, stairs=stairs,
        lifts=sa.lifts + sb.lifts, address=address(spec, blk), void_deck_uses=void_deck_uses(spec),
        roof_items=[("tank", roof_tank(typ, corner_stack), TANK_H)],
        meta={"lift_lobbies": lobbies, "entrances": sa.fronts + tb(sb.fronts) + [sa.ends[1]] + tb(sb.ends[1:]),
              "stacks": order, "corner_stack": corner_stack, "mirror": bool(spec.get("mirror", False)),
              "wing_lengths": [round(sa.x1 - sa.x0, 4), round(sb.x1 - sb.x0, 4)], "lightwell": gap,
              "amenities": {st: list(v) for st, v in sorted(amen.items())}, "furniture": furniture,
              "letterboxes": letterboxes})
    x0, _, _, y1 = plate_ops.bounds(typ)
    plan = plate_ops.transform_plan(plan, plate_ops.translation(-x0, -y1))
    if spec.get("mirror"):
        plan = plate_ops.transform_plan(plan, plate_ops.mirror_x(0.0))
    return plan
