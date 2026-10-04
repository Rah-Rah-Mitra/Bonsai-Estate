"""The original Blk 123 point block, unchanged (legacy/toolkit/hdb_block.py ``BlockBuilder.build``).

It is the regression baseline: ``estate.cmd legacy`` must reproduce the element counts of hdb_block.ifc
(1030 walls, 446 doors, 418 windows, 496 spaces, 44 flats) and two builds must be byte-identical.
"""
from __future__ import annotations

import random

import numpy as np
import shapely
from shapely.geometry import Point, box
from shapely.geometry.polygon import orient

from estate.geom.walls import WallSpec, attach, frame, ring_walls, translate
from estate.ifc.stairs import flight_run, stair
from estate.ifc.writer import IfcWriter
from estate.rules import (CORE_HALF, DOOR_H, FTF, LIFT_D, LOBBY_HALF, PARAPET_H, SLAB, STAIR_D, UNIT_D, UNIT_W,
                          VOID_DECK_FTF)

ROOMS = [
    ("LD", "Living / Dining", box(5.8, 2.8, 11.0, 8.6)),
    ("K", "Kitchen", box(7.4, 0.0, 9.8, 2.8)),
    ("SY", "Service Yard", box(9.8, 0.0, 11.0, 2.8)),
    ("F", "Foyer", box(5.8, 0.0, 7.4, 2.8)),
    ("HS", "Household Shelter", box(4.2, 0.0, 5.8, 1.6)),
    ("CB", "Common Bathroom", box(4.2, 1.6, 5.8, 3.4)),
    ("H", "Hall", box(1.6, 3.4, 5.8, 4.6)),
    ("B3", "Bedroom 3", box(0.0, 0.0, 4.2, 2.4).union(box(1.6, 2.4, 4.2, 3.4))),
    ("MBa", "Master Bathroom", box(0.0, 2.4, 1.6, 4.6)),
    ("MB", "Master Bedroom", box(0.0, 4.6, 3.2, 8.6)),
    ("B2", "Bedroom 2", box(3.2, 4.6, 5.8, 8.6)),
]
PARTITIONS = [
    (5.8, 0.2, 5.8, 1.7, 0.2, "HS"), (4.2, 0.2, 4.2, 1.7, 0.2, "HS"), (4.1, 1.6, 5.9, 1.6, 0.2, "HS"),
    (5.8, 1.7, 5.8, 3.45, 0.1, "INT"), (4.2, 1.7, 4.2, 3.45, 0.1, "INT"), (5.8, 4.55, 5.8, 8.4, 0.1, "INT"),
    (7.35, 2.8, 10.8, 2.8, 0.1, "INT"), (7.4, 0.2, 7.4, 2.85, 0.1, "INT"), (9.8, 0.2, 9.8, 2.85, 0.1, "INT"),
    (1.55, 3.4, 5.85, 3.4, 0.1, "INT"), (0.2, 2.4, 1.65, 2.4, 0.1, "INT"), (1.6, 2.35, 1.6, 4.65, 0.1, "INT"),
    (0.2, 4.6, 5.85, 4.6, 0.1, "INT"), (3.2, 4.65, 3.2, 8.4, 0.1, "INT"),
]
DOORS = [
    ("Main door", 6.6, 0.1, 1.0, "main"),
    ("Household shelter door", 5.8, 0.8, 0.8, "shelter"),
    ("Kitchen door", 7.4, 1.95, 0.9, "internal"),
    ("Service yard door", 9.8, 2.1, 0.8, "internal"),
    ("Bedroom 3 door", 2.85, 3.4, 0.9, "internal"),
    ("Common bathroom door", 5.0, 3.4, 0.8, "bath"),
    ("Master bathroom door", 0.95, 4.6, 0.8, "bath"),
    ("Master bedroom door", 2.65, 4.6, 0.9, "internal"),
    ("Bedroom 2 door", 4.15, 4.6, 0.9, "internal"),
]
WINDOWS = [
    ("Master bedroom window", 1.7, 8.5, 1.8, 1.0, 1.3),
    ("Bedroom 2 window", 4.5, 8.5, 1.4, 1.0, 1.3),
    ("Living window", 7.3, 8.5, 1.8, 1.0, 1.3),
    ("Dining window", 9.6, 8.5, 1.6, 1.0, 1.3),
    ("Living side window", 10.9, 5.2, 2.4, 1.0, 1.3),
    ("Service yard opening", 10.9, 1.4, 2.0, 1.0, 1.3),
    ("Kitchen window", 8.6, 0.1, 1.4, 1.2, 1.0),
    ("Bedroom 3 high window", 1.9, 0.1, 1.8, 1.5, 0.6),
    ("Master bedroom side window", 0.1, 7.2, 1.6, 1.0, 1.3),
    ("Master bathroom vent window", 0.1, 3.8, 0.6, 1.6, 0.6),
]
AC_LEDGES = [(0.7, 2.7), (3.7, 5.3)]
UNITS = {"101": (-1, 1), "103": (1, 1), "105": (1, -1), "107": (-1, -1)}


def build_legacy(storeys=12, schema="IFC4X3", seed=1, georef=None):
    """Return (writer, names, ffl) for the legacy point block, exactly as hdb_block.py built it."""
    W = IfcWriter(schema=schema, project_name="HDB Point Block Demo", site_name="Demo Site", georef=georef,
                  guid_key=f"legacy/{storeys}/{schema}/{seed}", author=("",))
    rng = random.Random(seed)
    m = W.m
    n_res = storeys - 1
    names = [f"L{i}" for i in range(1, storeys + 1)] + ["RF"]
    ffl = [0.0, VOID_DECK_FTF] + [VOID_DECK_FTF + FTF * i for i in range(1, storeys)]
    ffl = ffl[:len(names)]

    bldg = W.building("Blk 123", "HDB-style point block (demo)",
                      dict(AddressLines=["Blk 123 Sample Avenue 1"], Town="Singapore", PostalCode="560123",
                           Country="Singapore"))
    storeys_ = W.storeys(bldg, names, ffl)

    X, Y = CORE_HALF + UNIT_W, LOBBY_HALF + UNIT_D
    north_notch = box(-CORE_HALF, LOBBY_HALF + LIFT_D, CORE_HALF, Y)
    south_notch = box(-CORE_HALF, -Y, CORE_HALF, -LOBBY_HALF - STAIR_D)
    footprint = box(-X, -Y, X, Y).difference(north_notch).difference(south_notch)
    lifts = [box(-CORE_HALF, LOBBY_HALF, 0, LOBBY_HALF + LIFT_D), box(0, LOBBY_HALF, CORE_HALF, LOBBY_HALF + LIFT_D)]
    stairs = [box(-CORE_HALF, -LOBBY_HALF - STAIR_D, 0, -LOBBY_HALF), box(0, -LOBBY_HALF - STAIR_D, CORE_HALF, -LOBBY_HALF)]
    core = shapely.union_all(lifts + stairs)
    lobby = box(-X, -LOBBY_HALF, X, LOBBY_HALF)

    def T(sx, sy, x, y):
        return np.array([sx * (CORE_HALF + x), sy * (LOBBY_HALF + y)])

    def Tpoly(sx, sy, poly):
        return orient(shapely.affinity.affine_transform(poly, [sx, 0, 0, sy, sx * CORE_HALF, sy * LOBBY_HALF]), 1.0)

    def outside_ok(pt):
        p = Point(pt)
        return (not footprint.buffer(-1e-6).contains(p)) or lobby.buffer(0.01).contains(p)

    runs = [flight_run(ffl[i + 1] - ffl[i]) for i in range(len(ffl) - 1)]

    for li, (st, z) in enumerate(zip(storeys_, ffl)):
        nm = names[li]
        top = ffl[li + 1] if li + 1 < len(ffl) else None
        walls = []
        is_res = 1 <= li <= n_res
        is_roof = nm == "RF"
        sh = (top - z) if top is not None else None

        if li == 0:
            W.slab(footprint, 0.0, "L1 void deck slab", st, "BASESLAB")
        elif is_roof:
            W.slab(footprint.difference(core), z, "Roof slab", st, "ROOF", "roof")
        else:
            W.slab(footprint.difference(core), z, f"{nm} floor slab", st, "FLOOR")

        core_h = sh if not is_roof else 3.0
        for k, lp in enumerate(lifts):
            walls += ring_walls(lp, 0.2, z, core_h if not is_roof else 3.2, f"{nm}-LIFT{k + 1}", "CORE", False, st)
        for k, sp in enumerate(stairs):
            walls += ring_walls(sp, 0.2, z, core_h, f"{nm}-STAIR{k + 1}", "CORE", False, st)
        if is_roof:
            for k, lp in enumerate(lifts):
                W.slab(lp, z + 3.4, f"Lift motor room {k + 1} roof", st, "ROOF", "roof")
            for k, sp in enumerate(stairs):
                W.slab(sp, z + 3.2, f"Stair tower {k + 1} roof", st, "ROOF", "roof")

        if is_res:
            for stack, (sx, sy) in UNITS.items():
                uid = f"#{li + 1:02d}-{stack}"
                env = Tpoly(sx, sy, box(0, 0, UNIT_W, UNIT_D))
                walls += ring_walls(env, 0.2, z, sh - SLAB, f"{uid} EXT", "EXT", True, st)
                for (x0, y0, x1, y1, t, key) in PARTITIONS:
                    p0, p1 = T(sx, sy, x0, y0), T(sx, sy, x1, y1)
                    u = (p1 - p0) / np.linalg.norm(p1 - p0)
                    walls.append(WallSpec(p0, u, float(np.linalg.norm(p1 - p0)), t, -t / 2, z, sh - SLAB,
                                          f"{uid} {key}-{len(walls)}", key, False, st))
            walls += [WallSpec(np.array([X, -LOBBY_HALF]), np.array([0.0, 1.0]), 2 * LOBBY_HALF, 0.2, 0.0, z,
                               PARAPET_H, f"{nm} lobby parapet E", "PARAPET", True, st, climbable=True),
                      WallSpec(np.array([-X, LOBBY_HALF]), np.array([0.0, -1.0]), 2 * LOBBY_HALF, 0.2, 0.0, z,
                               PARAPET_H, f"{nm} lobby parapet W", "PARAPET", True, st, climbable=True)]

        if is_roof:
            roof_poly = footprint.difference(core)

            def along_core(p0, p1):
                return core.boundary.distance(Point((p0 + p1) / 2)) < 1e-6
            walls += ring_walls(roof_poly, 0.2, z, PARAPET_H, "RF parapet", "PARAPET", True, st,
                                skip=along_core, climbable=True)

        if is_res:
            for stack, (sx, sy) in UNITS.items():
                uid = f"#{li + 1:02d}-{stack}"
                for (dn, cx, cy, wd, kind) in DOORS:
                    attach(walls, T(sx, sy, cx, cy), wd, 0.0, DOOR_H, "door", f"{uid} {dn}", kind)
                for (wn, cx, cy, wd, sill, hh) in WINDOWS:
                    c = T(sx, sy, cx, cy)
                    w = None
                    for cand in walls:
                        if cand.locate(c, wd) is not None and cand.type_key == "EXT":
                            w = cand
                            break
                    if w is None:
                        continue
                    out = -w.n * 0.6
                    ends = [c + w.u * (wd / 2 - 0.05) + out, c - w.u * (wd / 2 - 0.05) + out]
                    if all(outside_ok(e) for e in ends):
                        attach([w], c, wd, sill, hh, "window", f"{uid} {wn}")
        if not is_roof:
            for k, lp in enumerate(lifts):
                cx = (lp.bounds[0] + lp.bounds[2]) / 2
                attach(walls, (cx, LOBBY_HALF + 0.1), 1.0, 0.0, DOOR_H, "door", f"{nm} lift {k + 1} landing door", "lift")
        for k, sp in enumerate(stairs):
            cx = (sp.bounds[0] + sp.bounds[2]) / 2
            attach(walls, (cx, -LOBBY_HALF - 0.1), 1.0, 0.0, DOOR_H, "door",
                   f"{nm} stair {k + 1} {'roof door' if is_roof else 'fire door'}", "roof" if is_roof else "stair")

        for w in walls:
            W.build_wall(w)
        wall_union = shapely.union_all([w.footprint() for w in walls])

        if top is not None:
            for k, sp in enumerate(stairs):
                bx0, by0, bx1, by1 = sp.bounds
                inner = (bx0 + 0.2, bx1 - 0.2, by1 - 0.2, by0 + 0.2)
                rn = runs[li + 1] if li + 1 < len(runs) else None
                stair(W, f"{nm} stair {k + 1}", inner, z, top, st, rn, guards=False)   # the baseline had none

        if li == 0:
            col = 0.6
            prof = m.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=col, YDim=col)
            k = 0
            import ifcopenshell.api.geometry as geometry
            for stack, (sx, sy) in UNITS.items():
                for (lx, ly) in [(0.3, 0.3), (5.5, 0.3), (10.7, 0.3), (0.3, 8.3), (5.5, 8.3), (10.7, 8.3)]:
                    p = T(sx, sy, lx, ly)
                    if Point(p).buffer(0.31).intersects(core):
                        continue
                    k += 1
                    rep = geometry.add_profile_representation(m, context=W.body, profile=prof, depth=sh - SLAB)
                    c = W.product("IfcColumn", f"L1 column C{k:02d}", rep, translate(p[0], p[1], 0.0), st, "COLUMN", "column")
                    W.props(c, "Pset_ColumnCommon", {"IsExternal": True, "LoadBearing": True})
            for k, lp in enumerate(lifts):
                bx0, by0, bx1, by1 = lp.bounds
                car = box(bx0 + 0.45, by0 + 0.35, bx1 - 0.45, by1 - 0.35)
                el = W.product("IfcTransportElement", f"Lift {k + 1} car", W.extrusion(car, 2.4, car.centroid.coords[0]),
                               translate(*car.centroid.coords[0], 0.05), st, "ELEVATOR", "lift")
                W.props(el, "Navigation", {"Elevator": True, "ServesLevels": ",".join(names[:-1])})
            vd = footprint.difference(core).difference(shapely.union_all([w.footprint() for w in walls]))
            W.space(vd, z, sh - SLAB, "L1-VOIDDECK", "Void deck", st, external=True)
        if is_res:
            W.space(lobby.difference(wall_union), z, sh - SLAB, f"{nm}-LOBBY", "Lift lobby / common corridor", st)

        if is_res:
            for stack, (sx, sy) in UNITS.items():
                uid = f"#{li + 1:02d}-{stack}"
                spaces = []
                for code, long_name, poly in ROOMS:
                    net = Tpoly(sx, sy, poly).difference(wall_union)
                    if net.geom_type == "MultiPolygon":
                        net = max(net.geoms, key=lambda g: g.area)
                    spaces.append(W.space(net, z, sh - SLAB, f"{uid} {code}", long_name, st, flat=uid, room=code))
                W.zone(uid, "4-room flat", spaces, {"FlatType": "4-Room", "UnitNumber": uid, "GrossArea": UNIT_W * UNIT_D})
                for (x0, x1) in AC_LEDGES:
                    led = Tpoly(sx, sy, box(x0, UNIT_D, x1, UNIT_D + 0.7))
                    el = W.slab(led, z, f"{uid} AC ledge", st, "USERDEFINED", "slab", 0.15, "AC ledge")
                    W.props(el, "Navigation", {"ClimbableTopEdge": True, "TopEdgeHeightAboveFloor": 0.0})
                    outer = Tpoly(sx, sy, box(x0, UNIT_D + 0.65, x1, UNIT_D + 0.7))
                    c = outer.centroid
                    rl = W.product("IfcRailing", f"{uid} AC ledge railing", W.extrusion(outer, 1.0, (c.x, c.y)),
                                   translate(c.x, c.y, z), st, "GUARDRAIL", "alu")
                    W.props(rl, "Navigation", {"ClimbableTopEdge": True, "TopEdgeHeightAboveFloor": 1.0})

        if is_roof:
            tank = box(5.5, 4.0, 9.5, 7.0)
            el = W.product("IfcTank", "Roof water tank", W.extrusion(tank, 2.2, tank.centroid.coords[0]),
                           translate(*tank.centroid.coords[0], z), st, "STORAGE", "tank")
            W.props(el, "Navigation", {"ClimbableTopEdge": True, "TopEdgeHeightAboveFloor": 2.2})

    ground = box(-34, -28, 34, 28).difference(footprint)
    W.product("IfcGeographicElement", "Terrain", W.extrusion(ground, 0.3, (0.0, 0.0)),
              translate(0.0, 0.0, -0.45), W.site, "TERRAIN", "grass")
    W.slab(box(4.0, -28.0, 7.0, -Y), -0.12, "Footpath to void deck", W.site, "USERDEFINED", "paving", 0.1, "Footpath")
    W.slab(box(-34, -28, 34, -21), -0.10, "Sample Avenue 1", W.site, "USERDEFINED", "asphalt", 0.1, "Road")
    from estate.ifc.vegetation import tree_types
    ttypes = tree_types(W)
    for (tx, ty) in [(-20, -17), (-12, -17), (12, -17), (20, -17), (-24, 0), (24, 0), (-20, 16), (0, 16), (20, 16),
                     (-10, 20), (10, 20)]:
        t = W.product("IfcGeographicElement", "Tree", None, frame((tx, ty, -0.15), (1, 0)), W.site)
        W.batch_type.setdefault(rng.choice(ttypes), []).append(t)

    W.flush()
    return W, names, ffl
