"""Neighbourhood centre (NC 514): an open-sided hawker centre and a two-storey shop block, authored as one
federated IFC with the estate's shared project / site identity.

Hawker centre: a column grid carries a big roof slab 6.5 m clear with a raised clerestory over the middle; two
stall islands (two back-to-back rows of 3 x 3 m stalls each) sit in the hall with roller shutters on their
fronts and a counter inside; the rest of the floor is seating (4-seat tables with fixed stools, typed and
mapped so 600 seats cost two representation maps). Toilets and the bin centre (served from Sample Street 51 on
the west) close the west end. Shop block: ground-floor shops open onto a 2.4 m covered walkway along the north
front, which is roofed by the upper-floor corridor (open, with a parapet); a stair and a lift in the east-end
core serve the upper floor, and a second stair at the west end gives the upper corridor an escape route in
either direction (no dead end; it lands on the walkway, which is open to the outside). The rest of the [nc]
rect is a paved forecourt slab, so the hawker hall's open sides, the shop fronts and the rear service doors all
step out on to a surface at the same finished floor 0.0 as everything else at ground level (step-free).

Walls come from floor plates (blocks/plate.py + blocks/builder.derive): stalls, toilets and the bin centre are
AMENITY faces, the hall floor is VOID faces (no walls between them, open to the outside), shops are ROOM faces
(party walls between units), the walkway / corridor are CORRIDOR faces (parapet on the upper floor). Every face a
person can enter, stairs included, becomes an IfcSpace, and doors record the spaces on both sides.
Stall shutters and the glazed shop doors get their own door types: ifcopenshell's door representation has no
rolling shutter, so the shutter type is a curtain between guide rails under a coil box (OperationType
ROLLINGUP), and the shop doors' leaves are styled as glass in an aluminium frame.
Building-local frame: origin at the centre of the [nc] rect, +x east, +y north.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

import ifcopenshell.api.geometry as geometry
import ifcopenshell.api.material as material
import ifcopenshell.api.root as root
import ifcopenshell.geom

from estate import config, env, guids
from estate.blocks.plate import (AMENITY, CORRIDOR, LIFT, LOBBY, OUT, PLANT, ROOM, STAIR, VOID, DoorSpec, Face, Plate,
                                 WindowSpec)
from estate.geom.walls import translate
from estate.ifc.stairs import stair
from estate.rules import DOOR_H, DOOR_KINDS, PARAPET_H, SLAB
from estate.site.mscp import (column, count_elements, derived, element_type, ensure_material, heal_entity_hash,
                              lift_car, new_writer, perimeter, place_typed, plate_walls, publish, slabs, solid_box,
                              space_lookup, style_door_types, to_estate, unshare_representations)

STALL = 3.0                     # stall module (wall centrelines)
STALL_H = 3.0                   # stall / toilet wall height (own roof slab on top)
HALL_CLEAR = 6.5                # hawker hall: clear height to the roof slab soffit
ROOF_T = 0.3
CLERESTORY = 1.2                # clear gap between the lower roof and the raised clerestory roof
TABLE_PITCH = 2.4
SHOP_FTF = (4.2, 4.0)           # ground, upper floor-to-floor
CORE_W = 5.8                    # shop block east-end core (stair 3.0 + lift 2.8)
STAIR_BAY = 3.0                 # shop block west-end escape stair (enclosure 3.0 x 6.0, as the east one)
STAIR_D = 6.0
WALK_D = 2.5                    # walkway face depth: 2.4 m clear in front of the 0.2 m shopfront wall
FORECOURT_H = 3.0               # height of the forecourt IfcSpace


# ============================================================================= door types
def _door_type(W, kind, width, height, rep, mat_key, element_type):
    """A door type registered under the writer's own cache key (IfcWriter.door_type), so every door of this kind
    and size placed afterwards is typed by it; name and GlobalId follow the writer's scheme."""
    optype = DOOR_KINDS[kind][0]
    name = f"D-{kind}-{optype}-{int(round(width * 1000))}x{int(round(height * 1000))}"
    t = root.create_entity(W.m, "IfcDoorType", name=name, predefined_type="DOOR")
    W.stable_id(t, f"IfcDoorType/{t.Name}")
    t.OperationType = optype
    t.ElementType = element_type
    geometry.assign_representation(W.m, product=t, representation=rep)
    material.assign_material(W.m, products=[t], type="IfcMaterial", material=W.mat[mat_key][0])
    W.door_types[(kind, optype, round(width, 3), round(height, 3))] = t
    return t


def shutter_type(W, width, height):
    """Roller shutter (closed): two guide rails and a 30 mm curtain under a 0.3 m coil box, all inside the opening.
    Door frame as IfcWriter places it: x along the wall from the opening start, y across the wall from 0.05 m
    short of its centre line, so y 0..0.1 stays within even a 100 mm wall."""
    box_h = 0.3
    items = [solid_box(W, 0.03, 0.05, 0.0, 0.06, 0.08, height - box_h),
             solid_box(W, width - 0.03, 0.05, 0.0, 0.06, 0.08, height - box_h),
             solid_box(W, width / 2, 0.05, 0.01, width - 0.08, 0.03, height - box_h - 0.01),   # curtain (the leaf)
             solid_box(W, width / 2, 0.05, height - box_h, width, 0.1, box_h)]
    for it in items:
        W.m.createIfcStyledItem(it, [W.mat["shutter"][1]], None)
    rep = W.m.createIfcShapeRepresentation(W.body, "Body", "SweptSolid", items)
    return _door_type(W, "shutter", width, height, rep, "shutter", "Roller shutter")


def glazed_door_type(W, kind, width, height):
    """Glazed aluminium door: ifcopenshell's door representation with the leaves (items taller than half the door
    and narrower than the lining) styled as glass, the lining, casings and threshold as the aluminium frame."""
    rep = geometry.add_door_representation(W.m, context=W.body, overall_height=height, overall_width=width,
                                           operation_type=DOOR_KINDS[kind][0])
    ensure_material(W, "frame")
    s = ifcopenshell.geom.settings()
    for it in rep.Items:
        v = np.array(ifcopenshell.geom.create_shape(s, it).verts).reshape(-1, 3)
        leaf = 0.3 < np.ptp(v[:, 0]) < width - 0.04 and np.ptp(v[:, 2]) > height / 2
        W.m.createIfcStyledItem(it, [W.mat["glass" if leaf else "frame"][1]], None)
    return _door_type(W, kind, width, height, rep, "frame", "Glazed shop door")


def own_door_types(W, plate):
    """Register the NC's own door types for the shutters and shop doors of a plate (before its walls are built)."""
    for d in plate.doors:
        key = (d.kind, DOOR_KINDS[d.kind][0], round(d.width, 3), round(d.height, 3))
        if key in W.door_types:
            continue
        if d.kind == "shutter":
            shutter_type(W, d.width, d.height)
        elif d.kind == "shop":
            glazed_door_type(W, d.kind, d.width, d.height)


# ============================================================================= hawker centre
def _local(rect, origin, inset=0.1):
    """A config rect in building-local coordinates, inset so centred walls on its edges stay inside it."""
    return (rect[0] - origin[0] + inset, rect[1] - origin[1] + inset, rect[2] - origin[0] - inset, rect[3] - origin[1] - inset)


def hawker_layout(cfg: dict, origin) -> dict:
    """Hall rect, stall islands (north / south), toilet strip and bin centre in building-local coordinates."""
    h = cfg["nc"]["hawker"]
    x0, y0, x1, y1 = _local(h["rect"], origin)
    n = int(h.get("stalls", 48))
    per_row = n // 4
    if per_row * 4 != n:
        raise ValueError("hawker stalls must be a multiple of 4 (two islands of two back-to-back rows)")
    isl_len = per_row * STALL
    west = 12.0                                         # toilets / bin zone
    if x1 - x0 < isl_len + west + 6.0 or y1 - y0 < 36.0:
        raise ValueError(f"hawker rect {h['rect']} too small for {n} stalls")
    xa = x0 + west                                      # islands right after the west zone, seating east of them
    islands = [(y1 - 15.0, y1 - 9.0), (y0 + 9.0, y0 + 15.0)]          # north, south (y ranges, 6 m deep)
    return dict(rect=(x0, y0, x1, y1), xa=xa, xb=xa + isl_len, per_row=per_row, islands=islands,
                toilets_x=(x0 + 3.5, x0 + 9.5), bin=box(x0, y0, x0 + 6.0, y0 + 6.0), seats=int(h.get("seats", 600)))


def hawker_plate(hl) -> tuple[Plate, list]:
    """Stalls, toilets, bin centre and the hall floor zones; returns (plate, stall records)."""
    x0, y0, x1, y1 = hl["rect"]
    xa, xb = hl["xa"], hl["xb"]
    (n0, n1), (s0, s1) = hl["islands"]
    tx0, tx1 = hl["toilets_x"]
    p = Plate()
    stalls = []
    k = 0
    # island rows: (y_lo, y_hi, front y, facing zone)
    rows = [(n0 + 3.0, n1, n1, "SEAT_N"), (n0, n0 + 3.0, n0, "SEAT_M"), (s0 + 3.0, s1, s1, "SEAT_M"), (s0, s0 + 3.0, s0, "SEAT_S")]
    for ya, yb, fy, zone in rows:
        for i in range(hl["per_row"]):
            k += 1
            fid = f"S{k:02d}"
            xs = xa + i * STALL
            p.add(Face(fid, box(xs, ya, xs + STALL, yb), AMENITY, name=f"Hawker stall #01-{k:02d}",
                       meta={"use": "stall", "unit": f"#01-{k:02d}"}))
            cx = xs + STALL / 2
            p.doors.append(DoorSpec(fid, zone, (cx, fy), 2.4, "shutter", zone, (cx - 1.2, fy), f"stall #01-{k:02d} shutter", 2.4))
            stalls.append(dict(fid=fid, unit=f"#01-{k:02d}", cx=cx, front=fy, north=fy == yb))
    mid0, mid1 = s1, n0
    rooms = [("TF", box(tx0, mid0, tx1, mid0 + 6.0), "Toilet (female)"),
             ("AT", box(tx0, mid0 + 6.0, tx1, mid0 + 9.0), "Accessible toilet"),
             ("TM", box(tx0, mid0 + 9.0, tx1, mid1), "Toilet (male)")]
    for fid, poly, nm in rooms:
        p.add(Face(fid, poly, AMENITY, name=nm, meta={"use": "toilet"}))
        cy = (poly.bounds[1] + poly.bounds[3]) / 2
        p.doors.append(DoorSpec(fid, "SEAT_M", (tx1, cy), 1.0, "bath", fid, (tx1, cy - 0.5), f"{nm} door", DOOR_H))
    b = hl["bin"]
    p.add(Face("BIN", b, AMENITY, name="Bin centre", meta={"use": "bin"}))
    by = (b.bounds[1] + b.bounds[3]) / 2
    p.doors.append(DoorSpec("BIN", OUT, (x0, by), 1.8, "service", "BIN", (x0, by - 0.9), "bin centre door", DOOR_H))
    zones = [("SEAT_N", box(x0, n1, x1, y1), "Seating area north"),
             ("W1", box(x0, n0, xa, n1), "Seating area north-west"),
             ("E1", box(xb, n0, x1, n1), "Seating area north-east"),
             ("WPATH", box(x0, mid0, tx0, mid1), "West passage"),
             ("SEAT_M", box(tx1, mid0, x1, mid1), "Seating area central"),
             ("W2", box(x0, s0, xa, s1), "Seating area south-west"),
             ("E2", box(xb, s0, x1, s1), "Seating area south-east"),
             ("SEAT_S", box(x0, y0, x1, s0).difference(b), "Seating area south")]
    for fid, poly, nm in zones:
        p.add(Face(fid, poly, VOID, name=nm, meta={"use": "seating"}))
    return p, stalls


def hawker_columns(hl, dp) -> list:
    x0, y0, x1, y1 = hl["rect"]
    rooms = unary_union([f.poly for f in dp.plate.faces.values() if f.kind == AMENITY]).buffer(1.0)
    pts = []
    for i in range(9):
        for j in range(7):
            x = x0 + 0.3 + i * (x1 - x0 - 0.6) / 8
            y = y0 + 0.3 + j * (y1 - y0 - 0.6) / 6
            if not rooms.intersects(Point(x, y).buffer(0.3)):
                pts.append((round(x, 3), round(y, 3)))
    return pts


def table_spots(hl, dp, cols) -> list:
    """Table centres (4 stools each) on a 2.4 m grid, clear of stall queues, rooms, columns and the hall edge."""
    x0, y0, x1, y1 = hl["rect"]
    islands = unary_union([box(hl["xa"], a, hl["xb"], b) for a, b in hl["islands"]]).buffer(3.0, join_style="mitre")
    others = unary_union([f.poly for f in dp.plate.faces.values() if f.kind == AMENITY and f.meta.get("use") != "stall"])
    keep_out = unary_union([islands, others.buffer(2.0, join_style="mitre"),
                            box(x0, hl["islands"][1][1], hl["toilets_x"][0], hl["islands"][0][0])]
                           + [Point(c).buffer(0.9) for c in cols])
    inner = box(x0, y0, x1, y1).buffer(-1.2, join_style="mitre")
    zones = [(fid, f.poly) for fid, f in sorted(dp.plate.faces.items()) if f.kind == VOID]
    out = []
    y = y0 + 2.0
    while y < y1 - 1.0:
        x = x0 + 2.0
        while x < x1 - 1.0:
            fp = box(x - 0.8, y - 0.8, x + 0.8, y + 0.8)
            if inner.contains(fp) and not keep_out.intersects(fp):
                zone = next((fid for fid, poly in zones if poly.contains(Point(x, y))), None)
                if zone:
                    out.append((zone, round(x, 3), round(y, 3)))
            x += TABLE_PITCH
        y += TABLE_PITCH
    order = {"SEAT_M": 0, "SEAT_N": 1, "SEAT_S": 2, "E1": 3, "E2": 4, "W1": 5, "W2": 6}
    return sorted(out, key=lambda t: (order.get(t[0], 9), t[2], t[1]))


def hawker_space_names(plate) -> dict:
    """Face id -> IfcSpace name: stalls by unit number, everything else 'L1-<face>'."""
    return {fid: f.meta["unit"] if f.meta.get("use") == "stall" else f"L1-{fid}" for fid, f in plate.faces.items()}


def _build_hawker(W, cfg, hl, st, zones_out, around):
    x0, y0, x1, y1 = hl["rect"]
    plate, stalls = hawker_plate(hl)
    dp = derived(plate, "ground")
    hall = box(x0, y0, x1, y1)
    W.slab(hall.buffer(0.1, join_style="mitre"), 0.0, "Hawker centre floor slab", st, "BASESLAB", "paving")
    own_door_types(W, plate)
    plate_walls(W, dp, 0.0, STALL_H, st, "Hawker", space_lookup(hawker_space_names(plate), around))
    # stall, toilet and bin-centre roofs
    (n0, n1), (s0, s1) = hl["islands"]
    for nm, poly in (("stall island north", box(hl["xa"], n0, hl["xb"], n1)),
                     ("stall island south", box(hl["xa"], s0, hl["xb"], s1)),
                     ("toilet block", unary_union([plate.faces[k].poly for k in ("TF", "AT", "TM")])),
                     ("bin centre", plate.faces["BIN"].poly)):
        W.slab(poly.buffer(0.1, join_style="mitre"), STALL_H + 0.15, f"Hawker {nm} roof", st, "ROOF", "roof", 0.15)
    # column grid and the roof with a raised clerestory over the central seating
    cols = hawker_columns(hl, dp)
    for k, (x, y) in enumerate(cols, 1):
        column(W, x, y, 0.0, HALL_CLEAR, 0.5, 0.5, st, f"Hawker column C{k:02d}")
    mid0, mid1 = hl["islands"][1][1], hl["islands"][0][0]
    cy = (mid0 + mid1) / 2
    strip = box(hl["xa"] - 4.5, cy - 3.0, hl["xb"] + 4.5, cy + 3.0)
    lower_top = HALL_CLEAR + ROOF_T
    W.slab(hall.buffer(0.1, join_style="mitre").difference(strip), lower_top, "Hawker centre roof", st, "ROOF", "roof",
           ROOF_T)
    upper = strip.buffer(0.75, join_style="mitre")
    W.slab(upper, lower_top + CLERESTORY + 0.25, "Hawker centre clerestory roof", st, "ROOF", "metal_roof", 0.25)
    sx0, sy0, sx1, sy1 = strip.bounds
    n_post = 0
    for i in range(7):
        x = sx0 + i * (sx1 - sx0) / 6
        for y in (sy0 - 0.3, sy1 + 0.3):
            n_post += 1
            column(W, x, y, lower_top, CLERESTORY, 0.3, 0.3, st, f"Hawker clerestory post P{n_post:02d}", "steel")
    # furniture: counters in the stalls, tables with fixed stools in the seating areas
    counter = element_type(W, "IfcFurnitureType", "Hawker stall counter 1.5 m", "USERDEFINED",
                           [("box", 0.0, 0.025, 0.0, 1.5, 0.55, 0.85, "tile"), ("box", 0.0, 0.0, 0.85, 1.5, 0.6, 0.05, "lift")],
                           mat="lift", element_type_name="Counter")
    table = element_type(W, "IfcFurnitureType", "Hawker table 0.8 m (4 seats)", "TABLE",
                         [("cyl", 0.0, 0.0, 0.0, 0.06, 0.71, "steel"), ("box", 0.0, 0.0, 0.71, 0.8, 0.8, 0.04, "tile")],
                         mat="tile")
    stool = element_type(W, "IfcFurnitureType", "Hawker fixed stool", "CHAIR",
                         [("cyl", 0.0, 0.0, 0.0, 0.04, 0.42, "steel"), ("cyl", 0.0, 0.0, 0.42, 0.17, 0.04, "tile")],
                         mat="tile")
    for s in stalls:        # counter along the shutter wall, 1.1 m of the 2.4 m shutter left clear to walk in
        if s["north"]:
            place_typed(W, counter, "IfcFurniture", f"Stall {s['unit']} counter", s["cx"] - 0.65, s["front"] - 0.4, 0.0, st)
        else:
            place_typed(W, counter, "IfcFurniture", f"Stall {s['unit']} counter", s["cx"] + 0.65, s["front"] + 0.4, 0.0, st,
                        u=(-1.0, 0.0))
    spots = table_spots(hl, dp, cols)
    seats = 0
    for n, (zone, x, y) in enumerate(spots, 1):
        if seats >= hl["seats"]:
            break
        place_typed(W, table, "IfcFurniture", f"{zone} table T{n:03d}", x, y, 0.0, st)
        for dx, dy in ((0.6, 0.0), (0.0, 0.6), (-0.6, 0.0), (0.0, -0.6)):
            place_typed(W, stool, "IfcFurniture", f"{zone} table T{n:03d} stool", x + dx, y + dy, 0.0, st)
            seats += 1
    # spaces and zones
    sps = []
    names = hawker_space_names(plate)
    for fid, net in sorted(dp.spaces.items()):
        f = plate.faces[fid]
        use = f.meta.get("use")
        hh = STALL_H if use in ("stall", "toilet", "bin") else HALL_CLEAR
        code = f.meta.get("unit", fid)
        sp = W.space(net, 0.0, hh, names[fid], f.name, st)
        if use == "stall":
            W.props(sp, "SampleCity_Room", {"UnitNumber": code, "RoomCode": "STALL", "Category": "commercial"})
        sps.append(sp)
    zones_out.append(W.zone("Hawker centre", cfg["nc"].get("name", "Hawker centre"), sps,
                            {"Use": "Hawker centre", "Stalls": len(stalls), "Seats": seats}, pset_name="SampleCity_Zone"))
    return dict(stalls=len(stalls), seats=seats, tables=seats // 4, columns=len(cols), clerestory_posts=n_post,
                toilets=3)


# ============================================================================= shop block
def shop_layout(cfg: dict, origin) -> dict:
    """Units, walkway line and the two cores of the shop block: the east core (stair 1, lift, lobby, M&E / refuse
    room) and the west escape stair (stair 2) at the walkway end of a 3 m bay; the first unit wraps round it."""
    s = cfg["nc"]["shops"]
    x0, y0, x1, y1 = _local(s["rect"], origin)
    storeys = int(s.get("storeys", 2))
    per_floor = int(s.get("units", 20)) // storeys
    unit_w = round((x1 - x0 - CORE_W - STAIR_BAY) / per_floor, 4)
    if unit_w < 4.0 or storeys != 2:
        raise ValueError(f"shop block {s['rect']}: {per_floor} units per floor do not fit (or storeys != 2)")
    yw = y1 - WALK_D
    cx0 = x1 - CORE_W
    ux0 = x0 + STAIR_BAY
    core = {"LOBBY": box(cx0, 0.0, x1, yw), "STAIR1": box(cx0, -6.0, cx0 + 3.0, 0.0), "LIFT": box(cx0 + 3.0, -2.5, x1, 0.0),
            "PLANT": box(cx0, y0, x1, -6.0).union(box(cx0 + 3.0, -6.0, x1, -2.5)),
            "STAIR2": box(x0, yw - STAIR_D, ux0, yw)}
    units = [box(ux0 + i * unit_w, y0, ux0 + (i + 1) * unit_w, yw) for i in range(per_floor)]
    ys = yw - STAIR_D
    units[0] = Polygon([(x0, y0), (ux0 + unit_w, y0), (ux0 + unit_w, yw), (ux0, yw), (ux0, ys), (x0, ys)])
    return dict(rect=(x0, y0, x1, y1), storeys=storeys, per_floor=per_floor, unit_w=unit_w, yw=yw, ux0=ux0, core=core,
                units=units)


def shop_space_names(sl, level: int, plate) -> dict:
    """Face id -> IfcSpace name on floor `level`: units by unit number, common faces '<storey>-<face>', the lift
    shaft and the doorless upper-floor plant room as '(shaft)' (no IfcSpace)."""
    nm = f"L{level + 1}"
    out = {}
    for fid, f in plate.faces.items():
        if f.kind == ROOM:
            out[fid] = f.flat
        elif f.kind == LIFT or (f.kind == PLANT and level > 0):
            out[fid] = f"{nm}-{fid} (shaft)"
        else:
            out[fid] = f"{nm}-{fid}"
    return out


def shop_plate(sl, level: int) -> tuple[Plate, list]:
    """Shops, walkway / corridor and the two cores of floor `level` (0 ground, 1 upper)."""
    x0, y0, x1, y1 = sl["rect"]
    yw, uw = sl["yw"], sl["unit_w"]
    p = Plate()
    walk = "WALK" if level == 0 else "CORR"
    p.add(Face(walk, box(x0, yw, x1, y1), CORRIDOR, name="Covered walkway" if level == 0 else "Upper floor corridor"))
    units = []
    glaze_w = round(uw - 2.8, 3)               # shopfront glazing between the door and the next party wall
    for i, poly in enumerate(sl["units"]):
        xa = sl["ux0"] + i * uw
        unit = f"#01-{49 + i:02d}" if level == 0 else f"#02-{1 + i:02d}"
        fid = f"U{level + 1}{i + 1:02d}"
        p.add(Face(fid, poly, ROOM, flat=unit, room="SHOP", name=f"Shop unit {unit}", meta={"category": "commercial"}))
        dx = xa + 1.25
        p.doors.append(DoorSpec(fid, walk, (dx, yw), 1.8, "shop", fid, (dx - 0.9, yw), f"shop {unit} shopfront door", 2.4))
        p.windows.append(WindowSpec(fid, (xa + 2.45 + glaze_w / 2, yw), glaze_w, 0.3, 2.4, f"shop {unit} shopfront glazing",
                                    required=True))
        if level == 0:
            p.doors.append(DoorSpec(fid, OUT, (xa + uw - 1.4, y0), 1.0, "service", fid, (xa + uw - 0.9, y0),
                                    f"shop {unit} rear service door", DOOR_H))
        else:
            p.windows.append(WindowSpec(fid, (xa + uw / 2 - 0.2, y0), 2.4, 0.9, 1.5, f"shop {unit} rear window", required=True))
        units.append((fid, unit))
    c = sl["core"]
    p.add(Face("LOBBY", c["LOBBY"], LOBBY, name="Shop block lift lobby"))
    p.add(Face("STAIR1", c["STAIR1"], STAIR, name="Shop block east stair"))
    p.add(Face("STAIR2", c["STAIR2"], STAIR, name="Shop block west stair"))
    p.add(Face("LIFT", c["LIFT"], LIFT, name="Shop block lift shaft"))
    p.add(Face("PLANT", c["PLANT"], PLANT, name="M&E / refuse room"))
    sb, lb, wb = c["STAIR1"].bounds, c["LIFT"].bounds, c["STAIR2"].bounds
    scx, lcx, wcx = (sb[0] + sb[2]) / 2, (lb[0] + lb[2]) / 2, (wb[0] + wb[2]) / 2
    p.doors.append(DoorSpec("STAIR1", "LOBBY", (scx, 0.0), 1.0, "stair", "STAIR1", (scx - 0.5, 0.0), "east stair fire door",
                            DOOR_H))
    # west stair: on the upper floor the door opens into the stair, at the walkway it opens out (escape direction)
    p.doors.append(DoorSpec("STAIR2", walk, (wcx, yw), 1.0, "stair", "STAIR2" if level else walk, (wcx - 0.5, yw),
                            "west stair fire door", DOOR_H))
    p.doors.append(DoorSpec("LIFT", "LOBBY", (lcx, 0.0), 1.0, "lift", "LIFT", (lcx - 0.5, 0.0), "lift 1 landing door",
                            DOOR_H))
    if level == 0:
        pb = c["PLANT"].bounds
        p.doors.append(DoorSpec("PLANT", OUT, ((pb[0] + pb[2]) / 2, y0), 1.5, "service", "PLANT",
                                ((pb[0] + pb[2]) / 2 - 0.75, y0), "M&E / refuse room door", DOOR_H))
        p.open(walk, OUT)               # open covered walkway at the front and both ends
        p.open("LOBBY", OUT)            # lobby entrance from the east
    return p, units


def _net(dp, fid):
    """Net floor polygon of a face; derive() leaves plant rooms out of dp.spaces, as it does lift shafts."""
    if fid in dp.spaces:
        return dp.spaces[fid]
    net = dp.plate.faces[fid].poly.difference(dp.wall_union)
    return max(getattr(net, "geoms", [net]), key=lambda g: g.area)


def _build_shops(W, cfg, sl, storeys, ffl, zones_out, names, around):
    x0, y0, x1, y1 = sl["rect"]
    c = sl["core"]
    outer = box(x0 - 0.1, y0 - 0.1, x1 + 0.1, y1 + 0.1)
    holes = unary_union([c[k].buffer(-0.1, join_style="mitre") for k in ("STAIR1", "STAIR2", "LIFT")])
    common, shops = [], 0
    for level in range(sl["storeys"]):
        st, z = storeys[level], ffl[level]
        nm = names[level]
        plate, units = shop_plate(sl, level)
        dp = derived(plate, "ground" if level == 0 else "typical")
        if level == 0:
            W.slab(outer, 0.0, "Shop block ground slab", st, "BASESLAB")
        else:
            inner = box(x0 - 0.1, y0 - 0.1, x1 + 0.1, sl["yw"]).difference(holes)
            slabs(W, inner, z, f"{nm} shop floor slab", st, "FLOOR")
            W.slab(box(x0 - 0.1, sl["yw"], x1 + 0.1, y1 + 0.1), z, f"{nm} corridor slab (walkway canopy)", st,
                   "FLOOR", "slab", SLAB, None)
        wall_h = ffl[level + 1] - z - SLAB
        own_door_types(W, plate)
        space_names = shop_space_names(sl, level, plate)
        plate_walls(W, dp, z, wall_h, st, nm, space_lookup(space_names, around if level == 0 else ()))
        if level == 0:
            for sid, label in (("STAIR1", "east"), ("STAIR2", "west")):
                sb = c[sid].bounds
                stair(W, f"{nm} shop block {label} stair", (sb[0] + 0.1, sb[2] - 0.1, sb[3] - 0.1, sb[1] + 0.1), z, ffl[1],
                      st, None, entry="N")
            lift_car(W, c["LIFT"], st, 1, names[:sl["storeys"]], "LIFT")
        hh = wall_h
        for fid, unit in units:
            sp = W.space(dp.spaces[fid], z, hh, unit, f"Shop unit {unit}", st, flat=unit, room="SHOP",
                         extra={"Category": "commercial"})
            zones_out.append(W.zone(unit, f"Shop unit {unit}", [sp],
                                    {"UnitNumber": unit, "Block": str(cfg["nc"]["blk"]), "Storey": nm,
                                     "Use": "Shop", "NetArea": round(dp.spaces[fid].area, 2)}, pset_name="SampleCity_Shop"))
            shops += 1
        walk = "WALK" if level == 0 else "CORR"
        common.append(W.space(dp.spaces[walk], z, hh, space_names[walk], plate.faces[walk].name, st, external=level == 0))
        # lobby, both stair enclosures and (ground floor, behind its service door) the M&E / refuse room
        for fid in ("LOBBY", "STAIR1", "STAIR2") + (("PLANT",) if level == 0 else ()):
            common.append(W.space(_net(dp, fid), z, hh, space_names[fid], plate.faces[fid].name, st))
    # roof
    rf, zr = storeys[-1], ffl[-1]
    W.slab(box(x0 - 0.1, y0 - 0.1, x1 + 0.1, y1 + 0.1), zr, "Shop block roof", rf, "ROOF", "roof")
    perimeter(W, (x0, y0, x1, y1), zr, PARAPET_H, rf, "RF shop block roof parapet")
    zones_out.append(W.zone("Shop block common areas", "Shop block walkway, corridor, lobbies and stairs", common,
                            {"Use": "Circulation"}, pset_name="SampleCity_Zone"))
    return dict(shops=shops)


# ============================================================================= entry point
def forecourt(cfg: dict, origin, hl, sl):
    """The [nc] rect minus the hawker hall and shop block slabs (building-local), paved at finished floor 0.0."""
    r = cfg["nc"]["rect"]
    x0, y0, x1, y1 = sl["rect"]
    site = box(r[0] - origin[0], r[1] - origin[1], r[2] - origin[0], r[3] - origin[1])
    return site.difference(box(*hl["rect"]).buffer(0.1, join_style="mitre")).difference(
        box(x0 - 0.1, y0 - 0.1, x1 + 0.1, y1 + 0.1))


def build_nc_ifc(cfg: dict | None = None, out_path=None, schema: str = "IFC4X3") -> dict:
    """Author NC 514 (hawker centre + shop block) into its own IFC (model/NC_514/NC_514.ifc by default) and publish
    its entrances next to it (<id>.build.json)."""
    cfg = cfg or config.load()
    c = cfg["nc"]
    blk = str(c["blk"])
    tid = config.target_id("nc", blk)
    out_path = Path(out_path) if out_path else env.MODEL / tid / f"{tid}.ifc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = c["rect"]
    origin = ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
    hl = hawker_layout(cfg, origin)
    sl = shop_layout(cfg, origin)
    fc = forecourt(cfg, origin, hl, sl)
    names = ["L1", "L2", "RF"]
    ffl = [0.0, SHOP_FTF[0], SHOP_FTF[0] + SHOP_FTF[1]]
    street = cfg["estate"].get("street", "")
    W = new_writer(cfg, schema, f"{cfg['estate']['code']}/{tid}")
    zones = []
    try:
        bldg = W.building(f"NC {blk}", c.get("name", f"Neighbourhood centre {blk}"),
                          dict(AddressLines=[f"Blk {blk} {street}".strip()], Town="Singapore",
                               PostalCode=f"560{blk[-3:]}", Country="Singapore"),
                          guid=guids.from_text(f"{cfg['estate']['code']}/{tid}/IfcBuilding"))
        storeys = W.storeys(bldg, names, ffl)
        # forecourt: one paving slab and one external space; ground-floor doors to OUT record it when they open on it
        slabs(W, fc, 0.0, "NC forecourt paving", storeys[0], "BASESLAB", "paving")
        W.space(fc, 0.0, FORECOURT_H, "L1-FORECOURT", "Neighbourhood centre forecourt", storeys[0], external=True)
        around = [("L1-FORECOURT", fc)]
        hinfo = _build_hawker(W, cfg, hl, storeys[0], zones, around)
        sinfo = _build_shops(W, cfg, sl, storeys, ffl, zones, names, around)
        W.props(bldg, "SampleCity_Building", {"Block": blk, "Typology": "NC", "Storeys": 2, "HawkerStalls": hinfo["stalls"],
                                             "Seats": hinfo["seats"], "Shops": sinfo["shops"]})
        style_door_types(W)
        W.flush()
        geometry.edit_object_placement(W.m, product=bldg, matrix=translate(origin[0], origin[1], 0.0),
                                       should_transform_children=True)
        unshare_representations(W)
        W.write(out_path)
        counts = count_elements(W.m)
    finally:
        W.close()
        heal_entity_hash()
    hx0, hy0, hx1, hy1 = c["hawker"]["rect"]
    sx0, sy0, sx1, sy1 = c["shops"]["rect"]
    lob = sl["core"]["LOBBY"].bounds
    lob_y = origin[1] + (lob[1] + lob[3]) / 2
    walk_y = sy1 - WALK_D / 2
    stairs = [sl["core"][k].bounds for k in ("STAIR1", "STAIR2")]
    info = {"id": tid, "path": str(out_path), "schema": schema, "levels": dict(zip(names, ffl)),
            "stalls": hinfo["stalls"], "seats": hinfo["seats"], "tables": hinfo["tables"], "shops": sinfo["shops"],
            "entrances": {"hawker_open_sides": [[(hx0 + hx1) / 2, hy1], [(hx0 + hx1) / 2, hy0], [hx1, (hy0 + hy1) / 2],
                                                [hx0, (hy0 + hy1) / 2]],
                          "shop_walkway": [[sx0, walk_y], [(sx0 + sx1) / 2, sy1], [sx1, walk_y]],
                          "shop_lobby": [[sx1, round(lob_y, 3)]],
                          "bin_centre_service": [[hx0, hy0 + 3.0]]},
            "lift_lobbies": to_estate(origin, [((lob[0] + lob[2]) / 2, (lob[1] + lob[3]) / 2)]),
            "stair_doors": to_estate(origin, [((stairs[0][0] + stairs[0][2]) / 2, stairs[0][3]),
                                              ((stairs[1][0] + stairs[1][2]) / 2, stairs[1][3])]),
            "forecourt_m2": round(fc.area, 1), "hawker": hinfo, "counts": counts}
    publish(info, out_path)
    return info
