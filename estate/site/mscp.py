"""Multi-storey car park (MSCP 513): parking decks, a switchback ramp bay, three stair cores, two lifts with a
lift lobby on every deck and a roof garden, authored as its own federated IFC.

Why this layout: across its 40 m width the rect takes exactly two 6 m aisles with four rows of 2.4 x 4.8 m lots
(31.2 m), so the aisles run north-south and the 8.4 m left over on the east side, towards the residential
precinct, carries the cores, a pedestrian walkway and motorcycle lots. Straight ramps cannot be stacked in a
single lane (each would land on the next one), so two parallel ramp lanes at the north end cross the building
east-west and switch back at turning landings: lane A rises east (L1-L2, L3-L4, ...), lane B rises west
(L2-L3, L4-L5, ...). Every ramp is 1:8 with 1:16 transitions at both ends. The lanes are enclosed by
full-height RC walls; a landing edge carries a parapet wherever its lane does not connect at that level.
Vehicles enter and leave on the south face (Link St side), pedestrians through the lift lobby on the east face.
Stairs 1 and 2 sit in the east strip; stair 3 takes the two south-west corner lots of every deck so the west
half of each deck has its own escape route. At L1 it discharges into the pedestrian bay left beside it, which
is open to the street side next to the vehicle gate.

Building-local frame: origin at the rect centre, +x east, +y north, decks every ``deck_height`` from finished
ground. The helpers at the top (derived-plate walls, straight walls, sloped prisms, typed furniture, door
FromSpace / ToSpace naming) are shared with the neighbourhood centre (centre.py).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Point, box
from shapely.ops import unary_union

import ifcopenshell
import ifcopenshell.api.aggregate as aggregate
import ifcopenshell.api.geometry as geometry
import ifcopenshell.api.material as material
import ifcopenshell.api.root as root
import ifcopenshell.api.style as style

from estate import config, env, guids
from estate.blocks.builder import derive
from estate.blocks.plate import LIFT, LOBBY, OUT, PLANT, REFUSE, STAIR, VOID, DoorSpec, Face, Plate
from estate.geom.walls import Opening, WallSpec, frame, translate
from estate.ifc.joins import joined_ends, plate_joins, run_axis, segment_joins
from estate.ifc.stairs import flight_run, stair
from estate.ifc.vegetation import tree_types
from estate.ifc.writer import IfcWriter
from estate.rules import DOOR_H, DOOR_KINDS, PALETTE, PARAPET_H, SLAB, WALL_TYPES

TOWER_H = {STAIR: 3.0, LIFT: 3.2, REFUSE: 3.0, PLANT: 3.0}

LOT_W, LOT_L, AISLE = 2.4, 4.8, 6.0
MOTO_W, MOTO_L = 1.0, 2.4
RAMP_LANE = 6.6                 # clear between lane walls: 6.0 m two-way carriageway + 0.3 m kerbs
RAMP_GRADE, TRANS_GRADE, TRANS_L = 1 / 8, 1 / 16, 2.4
RAMP_T, KERB_W, KERB_H = 0.25, 0.3, 0.15
COLUMN_BAY = 4                  # lots between columns along a row (9.6 m) x ~16 m across
GARDEN_SPECIES = [("Sea apple", 2.6, 2.0), ("Yellow flame", 3.2, 2.2), ("Tembusu (young)", 3.6, 1.8)]


_ENTITY = ifcopenshell.ifcopenshell_wrapper.entity_instance
_ENTITY_HASH = _ENTITY.__dict__.get("__hash__")


# ============================================================================= shared helpers
def heal_entity_hash():
    """estate.guids.deterministic() (entered by IfcWriter) puts entity_instance.__hash__ back as an unbound builtin,
    after which any set of entities in this process raises (EXPRESS-rule validation, later api calls). Restore the
    original method; a no-op once guids.py restores the class attribute itself."""
    if type(_ENTITY_HASH).__name__ == "instancemethod" and _ENTITY.__dict__.get("__hash__") is not _ENTITY_HASH:
        _ENTITY.__hash__ = _ENTITY_HASH


def new_writer(cfg: dict, schema: str, key: str) -> IfcWriter:
    """IfcWriter with the estate's shared IfcProject / IfcSite identity and deterministic GUIDs."""
    return IfcWriter(schema=schema, guid_key=f"{key}/{schema}", guid_base=key, typed_openings=True,
                     **config.identity(cfg))


def derived(plate: Plate, level: str):
    dp = derive(plate, level)
    if dp.errors:
        raise ValueError(f"plate ({level}): " + "; ".join(dp.errors[:8]))
    return dp


def plate_walls(W, dp, z, wall_h, storey, prefix, space_of=None):
    """Walls (with their doors and windows) of a derived plate at elevation z (as builder.build_building: walls
    are named '<prefix> <TYPE> <left face>|<right face>' and keyed by their end points for stable GlobalIds).
    space_of(face id, probe xy) names the IfcSpace on each side of a door for its Navigation FromSpace / ToSpace;
    the probe is a point 0.3 m into that side, so faces without a space of their own (OUT, open walkways) can be
    resolved against the spaces around them. The walls are joined as the residential blocks' (ifc/joins.py)."""
    def height(r):
        if r.height == "storey":
            return wall_h
        return PARAPET_H if r.height == "parapet" else TOWER_H.get(r.meta.get("core_kind"), 3.0)

    ops_by_run = {}
    for o in dp.openings:
        ops_by_run.setdefault(o.run, []).append(o)
    wall_joins = plate_joins(dp.runs, height)
    joined = joined_ends(wall_joins)
    built, walls_by_run = [], {}
    for r in dp.runs:
        if not r.wall or r.wall == "RAILING":
            continue
        h = height(r)
        u = r.u
        w = WallSpec(np.asarray(r.p0, float) - u * r.ext0, u, r.length + r.ext0 + r.ext1, r.t, -r.t / 2, z, h,
                     f"{prefix} {r.wall} {r.left}|{r.right}", r.wall, r.external, storey, r.climbable,
                     faces=(r.left, r.right),
                     meta={"key": f"{prefix}/{r.wall}/{r.p0[0]:.3f},{r.p0[1]:.3f}/{r.p1[0]:.3f},{r.p1[1]:.3f}",
                           "axis": run_axis(r, joined)})
        n = np.array([-u[1], u[0]])                    # left of p0 -> p1
        for o in sorted(ops_by_run.get(r.idx, []), key=lambda o: o.s0):
            meta = dict(o.meta, hinge=o.hinge)
            if space_of is not None and o.kind == "door":
                c = np.asarray(r.p0, float) + u * (o.s0 + o.s1) / 2
                for k in ("from", "to"):
                    if k in o.meta:
                        side = 1.0 if o.meta[k] == r.left else -1.0
                        meta[k] = space_of(o.meta[k], tuple(c + side * 0.3 * n))
            w.openings.append(Opening(o.s0 + r.ext0, o.s1 + r.ext0, o.sill, o.height, o.kind, f"{prefix} {o.name}",
                                      o.door_kind, o.swing, meta))
        built.append(W.build_wall(w))
        walls_by_run[r.idx] = built[-1][0]
    W.connect_walls(walls_by_run, wall_joins)
    return built


def space_lookup(names: dict, around=()):
    """A space_of for plate_walls: the face's own IfcSpace name from `names`, else the space of `around`
    [(name, polygon)] that contains the probe point (deck aisles, forecourt), else 'outside'."""
    def of(fid, xy):
        if fid in names:
            return names[fid]
        p = Point(xy)
        return next((nm for nm, poly in around if poly.contains(p)), "outside")
    return of


def ensure_material(W, key):
    """An IfcMaterial for a palette entry the writer keeps as a style only (rules.NO_MATERIAL), so doors of that
    palette key (glazed aluminium shop doors) carry a material; a no-op once the writer makes it one itself."""
    mt, st = W.mat[key]
    if mt is None:
        mt = material.add_material(W.m, name=PALETTE[key][0], category=key)
        style.assign_material_style(W.m, material=mt, style=st, context=W.body)
        W.mat[key] = (mt, st)
    return mt


def style_door_types(W):
    """Writer door types carry a material but unstyled items; style every item with its material's colour so
    viewers that read item styles (Datasmith, glTF export) show the door as Bonsai does."""
    styled = {si.Item.id() for si in W.m.by_type("IfcStyledItem") if si.Item is not None}
    for (kind, *_), t in W.door_types.items():
        st = W.mat[DOOR_KINDS[kind][1]][1]
        for rm in t.RepresentationMaps or []:
            for it in rm.MappedRepresentation.Items:
                if it.id() not in styled:
                    W.m.createIfcStyledItem(it, [st], None)


def publish(info: dict, out_path: Path):
    """<stem>.build.json next to the IFC: the entrances, lift lobbies and stair doors the masterplan and the site
    builder read back (model/<id>/<id>.build.json in the pipeline build). It records the IFC's path project-relative
    (env.rel), so the published file names no folder of the machine that built it; info keeps the full path."""
    Path(out_path).with_name(f"{Path(out_path).stem}.build.json").write_text(
        json.dumps(dict(info, path=env.rel(info["path"])), indent=1, default=str), encoding="utf-8")


def straight_wall(W, p, q, z, h, storey, name, type_key="PARAPET", external=True, climbable=None, defer=None):
    """A centred wall from p to q (2D points). With a `defer` list the WallSpec is collected there for
    build_joined instead of being built."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = float(np.linalg.norm(q - p))
    t = WALL_TYPES[type_key][1]
    climbable = (type_key == "PARAPET") if climbable is None else climbable
    w = WallSpec(p, (q - p) / L, L, t, -t / 2, z, h, name, type_key, external, storey, climbable)
    if defer is not None:
        defer.append(w)
        return w
    return W.build_wall(w)


def build_joined(W, specs):
    """Build free-standing walls collected by straight_wall / perimeter(defer=...) and join those that meet
    (perimeter corners, walls butting into a parapet): ifc/joins.segment_joins sets their reference lines."""
    wall_joins = segment_joins(specs)
    built = [W.build_wall(w) for w in specs]
    W.connect_walls({i: b[0] for i, b in enumerate(built)}, wall_joins)
    return built


def side_segments(a, b, skips):
    """The interval [a, b] minus the skip intervals."""
    segs = [(a, b)]
    for s0, s1 in sorted(skips):
        nxt = []
        for c, d in segs:
            if s1 <= c or s0 >= d:
                nxt.append((c, d))
                continue
            if s0 > c:
                nxt.append((c, s0))
            if s1 < d:
                nxt.append((s1, d))
        segs = nxt
    return [(c, d) for c, d in segs if d - c > 0.05]


def perimeter(W, rect, z, h, storey, name, skips=None, type_key="PARAPET", defer=None):
    """Walls centred on the edges of rect (centreline box x0, y0, x1, y1); skips = {side: [(a, b)]} along the
    side's axis (x for S/N, y for W/E). South/north sides run through the corners. `defer` as straight_wall."""
    x0, y0, x1, y1 = rect
    t2 = WALL_TYPES[type_key][1] / 2
    skips = skips or {}
    out = []
    for side, (a, b), mk in (("S", (x0 - t2, x1 + t2), lambda c, d: ((c, y0), (d, y0))),
                             ("N", (x0 - t2, x1 + t2), lambda c, d: ((c, y1), (d, y1))),
                             ("W", (y0 + t2, y1 - t2), lambda c, d: ((x0, c), (x0, d))),
                             ("E", (y0 + t2, y1 - t2), lambda c, d: ((x1, c), (x1, d)))):
        for k, (c, d) in enumerate(side_segments(a, b, skips.get(side, []))):
            p, q = mk(c, d)
            out.append(straight_wall(W, p, q, z, h, storey, f"{name} {side}{k + 1}", type_key, defer=defer))
    return out


def slabs(W, geom, top, name, storey, predefined="FLOOR", mat="slab", depth=SLAB, object_type=None):
    parts = [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon" and g.area > 0.01]
    parts.sort(key=lambda g: (round(g.bounds[0], 3), round(g.bounds[1], 3)))
    return [W.slab(g, top, name if len(parts) == 1 else f"{name} {k + 1}", storey, predefined, mat, depth, object_type)
            for k, g in enumerate(parts)]


def column(W, x, y, z, h, dx, dy, storey, name, mat="column"):
    prof = W.m.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=float(dx), YDim=float(dy))
    rep = geometry.add_profile_representation(W.m, context=W.body, profile=prof, depth=float(h))
    c = W.product("IfcColumn", name, rep, translate(x, y, z), storey, "COLUMN", mat)
    W.props(c, "Pset_ColumnCommon", {"IsExternal": False, "LoadBearing": True})
    return c


def sloped_rep(W, start_xy, direction, width, profile):
    """Body of a prism whose section `profile` [(s, z), ...] lies in the vertical plane through start_xy along
    the horizontal unit `direction`, extruded `width` to the right of that direction (ramps, kerbs)."""
    m = W.m
    R = (float(direction[0]), float(direction[1]), 0.0)
    Z = (R[1], -R[0], 0.0)                       # cross(R, up): right-hand side, local Y = up
    solid = m.createIfcExtrudedAreaSolid(m.createIfcArbitraryClosedProfileDef("AREA", None, W._curve(profile)),
                                         W._place3d(axis=Z, ref=R), m.createIfcDirection((0.0, 0.0, 1.0)), float(width))
    return m.createIfcShapeRepresentation(W.body, "Body", "SweptSolid", [solid])


def solid_box(W, cx, cy, z0, dx, dy, h):
    m = W.m
    prof = m.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=float(dx), YDim=float(dy),
                           Position=m.createIfcAxis2Placement2D(m.createIfcCartesianPoint((float(cx), float(cy)))))
    return m.createIfcExtrudedAreaSolid(prof, W._place3d((0.0, 0.0, z0)), m.createIfcDirection((0.0, 0.0, 1.0)), float(h))


def solid_cyl(W, cx, cy, z0, r, h):
    m = W.m
    prof = m.create_entity("IfcCircleProfileDef", ProfileType="AREA", Radius=float(r),
                           Position=m.createIfcAxis2Placement2D(m.createIfcCartesianPoint((float(cx), float(cy)))))
    return m.createIfcExtrudedAreaSolid(prof, W._place3d((0.0, 0.0, z0)), m.createIfcDirection((0.0, 0.0, 1.0)), float(h))


def element_type(W, cls, name, predefined, parts, mat=None, element_type_name=None, props=None):
    """A typed product (IfcFurnitureType, ...) with one RepresentationMap made of boxes and cylinders.
    parts: [("box", cx, cy, z0, dx, dy, h, palette) | ("cyl", cx, cy, z0, r, h, palette)]."""
    m = W.m
    items = []
    for p in parts:
        solid = solid_box(W, *p[1:7]) if p[0] == "box" else solid_cyl(W, *p[1:6])
        m.createIfcStyledItem(solid, [W.mat[p[-1]][1]], None)
        items.append(solid)
    t = root.create_entity(m, cls, name=name, predefined_type=predefined)
    if predefined == "USERDEFINED" and element_type_name:
        t.ElementType = element_type_name
    geometry.assign_representation(m, product=t, representation=m.createIfcShapeRepresentation(
        W.body, "Body", "SweptSolid", items))
    if mat and W.mat[mat][0] is not None:
        material.assign_material(m, products=[t], type="IfcMaterial", material=W.mat[mat][0])
    if props:
        W.props(t, props[0], props[1])
    return t


def place_typed(W, t, cls, name, x, y, z, storey, u=(1.0, 0.0), object_type=None):
    """An occurrence of type t (its representation is mapped at flush)."""
    el = W.product(cls, name, None, frame((x, y, z), u), storey, object_type=object_type)
    W.batch_type.setdefault(t, []).append(el)
    return el


def lift_car(W, shaft_poly, storey, k, levels, shaft_id):
    lb = shaft_poly.buffer(-0.1, join_style="mitre").bounds
    car = box(lb[0] + 0.25, lb[1] + 0.25, lb[2] - 0.25, lb[3] - 0.25)
    c = car.centroid
    el = W.product("IfcTransportElement", f"Lift {k} car", W.extrusion(car, 2.4, (c.x, c.y)), translate(c.x, c.y, 0.05),
                   storey, "ELEVATOR", "lift")
    W.props(el, "Navigation", {"Elevator": True, "ServesLevels": ",".join(levels), "Shaft": shaft_id})
    W.props(el, "Pset_TransportElementCommon", {"CapacityPeople": 13, "FireExit": False})
    return el


def unshare_representations(W):
    """IfcWriter (typed_openings) reuses one opening body per size; EXPRESS allows a shape representation in a
    single product definition, so every further user gets its own representation (the solids stay shared)."""
    m = W.m
    for rep in list(m.by_type("IfcShapeRepresentation")):
        users = sorted(rep.OfProductRepresentation, key=lambda e: e.id())
        for pds in users[1:]:
            new = m.createIfcShapeRepresentation(rep.ContextOfItems, rep.RepresentationIdentifier,
                                                 rep.RepresentationType, rep.Items)
            pds.Representations = [new if r.id() == rep.id() else r for r in pds.Representations]


def to_estate(origin, pts):
    return [[round(origin[0] + float(x), 3), round(origin[1] + float(y), 3)] for x, y in pts]


def count_elements(f) -> dict:
    out = {c: len(f.by_type(c)) for c in ("IfcWall", "IfcDoor", "IfcWindow", "IfcSlab", "IfcColumn", "IfcStair",
                                          "IfcStairFlight", "IfcRamp", "IfcRampFlight", "IfcRailing", "IfcSpace",
                                          "IfcZone", "IfcFurniture", "IfcTransportElement", "IfcOpeningElement",
                                          "IfcBuildingStorey", "IfcGeographicElement", "IfcBuildingElementProxy")}
    if f.schema != "IFC4":
        out["IfcKerb"] = len(f.by_type("IfcKerb"))
    out["IfcElement"] = len(f.by_type("IfcElement"))
    return out


# ============================================================================= layout
def layout(cfg: dict) -> dict:
    """Every rectangle of the car park in building-local coordinates (shared by the IFC and the drawings)."""
    c = cfg["mscp"]
    x0r, y0r, x1r, y1r = c["rect"]
    hx, hy = (x1r - x0r) / 2, (y1r - y0r) / 2
    if hx * 2 < 39.0 or hy * 2 < 80.0:
        raise ValueError(f"MSCP rect {c['rect']} is too small for the 40 x 100 m deck layout")
    H = float(c.get("deck_height", 3.0))
    xi0, xi1, yi0, yi1 = -hx + 0.2, hx - 0.2, -hy + 0.2, hy - 0.2
    # ---- across (x): R1 | A1 | R2 R3 | A2 | R4 | walkway | east strip with cores
    xs = [xi0]
    for d in (LOT_L, AISLE, LOT_L, LOT_L, AISLE, LOT_L):
        xs.append(round(xs[-1] + d, 4))
    R1, A1, R2, R3, A2, R4 = [(xs[i], xs[i + 1]) for i in range(6)]
    walk = (xs[6], round(xs[6] + 2.8, 4))
    cx0, cfac = walk[1], hx - 0.1            # core faces from the walkway to the facade wall centreline
    # ---- along (y): south lots | south cross aisle | rows | north cross aisle | ramp lanes A, B
    w3 = hy - 0.1
    w2 = round(w3 - RAMP_LANE - 0.2, 4)
    w1 = round(w2 - RAMP_LANE - 0.2, 4)
    laneA, laneB = (w1 + 0.1, w2 - 0.1), (w2 + 0.1, w3 - 0.1)
    nca = (round(w1 - 0.1 - AISLE, 4), round(w1 - 0.1, 4))
    srow = (yi0, round(yi0 + LOT_L, 4))
    n_lots = int(math.floor((nca[0] - (srow[1] + AISLE)) / LOT_W + 1e-9))
    rows_y = (round(nca[0] - n_lots * LOT_W, 4), nca[0])
    sca = (srow[1], rows_y[0])
    # ---- ramps: 1:16 transitions + 1:8 main run
    ramp_len = round(2 * TRANS_L + (H - 2 * TRANS_L * TRANS_GRADE) / RAMP_GRADE, 4)
    rx = ramp_len / 2
    if rx > xi1 - 6.0:
        raise ValueError("ramp too long for the deck width")
    # ---- cores (east strip)
    lob = (-9.5, -6.5)
    core = {
        "LOBBY": box(cx0, lob[0], cfac, lob[1]),
        "LIFT1": box(cx0, lob[1], round((cx0 + cfac) / 2, 4), lob[1] + 2.5),
        "LIFT2": box(round((cx0 + cfac) / 2, 4), lob[1], cfac, lob[1] + 2.5),
        "STAIR1": box(cx0, lob[0] - 5.6, cx0 + 2.8, lob[0]),
        "SWITCH": box(cx0 + 2.8, lob[0] - 2.8, cfac, lob[0]),
        "WALKA": box(walk[0], lob[0] - 5.6, walk[1], lob[1] + 2.5),
        "STAIR2": box(cx0, rows_y[1] - 5.6, cx0 + 2.8, rows_y[1]),
        # south-west corner, walls on the facade lines; its door opens north onto the south cross aisle
        "STAIR3": box(-hx + 0.1, -hy + 0.1, round(-hx + 2.9, 4), round(-hy + 5.7, 4)),
    }
    # the south row loses the lots the stair 3 enclosure touches; the strip left beside it is a pedestrian bay
    s3 = core["STAIR3"].bounds
    first = xi0 + math.ceil((s3[2] + 0.1 - xi0) / LOT_W - 1e-9) * LOT_W
    s3_bay = box(round(s3[2] + 0.1, 4), srow[0], round(first, 4), srow[1])
    entrance_v = (A1[0], round(A1[0] + 7.2, 4))          # two 3.6 m lanes in line with aisle A1
    return dict(H=H, hx=hx, hy=hy, xi=(xi0, xi1), yi=(yi0, yi1), R=[R1, R2, R3, R4], A=[A1, A2], walk=walk,
                strip=(walk[1], xi1), rows_y=rows_y, n_lots=n_lots, srow=srow, sca=sca, nca=nca,
                walls=(w1, w2, w3), lanes={"A": laneA, "B": laneB}, ramp_len=ramp_len, rx=rx, core=core,
                stairs=sorted(k for k in core if k.startswith("STAIR")), s_row_from=round(first, 4), s3_bay=s3_bay,
                lobby_y=lob, entrance_v=entrance_v,
                origin=((x0r + x1r) / 2, (y0r + y1r) / 2), decks=int(c.get("decks", 7)))


def ramps(lay) -> list[dict]:
    """Switchback ramps: from level i (0-based) to i+1, lane A rising east for even i, lane B rising west."""
    out = []
    for i in range(lay["decks"] - 1):
        lane = "A" if i % 2 == 0 else "B"
        east = lane == "A"
        out.append(dict(i=i, lane=lane, x_from=-lay["rx"] if east else lay["rx"], x_to=lay["rx"] if east else -lay["rx"],
                        start_end="W" if east else "E", arrive_end="E" if east else "W"))
    return out


def lane_void(lay, lane, level, rps):
    """Is the lane open (no deck slab) at this level? Yes when a ramp of the lane starts or arrives there."""
    if level == 0 or level >= lay["decks"]:
        return False
    return any(r["lane"] == lane and level in (r["i"], r["i"] + 1) for r in rps)


def lane_connected(lane, end, level, rps):
    return any(r["lane"] == lane and ((r["i"] == level and r["start_end"] == end) or
                                      (r["i"] + 1 == level and r["arrive_end"] == end)) for r in rps)


def lots(lay, level: int) -> list[tuple]:
    """Car lots of a deck as (row id, polygon); level 0 loses the lots in front of the vehicle gate, every deck
    the south-row lots at the stair 3 corner."""
    out = []
    xi0, xi1 = lay["xi"]
    n_s = int(math.floor((xi1 - xi0) / LOT_W + 1e-9))
    for j in range(n_s):
        x0 = xi0 + j * LOT_W
        if x0 < lay["s_row_from"] - 1e-6:
            continue
        if level == 0 and x0 < lay["entrance_v"][1] - 1e-6 and x0 + LOT_W > lay["entrance_v"][0] + 1e-6:
            continue
        out.append(("S", box(x0, lay["srow"][0], x0 + LOT_W, lay["srow"][1])))
    y0 = lay["rows_y"][0]
    for k, (a, b) in enumerate(lay["R"], 1):
        for j in range(lay["n_lots"]):
            out.append((f"R{k}", box(a, y0 + j * LOT_W, b, y0 + (j + 1) * LOT_W)))
    return out


def moto_lots(lay) -> list:
    xi1 = lay["xi"][1]
    c = lay["core"]
    blocked = unary_union([c["STAIR1"], c["SWITCH"], c["LOBBY"], c["LIFT1"], c["LIFT2"]]).buffer(0.1)
    y_lo, y_hi = lay["rows_y"][0], c["STAIR2"].bounds[1] - 0.6
    out, y = [], y_lo
    while y + MOTO_W <= y_hi + 1e-6:
        p = box(xi1 - MOTO_L, y, xi1, y + MOTO_W)
        if not p.intersects(blocked):
            out.append(p)
        y = round(y + MOTO_W, 4)
    return out


def deck_spaces(lay, level: int, mlots) -> list[tuple]:
    """Spaces of a parking deck as (code, polygon, long name, kind): car lots, motorcycle lots, aisles and
    pedestrian areas (the lobby and stair enclosures come from the core plate)."""
    nm = f"L{level + 1}"
    core = lay["core"]
    out = [(f"{n:03d}", poly, f"Car lot {nm}-{n:03d} ({row})", "lot") for n, (row, poly) in enumerate(lots(lay, level), 1)]
    out += [(f"M{k:02d}", poly, f"Motorcycle lot {nm}-M{k:02d}", "moto") for k, poly in enumerate(mlots, 1)]
    xi0, xi1 = lay["xi"]
    yi1 = lay["yi"][1]
    rx = lay["rx"]
    A1, A2 = lay["A"]
    ry0, ry1 = lay["rows_y"]
    s3 = core["STAIR3"].buffer(0.1, join_style="mitre")
    out += [("A1", box(A1[0], ry0, A1[1], ry1), "Drive aisle 1", "aisle"),
            ("A2", box(A2[0], ry0, A2[1], ry1), "Drive aisle 2", "aisle"),
            ("SCA", box(xi0, lay["sca"][0], xi1, lay["sca"][1]).difference(s3), "South cross aisle", "aisle"),
            ("NCA", box(xi0, lay["nca"][0], xi1, lay["nca"][1]), "North cross aisle", "aisle"),
            ("LW", box(xi0, lay["nca"][1], -rx, yi1), "Ramp landing west", "aisle"),
            ("LE", box(rx, lay["nca"][1], xi1, yi1), "Ramp landing east", "aisle"),
            ("WALK", box(lay["walk"][0], ry0, lay["walk"][1], ry1), "Pedestrian walkway", "walk"),
            ("S3BAY", lay["s3_bay"], "Stair 3 pedestrian bay", "walk")]
    strip = box(lay["strip"][0], ry0, xi1, ry1).difference(unary_union(
        [core[k] for k in ("LOBBY", "LIFT1", "LIFT2", "STAIR1", "SWITCH", "STAIR2")]).buffer(0.1)).difference(
        unary_union(mlots))
    for k, part in enumerate(sorted(getattr(strip, "geoms", [strip]), key=lambda g: g.bounds[1]), 1):
        if part.area > 1.0:
            out.append((f"MA{k}", part, f"Motorcycle aisle {k}", "aisle"))
    if level == 0:
        ev = lay["entrance_v"]
        out.append(("GATE", box(ev[0], lay["srow"][0], ev[1], lay["srow"][1]), "Vehicle entrance / exit", "aisle"))
    # aisles stop at the core wall faces, so a probe across a core door meets the stair or lobby on its far side
    walls = unary_union([g for k, g in core.items() if k != "WALKA"]).buffer(0.1, join_style="mitre")
    return [(code, poly if kind in ("lot", "moto") else poly.difference(walls), long_name, kind)
            for code, poly, long_name, kind in out]


def core_plate(lay, level: str) -> Plate:
    """Lift lobby, lift shafts, stair enclosures and switch room; level in ground | typical | roof."""
    c = lay["core"]
    p = Plate()
    roof = level == "roof"
    for k in (1, 2):
        p.add(Face(f"LIFT{k}", c[f"LIFT{k}"], LIFT, name=f"Lift {k} shaft"))
    for k, sid in enumerate(lay["stairs"], 1):
        p.add(Face(sid, c[sid], STAIR, name=f"Stair {k}"))
    if not roof:
        p.add(Face("LOBBY", c["LOBBY"], LOBBY, name="Lift lobby"))
        p.add(Face("SWITCH", c["SWITCH"], PLANT, name="Switch room"))
        p.add(Face("WALKA", c["WALKA"], VOID, name="Pedestrian walkway"))
    access = OUT if roof else "LOBBY"
    ly0, ly1 = lay["lobby_y"]
    for k in (1, 2):
        b = c[f"LIFT{k}"].bounds
        cx = round((b[0] + b[2]) / 2, 4)
        p.doors.append(DoorSpec(f"LIFT{k}", access, (cx, ly1), 1.0, "lift", f"LIFT{k}", (cx - 0.5, ly1),
                                f"lift {k} landing door", DOOR_H))
    kind = "roof" if roof else "stair"
    b = c["STAIR1"].bounds
    cx = round((b[0] + b[2]) / 2, 4)
    p.doors.append(DoorSpec("STAIR1", access, (cx, ly0), 1.0, kind, "STAIR1", (cx - 0.5, ly0),
                            f"stair 1 {'roof door' if roof else 'fire door'}", DOOR_H))
    for k in (2, 3):
        b = c[f"STAIR{k}"].bounds
        cx = round((b[0] + b[2]) / 2, 4)
        p.doors.append(DoorSpec(f"STAIR{k}", OUT, (cx, b[3]), 1.0, kind, f"STAIR{k}", (cx - 0.5, b[3]),
                                f"stair {k} {'roof door' if roof else 'fire door'}", DOOR_H))
    if level == "ground":
        # stair 3 discharges into the pedestrian bay beside it, open to the street on the south: a door in its
        # east wall on the bottom landing, where flight 1 comes down (the south end lies under the half landing)
        b = c["STAIR3"].bounds
        dy = round(lay["srow"][1] - 0.5, 4)
        p.doors.append(DoorSpec("STAIR3", OUT, (b[2], dy), 1.0, "stair", OUT, (b[2], dy + 0.5),
                                "stair 3 discharge door", DOOR_H))
    if not roof:
        b = c["SWITCH"].bounds
        cx = round((b[0] + b[2]) / 2, 4)
        p.doors.append(DoorSpec("SWITCH", "LOBBY", (cx, ly0), 1.0, "locked", "SWITCH", (cx + 0.5, ly0),
                                "switch room door", DOOR_H))
    if level == "ground":
        p.open("LOBBY", OUT)                    # pedestrian entrance from the east
    return p


def columns_at(lay) -> list[tuple]:
    """(x, y, dx, dy): ~9.6 m x 16 m grid on the row backs and facades, plus landing and south facade columns."""
    xi0, xi1 = lay["xi"]
    hy = lay["hy"]
    xw, xe = xi0 + 0.1, xi1 - 0.1
    lines = [(xw, 0.4, 0.6), (lay["R"][1][1], 0.5, 0.5), (lay["R"][3][1], 0.5, 0.5), (xe, 0.4, 0.6)]
    y0 = lay["rows_y"][0]
    ys = [round(y0 + k * COLUMN_BAY * LOT_W, 4) for k in range(lay["n_lots"] // COLUMN_BAY + 1)]
    pts = [(x, y, dx, dy) for x, dx, dy in lines for y in ys]
    pts += [(x, -hy + 0.3, dy, dx) for x, dx, dy in lines]
    pts += [(x, y, 0.4, 0.6) for x in (xw, xe) for y in (lay["walls"][0], lay["walls"][1], hy - 0.3)]
    blocked = unary_union([g for k, g in lay["core"].items() if k != "WALKA"]).buffer(0.3)
    out = []
    for p in sorted(set((round(x, 3), round(y, 3), dx, dy) for x, y, dx, dy in pts)):
        if not blocked.contains(box(p[0] - 0.01, p[1] - 0.01, p[0] + 0.01, p[1] + 0.01)):
            out.append(p)
    return out


# ============================================================================= IFC
def _ramp(W, lay, r, z0, storey, landing, names):
    H = lay["H"]
    L = lay["ramp_len"]
    ya, yb = lay["lanes"][r["lane"]]
    d = 1.0 if r["x_to"] > r["x_from"] else -1.0
    zt = TRANS_L * TRANS_GRADE
    pts = [(0.0, z0), (TRANS_L, z0 + zt), (L - TRANS_L, z0 + H - zt), (L, z0 + H)]
    start = (r["x_from"], yb if d > 0 else ya)
    name = f"Ramp {names[r['i']]}-{names[r['i'] + 1]} (lane {r['lane']})"
    el = W.product("IfcRamp", name, None, translate(r["x_from"], ya, z0), storey, "STRAIGHT_RUN_RAMP")
    W.props(el, "Pset_RampCommon", {"RequiredSlope": math.atan(RAMP_GRADE), "HandicapAccessible": False,
                                    "IsExternal": False, "FireExit": False, "HasNonSkidSurface": True})
    W.props(el, "Navigation", {"Ramp": True, "Vehicular": True, "MaxGradient": RAMP_GRADE,
                               "FromLevel": names[r["i"]], "ToLevel": names[r["i"] + 1]})
    parts = []
    for k, (a, b) in enumerate(zip(pts[:-1], pts[1:]), 1):
        prof = [a, b, (b[0], b[1] - RAMP_T), (a[0], a[1] - RAMP_T)]
        fl = W.product("IfcRampFlight", f"{name} flight {k}", sloped_rep(W, start, (d, 0.0), yb - ya, prof),
                       translate(start[0], start[1], 0.0), None, "STRAIGHT", "slab")
        g = (b[1] - a[1]) / (b[0] - a[0])
        W.props(fl, "Pset_RampFlightCommon", {"Slope": math.atan(g), "ClearWidth": round(yb - ya - 2 * KERB_W, 3)})
        W.props(fl, "Navigation", {"Gradient": round(g, 4), "Transition": k != 2})
        parts.append(fl)
    kprof = pts + [(s, z + KERB_H) for s, z in reversed(pts)]
    cls, pre, ot = ("IfcKerb", None, None) if W.schema != "IFC4" else ("IfcBuildingElementProxy", "USERDEFINED", "Kerb")
    for side, (ka, kb) in (("north", (yb - KERB_W, yb)), ("south", (ya, ya + KERB_W))):
        ks = (r["x_from"], kb if d > 0 else ka)
        kerb = W.product(cls, f"{name} kerb {side}", sloped_rep(W, ks, (d, 0.0), KERB_W, kprof),
                         translate(ks[0], ks[1], 0.0), None, pre, "kerb", ot)
        if W.schema != "IFC4":
            W.props(kerb, "Pset_KerbCommon", {"Upstand": KERB_H, "Mountable": False})
        parts.append(kerb)
    if landing is not None:
        parts.append(landing)
    aggregate.assign_object(W.m, products=parts, relating_object=el)
    return el


def _garden(W, lay, z, st):
    """Roof garden: lawn panels, perimeter planters, trees on the lawns and benches along the promenade."""
    xi0, xi1 = lay["xi"]
    yi0, yi1 = lay["yi"]
    A1, A2 = lay["A"]
    # lawns between the perimeter path and the promenade, cut by cross paths
    lawn_x = [(xi0 + 2.8, A1[1] + 3.0), (A1[1] + 7.0, A2[1] + 2.4)]
    y_cuts = [(yi0 + 9.8, -17.0), (-15.0, 7.0), (9.0, lay["nca"][1] - 3.0)]
    tt = tree_types(W, GARDEN_SPECIES)
    bench = element_type(W, "IfcFurnitureType", "Garden bench 1.8 m", "USERDEFINED",
                         [("box", 0.0, 0.0, 0.40, 1.8, 0.45, 0.06, "timber"),
                          ("box", -0.8, 0.0, 0.0, 0.08, 0.4, 0.40, "steel"),
                          ("box", 0.8, 0.0, 0.0, 0.08, 0.4, 0.40, "steel"),
                          ("box", 0.0, 0.2, 0.46, 1.8, 0.05, 0.40, "timber")], mat="timber", element_type_name="Bench")
    n_tree = n_bench = 0
    lawns = []
    for i, (xa, xb) in enumerate(lawn_x):
        for j, (ya, yb) in enumerate(y_cuts):
            p = box(xa, ya, xb, yb)
            lawns.append(p)
            el = W.box_element("IfcGeographicElement", f"RF roof garden lawn {i + 1}.{j + 1}", p, z, 0.1, st,
                               "TERRAIN", "grass")
            W.props(el, "Navigation", {"Walkable": True})
            for f in (0.25, 0.75):
                ty = ya + (yb - ya) * f
                t = tt[(n_tree) % len(tt)]
                place_typed(W, t, "IfcGeographicElement", "Roof garden tree", (xa + xb) / 2, ty, z + 0.1, st)
                n_tree += 1
    # benches facing the lawns along the central promenade (between the two lawn strips)
    px0, px1 = lawn_x[0][1], lawn_x[1][0]
    for (ya, yb) in y_cuts:
        for f in (0.3, 0.7):
            y = ya + (yb - ya) * f
            place_typed(W, bench, "IfcFurniture", "Roof garden bench", px0 + 0.5, y, z, st, u=(0.0, -1.0))
            place_typed(W, bench, "IfcFurniture", "Roof garden bench", px1 - 0.5, y, z, st, u=(0.0, 1.0))
            n_bench += 2
    # planters along the west, south and north parapets (gaps keep the perimeter path open), kept 1.5 m clear of
    # the stair towers so their roof doors open onto the path
    keep_out = unary_union([lay["core"][k] for k in lay["stairs"]]).buffer(1.5, join_style="mitre")
    planters = []
    for k in range(8):
        y = yi0 + 4.0 + k * 11.8
        planters.append(box(xi0, y, xi0 + 1.2, min(y + 9.8, yi1 - 4.0)))
    for x0 in (xi0 + 4.0, xi0 + 16.0):
        planters.append(box(x0, yi0, x0 + 9.8, yi0 + 1.2))
        planters.append(box(x0, yi1 - 1.2, x0 + 9.8, yi1))
    trimmed = []
    for p in planters:
        if p.intersects(keep_out):
            rest = [g for g in getattr(p.difference(keep_out), "geoms", [p.difference(keep_out)]) if g.area > 2.0]
            if not rest:
                continue
            p = box(*max(rest, key=lambda g: g.area).bounds)
        trimmed.append(p)
    planters = trimmed
    for k, p in enumerate(planters, 1):
        el = W.box_element("IfcBuildingElementProxy", f"RF planter {k:02d}", p, z, 0.6, st, "USERDEFINED", "accent",
                           "Planter")
        W.props(el, "Navigation", {"ClimbableTopEdge": True, "TopEdgeHeightAboveFloor": 0.6})
    return dict(lawns=len(lawns), trees=n_tree, benches=n_bench, planters=len(planters))


def _markings(W, lay, z, st, name, gate=None):
    """Lot separator lines of one deck as a single proxy (one item per line keeps the element count low)."""
    m = W.m
    items = []
    xi0, xi1 = lay["xi"]
    n_s = int(math.floor((xi1 - xi0) / LOT_W + 1e-9))
    for j in range(n_s + 1):
        x = xi0 + j * LOT_W
        if gate is not None and gate[0] - 1e-6 <= x <= gate[1] + 1e-6:
            continue
        if x < lay["s_row_from"] - 1e-6:          # stair 3 enclosure and its pedestrian bay
            continue
        items.append(solid_box(W, xi0 + j * LOT_W, lay["srow"][0] + LOT_L / 2 + 0.1, z, 0.1, LOT_L - 0.2, 0.005))
    y0 = lay["rows_y"][0]
    for a, b in lay["R"]:
        for j in range(lay["n_lots"] + 1):
            items.append(solid_box(W, (a + b) / 2, y0 + j * LOT_W, z, LOT_L - 0.2, 0.1, 0.005))
    rep = m.createIfcShapeRepresentation(W.body, "Body", "SweptSolid", items)
    el = W.product("IfcBuildingElementProxy", name, rep, translate(0.0, 0.0, 0.0), st, "USERDEFINED", "marking",
                   "Parking bay marking")
    return el


def build_mscp_ifc(cfg: dict | None = None, out_path=None, schema: str = "IFC4X3") -> dict:
    """Author MSCP 513 into its own IFC (model/MSCP_513/MSCP_513.ifc by default). Returns an info dict."""
    cfg = cfg or config.load()
    c = cfg["mscp"]
    blk = str(c["blk"])
    tid = config.target_id("mscp", blk)
    out_path = Path(out_path) if out_path else env.MODEL / tid / f"{tid}.ifc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lay = layout(cfg)
    H, decks = lay["H"], lay["decks"]
    hx, hy = lay["hx"], lay["hy"]
    xi0, xi1 = lay["xi"]
    yi0, yi1 = lay["yi"]
    names = [f"L{i + 1}" for i in range(decks)] + ["RF"]
    ffl = [round(i * H, 4) for i in range(decks + 1)]
    rps = ramps(lay)
    plates = {lvl: derived(core_plate(lay, lvl), lvl) for lvl in ("ground", "typical", "roof")}
    core = lay["core"]
    w1, w2, w3 = lay["walls"]
    rx = lay["rx"]
    street = cfg["estate"].get("street", "")

    W = new_writer(cfg, schema, f"{cfg['estate']['code']}/{tid}")
    try:
        bldg = W.building(f"MSCP {blk}", c.get("name", f"Multi-storey car park {blk}"),
                          dict(AddressLines=[f"Blk {blk} {street}".strip()], Town="Singapore",
                               PostalCode=f"560{blk[-3:]}", Country="Singapore"),
                          guid=guids.from_text(f"{cfg['estate']['code']}/{tid}/IfcBuilding"))
        storeys = W.storeys(bldg, names, ffl)
        runs_next = [flight_run(ffl[i + 1] - ffl[i]) for i in range(len(ffl) - 1)]
        shafts = ["LIFT1", "LIFT2"] + lay["stairs"]
        holes = unary_union([core[k].buffer(-0.1, join_style="mitre") for k in shafts])
        towers = unary_union([core[k].buffer(0.1, join_style="mitre") for k in shafts])
        footprint = box(-hx, -hy, hx, hy)
        landing_zone = {"W": box(-hx, w1 - 0.1, -rx, hy), "E": box(rx, w1 - 0.1, hx, hy)}
        arriving = {r["i"] + 1: r for r in rps}
        lots_per_deck, moto_total, n_cols = {}, 0, 0
        stair_risers = {}
        zones = []
        mlots = moto_lots(lay)
        cols = columns_at(lay)
        s3 = core["STAIR3"].bounds

        for li, (st, z) in enumerate(zip(storeys, ffl)):
            nm = names[li]
            roof = nm == "RF"
            level = "ground" if li == 0 else "roof" if roof else "typical"
            dp = plates[level]
            spaces = []
            # ---- slabs (landings that a ramp arrives at go into that ramp's aggregate)
            landing_slab = None
            if li == 0:
                W.slab(footprint, 0.0, "L1 ground deck slab", st, "BASESLAB")
            elif roof:
                slabs(W, footprint.difference(holes), z, "Roof garden slab", st, "ROOF", "roof")
            else:
                g = footprint.difference(holes)
                for lane in ("A", "B"):
                    if lane_void(lay, lane, li, rps):
                        ya, yb = lay["lanes"][lane]
                        g = g.difference(box(-rx, ya, rx, yb))
                if li in arriving:
                    lz = landing_zone[arriving[li]["arrive_end"]]
                    g = g.difference(lz)
                    landing_slab = W.slab(lz, z, f"{nm} ramp landing {arriving[li]['arrive_end']}", None, "LANDING")
                slabs(W, g, z, f"{nm} deck slab", st, "FLOOR")
            # ---- the ramp arriving here (contained in the storey below) takes this level's landing slab
            if li in arriving:
                r = arriving[li]
                _ramp(W, lay, r, ffl[li - 1], storeys[li - 1], landing_slab, names)

            # ---- the storey's spaces, named before the walls so doors can record FromSpace / ToSpace
            sh = 2.8 if not roof else 2.4
            if roof:
                deck = [("GARDEN", box(xi0, yi0, xi1, yi1).difference(towers), "Roof garden", "external")]
            else:
                deck = deck_spaces(lay, li, mlots)
            own = {} if roof else {"LOBBY": f"{nm}-LOBBY"}
            own.update({sid: f"{nm}-{sid}" for sid in lay["stairs"]})
            own.update({k: f"{nm}-{k} (shaft)" for k in ("LIFT1", "LIFT2", "SWITCH")})
            space_of = space_lookup(own, [(f"{nm}-{code}", poly) for code, poly, _, _ in deck])

            wall_h = (ffl[li + 1] - z - SLAB) if not roof else 3.0
            # ---- cores
            plate_walls(W, dp, z, wall_h, st, nm, space_of)
            # ---- perimeter parapets (none where a core wall stands on the facade line)
            skips = {"W": [(s3[1] - 0.2, s3[3] + 0.1)], "S": [(s3[0] - 0.2, s3[2] + 0.1)]}
            if roof:
                skips["E"] = [(core["LIFT2"].bounds[1], core["LIFT2"].bounds[3])]
            else:
                skips.update({"N": [(-rx, rx)], "E": [(core["SWITCH"].bounds[1], core["LIFT2"].bounds[3])]})
                if li == 0:          # the stair 3 bay (pedestrian exit) and the vehicle gate open on to the street
                    skips["S"] += [(s3[2] + 0.1, lay["s_row_from"]), lay["entrance_v"]]
            loose = []
            perimeter(W, (-hx + 0.1, -hy + 0.1, hx - 0.1, hy - 0.1), z, PARAPET_H, st, f"{nm} parapet", skips,
                      defer=loose)
            if not roof:
                # ramp lane walls (full storey height) and landing-edge parapets where a lane is not connected
                for y, nmw in ((w1, "lane A south wall"), (w2, "ramp lane spine wall"), (w3, "lane B north wall")):
                    straight_wall(W, (-rx, y), (rx, y), z, wall_h, st, f"{nm} {nmw}", "EXT", climbable=False,
                                  defer=loose)
                for lane, (ya, yb) in sorted(lay["lanes"].items()):
                    for end, x in (("W", -rx - 0.1), ("E", rx + 0.1)):
                        if not lane_connected(lane, end, li, rps):
                            straight_wall(W, (x, ya - 0.2), (x, yb + 0.2), z, PARAPET_H, st,
                                          f"{nm} landing parapet lane {lane} {end}", defer=loose)
            build_joined(W, loose)
            if not roof:
                # columns
                for k, (x, y, dx, dy) in enumerate(cols, 1):
                    column(W, x, y, z, wall_h, dx, dy, st, f"{nm} column C{k:02d}")
                    n_cols += 1
                _markings(W, lay, z, st, f"{nm} parking bay markings", lay["entrance_v"] if li == 0 else None)
            # ---- stairs to the next level
            if not roof:
                for k, sid in enumerate(lay["stairs"], 1):
                    x0, y0, x1, y1 = core[sid].bounds
                    rn = runs_next[li + 1] if li + 1 < len(runs_next) else None
                    stair_risers[f"{nm} stair {k}"] = stair(W, f"{nm} stair {k}", (x0 + 0.1, x1 - 0.1, y1 - 0.1, y0 + 0.1),
                                                            z, ffl[li + 1], st, rn, entry="N")
            if li == 0:
                for k in (1, 2):
                    lift_car(W, core[f"LIFT{k}"], st, k, names, f"LIFT{k}")
            if roof:
                for k, sid in enumerate(lay["stairs"], 1):
                    W.slab(core[sid].buffer(0.1, join_style="mitre"), z + 3.2, f"Stair {k} tower roof", st, "ROOF", "roof")
                for k in (1, 2):
                    W.slab(core[f"LIFT{k}"].buffer(0.1, join_style="mitre"), z + 3.4, f"Lift {k} motor room roof", st,
                           "ROOF", "roof")
            # ---- spaces
            for code, poly, long_name, kind in deck:
                if kind in ("lot", "moto"):
                    sp = W.space(poly, z, sh, f"{nm}-{code}", long_name, st, predefined="PARKING")
                    W.props(sp, "Pset_SpaceParking", {"ParkingUse": "Car" if kind == "lot" else "Motorcycle",
                                                      "ParkingUnits": 1, "IsAisle": False})
                elif kind == "aisle":
                    sp = W.space(poly, z, sh, f"{nm}-{code}", long_name, st, predefined="PARKING")
                    W.props(sp, "Pset_SpaceParking", {"ParkingUse": "Car", "IsAisle": True, "IsOneWay": False})
                else:
                    sp = W.space(poly, z, sh, f"{nm}-{code}", long_name, st, external=kind == "external")
                spaces.append(sp)
            if not roof:
                lots_per_deck[nm] = sum(kind == "lot" for *_, kind in deck)
                moto_total += len(mlots)
                spaces.append(W.space(dp.spaces["LOBBY"], z, sh, f"{nm}-LOBBY", "Lift lobby", st))
            for k, sid in enumerate(lay["stairs"], 1):      # stair enclosures (engine portals need both sides)
                spaces.append(W.space(dp.spaces[sid], z, 3.0 if roof else wall_h, f"{nm}-{sid}", f"Stair {k}", st))
            if roof:
                garden = _garden(W, lay, z, st) if c.get("roof_garden", True) else {}
            zones.append(W.zone(f"MSCP {blk} {nm}", f"{'Roof garden' if roof else 'Parking deck ' + nm}", spaces,
                                {"Block": blk, "Storey": nm, "Use": "Roof garden" if roof else "Car park deck",
                                 "CarLots": lots_per_deck.get(nm, 0)}, pset_name="SampleCity_Zone"))

        total = sum(lots_per_deck.values())
        W.props(bldg, "SampleCity_Building", {"Block": blk, "Typology": "MSCP", "Storeys": decks,
                                             "CarLots": total, "MotorcycleLots": moto_total})
        style_door_types(W)
        W.flush()
        origin = lay["origin"]
        geometry.edit_object_placement(W.m, product=bldg, matrix=translate(origin[0], origin[1], 0.0),
                                       should_transform_children=True)
        unshare_representations(W)
        W.write(out_path)
        counts = count_elements(W.m)
    finally:
        W.close()
        heal_entity_hash()
    ly0, ly1 = lay["lobby_y"]
    ped = (hx, (ly0 + ly1) / 2)
    ev = lay["entrance_v"]
    veh = ((ev[0] + ev[1]) / 2, -hy)
    lob = core["LOBBY"].centroid
    sd = [((core[k].bounds[0] + core[k].bounds[2]) / 2, ly0 if k == "STAIR1" else core[k].bounds[3]) for k in lay["stairs"]]
    info = {"id": tid, "path": str(out_path), "schema": schema, "levels": dict(zip(names, ffl)),
            "car_lots": total, "car_lots_per_deck": lots_per_deck, "motorcycle_lots": moto_total,
            "entrances": {"pedestrian": to_estate(origin, [ped]), "vehicle": to_estate(origin, [veh]),
                          "vehicle_gate_width": round(ev[1] - ev[0], 2)},
            "lift_lobbies": to_estate(origin, [(lob.x, lob.y)]),
            "stair_doors": to_estate(origin, sd),
            # escape route, not an entrance: where the stair 3 bay meets the street side (south face)
            "stair_discharge": to_estate(origin, [((s3[2] + 0.1 + lay["s_row_from"]) / 2, -hy)]),
            "ramps": [dict(lane=r["lane"], frm=names[r["i"]], to=names[r["i"] + 1], length=lay["ramp_len"],
                           gradient=RAMP_GRADE, transition=TRANS_GRADE) for r in rps],
            "stairs": {k: dict(risers=v[0], riser=round(v[1], 4)) for k, v in stair_risers.items()},
            "columns": n_cols, "garden": garden, "counts": counts}
    publish(info, out_path)
    return info
