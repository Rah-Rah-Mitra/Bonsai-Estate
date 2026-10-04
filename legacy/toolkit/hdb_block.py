#!/usr/bin/env python3
"""
hdb_block.py - one realistic, walkable HDB-style point block as an IFC (BIM) model.

Real layout
-----------
Every residential floor has four mirrored 4-room flats (~95 m2 gross, inside HDB's 85-95 m2 range):
living/dining, foyer, kitchen + service yard, household shelter, hall, three bedrooms, master and
common bathrooms - each an IfcSpace grouped into an IfcZone per flat ("#05-101"). Flats open onto a
naturally ventilated lift lobby with two lifts and two common staircases. Ground level is a void deck.
The roof has parapets, stair towers with roof doors, lift motor rooms and a water tank.

Design rules used (Singapore Approved Document, BCA)
---------------------------------------------------
risers <= 175 mm (uniform per flight), goings >= 275 mm on common stairs, <= 18 risers per flight,
stair and landing clear width >= 1.0 m, headroom >= 2.0 m, barriers/sills >= 1.0 m, ceilings >= 2.4 m
(HDB uses 2.6 m, so floor-to-floor is 2.8 m with a 200 mm slab).

Walkable
--------
Doors are real openings (>= 0.8 m) with door leaves you can strip or animate; floor slabs are cut at
stair cores; every storey and the roof are linked by stairs; lifts are shafts with closed landing
doors (swap for elevator logic in an engine). Pset "Navigation" marks passable doors and climbable
edges (parapets, AC ledges) for export.

Usage
-----
    python hdb_block.py -o hdb_block.ifc                    # 12 storeys, IFC4X3
    python hdb_block.py --storeys 16 --schema IFC4 -o b.ifc # IFC4 for Unreal's Datasmith IFC importer
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box
from shapely.geometry.polygon import orient

import ifcopenshell
import ifcopenshell.api.aggregate as aggregate
import ifcopenshell.api.context as context
import ifcopenshell.api.feature as feature
import ifcopenshell.api.geometry as geometry
import ifcopenshell.api.georeference as georeference
import ifcopenshell.api.group as group
import ifcopenshell.api.material as material
import ifcopenshell.api.project as project
import ifcopenshell.api.pset as pset
import ifcopenshell.api.root as root
import ifcopenshell.api.spatial as spatial
import ifcopenshell.api.style as style
import ifcopenshell.api.type as ifctype
import ifcopenshell.api.unit as unit
import ifcopenshell.util.element as uel

# ----------------------------------------------------------------------------- design rules
SLAB = 0.20
FTF = 2.80                      # 2.6 m ceiling + slab
VOID_DECK_FTF = 3.60
RISER_MAX = 0.175               # Approved Document: riser <= 175 mm
GOING = 0.28                    # common stairs: going >= 275 mm
MAX_RISERS_PER_FLIGHT = 18
FLIGHT_W = 1.15                 # >= 1.0 m clear
WELL = 0.20
LANDING = 1.30                  # >= 1.0 m
WAIST = 0.15
DOOR_H = 2.10
PARAPET_H = 1.10                # barrier >= 1.0 m

UNIT_W, UNIT_D = 11.0, 8.6      # flat envelope (x from core, y from lobby)
CORE_HALF = 2.9                 # half width of the central core strip
LOBBY_HALF = 1.5                # 3.0 m lift lobby / corridor
LIFT_D = 2.7                    # lift shaft outer depth
STAIR_D = 5.8                   # stair enclosure outer depth (5.4 m clear)

# ----------------------------------------------------------------------------- 4-room flat template
# local coordinates: x runs from the core side (0) to the end facade (11.0); y runs from the lobby (0)
# to the outer facade (8.6). Envelope walls are 200 mm inside these lines.
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
# internal walls: (x0, y0, x1, y1, thickness, wall type). Household shelter walls are 200 mm RC.
PARTITIONS = [
    (5.8, 0.2, 5.8, 1.7, 0.2, "HS"), (4.2, 0.2, 4.2, 1.7, 0.2, "HS"), (4.1, 1.6, 5.9, 1.6, 0.2, "HS"),
    (5.8, 1.7, 5.8, 3.45, 0.1, "INT"),   # common bath | foyer, living
    (4.2, 1.7, 4.2, 3.45, 0.1, "INT"),   # bedroom 3 | common bath
    (5.8, 4.55, 5.8, 8.4, 0.1, "INT"),   # bedroom 2 | living  (gap 3.45-4.55 = hall archway)
    (7.35, 2.8, 10.8, 2.8, 0.1, "INT"),  # kitchen, service yard | living (foyer is open to living)
    (7.4, 0.2, 7.4, 2.85, 0.1, "INT"),   # foyer | kitchen
    (9.8, 0.2, 9.8, 2.85, 0.1, "INT"),   # kitchen | service yard
    (1.55, 3.4, 5.85, 3.4, 0.1, "INT"),  # bedroom 3, common bath | hall
    (0.2, 2.4, 1.65, 2.4, 0.1, "INT"),   # bedroom 3 | master bath
    (1.6, 2.35, 1.6, 4.65, 0.1, "INT"),  # master bath | bedroom 3, hall
    (0.2, 4.6, 5.85, 4.6, 0.1, "INT"),   # bedrooms | hall, master bath
    (3.2, 4.65, 3.2, 8.4, 0.1, "INT"),   # master bedroom | bedroom 2
]
# doors: (name, centre x, centre y, clear width, kind)
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
# windows: (name, centre x, centre y, width, sill, height). Skipped automatically if they would open
# onto another enclosed space; windows facing the common lobby are allowed (as in real HDB blocks).
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
AC_LEDGES = [(0.7, 2.7), (3.7, 5.3)]     # x-ranges on the outer facade, 0.7 m deep
UNITS = {"101": (-1, 1), "103": (1, 1), "105": (1, -1), "107": (-1, -1)}   # stack: (sx, sy)

PALETTE = {  # key: (name, rgb, transparency)
    "ext": ("Painted RC - HDB off-white", (0.93, 0.91, 0.86), 0.0),
    "int": ("Lightweight panel - white", (0.96, 0.95, 0.92), 0.0),
    "hs": ("RC - household shelter", (0.80, 0.79, 0.76), 0.0),
    "core": ("Painted RC - core accent", (0.62, 0.70, 0.62), 0.0),
    "slab": ("RC slab", (0.70, 0.70, 0.68), 0.0),
    "roof": ("Roof waterproofing", (0.42, 0.43, 0.45), 0.0),
    "stair": ("RC stair - granolithic", (0.78, 0.76, 0.72), 0.0),
    "steel": ("Mild steel - painted", (0.25, 0.27, 0.30), 0.0),
    "timber": ("Timber door", (0.58, 0.40, 0.25), 0.0),
    "blast": ("Steel blast door", (0.55, 0.57, 0.58), 0.0),
    "lift": ("Stainless steel", (0.78, 0.80, 0.82), 0.0),
    "glass": ("Glass", (0.55, 0.74, 0.85), 0.6),
    "frame": ("Aluminium frame", (0.80, 0.81, 0.82), 0.0),
    "alu": ("Aluminium railing", (0.74, 0.76, 0.78), 0.0),
    "column": ("RC column", (0.85, 0.83, 0.78), 0.0),
    "grass": ("Grass", (0.42, 0.60, 0.33), 0.0),
    "paving": ("Concrete paving", (0.74, 0.72, 0.68), 0.0),
    "asphalt": ("Asphalt", (0.22, 0.22, 0.24), 0.0),
    "tank": ("Water tank - GRP", (0.45, 0.62, 0.78), 0.0),
    "space": ("Space", (0.95, 0.90, 0.55), 0.85),
    "bark": ("Bark", (0.40, 0.28, 0.18), 0.0),
    "foliage": ("Foliage", (0.24, 0.48, 0.22), 0.0),
}
WALL_TYPES = {  # key: (type name, thickness, palette key, predefined type)
    "EXT": ("EXT-200 RC painted", 0.20, "ext", "SOLIDWALL"),
    "INT": ("INT-100 lightweight partition", 0.10, "int", "PARTITIONING"),
    "HS": ("HS-200 RC household shelter", 0.20, "hs", "SHEAR"),
    "CORE": ("CORE-200 RC lift / stair core", 0.20, "core", "SHEAR"),
    "PARAPET": ("PAR-200 RC parapet", 0.20, "ext", "PARAPET"),
}
DOOR_KINDS = {  # kind: (operation type, palette key, passable for people, fire rating)
    "main": ("SINGLE_SWING_LEFT", "timber", True, None),
    "internal": ("SINGLE_SWING_LEFT", "timber", True, None),
    "bath": ("SINGLE_SWING_LEFT", "timber", True, None),
    "shelter": ("SINGLE_SWING_LEFT", "blast", True, None),
    "stair": ("SINGLE_SWING_LEFT", "steel", True, "1 hr"),
    "roof": ("SINGLE_SWING_LEFT", "steel", True, None),
    "lift": ("DOUBLE_DOOR_SLIDING", "lift", False, "1 hr"),
}


# ----------------------------------------------------------------------------- helpers
def frame(origin, u) -> np.ndarray:
    x = np.array([u[0], u[1], 0.0])
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, 1.0])
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, np.cross(z, x), z, origin
    return m


def translate(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


@dataclass
class Opening:
    s0: float
    s1: float
    sill: float
    height: float
    kind: str           # "door" | "window"
    name: str
    door_kind: str = ""


@dataclass
class WallSpec:
    p0: np.ndarray
    u: np.ndarray
    length: float
    t: float
    lo: float           # local-y start of the thickness band (0 = left aligned, -t/2 = centred)
    z0: float
    h: float
    name: str
    type_key: str
    external: bool
    storey: object
    climbable: bool = False
    openings: list = field(default_factory=list)

    @property
    def n(self):
        return np.array([-self.u[1], self.u[0]])

    def footprint(self) -> Polygon:
        a, b = self.p0 + self.n * self.lo, self.p0 + self.n * (self.lo + self.t)
        e = self.u * self.length
        return Polygon([a, a + e, b + e, b])

    def locate(self, c, width):
        """Return the along-wall position of point c if an opening of `width` centred there fits."""
        d = np.asarray(c, float) - self.p0
        a, b = float(d @ self.u), float(d @ self.n)
        if a - width / 2 < -1e-6 or a + width / 2 > self.length + 1e-6:
            return None
        if not (self.lo - 0.06 <= b <= self.lo + self.t + 0.06):
            return None
        return a, abs(b - (self.lo + self.t / 2))


def ring_walls(poly: Polygon, t, z0, h, name, type_key, external, storey, skip=None, climbable=False):
    """Walls along a polygon's exterior, thickness inside, butt-jointed corners (orthogonal rings)."""
    pts = [np.array(p) for p in list(orient(poly, 1.0).exterior.coords)[:-1]]
    pts = [p for i, p in enumerate(pts)
           if abs(np.cross(np.append(p - pts[i - 1], 0), np.append(pts[(i + 1) % len(pts)] - p, 0))[2]) > 1e-9]
    out, n = [], len(pts)
    for i in range(n):
        p0, p1, prev = pts[i], pts[(i + 1) % n], pts[i] - pts[i - 1]
        if skip is not None and skip(p0, p1):
            continue
        seg = p1 - p0
        L = float(np.linalg.norm(seg))
        u, up = seg / L, prev / np.linalg.norm(prev)
        adj = t if (up[0] * u[1] - up[1] * u[0]) > 0 else -t   # trim convex, extend reflex corners
        out.append(WallSpec(p0 + u * adj, u, L - adj, t, 0.0, z0, h, f"{name}-W{i + 1:02d}", type_key,
                            external, storey, climbable))
    return out


# ----------------------------------------------------------------------------- builder
class BlockBuilder:
    def __init__(self, a):
        self.a = a
        self.rng = random.Random(a.seed)
        self.schema = a.schema
        self.m = project.create_file(version=a.schema)
        self.batch_contain, self.batch_mat, self.batch_type = {}, {}, {}
        self.counts = {}
        self.centre_walls = []
        self._setup()

    # ------------------------------------------------------------------ setup
    def _setup(self):
        m = self.m
        self.project = root.create_entity(m, "IfcProject", name="HDB Point Block Demo")
        unit.assign_unit(m, units=[unit.add_si_unit(m, unit_type=t) for t in
                                   ("LENGTHUNIT", "AREAUNIT", "VOLUMEUNIT", "PLANEANGLEUNIT")])
        model3d = context.add_context(m, context_type="Model")
        self.body = context.add_context(m, "Model", "Body", "MODEL_VIEW", parent=model3d)
        self.axis = context.add_context(m, "Model", "Axis", "GRAPH_VIEW", parent=model3d)
        if self.a.georef:
            georeference.add_georeferencing(m, ifc_class="IfcMapConversion", name="EPSG:3414")
            georeference.edit_georeferencing(m, projected_crs={"Name": "EPSG:3414"}, coordinate_operation={
                "Eastings": self.a.eastings, "Northings": self.a.northings, "OrthogonalHeight": 0.0})
        self.site = root.create_entity(m, "IfcSite", name="Demo Site")
        aggregate.assign_object(m, products=[self.site], relating_object=self.project)
        geometry.edit_object_placement(m, product=self.site)

        self.mat = {}
        for key, (name, rgb, tr) in PALETTE.items():
            st = style.add_style(m, name=name)
            style.add_surface_style(m, style=st, ifc_class="IfcSurfaceStyleShading", attributes={
                "SurfaceColour": {"Name": None, "Red": rgb[0], "Green": rgb[1], "Blue": rgb[2]},
                "Transparency": tr})
            mt = None
            if key not in ("space", "bark", "foliage", "frame"):
                mt = material.add_material(m, name=name, category=key)
                style.assign_material_style(m, material=mt, style=st, context=self.body)
            self.mat[key] = (mt, st)

        self.wall_types = {}
        for key, (name, t, pkey, pre) in WALL_TYPES.items():
            wt = root.create_entity(m, "IfcWallType", name=name, predefined_type=pre)
            ls = material.add_material_set(m, name=name, set_type="IfcMaterialLayerSet")
            layer = material.add_layer(m, layer_set=ls, material=self.mat[pkey][0])
            material.edit_layer(m, layer=layer, attributes={"LayerThickness": t})
            material.assign_material(m, products=[wt], type="IfcMaterialLayerSet", material=ls)
            self.wall_types[key] = wt

    # ------------------------------------------------------------------ primitives
    def _n(self, cls):
        self.counts[cls] = self.counts.get(cls, 0) + 1

    def contain(self, structure, el):
        self.batch_contain.setdefault(structure, []).append(el)

    def use_material(self, key, el):
        self.batch_mat.setdefault(key, []).append(el)

    def props(self, product, name, values):
        p = pset.add_pset(self.m, product=product, name=name)
        pset.edit_pset(self.m, pset=p, properties=values)

    def _curve(self, coords):
        pts = [self.m.createIfcCartesianPoint((float(x), float(y))) for x, y in coords]
        return self.m.createIfcPolyline(pts + [pts[0]])

    def _profile(self, poly: Polygon):
        poly = orient(poly, 1.0)
        outer = self._curve(list(poly.exterior.coords)[:-1])
        if poly.interiors:
            return self.m.createIfcArbitraryProfileDefWithVoids(
                "AREA", None, outer, [self._curve(list(r.coords)[:-1]) for r in poly.interiors])
        return self.m.createIfcArbitraryClosedProfileDef("AREA", None, outer)

    def _place3d(self, loc=(0.0, 0.0, 0.0), axis=None, ref=None):
        m = self.m
        return m.createIfcAxis2Placement3D(
            m.createIfcCartesianPoint(tuple(float(v) for v in loc)),
            m.createIfcDirection(tuple(float(v) for v in axis)) if axis is not None else None,
            m.createIfcDirection(tuple(float(v) for v in ref)) if ref is not None else None)

    def extrusion(self, poly: Polygon, depth, origin_xy):
        local = shapely.affinity.translate(poly, -origin_xy[0], -origin_xy[1])
        solid = self.m.createIfcExtrudedAreaSolid(self._profile(local), self._place3d(),
                                                  self.m.createIfcDirection((0.0, 0.0, 1.0)), float(depth))
        return self.m.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", [solid])

    def product(self, cls, name, rep, matrix, container, predefined=None, mat=None, object_type=None):
        el = root.create_entity(self.m, cls, name=name, predefined_type=predefined)
        if object_type:
            el.ObjectType = object_type
        geometry.edit_object_placement(self.m, product=el, matrix=matrix)
        if rep is not None:
            geometry.assign_representation(self.m, product=el, representation=rep)
        if container is not None:
            self.contain(container, el)
        if mat:
            self.use_material(mat, el)
        self._n(cls)
        return el

    def slab(self, poly, top, name, container, predefined="FLOOR", mat="slab", depth=SLAB, object_type=None):
        c = poly.centroid
        el = self.product("IfcSlab", name, self.extrusion(poly, depth, (c.x, c.y)),
                          translate(c.x, c.y, top - depth), container, predefined, mat, object_type)
        return el

    # ------------------------------------------------------------------ walls, doors, windows
    def build_wall(self, w: WallSpec):
        M = frame((w.p0[0], w.p0[1], w.z0), w.u)
        rep = geometry.add_wall_representation(self.m, context=self.body, length=w.length, height=w.h,
                                               thickness=w.t, offset=w.lo)
        pre = WALL_TYPES[w.type_key][3]
        wall = self.product("IfcWall", w.name, rep, M, w.storey, predefined=pre)
        geometry.assign_representation(self.m, product=wall, representation=geometry.add_axis_representation(
            self.m, context=self.axis, axis=[(0.0, 0.0), (w.length, 0.0)]))
        self.batch_type.setdefault(self.wall_types[w.type_key], []).append(wall)
        if w.lo != 0.0:
            self.centre_walls.append((wall, w.lo))
        self.props(wall, "Pset_WallCommon", {"IsExternal": w.external, "LoadBearing": w.type_key in ("HS", "CORE")})
        if w.climbable:
            self.props(wall, "Navigation", {"ClimbableTopEdge": True, "TopEdgeHeightAboveFloor": round(w.h, 2)})
        for op in w.openings:
            self.build_opening(wall, w, M, op)
        return wall

    def build_opening(self, wall, w: WallSpec, M, op: Opening):
        width = op.s1 - op.s0
        oe = root.create_entity(self.m, "IfcOpeningElement", name=f"{op.name} opening")
        geometry.edit_object_placement(self.m, product=oe, matrix=M @ translate(op.s0, w.lo - 0.1, op.sill))
        geometry.assign_representation(self.m, product=oe, representation=geometry.add_wall_representation(
            self.m, context=self.body, length=width, height=op.height, thickness=w.t + 0.2))
        feature.add_feature(self.m, feature=oe, element=wall)
        if op.kind == "door":
            optype, pkey, passable, fire = DOOR_KINDS[op.door_kind]
            rep = geometry.add_door_representation(self.m, context=self.body, overall_height=op.height,
                                                   overall_width=width, operation_type=optype)
            el = self.product("IfcDoor", op.name, rep, M @ translate(op.s0, w.lo + w.t / 2 - 0.05, op.sill),
                              w.storey, predefined="DOOR", mat=pkey)
            el.OperationType = optype
            el.OverallWidth, el.OverallHeight = width, op.height
            common = {"IsExternal": w.external, "HandicapAccessible": width >= 0.85}
            if fire:
                common["FireRating"] = fire
            self.props(el, "Pset_DoorCommon", common)
            self.props(el, "Navigation", {"Passable": passable, "Interactive": True, "DoorKind": op.door_kind,
                                          "ClearWidth": round(width, 3)})
        else:
            rep = geometry.add_window_representation(self.m, context=self.body, overall_height=op.height,
                                                     overall_width=width)
            for item in rep.Items:
                if item.SweptArea.is_a("IfcArbitraryProfileDefWithVoids"):
                    self.m.createIfcStyledItem(item, [self.mat["frame"][1]], None)
            el = self.product("IfcWindow", op.name, rep, M @ translate(op.s0, w.lo + w.t / 2 - 0.0875, op.sill),
                              w.storey, predefined="WINDOW", mat="glass")
            el.OverallWidth, el.OverallHeight = width, op.height
            self.props(el, "Pset_WindowCommon", {"IsExternal": True})
        feature.add_filling(self.m, opening=oe, element=el)

    def attach(self, walls, centre, width, sill, height, kind, name, door_kind=""):
        best = None
        for w in walls:
            r = w.locate(centre, width)
            if r is not None and (best is None or r[1] < best[1]):
                best = (w, r[0], r[1])
        if best is None:
            raise ValueError(f"no wall found for {name} at {centre}")
        w, a, _ = best
        w.openings.append(Opening(a - width / 2, a + width / 2, sill, height, kind, name, door_kind))
        return w

    # ------------------------------------------------------------------ stairs
    def flight(self, name, x_lo, x_hi, y_start, direction, z0, n, r, container):
        """Stepped RC flight: n risers of height r, (n-1) goings, running north (+1) or south (-1)."""
        m = self.m
        dv = WAIST * math.sqrt(1 + (r / GOING) ** 2)
        prof = [(0.0, 0.0), (0.0, r)]
        for i in range(1, n):
            prof += [(i * GOING, i * r), (i * GOING, (i + 1) * r)]
        prof += [((n - 1) * GOING, (n - 1) * r - dv), (dv * GOING / r, 0.0)]
        R = np.array([0.0, float(direction), 0.0])
        Z = np.cross(R, [0.0, 0.0, 1.0])
        x_side = x_lo if Z[0] > 0 else x_hi
        solid = m.createIfcExtrudedAreaSolid(
            m.createIfcArbitraryClosedProfileDef("AREA", None, self._curve(prof)),
            self._place3d(axis=Z, ref=R), m.createIfcDirection((0.0, 0.0, 1.0)), float(x_hi - x_lo))
        rep = m.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", [solid])
        el = self.product("IfcStairFlight", name, rep, translate(x_side, y_start, z0), None, "STRAIGHT", "stair")
        el.NumberOfRisers, el.NumberOfTreads = n, n - 1
        el.RiserHeight, el.TreadLength = round(r, 4), GOING
        self.props(el, "Pset_StairFlightCommon", {"NumberOfRiser": n, "NumberOfTreads": n - 1,
                                                  "RiserHeight": round(r, 4), "TreadLength": GOING})
        # balustrade on the well side, 900 mm above the pitch line
        inner = x_lo if x_side == x_hi else x_hi          # well-side edge of this flight
        t = 0.04
        rail_side = inner + t if Z[0] < 0 else inner - t
        rail_prof = [(0.0, r - 0.3), ((n - 1) * GOING, n * r - 0.3), ((n - 1) * GOING, n * r + 0.9), (0.0, r + 0.9)]
        rsolid = m.createIfcExtrudedAreaSolid(
            m.createIfcArbitraryClosedProfileDef("AREA", None, self._curve(rail_prof)),
            self._place3d(axis=Z, ref=R), m.createIfcDirection((0.0, 0.0, 1.0)), t)
        rail = self.product("IfcRailing", name + " balustrade",
                            m.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", [rsolid]),
                            translate(rail_side, y_start, z0), None, "BALUSTRADE", "steel")
        return el, rail

    def stair(self, name, inner, z_lo, z_hi, storey, run_next, with_lower_landing=False):
        """Half-turn (dog-leg) common stair inside an enclosure. inner = (x0, x1, y_north, y_south)."""
        x0, x1, yN, yS = inner
        depth = yN - yS
        d_mid = depth - LANDING
        h = z_hi - z_lo
        N = math.ceil(h / RISER_MAX - 1e-9)
        r = h / N
        N1, N2 = math.ceil(N / 2), N - math.ceil(N / 2)
        assert max(N1, N2) <= MAX_RISERS_PER_FLIGHT and (N1 - 1) * GOING <= d_mid - LANDING + 1e-6, name
        run1, run2 = (N1 - 1) * GOING, (N2 - 1) * GOING
        st = self.product("IfcStair", name, None, translate(x0, yS, z_lo), storey, "HALF_TURN_STAIR")
        self.props(st, "Pset_StairCommon", {"NumberOfRiser": N, "RiserHeight": round(r, 4), "TreadLength": GOING,
                                            "RequiredHeadroom": 2.0, "FireExit": True, "IsExternal": False})
        parts = []
        z_mid = z_lo + N1 * r
        east, west = (x1 - FLIGHT_W, x1), (x0, x0 + FLIGHT_W)
        # flight 1: east half, from the floor landing southwards up to the mid landing
        parts += self.flight(f"{name} flight 1", *east, yN - (d_mid - run1), -1, z_lo, N1, r, storey)
        parts.append(self.slab(box(x0, yS, x1, yN - d_mid), z_mid, f"{name} mid landing", None, "LANDING", "stair"))
        # flight 2: west half, from the mid landing northwards up to the next floor
        parts += self.flight(f"{name} flight 2", *west, yN - d_mid, 1, z_mid, N2, r, storey)
        # upper floor landing: reaches the arrival of flight 2 and the departure of the next flight 1
        reach_w = d_mid - run2
        reach_e = d_mid - run_next if run_next is not None else reach_w
        land = box(west[0], yN - reach_w, west[1], yN).union(box(east[0], yN - reach_e, east[1], yN)).union(
            box(x0, yN - min(reach_w, reach_e), x1, yN))
        parts.append(self.slab(land, z_hi, f"{name} floor landing", None, "LANDING", "stair"))
        aggregate.assign_object(self.m, products=parts, relating_object=st)
        return N, r

    @staticmethod
    def flight_run(h):
        N = math.ceil(h / RISER_MAX - 1e-9)
        return (math.ceil(N / 2) - 1) * GOING

    # ------------------------------------------------------------------ trees (typed, low poly)
    def tree_types(self):
        m = self.m
        self.ttypes = []
        p = (1 + 5 ** 0.5) / 2
        base = [(-1, p, 0), (1, p, 0), (-1, -p, 0), (1, -p, 0), (0, -1, p), (0, 1, p), (0, -1, -p), (0, 1, -p),
                (p, 0, -1), (p, 0, 1), (-p, 0, -1), (-p, 0, 1)]
        faces0 = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
                  (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
                  (6, 2, 10), (8, 6, 7), (9, 8, 1)]
        for tname, h, rad in [("Rain tree (young)", 3.0, 2.6), ("Angsana", 3.6, 3.2)]:
            v = [np.array(x, float) / np.linalg.norm(x) for x in base]
            cache, faces = {}, []

            def mid(a, b):
                k = (min(a, b), max(a, b))
                if k not in cache:
                    c = (v[a] + v[b]) / 2
                    v.append(c / np.linalg.norm(c))
                    cache[k] = len(v) - 1
                return cache[k]
            for a, b, c in faces0:
                ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
                faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
            pts = [m.createIfcCartesianPoint(tuple(float(q) for q in x * rad + np.array([0, 0, h + rad * 0.7])))
                   for x in v]
            crown = m.createIfcFacetedBrep(m.createIfcClosedShell([m.createIfcFace([m.createIfcFaceOuterBound(
                m.createIfcPolyLoop([pts[i] for i in f]), True)]) for f in faces]))
            octo = [(0.25 * math.cos(t), 0.25 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
            trunk = m.createIfcExtrudedAreaSolid(m.createIfcArbitraryClosedProfileDef("AREA", None, self._curve(octo)),
                                                 self._place3d(), m.createIfcDirection((0.0, 0.0, 1.0)), h)
            m.createIfcStyledItem(trunk, [self.mat["bark"][1]], None)
            m.createIfcStyledItem(crown, [self.mat["foliage"][1]], None)
            pre, ot = ("VEGETATION", None) if self.schema == "IFC4X3" else ("USERDEFINED", "Vegetation")
            tt = root.create_entity(m, "IfcGeographicElementType", name=tname, predefined_type=pre)
            if ot:
                tt.ElementType = ot
            geometry.assign_representation(m, product=tt, representation=m.createIfcShapeRepresentation(
                self.body, "Body", "SolidModel", [trunk, crown]))
            self.ttypes.append(tt)

    # ------------------------------------------------------------------ the block
    def build(self):
        a = self.a
        m = self.m
        n_res = a.storeys - 1
        names = [f"L{i}" for i in range(1, a.storeys + 1)] + ["RF"]
        ffl = [0.0, VOID_DECK_FTF] + [VOID_DECK_FTF + FTF * i for i in range(1, a.storeys)]
        ffl = ffl[:len(names)]

        bldg = root.create_entity(m, "IfcBuilding", name="Blk 123")
        bldg.LongName = "HDB-style point block (demo)"
        bldg.BuildingAddress = m.create_entity("IfcPostalAddress", AddressLines=["Blk 123 Sample Avenue 1"],
                                               Town="Singapore", PostalCode="560123", Country="Singapore")
        aggregate.assign_object(m, products=[bldg], relating_object=self.site)
        geometry.edit_object_placement(m, product=bldg)
        storeys = []
        for nm, z in zip(names, ffl):
            s = root.create_entity(m, "IfcBuildingStorey", name=nm)
            s.Elevation = z
            geometry.edit_object_placement(m, product=s, matrix=translate(0, 0, z))
            storeys.append(s)
            self._n("IfcBuildingStorey")
        aggregate.assign_object(m, products=storeys, relating_object=bldg)

        # plan geometry (world, building centred on origin)
        X, Y = CORE_HALF + UNIT_W, LOBBY_HALF + UNIT_D
        north_notch = box(-CORE_HALF, LOBBY_HALF + LIFT_D, CORE_HALF, Y)
        south_notch = box(-CORE_HALF, -Y, CORE_HALF, -LOBBY_HALF - STAIR_D)
        footprint = box(-X, -Y, X, Y).difference(north_notch).difference(south_notch)
        lifts = [box(-CORE_HALF, LOBBY_HALF, 0, LOBBY_HALF + LIFT_D), box(0, LOBBY_HALF, CORE_HALF, LOBBY_HALF + LIFT_D)]
        stairs = [box(-CORE_HALF, -LOBBY_HALF - STAIR_D, 0, -LOBBY_HALF), box(0, -LOBBY_HALF - STAIR_D, CORE_HALF, -LOBBY_HALF)]
        core = shapely.union_all(lifts + stairs)
        lobby = box(-X, -LOBBY_HALF, X, LOBBY_HALF)
        self.plan = dict(footprint=footprint, lifts=lifts, stairs=stairs, lobby=lobby)

        def T(sx, sy, x, y):
            return np.array([sx * (CORE_HALF + x), sy * (LOBBY_HALF + y)])

        def Tpoly(sx, sy, poly):
            return orient(shapely.affinity.affine_transform(
                poly, [sx, 0, 0, sy, sx * CORE_HALF, sy * LOBBY_HALF]), 1.0)

        def outside_ok(pt):  # a window may open to outdoor air or onto the common lobby
            p = Point(pt)
            return (not footprint.buffer(-1e-6).contains(p)) or lobby.buffer(0.01).contains(p)

        runs = [self.flight_run(ffl[i + 1] - ffl[i]) for i in range(len(ffl) - 1)]

        for li, (st, z) in enumerate(zip(storeys, ffl)):
            nm = names[li]
            top = ffl[li + 1] if li + 1 < len(ffl) else None
            walls: list[WallSpec] = []
            is_res = 1 <= li <= n_res
            is_roof = nm == "RF"
            sh = (top - z) if top is not None else None

            # ---- slabs
            if li == 0:
                self.slab(footprint, 0.0, "L1 void deck slab", st, "BASESLAB")
            elif is_roof:
                self.slab(footprint.difference(core), z, "Roof slab", st, "ROOF", "roof")
            else:
                self.slab(footprint.difference(core), z, f"{nm} floor slab", st, "FLOOR")

            # ---- cores: lift shafts and stair enclosures (full storey height; no slab inside)
            core_h = sh if not is_roof else 3.0
            for k, lp in enumerate(lifts):
                walls += ring_walls(lp, 0.2, z, core_h if not is_roof else 3.2, f"{nm}-LIFT{k + 1}", "CORE", False, st)
            for k, sp in enumerate(stairs):
                walls += ring_walls(sp, 0.2, z, core_h, f"{nm}-STAIR{k + 1}", "CORE", False, st)
            if is_roof:
                for k, lp in enumerate(lifts):
                    self.slab(lp, z + 3.4, f"Lift motor room {k + 1} roof", st, "ROOF", "roof")
                for k, sp in enumerate(stairs):
                    self.slab(sp, z + 3.2, f"Stair tower {k + 1} roof", st, "ROOF", "roof")

            # ---- flats
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
                # lobby end parapets
                walls += [WallSpec(np.array([X, -LOBBY_HALF]), np.array([0.0, 1.0]), 2 * LOBBY_HALF, 0.2, 0.0, z,
                                   PARAPET_H, f"{nm} lobby parapet E", "PARAPET", True, st, climbable=True),
                          WallSpec(np.array([-X, LOBBY_HALF]), np.array([0.0, -1.0]), 2 * LOBBY_HALF, 0.2, 0.0, z,
                                   PARAPET_H, f"{nm} lobby parapet W", "PARAPET", True, st, climbable=True)]

            # ---- roof parapet (skip edges that run along the stair towers / lift rooms)
            if is_roof:
                roof_poly = footprint.difference(core)

                def along_core(p0, p1):
                    mid = Point((p0 + p1) / 2)
                    return core.boundary.distance(mid) < 1e-6
                walls += ring_walls(roof_poly, 0.2, z, PARAPET_H, "RF parapet", "PARAPET", True, st,
                                    skip=along_core, climbable=True)

            # ---- openings: flat doors and windows
            if is_res:
                for stack, (sx, sy) in UNITS.items():
                    uid = f"#{li + 1:02d}-{stack}"
                    for (dn, cx, cy, wd, kind) in DOORS:
                        self.attach(walls, T(sx, sy, cx, cy), wd, 0.0, DOOR_H, "door", f"{uid} {dn}", kind)
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
                            self.attach([w], c, wd, sill, hh, "window", f"{uid} {wn}")
            # lift landing doors and stair doors on the lobby side
            if not is_roof:
                for k, lp in enumerate(lifts):
                    cx = (lp.bounds[0] + lp.bounds[2]) / 2
                    self.attach(walls, (cx, LOBBY_HALF + 0.1), 1.0, 0.0, DOOR_H, "door", f"{nm} lift {k + 1} landing door", "lift")
            for k, sp in enumerate(stairs):
                cx = (sp.bounds[0] + sp.bounds[2]) / 2
                self.attach(walls, (cx, -LOBBY_HALF - 0.1), 1.0, 0.0, DOOR_H, "door",
                            f"{nm} stair {k + 1} {'roof door' if is_roof else 'fire door'}", "roof" if is_roof else "stair")

            # ---- create walls (+ openings)
            for w in walls:
                self.build_wall(w)
            wall_union = shapely.union_all([w.footprint() for w in walls])

            # ---- stairs up to the next level
            if top is not None:
                for k, sp in enumerate(stairs):
                    bx0, by0, bx1, by1 = sp.bounds
                    inner = (bx0 + 0.2, bx1 - 0.2, by1 - 0.2, by0 + 0.2)
                    rn = runs[li + 1] if li + 1 < len(runs) else None
                    self.stair(f"{nm} stair {k + 1}", inner, z, top, st, rn)

            # ---- void deck columns, lift cars, site-level spaces
            if li == 0:
                col = 0.6
                prof = m.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=col, YDim=col)
                k = 0
                for stack, (sx, sy) in UNITS.items():
                    for (lx, ly) in [(0.3, 0.3), (5.5, 0.3), (10.7, 0.3), (0.3, 8.3), (5.5, 8.3), (10.7, 8.3)]:
                        p = T(sx, sy, lx, ly)
                        if Point(p).buffer(0.31).intersects(core):
                            continue
                        k += 1
                        rep = geometry.add_profile_representation(m, context=self.body, profile=prof, depth=sh - SLAB)
                        c = self.product("IfcColumn", f"L1 column C{k:02d}", rep, translate(p[0], p[1], 0.0), st,
                                         "COLUMN", "column")
                        self.props(c, "Pset_ColumnCommon", {"IsExternal": True, "LoadBearing": True})
                for k, lp in enumerate(lifts):
                    bx0, by0, bx1, by1 = lp.bounds
                    car = box(bx0 + 0.45, by0 + 0.35, bx1 - 0.45, by1 - 0.35)
                    el = self.product("IfcTransportElement", f"Lift {k + 1} car", self.extrusion(car, 2.4, car.centroid.coords[0]),
                                      translate(*car.centroid.coords[0], 0.05), st, "ELEVATOR", "lift")
                    self.props(el, "Navigation", {"Elevator": True, "ServesLevels": ",".join(names[:-1])})
                vd = footprint.difference(core).difference(shapely.union_all([w.footprint() for w in walls]))
                self.space(vd, z, sh - SLAB, "L1-VOIDDECK", "Void deck", st, external=True)
            if is_res:
                self.space(lobby.difference(wall_union), z, sh - SLAB, f"{nm}-LOBBY", "Lift lobby / common corridor", st)

            # ---- rooms (IfcSpace) grouped per flat (IfcZone); AC ledges
            if is_res:
                for stack, (sx, sy) in UNITS.items():
                    uid = f"#{li + 1:02d}-{stack}"
                    zone = root.create_entity(m, "IfcZone", name=uid)
                    zone.LongName = "4-room flat"
                    spaces = []
                    for code, long_name, poly in ROOMS:
                        net = Tpoly(sx, sy, poly).difference(wall_union)
                        if net.geom_type == "MultiPolygon":
                            net = max(net.geoms, key=lambda g: g.area)
                        sp = self.space(net, z, sh - SLAB, f"{uid} {code}", long_name, st, flat=uid, room=code)
                        spaces.append(sp)
                    group.assign_group(m, products=spaces, group=zone)
                    self.props(zone, "SampleCity_Flat", {"FlatType": "4-Room", "UnitNumber": uid,
                                                         "GrossArea": UNIT_W * UNIT_D})
                    self._n("IfcZone")
                    for (x0, x1) in AC_LEDGES:
                        led = Tpoly(sx, sy, box(x0, UNIT_D, x1, UNIT_D + 0.7))
                        pre, ot = "USERDEFINED", "AC ledge"
                        el = self.slab(led, z, f"{uid} AC ledge", st, pre, "slab", 0.15, ot)
                        self.props(el, "Navigation", {"ClimbableTopEdge": True, "TopEdgeHeightAboveFloor": 0.0})
                        # 1.0 m railing on the outer edge
                        outer = Tpoly(sx, sy, box(x0, UNIT_D + 0.65, x1, UNIT_D + 0.7))
                        c = outer.centroid
                        rl = self.product("IfcRailing", f"{uid} AC ledge railing", self.extrusion(outer, 1.0, (c.x, c.y)),
                                          translate(c.x, c.y, z), st, "GUARDRAIL", "alu")
                        self.props(rl, "Navigation", {"ClimbableTopEdge": True, "TopEdgeHeightAboveFloor": 1.0})

            # ---- roof extras
            if is_roof:
                tank = box(5.5, 4.0, 9.5, 7.0)
                el = self.product("IfcTank", "Roof water tank", self.extrusion(tank, 2.2, tank.centroid.coords[0]),
                                  translate(*tank.centroid.coords[0], z), st, "STORAGE", "tank")
                self.props(el, "Navigation", {"ClimbableTopEdge": True, "TopEdgeHeightAboveFloor": 2.2})

        # ---- site: ground, footpath, road, trees
        ground = box(-34, -28, 34, 28).difference(footprint)
        self.product("IfcGeographicElement", "Terrain", self.extrusion(ground, 0.3, (0.0, 0.0)),
                     translate(0.0, 0.0, -0.45), self.site, "TERRAIN", "grass")
        self.slab(box(4.0, -28.0, 7.0, -Y), -0.12, "Footpath to void deck", self.site, "USERDEFINED", "paving", 0.1, "Footpath")
        self.slab(box(-34, -28, 34, -21), -0.10, "Sample Avenue 1", self.site, "USERDEFINED", "asphalt", 0.1, "Road")
        self.tree_types()
        for (tx, ty) in [(-20, -17), (-12, -17), (12, -17), (20, -17), (-24, 0), (24, 0), (-20, 16), (0, 16), (20, 16),
                         (-10, 20), (10, 20)]:
            t = self.product("IfcGeographicElement", "Tree", None, frame((tx, ty, -0.15), (1, 0)), self.site)
            self.batch_type.setdefault(self.rng.choice(self.ttypes), []).append(t)

        self._flush()
        return names, ffl

    def space(self, poly, z, h, name, long_name, storey, external=False, flat=None, room=None):
        m = self.m
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        sp = root.create_entity(m, "IfcSpace", name=name, predefined_type="INTERNAL" if not external else "EXTERNAL")
        sp.LongName = long_name
        c = poly.centroid
        geometry.edit_object_placement(m, product=sp, matrix=translate(c.x, c.y, z))
        rep = self.extrusion(poly, h, (c.x, c.y))
        geometry.assign_representation(m, product=sp, representation=rep)
        style.assign_representation_styles(m, shape_representation=rep, styles=[self.mat["space"][1]])
        aggregate.assign_object(m, products=[sp], relating_object=storey)
        self.props(sp, "Pset_SpaceCommon", {"IsExternal": external, "Reference": long_name})
        q = pset.add_qto(m, product=sp, name="Qto_SpaceBaseQuantities")
        pset.edit_qto(m, qto=q, properties={"NetFloorArea": round(poly.area, 2), "Height": round(h, 3)})
        if flat:
            self.props(sp, "SampleCity_Room", {"UnitNumber": flat, "RoomCode": room})
        self._n("IfcSpace")
        return sp

    def _flush(self):
        for structure, els in self.batch_contain.items():
            spatial.assign_container(self.m, products=els, relating_structure=structure)
        for key, els in self.batch_mat.items():
            if self.mat[key][0] is not None:
                material.assign_material(self.m, products=els, type="IfcMaterial", material=self.mat[key][0])
        for t, occ in self.batch_type.items():
            ifctype.assign_type(self.m, related_objects=occ, relating_type=t)
        for wall, lo in self.centre_walls:   # keep Bonsai's layer usage consistent with centred geometry
            usage = uel.get_material(wall)
            if usage is not None and usage.is_a("IfcMaterialLayerSetUsage"):
                usage.OffsetFromReferenceLine = lo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="hdb_block.ifc")
    ap.add_argument("--storeys", type=int, default=12, help="storeys incl. the void deck (roof added on top)")
    ap.add_argument("--schema", choices=["IFC4X3", "IFC4"], default="IFC4X3")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--georef", action="store_true", help="add an SVY21 map conversion")
    ap.add_argument("--eastings", type=float, default=30000.0)
    ap.add_argument("--northings", type=float, default=36000.0)
    a = ap.parse_args()
    b = BlockBuilder(a)
    names, ffl = b.build()
    b.m.write(a.out)
    print(f"Wrote {a.out}: {a.storeys} storeys + roof, roof level {ffl[-1]:.1f} m, schema {a.schema}")
    for k in sorted(b.counts):
        print(f"  {k:<22}{b.counts[k]:>6}")


if __name__ == "__main__":
    main()
