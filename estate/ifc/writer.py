"""IFC authoring core shared by every building and the site (lifted from BlockBuilder in legacy/toolkit/hdb_block.py).

Elements are authored with ifcopenshell.api (the library underneath Bonsai's operators). Placements pass
through a transform stack so a component (a stair, a wing) can be authored in its own frame and rotated by
multiples of 90 degrees. Containment, materials and types are batched and flushed once per file.

With ``parametric`` (the default; the legacy baseline switches it off) the walls, doors and windows also carry
what Bonsai's parametric tools edit (see ifc/joins.py): every wall its own IfcMaterialLayerSetUsage and a
Plan/Axis/GRAPH_VIEW reference line, wall joins as IfcRelConnectsPathElements (``connect_walls``), and every door
and window type its BBIM_Door / BBIM_Window data, from which its representation is built.
"""
from __future__ import annotations

import contextlib
import math
import os

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

import ifcopenshell
import ifcopenshell.guid
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

from estate import guids
from estate.geom.walls import Opening, WallSpec, translate
from estate.ifc import joins
from estate.rules import DOOR_KINDS, NO_MATERIAL, PALETTE, SLAB, WALL_TYPES

FIXED_TIMESTAMP = "2026-10-04T00:00:00+08:00"
# stable keys use one class name for kinds that change class between IFC4X3 and IFC4 (site works), so the IFC4
# copy keeps the GlobalIds of the IFC4X3 master
KEY_CLASS = {"IfcPavement": "IfcSlab", "IfcCourse": "IfcSlab", "IfcKerb": "IfcCivilElement"}
MATERIAL_CATEGORY = {  # palette key -> IfcMaterial.Category (IFC's suggested categories)
    "ext": "concrete", "hs": "concrete", "core": "concrete", "slab": "concrete", "stair": "concrete",
    "column": "concrete", "party": "concrete", "accent": "concrete", "kerb": "concrete", "paving": "concrete",
    "int": "gypsum", "tile": "ceramic", "roof": "bitumen", "asphalt": "asphalt", "steel": "steel", "blast": "steel",
    "lift": "steel", "shutter": "steel", "metal_roof": "steel", "alu": "aluminium", "frame": "aluminium",
    "timber": "wood", "glass": "glass", "tank": "plastic", "grass": "earth", "marking": "paint", "rubber": "rubber",
}
GLAZED_DOOR_KINDS = {"balcony", "shop"}
DOOR_LINING = 0.05      # lining + stop inside the structural opening of IfcOpenShell's door representation


class IfcWriter:
    def __init__(self, *, schema="IFC4X3", project_name="Sample Town N5", project_guid=None, site_name="Sample Town N5 site",
                 site_guid=None, georef=None, guid_key="estate", guid_base=None, typed_openings=False,
                 author=("Sample Town N5 generator",), parametric=True):
        """guid_key seeds the GlobalId sequence used for relationships and unnamed entities (include the schema);
        guid_base (schema-agnostic, e.g. 'SN5/BLK_501') gives every named product, space, zone, type and property set
        a GlobalId derived from a stable key, so ids survive unrelated edits and match between IFC4X3 and IFC4.
        parametric=False writes walls, doors and windows as the original toolkit did (Model/Axis line, one layer set
        usage per wall type, no joins, no BBIM data): the legacy baseline must stay byte-identical."""
        self.schema = schema
        self.guid_base = guid_base
        self._key_count = {}
        self._stable = set()
        self.typed_openings = typed_openings
        self.parametric = parametric
        self._plan_axis = None
        self._wall_lo = {}
        self._guids = guids.deterministic(guid_key)
        self._guids.__enter__()
        self.m = project.create_file(version=schema)
        h = self.m.header.file_name
        h.time_stamp = FIXED_TIMESTAMP
        h.author = list(author)
        h.organization = ["Sample Town N5"]
        h.preprocessor_version = f"IfcOpenShell {ifcopenshell.version}"
        h.originating_system = "Sample Town N5 estate generator (estate/, Bonsai 0.9 toolchain)"
        self.batch_contain, self.batch_mat, self.batch_type = {}, {}, {}
        self.counts = {}
        self.centre_walls = []
        self.xf = np.eye(4)
        self._xf_stack = []
        self.door_types, self.window_types = {}, {}
        self._setup(project_name, project_guid, site_name, site_guid, georef)

    # ------------------------------------------------------------------ setup
    def _setup(self, project_name, project_guid, site_name, site_guid, georef):
        m = self.m
        self.project = root.create_entity(m, "IfcProject", name=project_name)
        if project_guid:
            self.project.GlobalId = project_guid
        unit.assign_unit(m, units=[unit.add_si_unit(m, unit_type=t) for t in
                                   ("LENGTHUNIT", "AREAUNIT", "VOLUMEUNIT", "PLANEANGLEUNIT")])
        model3d = context.add_context(m, context_type="Model")
        self.body = context.add_context(m, "Model", "Body", "MODEL_VIEW", parent=model3d)
        # wall reference lines: ifcopenshell.util.representation.get_reference_line (and so Bonsai's wall tools)
        # reads Plan/Axis/GRAPH_VIEW only; that context is added with the first wall (axis_context)
        self.axis = None if self.parametric else context.add_context(m, "Model", "Axis", "GRAPH_VIEW", parent=model3d)
        if georef:
            georeference.add_georeferencing(m, ifc_class="IfcMapConversion", name=f"EPSG:{georef.get('epsg', 3414)}")
            op = {"Eastings": float(georef.get("eastings", 0.0)), "Northings": float(georef.get("northings", 0.0)),
                  "OrthogonalHeight": float(georef.get("height", 0.0))}
            north = float(georef.get("true_north_deg", 0.0))
            if north:
                op["XAxisAbscissa"], op["XAxisOrdinate"] = math.cos(math.radians(north)), math.sin(math.radians(north))
            georeference.edit_georeferencing(m, projected_crs={"Name": f"EPSG:{georef.get('epsg', 3414)}"},
                                             coordinate_operation=op)
        self.site = root.create_entity(m, "IfcSite", name=site_name)
        if site_guid:
            self.site.GlobalId = site_guid
        if georef:      # approximate WGS84 reference of the (placeholder) SVY21 origin, for viewers that want one
            self.site.RefLatitude, self.site.RefLongitude = (1, 21, 45, 0), (103, 49, 30, 0)
            self.site.RefElevation = float(georef.get("height", 0.0))
        aggregate.assign_object(m, products=[self.site], relating_object=self.project)
        geometry.edit_object_placement(m, product=self.site)

        self.mat = {}
        for key, (name, rgb, tr) in PALETTE.items():
            st = style.add_style(m, name=name)
            style.add_surface_style(m, style=st, ifc_class="IfcSurfaceStyleShading", attributes={
                "SurfaceColour": {"Name": None, "Red": rgb[0], "Green": rgb[1], "Blue": rgb[2]},
                "Transparency": tr})
            mt = None
            if key not in NO_MATERIAL:
                mt = material.add_material(m, name=name, category=MATERIAL_CATEGORY.get(key, key))
                style.assign_material_style(m, material=mt, style=st, context=self.body)
            self.mat[key] = (mt, st)

        self.wall_types = {}
        for key, (name, t, pkey, pre) in WALL_TYPES.items():
            wt = self.stable_id(root.create_entity(m, "IfcWallType", name=name, predefined_type=pre), f"IfcWallType/{name}")
            ls = material.add_material_set(m, name=name, set_type="IfcMaterialLayerSet")
            layer = material.add_layer(m, layer_set=ls, material=self.mat[pkey][0])
            attrs = {"LayerThickness": t}
            if self.parametric and key in joins.LAYER_PRIORITY:
                attrs["Priority"] = joins.LAYER_PRIORITY[key]
            material.edit_layer(m, layer=layer, attributes=attrs)
            material.assign_material(m, products=[wt], type="IfcMaterialLayerSet", material=ls)
            self.wall_types[key] = wt

    def stable_id(self, el, key):
        """Give el a GlobalId derived from guid_base + key (repeats get #1, #2 ... in creation order)."""
        if self.guid_base is None:
            return el
        n = self._key_count.get(key, 0)
        self._key_count[key] = n + 1
        el.GlobalId = guids.from_text(f"{self.guid_base}/{key}" + (f"#{n}" if n else ""))
        self._stable.add(el.id())
        return el

    def _finalise_ids(self):
        """Give every remaining rooted entity a GlobalId derived from what it is, not from creation order:
        named objects from (class, name), property sets from their owner, relationships from the object they
        relate and their first related object. Runs once before writing."""
        f = self.m
        roots = sorted(f.by_type("IfcRoot"), key=lambda e: e.id())
        explicit = {e.id() for e in (self.project, self.site)}
        done = self._stable | explicit
        for e in roots:                                   # objects, types, groups with a name
            if e.id() in done or e.is_a("IfcRelationship") or e.is_a("IfcPropertyDefinition"):
                continue
            if getattr(e, "Name", None):
                self.stable_id(e, f"{KEY_CLASS.get(e.is_a(), e.is_a())}/{e.Name}")
                done.add(e.id())
        for e in roots:                                   # property sets / quantities: from their owner
            if e.id() in done or not e.is_a("IfcPropertyDefinition"):
                continue
            owners = [r.RelatedObjects[0] for r in getattr(e, "DefinesOccurrence", ()) or () if r.RelatedObjects]
            owners += list(getattr(e, "DefinesType", ()) or ())
            if owners:
                self.stable_id(e, f"pset/{owners[0].GlobalId}/{e.Name}")
                done.add(e.id())

        def ident(x):
            if x is None:
                return "-"
            if x.is_a("IfcRoot"):
                return x.GlobalId
            # layer sets have a LayerSetName, not a Name: key them by it, not by their (order-dependent) STEP id
            return f"{x.is_a()}:{getattr(x, 'Name', None) or getattr(x, 'LayerSetName', None) or x.id()}"

        def entities(v):     # Relating* / Related* values that are entities (not RelatedConnectionType, priorities)
            vs = list(v) if isinstance(v, (list, tuple)) else [v]
            return [x for x in vs if isinstance(x, ifcopenshell.entity_instance)]
        for e in roots:                                   # relationships: from what they relate
            if e.id() in done or not e.is_a("IfcRelationship"):
                continue
            info = e.get_info(recursive=False)
            relating = [x for k, v in info.items() if k.startswith("Relating") for x in entities(v)]
            related = [x for k, v in info.items() if k.startswith("Related") for x in entities(v)]
            first = min((ident(x) for x in related), default="-")
            self.stable_id(e, f"{e.is_a()}/{ident(relating[0]) if relating else '-'}/{first}")

    def _purge_unused(self):
        """Wall types without occurrences (and their layer sets) and materials nothing uses."""
        import ifcopenshell.api.material as material_api
        import ifcopenshell.api.root as root_api
        for key, wt in list(self.wall_types.items()):
            if not any(r.RelatedObjects for r in getattr(wt, "Types", ()) or ()):
                sets = [r.RelatingMaterial for r in getattr(wt, "HasAssociations", ()) or ()
                        if r.is_a("IfcRelAssociatesMaterial")]
                root_api.remove_product(self.m, product=wt)
                for ms in sets:
                    if ms.is_a("IfcMaterialLayerSet") and not ms.AssociatedTo:
                        for layer in ms.MaterialLayers:
                            self.m.remove(layer)
                        self.m.remove(ms)
                del self.wall_types[key]
        used = set()
        for r in self.m.by_type("IfcRelAssociatesMaterial"):
            used.add(r.RelatingMaterial.id())
        for ly in self.m.by_type("IfcMaterialLayer"):
            if ly.Material is not None:
                used.add(ly.Material.id())
        for key, (mt, st) in list(self.mat.items()):
            if mt is not None and mt.id() not in used:
                material_api.remove_material(self.m, material=mt)
                self.mat[key] = (None, st)

    def close(self):
        if self._guids is not None:
            self._guids.__exit__(None, None, None)
            self._guids = None

    def write(self, path, purge=True):
        if self.guid_base is not None:
            if purge:
                self._purge_unused()
            self._finalise_ids()
        h = self.m.header.file_name
        h.time_stamp = FIXED_TIMESTAMP
        h.name = os.path.basename(str(path)).replace(".tmp.ifc", ".ifc")
        h.authorization = "none"
        self.m.write(str(path))

    # ------------------------------------------------------------------ transform stack
    @contextlib.contextmanager
    def transformed(self, M):
        self._xf_stack.append(self.xf)
        self.xf = self.xf @ M
        try:
            yield
        finally:
            self.xf = self._xf_stack.pop()

    def place(self, el, matrix=None):
        geometry.edit_object_placement(self.m, product=el, matrix=self.xf @ (np.eye(4) if matrix is None else matrix))

    # ------------------------------------------------------------------ spatial structure
    def building(self, name, long_name=None, address=None, guid=None):
        b = self.stable_id(root.create_entity(self.m, "IfcBuilding", name=name), f"IfcBuilding/{name}")
        if guid:
            b.GlobalId = guid
        b.LongName = long_name
        if address:
            b.BuildingAddress = self.m.create_entity("IfcPostalAddress", **address)
        aggregate.assign_object(self.m, products=[b], relating_object=self.site)
        geometry.edit_object_placement(self.m, product=b)
        return b

    def storeys(self, building, names, elevations):
        out = []
        for nm, z in zip(names, elevations):
            s = self.stable_id(root.create_entity(self.m, "IfcBuildingStorey", name=nm),
                               f"IfcBuildingStorey/{building.Name}/{nm}")
            s.Elevation = z
            geometry.edit_object_placement(self.m, product=s, matrix=translate(0, 0, z))
            out.append(s)
            self._n("IfcBuildingStorey")
        aggregate.assign_object(self.m, products=out, relating_object=building)
        return out

    # ------------------------------------------------------------------ primitives
    def _n(self, cls):
        self.counts[cls] = self.counts.get(cls, 0) + 1

    def contain(self, structure, el):
        self.batch_contain.setdefault(structure, []).append(el)

    def use_material(self, key, el):
        self.batch_mat.setdefault(key, []).append(el)

    def props(self, product, name, values):
        p = pset.add_pset(self.m, product=product, name=name)
        if self.guid_base is not None:
            p.GlobalId = guids.from_text(f"{product.GlobalId}/{name}")
        pset.edit_pset(self.m, pset=p, properties=values)
        return p

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

    def product(self, cls, name, rep, matrix, container, predefined=None, mat=None, object_type=None, key=None):
        el = self.stable_id(root.create_entity(self.m, cls, name=name, predefined_type=predefined),
                            f"{KEY_CLASS.get(cls, cls)}/{key or name}")
        if object_type:
            el.ObjectType = object_type
        self.place(el, matrix)
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
        return self.product("IfcSlab", name, self.extrusion(poly, depth, (c.x, c.y)),
                            translate(c.x, c.y, top - depth), container, predefined, mat, object_type)

    def box_element(self, cls, name, poly, z0, height, container, predefined=None, mat=None, object_type=None):
        """Any prismatic element: footprint polygon extruded from z0."""
        c = poly.centroid
        return self.product(cls, name, self.extrusion(poly, height, (c.x, c.y)), translate(c.x, c.y, z0),
                            container, predefined, mat, object_type)

    # ------------------------------------------------------------------ walls, doors, windows
    def axis_context(self):
        """Plan/Axis/GRAPH_VIEW (with its Plan context), created with the first wall so files without walls keep
        their contexts."""
        if self._plan_axis is None:
            plan = context.add_context(self.m, context_type="Plan")
            self._plan_axis = context.add_context(self.m, "Plan", "Axis", "GRAPH_VIEW", parent=plan)
        return self._plan_axis

    def build_wall(self, w: WallSpec):
        """One IfcWall (body extruded in the wall frame: x along w.u from w.p0, the thickness band from local y
        w.lo) with its openings. Parametric: the reference line runs along local y = 0 over w.meta['axis'] (the
        run's nodes; default the body ends) and the wall gets its own layer set usage at flush."""
        from estate.geom.walls import frame
        M = frame((w.p0[0], w.p0[1], w.z0), w.u)
        rep = geometry.add_wall_representation(self.m, context=self.body, length=w.length, height=w.h,
                                               thickness=w.t, offset=w.lo)
        pre = WALL_TYPES[w.type_key][3]
        wall = self.product("IfcWall", w.name, rep, M, w.storey, predefined=pre, key=w.meta.get("key"))
        if self.parametric:
            a0, a1 = w.meta.get("axis", (0.0, w.length))
            axis = geometry.add_axis_representation(self.m, context=self.axis_context(),
                                                    axis=[(float(a0) + 0.0, 0.0), (float(a1) + 0.0, 0.0)])
            self._wall_lo[wall.id()] = w.lo
        else:
            axis = geometry.add_axis_representation(self.m, context=self.axis, axis=[(0.0, 0.0), (w.length, 0.0)])
        geometry.assign_representation(self.m, product=wall, representation=axis)
        self.batch_type.setdefault(self.wall_types[w.type_key], []).append(wall)
        if w.lo != 0.0:
            self.centre_walls.append((wall, w.lo))
        self.props(wall, "Pset_WallCommon", {"IsExternal": w.external,
                                             "LoadBearing": w.type_key in ("EXT", "HS", "HS250", "CORE", "PARTY")})
        if w.climbable:
            self.props(wall, "Navigation", {"ClimbableTopEdge": True, "TopEdgeHeightAboveFloor": round(w.h, 2)})
        built = []
        for op in w.openings:
            built.append(self.build_opening(wall, w, M, op))
        return wall, built

    def connect_walls(self, walls: dict, wall_joins) -> list:
        """IfcRelConnectsPathElements for joins (ifc/joins.Join, keyed like `walls` {key: IfcWall}) through
        ifcopenshell.api.geometry.connect_path, as Bonsai's wall tools write them: no connection geometry, empty
        priorities. GlobalIds are keyed by the relating wall's GlobalId and end. A no-op for the legacy writer."""
        if not self.parametric:
            return []
        out = []
        n0 = len(self.m.by_type("IfcRelConnectsPathElements"))
        for j in wall_joins:
            a, b = walls.get(j.relating), walls.get(j.related)
            if a is None or b is None:
                continue
            rel = geometry.connect_path(self.m, relating_element=a, related_element=b,
                                        relating_connection=j.relating_end, related_connection=j.related_end)
            out.append(self.stable_id(rel, f"IfcRelConnectsPathElements/{a.GlobalId}/{j.relating_end}"))
        if len(self.m.by_type("IfcRelConnectsPathElements")) != n0 + len(out):
            # connect_path drops an earlier join that conflicts with a new one at the same wall end
            raise ValueError("wall joins replaced by a later join at the same wall end")
        return out

    def build_opening(self, wall, w: WallSpec, M, op: Opening):
        width = op.s1 - op.s0
        oe = self.stable_id(root.create_entity(self.m, "IfcOpeningElement", name=f"{op.name} opening"),
                            f"IfcOpeningElement/{op.name}")
        self.place(oe, M @ translate(op.s0, w.lo - 0.1, op.sill))
        geometry.assign_representation(self.m, product=oe, representation=self._opening_rep(width, op.height, w.t + 0.2))
        feature.add_feature(self.m, feature=oe, element=wall)
        if op.kind == "door":
            optype, pkey, passable, fire = DOOR_KINDS[op.door_kind]
            hinge_end = op.meta.get("hinge", "start")
            if op.swing < 0:    # leaf frame rotated 180 degrees: local x runs from s1 back to s0
                hinge_end = "end" if hinge_end == "start" else "start"
            if hinge_end == "end" and optype == "SINGLE_SWING_LEFT":
                optype = "SINGLE_SWING_RIGHT"
            if op.swing >= 0:
                Md = M @ translate(op.s0, w.lo + w.t / 2 - 0.05, op.sill)
            else:   # rotate the leaf 180 degrees about z: opens to the wall's right side
                Md = M @ translate(op.s1, w.lo + w.t / 2 + 0.05, op.sill) @ np.diag([-1.0, -1.0, 1.0, 1.0])
            if self.typed_openings:
                el = self.product("IfcDoor", op.name, None, Md, w.storey, predefined="DOOR", mat=pkey)
                self.batch_type.setdefault(self.door_type(op.door_kind, optype, width, op.height), []).append(el)
            else:
                rep, data = self.door_representation(optype, width, op.height)
                el = self.product("IfcDoor", op.name, rep, Md, w.storey, predefined="DOOR", mat=pkey)
                self.bbim_pset(el, "BBIM_Door", data)
            el.OperationType = optype
            el.OverallWidth, el.OverallHeight = width, op.height
            clear = round(width - 2 * DOOR_LINING, 3)
            common = {"IsExternal": w.external, "HandicapAccessible": clear >= 0.85}
            if fire:
                common["FireRating"] = fire
            self.props(el, "Pset_DoorCommon", common)
            nav = {"Passable": passable, "Interactive": True, "DoorKind": op.door_kind, "ClearWidth": clear,
                   "NominalWidth": round(width, 3)}
            if "from" in op.meta:
                nav["FromSpace"], nav["ToSpace"] = op.meta["from"], op.meta["to"]
            self.props(el, "Navigation", nav)
        else:
            Mw = M @ translate(op.s0, w.lo + w.t / 2 - 0.0875, op.sill)
            if self.typed_openings:
                el = self.product("IfcWindow", op.name, None, Mw, w.storey, predefined="WINDOW", mat="glass")
                self.batch_type.setdefault(self.window_type(width, op.height), []).append(el)
            else:
                rep, data = self.window_representation(width, op.height)
                el = self.product("IfcWindow", op.name, rep, Mw, w.storey, predefined="WINDOW", mat="glass")
                self.bbim_pset(el, "BBIM_Window", data)
            el.OverallWidth, el.OverallHeight = width, op.height
            self.props(el, "Pset_WindowCommon", {"IsExternal": True})
            if op.sill < 1.0:   # low sill: fixed lower pane / grille up to 1.0 m (Approved Document barrier height)
                self.props(el, "Navigation", {"BarrierHeight": 1.0, "Passable": False})
        feature.add_filling(self.m, opening=oe, element=el)
        return el

    def _opening_rep(self, width, height, thickness):
        if not self.typed_openings:
            return geometry.add_wall_representation(self.m, context=self.body, length=width, height=height,
                                                    thickness=thickness)
        # share the solid, not the representation (IfcShapeModel WR11: one product per representation)
        key = (round(width, 4), round(height, 4), round(thickness, 4))
        cache = self.__dict__.setdefault("_opening_items", {})
        if key not in cache:
            rep = geometry.add_wall_representation(self.m, context=self.body, length=width, height=height,
                                                   thickness=thickness)
            cache[key] = rep.Items
            return rep
        return self.m.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", cache[key])

    def _style_window_frame(self, rep):
        for item in rep.Items:
            if item.is_a("IfcExtrudedAreaSolid") and item.SweptArea.is_a("IfcArbitraryProfileDefWithVoids"):
                self.m.createIfcStyledItem(item, [self.mat["frame"][1]], None)

    def door_representation(self, optype, width, height):
        """(body representation, BBIM_Door data or None) of a door. Parametric: built from the data Bonsai's door
        tool edits (ifc/joins.door_data, the API defaults), through the same keyword mapping as Bonsai's
        update_door_modifier_representation; kinds Bonsai cannot rebuild (and the legacy writer) get no data."""
        data = joins.door_data(optype, width, height) if self.parametric else None
        if data is None:
            rep = geometry.add_door_representation(self.m, context=self.body, overall_height=height, overall_width=width,
                                                   operation_type=optype)
        else:
            rep = geometry.add_door_representation(self.m, context=self.body, **joins.door_kwargs(data))
        return rep, data

    def window_representation(self, width, height):
        """(body representation with its frame styled, BBIM_Window data or None) of a single-panel window, as
        door_representation."""
        data = joins.window_data(width, height) if self.parametric else None
        if data is None:
            rep = geometry.add_window_representation(self.m, context=self.body, overall_height=height, overall_width=width)
        else:
            rep = geometry.add_window_representation(self.m, context=self.body, **joins.window_kwargs(data))
        self._style_window_frame(rep)
        return rep, data

    def bbim_pset(self, element, name, data):
        """Bonsai's parametric data pset (BBIM_Door / BBIM_Window): one 'Data' property holding the JSON as an IfcText
        (an IfcLabel stops at 255 characters). On a type, its occurrences inherit it (tool.Parametric.is_door)."""
        if data is not None:
            self.props(element, name, {"Data": self.m.createIfcText(joins.data_text(data))})

    def door_type(self, kind, optype, width, height):
        key = (kind, optype, round(width, 3), round(height, 3))
        if key not in self.door_types:
            pkey = DOOR_KINDS[kind][1]
            t = root.create_entity(self.m, "IfcDoorType", name=f"D-{kind}-{optype}-{int(round(width * 1000))}x{int(round(height * 1000))}",
                                   predefined_type="DOOR")
            self.stable_id(t, f"IfcDoorType/{t.Name}")
            t.OperationType = optype
            rep, data = self.door_representation(optype, width, height)
            self._style_door_items(rep, kind, width, height)
            geometry.assign_representation(self.m, product=t, representation=rep)
            if self.mat[pkey][0] is not None:
                material.assign_material(self.m, products=[t], type="IfcMaterial", material=self.mat[pkey][0])
            self.bbim_pset(t, "BBIM_Door", data)
            self.door_types[key] = t
        return self.door_types[key]

    def _style_door_items(self, rep, kind, width, height):
        """Style every item of a door type (viewers that read item styles show it as Bonsai does); glazed kinds
        get glass leaves (items taller than half the door and narrower than the lining) in an aluminium frame."""
        pkey = DOOR_KINDS[kind][1]
        if kind not in GLAZED_DOOR_KINDS:
            for it in rep.Items:
                self.m.createIfcStyledItem(it, [self.mat[pkey][1]], None)
            return
        import ifcopenshell.geom
        s = ifcopenshell.geom.settings()
        for it in rep.Items:
            v = np.array(ifcopenshell.geom.create_shape(s, it).verts).reshape(-1, 3)
            leaf = len(v) and 0.3 < np.ptp(v[:, 0]) < width - 0.04 and np.ptp(v[:, 2]) > height / 2
            self.m.createIfcStyledItem(it, [self.mat["glass" if leaf else "frame"][1]], None)

    def window_type(self, width, height):
        key = (round(width, 3), round(height, 3))
        if key not in self.window_types:
            t = root.create_entity(self.m, "IfcWindowType", name=f"W-{int(round(width * 1000))}x{int(round(height * 1000))}",
                                   predefined_type="WINDOW")
            self.stable_id(t, f"IfcWindowType/{t.Name}")
            t.PartitioningType = "SINGLE_PANEL"
            rep, data = self.window_representation(width, height)
            geometry.assign_representation(self.m, product=t, representation=rep)
            material.assign_material(self.m, products=[t], type="IfcMaterial", material=self.mat["glass"][0])
            self.bbim_pset(t, "BBIM_Window", data)
            self.window_types[key] = t
        return self.window_types[key]

    # ------------------------------------------------------------------ spaces and zones
    def space(self, poly, z, h, name, long_name, storey, external=False, flat=None, room=None, extra=None,
              predefined=None):
        m = self.m
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        sp = self.stable_id(root.create_entity(m, "IfcSpace", name=name,
                                               predefined_type=predefined or ("INTERNAL" if not external else "EXTERNAL")),
                            f"IfcSpace/{name}")
        sp.LongName = long_name
        c = poly.centroid
        self.place(sp, translate(c.x, c.y, z))
        rep = self.extrusion(poly, h, (c.x, c.y))
        geometry.assign_representation(m, product=sp, representation=rep)
        style.assign_representation_styles(m, shape_representation=rep, styles=[self.mat["space"][1]])
        aggregate.assign_object(m, products=[sp], relating_object=storey)
        self.props(sp, "Pset_SpaceCommon", {"IsExternal": external, "Reference": long_name})
        q = pset.add_qto(m, product=sp, name="Qto_SpaceBaseQuantities")
        if self.guid_base is not None:
            q.GlobalId = guids.from_text(f"{sp.GlobalId}/Qto_SpaceBaseQuantities")
        pset.edit_qto(m, qto=q, properties={"NetFloorArea": round(poly.area, 2), "Height": round(h, 3)})
        if flat:
            vals = {"UnitNumber": flat, "RoomCode": room}
            if extra:
                vals.update(extra)
            self.props(sp, "SampleCity_Room", vals)
        self._n("IfcSpace")
        return sp

    def zone(self, name, long_name, spaces, props=None, pset_name="SampleCity_Flat"):
        z = self.stable_id(root.create_entity(self.m, "IfcZone", name=name), f"IfcZone/{name}")
        z.LongName = long_name
        group.assign_group(self.m, products=spaces, group=z)
        if props:
            self.props(z, pset_name, props)
        self._n("IfcZone")
        return z

    # ------------------------------------------------------------------ flush
    def flush(self):
        for structure, els in self.batch_contain.items():
            spatial.assign_container(self.m, products=els, relating_structure=structure)
        for key, els in self.batch_mat.items():
            if self.mat[key][0] is not None:
                material.assign_material(self.m, products=els, type="IfcMaterial", material=self.mat[key][0])
        for t, occ in self.batch_type.items():
            if self.parametric and t.is_a("IfcWallType"):
                ifctype.assign_type(self.m, related_objects=occ, relating_type=t, should_map_representations=False)
                self._own_usages(t, occ)
            else:
                ifctype.assign_type(self.m, related_objects=occ, relating_type=t)
        if not self.parametric:
            for wall, lo in self.centre_walls:   # keep Bonsai's layer usage consistent with centred geometry
                usage = uel.get_material(wall)
                if usage is not None and usage.is_a("IfcMaterialLayerSetUsage"):
                    usage.OffsetFromReferenceLine = lo
        self.batch_contain, self.batch_mat, self.batch_type, self.centre_walls = {}, {}, {}, []

    def _own_usages(self, wall_type, walls):
        """One IfcMaterialLayerSetUsage per wall (AXIS2, POSITIVE, offset = the wall's thickness band start): Bonsai
        edits a usage in place (offset, flip), so a usage shared by the walls of a type would move all of them.
        Bonsai's wall tools only treat a wall as parametric when the usage is the wall's own (get_usage_type
        reads the material without inheriting it from the type)."""
        layer_set = uel.get_material(wall_type)
        for wall in walls:
            usage = self.m.create_entity("IfcMaterialLayerSetUsage", ForLayerSet=layer_set, LayerSetDirection="AXIS2",
                                         DirectionSense="POSITIVE",
                                         OffsetFromReferenceLine=float(self._wall_lo.get(wall.id(), 0.0)))
            rel = self.m.create_entity("IfcRelAssociatesMaterial", GlobalId=ifcopenshell.guid.new(),
                                       RelatedObjects=[wall], RelatingMaterial=usage)
            self.stable_id(rel, f"IfcRelAssociatesMaterial/{wall.GlobalId}")
