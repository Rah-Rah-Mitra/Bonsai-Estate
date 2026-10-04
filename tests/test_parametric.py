"""Bonsai parametric data: wall joins, per-wall layer set usages, Plan/Axis reference lines, BBIM_Door / BBIM_Window.

The suite builds its own files under build/parametric_tests/ (the PT4 test block with and without the parametric
data, and the neighbourhood centre twice; about 25 s), so it never reads a stale build/t_pt4.ifc. Bonsai's own
wall regenerator (ifcopenshell.api.geometry.regenerate_wall_representation) and door / window mapping are run
without Blender; regenerated walls are compared by their sections in every height band, not only in plan (a plan
union cannot see a wall cut back to the face of a lower one). The Blender test opens a .blend made from the PT4
block in blender.exe with Bonsai, runs Bonsai's door editing and wall recalculation operators and takes about a
minute; set ESTATE_SKIP_BLENDER=1 to skip it.
"""
from __future__ import annotations

import json
import shutil
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estate.env import bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402
import shapely  # noqa: E402
import ifcopenshell  # noqa: E402
import ifcopenshell.api.geometry as geometry  # noqa: E402
import ifcopenshell.geom  # noqa: E402
import ifcopenshell.util.element as uel  # noqa: E402
import ifcopenshell.util.placement as upl  # noqa: E402
import ifcopenshell.util.representation as urep  # noqa: E402

from estate import config, env  # noqa: E402
from estate.ifc import joins  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
import make_t_pt4  # noqa: E402

OUT = env.BUILD / "parametric_tests"
TESTS = env.BUILD / "blender_tests" / "parametric"
SKIP_BLENDER = __import__("os").environ.get("ESTATE_SKIP_BLENDER") == "1" or not env.BLENDER_EXE.exists()
TOL = 1e-6
TYPICAL = "L5"
_BUILT = {}


# ----------------------------------------------------------------------------- builds (once per test run)
def build_pt4(out: Path, parametric=True) -> Path:
    """The PT4 test block exactly as tests/fixtures/make_t_pt4.py builds it; parametric=False is the writer as it
    was before the parametric data (the baseline the other checks compare with)."""
    from estate.blocks.builder import build_building, derive_plan
    from estate.blocks.point import plan_point
    from estate.ifc.writer import IfcWriter
    plan = plan_point(dict(make_t_pt4.SPEC))
    W = IfcWriter(guid_key="test/900", typed_openings=True, parametric=parametric, **config.identity(config.load()))
    build_building(W, plan, derive_plan(plan))
    out.parent.mkdir(parents=True, exist_ok=True)
    W.write(out)
    W.close()
    return out


def built(which: str) -> Path:
    """pt4 / pt4_again (parametric), pt4_flat (baseline), nc / nc_again (NC 514, the guid_base path)."""
    if which not in _BUILT:
        if which.startswith("pt4"):
            _BUILT[which] = build_pt4(OUT / which / "t_pt4.ifc", parametric=which != "pt4_flat")
        else:
            from estate.site.centre import build_nc_ifc
            out = OUT / which / "NC_514.ifc"
            out.parent.mkdir(parents=True, exist_ok=True)
            build_nc_ifc(config.load(), out)
            _BUILT[which] = out
    return _BUILT[which]


def typical_runs():
    from estate.blocks.builder import derive_plan
    from estate.blocks.point import plan_point
    return derive_plan(plan_point(dict(make_t_pt4.SPEC)))["typical"].runs


# ----------------------------------------------------------------------------- geometry helpers
def storey_of(el):
    c = uel.get_container(el)
    return c.Name if c is not None else None


def axis_world(w):
    """The wall's reference line (Plan/Axis/GRAPH_VIEW, as Bonsai reads it) in plan coordinates."""
    M = upl.get_local_placement(w.ObjectPlacement)
    return [(M @ np.array([p[0], p[1], 0.0, 1.0]))[:2] for p in urep.get_reference_line(w)]


def sections(f, els) -> dict:
    """{entity id: (plan footprint, bottom z, top z)} of the bodies (openings not cut), from the tessellation in
    world coordinates. Wall bodies are vertical prisms, so footprint and z range give the section at any height."""
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    s.set("disable-opening-subtractions", True)
    out = {}
    it = ifcopenshell.geom.iterator(s, f, 2, include=list(els))
    if it.initialize():
        while True:
            sh = it.get()
            v = np.array(sh.geometry.verts).reshape(-1, 3)
            tris = [shapely.Polygon(v[t][:, :2]) for t in np.array(sh.geometry.faces).reshape(-1, 3)]
            out[sh.id] = (shapely.union_all([t for t in tris if t.area > 1e-10]), float(v[:, 2].min()),
                          float(v[:, 2].max()))
            if not it.next():
                break
    return out


def section_differences(before: dict, after: dict, tol=1e-6) -> list:
    """Wall sections of a storey before / after (sections()): in every height band between the distinct body
    bottoms and tops, the union of the walls cut at the band's middle. [(z, missing m², extra m²)] for the bands
    that differ. A plan-only comparison cannot see a wall that is cut back to the face of a lower one."""
    zs = sorted({round(z, 6) for sec in (before, after) for _, z0, z1 in sec.values() for z in (z0, z1)})
    out = []
    for a, b in zip(zs, zs[1:]):
        z = (a + b) / 2
        ub = shapely.union_all([fp for fp, z0, z1 in before.values() if z0 < z < z1])
        ua = shapely.union_all([fp for fp, z0, z1 in after.values() if z0 < z < z1])
        missing, extra = ub.difference(ua).area, ua.difference(ub).area
        if missing > tol or extra > tol:
            out.append((round(z, 3), round(missing, 6), round(extra, 6)))
    return out


def body_height(w) -> float:
    item = urep.get_representation(w, "Model", "Body", "MODEL_VIEW").Items[0]
    while item.is_a("IfcBooleanResult"):
        item = item.FirstOperand
    return float(item.Depth)


def layer_priority(w) -> int:
    return uel.get_material(w, should_inherit=False).ForLayerSet.MaterialLayers[0].Priority


def item_vertices(item) -> np.ndarray:
    v = np.array(ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), item).verts).reshape(-1, 3)
    return v[np.lexsort(v.T[::-1])]


def same_items(items_a, items_b) -> bool:
    if len(items_a) != len(items_b):
        return False
    for a, b in zip(items_a, items_b):
        va, vb = item_vertices(a), item_vertices(b)
        if va.shape != vb.shape or not np.allclose(va, vb, atol=TOL):
            return False
    return True


# Bonsai's door / window property defaults (bonsai/bim/module/model/prop.py BIMDoorProperties /
# BIMWindowProperties) and its representation mapping (door.py / window.py update_*_modifier_representation),
# re-implemented here independently of estate/ifc/joins.py
BONSAI_DOOR = dict(door_type="SINGLE_SWING_LEFT", overall_height=2.0, overall_width=0.9, lining_depth=0.05,
                   lining_thickness=0.05, lining_offset=0.0, lining_to_panel_offset_x=0.025,
                   lining_to_panel_offset_y=0.025, transom_thickness=0.0, transom_offset=1.525, casing_thickness=0.075,
                   casing_depth=0.005, threshold_thickness=0.025, threshold_depth=0.1, threshold_offset=0.0,
                   panel_depth=0.035, panel_width_ratio=1.0, frame_thickness=0.035, frame_depth=0.035)
BONSAI_WINDOW = dict(window_type="SINGLE_PANEL", overall_height=0.9, overall_width=0.6, lining_depth=0.05,
                     lining_thickness=0.05, lining_offset=0.05, lining_to_panel_offset_x=0.025,
                     lining_to_panel_offset_y=0.025, mullion_thickness=0.05, first_mullion_offset=0.3,
                     second_mullion_offset=0.45, transom_thickness=0.05, first_transom_offset=0.3,
                     second_transom_offset=0.6, frame_depth=[0.035] * 3, frame_thickness=[0.035] * 3)
BONSAI_PANELS = {"SINGLE_PANEL": 1, "DOUBLE_PANEL_HORIZONTAL": 2, "DOUBLE_PANEL_VERTICAL": 2}


def bonsai_props(defaults: dict, data_text: str) -> dict:
    """What Bonsai's enable-editing puts in the property group: the defaults, then the pset data with its lining and
    panel dictionaries flattened (parametric_lifecycle.FeatureModifierEditMixin._enable_one)."""
    data = json.loads(data_text)
    data.update(data.pop("lining_properties"))
    data.update(data.pop("panel_properties"))
    return dict(defaults, **data)


def bonsai_door_kwargs(p: dict) -> dict:
    return {"operation_type": p["door_type"], "overall_height": p["overall_height"], "overall_width": p["overall_width"],
            "lining_properties": {"LiningDepth": p["lining_depth"], "LiningThickness": p["lining_thickness"],
                                  "LiningOffset": p["lining_offset"], "LiningToPanelOffsetX": p["lining_to_panel_offset_x"],
                                  "LiningToPanelOffsetY": p["lining_to_panel_offset_y"],
                                  "TransomThickness": p["transom_thickness"], "TransomOffset": p["transom_offset"],
                                  "CasingThickness": p["casing_thickness"], "CasingDepth": p["casing_depth"],
                                  "ThresholdThickness": p["threshold_thickness"], "ThresholdDepth": p["threshold_depth"],
                                  "ThresholdOffset": p["threshold_offset"]},
            "panel_properties": {"PanelDepth": p["panel_depth"], "PanelWidth": p["panel_width_ratio"],
                                 "FrameDepth": p["frame_depth"], "FrameThickness": p["frame_thickness"]}}


def bonsai_window_kwargs(p: dict) -> dict:
    return {"partition_type": p["window_type"], "overall_height": p["overall_height"], "overall_width": p["overall_width"],
            "lining_properties": {"LiningDepth": p["lining_depth"], "LiningThickness": p["lining_thickness"],
                                  "LiningOffset": p["lining_offset"], "LiningToPanelOffsetX": p["lining_to_panel_offset_x"],
                                  "LiningToPanelOffsetY": p["lining_to_panel_offset_y"],
                                  "MullionThickness": p["mullion_thickness"], "FirstMullionOffset": p["first_mullion_offset"],
                                  "SecondMullionOffset": p["second_mullion_offset"],
                                  "TransomThickness": p["transom_thickness"], "FirstTransomOffset": p["first_transom_offset"],
                                  "SecondTransomOffset": p["second_transom_offset"]},
            "panel_properties": [{"FrameDepth": p["frame_depth"][i], "FrameThickness": p["frame_thickness"][i]}
                                 for i in range(BONSAI_PANELS[p["window_type"]])]}


def parametric_summary(f) -> dict:
    """The parametric data of a file by GlobalId (for comparing a file with its IFC4 copy)."""
    rels = sorted((r.GlobalId, r.RelatingElement.GlobalId, r.RelatingConnectionType, r.RelatedElement.GlobalId,
                   r.RelatedConnectionType) for r in f.by_type("IfcRelConnectsPathElements"))
    usages = []
    for w in f.by_type("IfcWall"):
        m = uel.get_material(w, should_inherit=False)
        usages.append((w.GlobalId, m.is_a(), round(m.OffsetFromReferenceLine, 6), m.LayerSetDirection, m.DirectionSense,
                       [o.GlobalId for r in m.AssociatedTo for o in r.RelatedObjects]))
    axes = sorted((w.GlobalId, tuple(tuple(np.round(p, 6)) for p in urep.get_reference_line(w))) for w in f.by_type("IfcWall"))
    psets = sorted((t.GlobalId, n, uel.get_pset(t, n, "Data")) for cls in ("IfcDoorType", "IfcWindowType")
                   for t in f.by_type(cls) for n in ("BBIM_Door", "BBIM_Window") if uel.get_pset(t, n))
    prio = sorted((ls.LayerSetName, tuple(ly.Priority for ly in ls.MaterialLayers)) for ls in f.by_type("IfcMaterialLayerSet"))
    return dict(joins=rels, usages=sorted(usages), axes=axes, psets=psets, priorities=prio)


# ----------------------------------------------------------------------------- 1. joins, usages, reference lines
class TestWallJoins(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = ifcopenshell.open(str(built("pt4")))
        cls.walls = cls.f.by_type("IfcWall")
        cls.rels = cls.f.by_type("IfcRelConnectsPathElements")

    def ends_of(self, w):
        return dict(zip((joins.ATSTART, joins.ATEND), axis_world(w)))

    def test_relationships_are_bonsai_joins(self):
        """Ends are ATSTART / ATEND on the relating side; ATPATH only on the related side (T); no self joins, no
        priorities (Bonsai's regenerator crashes on them), both walls on one storey."""
        self.assertGreater(len(self.rels), 1000)
        for r in self.rels:
            self.assertIsNot(r.RelatingElement, r.RelatedElement)
            self.assertNotEqual(r.RelatingElement.id(), r.RelatedElement.id())
            self.assertIn(r.RelatingConnectionType, (joins.ATSTART, joins.ATEND))
            self.assertIn(r.RelatedConnectionType, (joins.ATSTART, joins.ATEND, joins.ATPATH))
            self.assertFalse(r.RelatingPriorities or r.RelatedPriorities)
            self.assertIsNone(r.ConnectionGeometry)
            self.assertEqual(storey_of(r.RelatingElement), storey_of(r.RelatedElement))
        kinds = Counter("T" if r.RelatedConnectionType == joins.ATPATH else "L" for r in self.rels)
        self.assertGreater(kinds["T"], 100)
        self.assertGreater(kinds["L"], 100)

    def test_joined_walls_alike_in_height(self):
        """Bonsai extrudes a rebuilt corner to each wall's own height, so joined walls have one height, except a T
        on a taller wall at least as strong as the stem (it fills the corner above the stem); in this block they
        all have one height (the lobby facade wall continued by the lobby parapet turns the corner into the facade
        wall, not the parapet)."""
        for r in self.rels:
            a, b = r.RelatingElement, r.RelatedElement
            if abs(body_height(a) - body_height(b)) > TOL:
                self.assertEqual(r.RelatedConnectionType, joins.ATPATH, a.Name)
                self.assertGreater(body_height(b), body_height(a), a.Name)
                self.assertGreaterEqual(layer_priority(b), layer_priority(a), a.Name)
        self.assertEqual([r.RelatingElement.Name for r in self.rels
                          if abs(body_height(r.RelatingElement) - body_height(r.RelatedElement)) > TOL], [])
        lobby = [r for r in self.rels if {r.RelatingElement.Name, r.RelatedElement.Name} ==
                 {f"{TYPICAL} EXT LOBBY|U107:SY", f"{TYPICAL} EXT @out|U107:LD"}]
        self.assertEqual([(r.RelatingConnectionType, r.RelatedConnectionType) for r in lobby],
                         [(joins.ATSTART, joins.ATEND)])          # an L at the lobby corner, not a T

    def test_one_join_per_touching_wall_end(self):
        """A wall end meeting a perpendicular wall's reference line has exactly one join unless a collinear wall
        continues from it; an end meeting nothing has none; no end has two. A continued end is joined only as the
        wall a corner turns into: of the corner wall's height, and the thicker of the collinear walls of that height
        (a shelter wall continued by a partition; the lobby facade wall continued by the lobby parapet)."""
        by_storey = defaultdict(list)
        for w in self.walls:
            by_storey[storey_of(w)].append(w)
        count = Counter()
        partner = {}
        for r in self.rels:
            a, b = r.RelatingElement, r.RelatedElement
            count[(a.id(), r.RelatingConnectionType)] += 1
            if r.RelatedConnectionType != joins.ATPATH:
                count[(b.id(), r.RelatedConnectionType)] += 1
                partner[(a.id(), r.RelatingConnectionType)] = (b, r.RelatedConnectionType)
                partner[(b.id(), r.RelatedConnectionType)] = (a, r.RelatingConnectionType)
        self.assertLessEqual(max(count.values()), 1)
        checked = Counter()
        for st, walls in by_storey.items():
            info = {w.id(): (w, axis_world(w)) for w in walls}
            for w, (p0, p1) in info.values():
                u = (p1 - p0) / np.linalg.norm(p1 - p0)
                t = uel.get_material(w, should_inherit=False).ForLayerSet.TotalThickness
                for role, e in ((joins.ATSTART, p0), (joins.ATEND, p1)):
                    touch, coll = False, []
                    for o, (q0, q1) in info.values():
                        if o is w:
                            continue
                        v = (q1 - q0) / np.linalg.norm(q1 - q0)
                        if abs(u @ v) > 1 - TOL and (np.linalg.norm(q0 - e) < TOL or np.linalg.norm(q1 - e) < TOL):
                            coll.append(o)
                        elif abs(u @ v) < TOL:
                            s = (e - q0) @ v
                            d = abs((e - q0) @ np.array([-v[1], v[0]]))
                            touch |= d < TOL and -TOL <= s <= np.linalg.norm(q1 - q0) + TOL
                    n = count[(w.id(), role)]
                    if not touch:
                        self.assertEqual(n, 0, f"{w.Name} {role} is free but joined")
                    elif not coll:
                        self.assertEqual(n, 1, f"{w.Name} {role} meets a wall but has {n} joins")
                    elif n:
                        o, k = partner[(w.id(), role)]
                        self.assertLess(np.linalg.norm(self.ends_of(o)[k] - e), TOL, w.Name)
                        self.assertLess(abs(body_height(o) - body_height(w)), TOL, w.Name)
                        level = [c for c in coll if abs(body_height(c) - body_height(w)) < TOL]
                        tc = max([uel.get_material(c, should_inherit=False).ForLayerSet.TotalThickness for c in level],
                                 default=0.0)
                        self.assertGreater(t, tc - TOL, f"{w.Name}: a continued end joined although not the thicker")
                    checked[(touch, bool(coll), n)] += 1
        self.assertGreater(checked[(True, False, 1)], 1000)

    def test_geometry_of_joins(self):
        """L: both ends at one node, at right angles. T: the stem's end on the bar's reference line, inside it (or at
        its end where an equally thick wall continues the bar)."""
        for r in self.rels:
            a, b = r.RelatingElement, r.RelatedElement
            e = self.ends_of(a)[r.RelatingConnectionType]
            q0, q1 = axis_world(b)
            v = (q1 - q0) / np.linalg.norm(q1 - q0)
            p0, p1 = axis_world(a)
            self.assertLess(abs(((p1 - p0) / np.linalg.norm(p1 - p0)) @ v), TOL, a.Name)
            if r.RelatedConnectionType != joins.ATPATH:
                self.assertLess(np.linalg.norm(self.ends_of(b)[r.RelatedConnectionType] - e), TOL, a.Name)
                continue
            s = (e - q0) @ v
            self.assertLess(abs((e - q0) @ np.array([-v[1], v[0]])), TOL, a.Name)
            self.assertTrue(-TOL <= s <= np.linalg.norm(q1 - q0) + TOL, a.Name)

    def test_typical_storeys_alike(self):
        per = Counter(storey_of(r.RelatingElement) for r in self.rels)
        typical = [per[f"L{i}"] for i in range(2, 13)]
        self.assertEqual(len(set(typical)), 1, per)
        kinds = {st: Counter((r.RelatingConnectionType, r.RelatedConnectionType) for r in self.rels
                             if storey_of(r.RelatingElement) == st) for st in ("L2", "L7", "L12")}
        self.assertEqual(kinds["L2"], kinds["L7"])
        self.assertEqual(kinds["L2"], kinds["L12"])

    def test_every_wall_has_its_own_usage(self):
        """Bonsai's wall tools need the wall's own IfcMaterialLayerSetUsage (AXIS2); it edits it in place, so it is
        never shared. Offset and sense put the layer band where the body is (centred on the reference line)."""
        seen = set()
        for w in self.walls:
            m = uel.get_material(w, should_inherit=False)
            self.assertTrue(m is not None and m.is_a("IfcMaterialLayerSetUsage"), w.Name)
            self.assertNotIn(m.id(), seen)
            seen.add(m.id())
            self.assertEqual([o for r in m.AssociatedTo for o in r.RelatedObjects], [w])
            self.assertEqual((m.LayerSetDirection, m.DirectionSense), ("AXIS2", "POSITIVE"))
            self.assertEqual(m.ForLayerSet.id(), uel.get_material(uel.get_type(w)).id())
            self.assertAlmostEqual(m.OffsetFromReferenceLine, -m.ForLayerSet.TotalThickness / 2, places=9)
        prio = {ls.LayerSetName: ls.MaterialLayers[0].Priority for ls in self.f.by_type("IfcMaterialLayerSet")}
        self.assertGreater(prio["CORE-200 RC lift / stair core"], prio["INT-100 lightweight partition"])

    def test_reference_lines_are_the_run_nodes(self):
        """get_reference_line (the Plan/Axis/GRAPH_VIEW line, Bonsai's input) runs between the run's nodes; the body
        keeps the corner extensions."""
        ctx = [c for c in self.f.by_type("IfcGeometricRepresentationSubContext") if c.ContextIdentifier == "Axis"]
        self.assertEqual([(c.ContextType, c.TargetView) for c in ctx], [("Plan", "GRAPH_VIEW")])
        segs = set()
        for w in self.walls:
            if storey_of(w) == TYPICAL:
                a, b = axis_world(w)
                segs.add((round(a[0], 4), round(a[1], 4), round(b[0], 4), round(b[1], 4)))
        runs = {(r.p0[0], r.p0[1], r.p1[0], r.p1[1]) for r in typical_runs() if r.wall and r.wall != "RAILING" and r.t > 0}
        self.assertEqual(segs, runs)


# ----------------------------------------------------------------------------- 2. Bonsai's wall regenerator
def regenerate_storey(test, f, walls):
    """Bonsai's regenerator on every wall of a storey: (sections before, sections after, reference lines kept)."""
    before = sections(f, walls)
    axes = {w.id(): axis_world(w) for w in walls}
    for w in walls:
        test.assertIsNotNone(geometry.regenerate_wall_representation(f, w), w.Name)
    after = sections(f, walls)
    kept = all(np.linalg.norm(p - q) < TOL for w in walls for p, q in zip(axes[w.id()], axis_world(w)))
    return before, after, kept


def synthetic_runs():
    """Six small corners 20 m apart, one per case of plate_joins' height rules (finished and merged as derive does):
    A a lobby-type corner (a storey-high wall on a facade wall continued by a parapet), B a storey-high wall on two
    collinear parapet-high walls, C a parapet on a core wall passing through, D a storey-high wall on a parapet
    passing through, E an L of a storey-high wall and a parapet, F an L of two storey-high walls."""
    from estate.geom.arrangement import Run, finish_runs, merge_collinear
    from estate.rules import WALL_TYPES

    def run(name, x0, y0, x1, y1, wall, height="storey"):
        return Run((float(x0), float(y0)), (float(x1), float(y1)), name, "out", wall, WALL_TYPES[wall][1], height)
    runs = [run("Astem", 0, 0, 5, 0, "EXT"), run("Afacade", 0, 0, 0, 4, "EXT"),
            run("Aparapet", 0, -3, 0, 0, "PARAPET", "parapet"),
            run("Bstem", 20, 0, 25, 0, "EXT"), run("Blow", 20, 0, 20, 4, "EXT", "parapet"),
            run("Bparapet", 20, -3, 20, 0, "PARAPET", "parapet"),
            run("Cparapet", 40, 0, 45, 0, "PARAPET", "parapet"), run("Ccore", 40, -3, 40, 0, "CORE"),
            run("Ccore", 40, 0, 40, 4, "CORE"),
            run("Dstem", 60, 0, 65, 0, "EXT"), run("Dparapet", 60, -3, 60, 0, "PARAPET", "parapet"),
            run("Dparapet", 60, 0, 60, 4, "PARAPET", "parapet"),
            run("Eparapet", 80, 0, 85, 0, "PARAPET", "parapet"), run("Ewall", 80, 0, 80, 4, "EXT"),
            run("Fx", 100, 0, 105, 0, "EXT"), run("Fy", 100, 0, 100, 4, "EXT")]
    for i, r in enumerate(runs):
        r.idx = i
    return merge_collinear(finish_runs(runs), [])[0]


class TestRegenerate(unittest.TestCase):
    def test_regenerated_storeys_keep_every_wall_section(self):
        """Bonsai's regenerator (regenerate_wall_representation, run by its wall tools) on every wall of a storey
        rebuilds the bodies from the reference lines, usages and joins alone. On the PT4 block's void deck, typical
        storey and roof (lobby corners, lift and stair towers continued by the roof parapet) and on every storey of
        the neighbourhood centre, the union of the wall sections is the generated one in every height band
        (missing and extra < 1e-6 m²; it is exact). The plan union alone (the spec's < 1 % criterion) would not
        see a wall cut back to the face of a lower one."""
        for which, only in (("pt4", ("L1", TYPICAL, "RF")), ("nc", None)):
            f = ifcopenshell.open(str(built(which)))
            by = defaultdict(list)
            for w in f.by_type("IfcWall"):
                by[storey_of(w)].append(w)
            for st in sorted(by, key=str):
                if only and st not in only:
                    continue
                with self.subTest(file=which, storey=st):
                    before, after, kept = regenerate_storey(self, f, by[st])
                    self.assertEqual(section_differences(before, after), [])
                    self.assertTrue(kept)                       # the reference lines stay on the nodes
                    ub = shapely.union_all([s[0] for s in before.values()])
                    ua = shapely.union_all([s[0] for s in after.values()])
                    self.assertLess(ub.symmetric_difference(ua).area, 0.01 * ub.area)
                    if which == "pt4" and st == TYPICAL:     # Bonsai really rebuilt the corners its own way
                        changed = sum(before[k][0].symmetric_difference(after[k][0]).area > 1e-6 for k in before)
                        self.assertGreater(changed, 20)

    def test_height_rules(self):
        """plate_joins joins walls only where Bonsai gives back the generated corner at every height; an extended end
        it leaves unjoined keeps its extension in the reference line. Built (mscp.plate_walls, as the car park and
        centre storeys) and regenerated in Bonsai, every corner keeps every wall section."""
        from estate.ifc.writer import IfcWriter
        from estate.rules import PARAPET_H
        from estate.site.mscp import plate_walls
        runs = synthetic_runs()
        name = {r.idx: r.left for r in runs}
        by = {r.left: r for r in runs}
        jn = joins.plate_joins(runs, lambda r: 2.8 if r.height == "storey" else PARAPET_H)
        got = sorted((name[j.relating], j.relating_end, name[j.related], j.related_end) for j in jn)
        self.assertEqual(got, [("Astem", joins.ATSTART, "Afacade", joins.ATSTART),       # L into the facade wall
                               ("Cparapet", joins.ATSTART, "Ccore", joins.ATPATH),       # T on the taller core
                               ("Fx", joins.ATSTART, "Fy", joins.ATSTART)])
        joined = joins.joined_ends(jn)
        for nm in ("Bstem", "Dstem", "Eparapet", "Ewall"):      # unjoined, extended: the line runs to the body end
            r = by[nm]
            self.assertGreater(r.ext0, 0.0, nm)
            self.assertEqual(joins.run_axis(r, joined), (0.0, r.ext0 + r.length), nm)
        self.assertEqual(joins.run_axis(by["Astem"], joined), joins.run_axis(by["Astem"]))
        W = IfcWriter(guid_key="test/heights", typed_openings=True, **config.identity(config.load()))
        st = W.storeys(W.building("Height rules"), ["L1"], [0.0])[0]
        plate_walls(W, SimpleNamespace(runs=runs, openings=[]), 0.0, 2.8, st, "H")
        W.flush()
        out = OUT / "heights" / "height_rules.ifc"
        out.parent.mkdir(parents=True, exist_ok=True)
        W.write(out)
        W.close()
        f = ifcopenshell.open(str(out))
        self.assertEqual(len(f.by_type("IfcRelConnectsPathElements")), 3)
        before, after, kept = regenerate_storey(self, f, f.by_type("IfcWall"))
        self.assertEqual(section_differences(before, after), [])
        self.assertTrue(kept)


# ----------------------------------------------------------------------------- 3. door / window data
class TestDoorWindowData(unittest.TestCase):
    def check_types(self, f):
        body = urep.get_context(f, "Model", "Body", "MODEL_VIEW")
        n = Counter()
        for t in f.by_type("IfcDoorType") + f.by_type("IfcWindowType"):
            door = t.is_a("IfcDoorType")
            name = "BBIM_Door" if door else "BBIM_Window"
            if door and t.OperationType not in joins.BONSAI_DOOR_TYPES:
                self.assertIsNone(uel.get_pset(t, name), t.Name)        # ROLLINGUP: Bonsai cannot rebuild it
                n["skipped"] += 1
                continue
            ps = uel.get_pset(t, name)
            self.assertIsNotNone(ps, t.Name)
            prop = next(p for d in t.HasPropertySets or () if d.Name == name for p in d.HasProperties)
            self.assertEqual((prop.Name, prop.NominalValue.is_a()), ("Data", "IfcText"))
            p = bonsai_props(BONSAI_DOOR if door else BONSAI_WINDOW, ps["Data"])
            kw = bonsai_door_kwargs(p) if door else bonsai_window_kwargs(p)
            add = geometry.add_door_representation if door else geometry.add_window_representation
            stored = t.RepresentationMaps[0].MappedRepresentation.Items
            self.assertTrue(same_items(stored, add(f, context=body, **kw).Items), t.Name)
            ours = joins.door_kwargs(json.loads(ps["Data"])) if door else joins.window_kwargs(json.loads(ps["Data"]))
            self.assertTrue(same_items(stored, add(f, context=body, **ours).Items), t.Name)
            self.assertEqual(p["overall_width"], round(p["overall_width"], 4))
            occ = [o for r in t.Types for o in r.RelatedObjects]
            self.assertTrue(occ and all(uel.get_pset(o, name) for o in occ))   # tool.Parametric.is_door / is_window
            self.assertEqual(t.OperationType if door else t.PartitioningType, p["door_type" if door else "window_type"])
            n["door" if door else "window"] += 1
        return n

    def test_types_are_built_from_their_data(self):
        """Every door / window type: Bonsai's own mapping of its BBIM data (defaults, then the data, then
        update_*_modifier_representation's keyword mapping) triangulates to the stored representation (1e-6)."""
        n = self.check_types(ifcopenshell.open(str(built("pt4"))))
        self.assertGreater(n["door"], 10)
        self.assertGreater(n["window"], 5)
        n = self.check_types(ifcopenshell.open(str(built("nc"))))
        self.assertEqual(n["skipped"], 1)                 # the stall roller shutter
        self.assertGreaterEqual(n["door"], 5)

    def test_type_geometry_unchanged(self):
        """The data are the API defaults, so the door and window types have the geometry they had before."""
        new, old = ifcopenshell.open(str(built("pt4"))), ifcopenshell.open(str(built("pt4_flat")))
        for cls in ("IfcDoorType", "IfcWindowType"):
            a = {t.Name: t for t in new.by_type(cls)}
            b = {t.Name: t for t in old.by_type(cls)}
            self.assertEqual(set(a), set(b))
            for k in a:
                self.assertTrue(same_items(a[k].RepresentationMaps[0].MappedRepresentation.Items,
                                           b[k].RepresentationMaps[0].MappedRepresentation.Items), k)

    def test_data_helpers(self):
        d = joins.door_data("SLIDING_TO_LEFT", 0.8999999999999999, 2.1)
        self.assertEqual(d["overall_width"], 0.9)
        self.assertNotIn("lining_to_panel_offset_x", d["lining_properties"])    # Bonsai leaves them out for sliding
        self.assertIsNone(joins.door_data("ROLLINGUP", 2.4, 2.4))
        self.assertEqual(len(joins.window_kwargs(joins.window_data(1.2, 1.0))["panel_properties"]), 1)


# ----------------------------------------------------------------------------- 4. determinism, IFC4, validation
class TestBuilds(unittest.TestCase):
    def test_two_builds_identical(self):
        self.assertEqual(built("pt4").read_bytes(), built("pt4_again").read_bytes())
        self.assertEqual(built("nc").read_bytes(), built("nc_again").read_bytes())

    def test_stable_global_ids(self):
        """guid_base files: every join and usage association has a GlobalId keyed by its wall (and end), every BBIM
        pset one keyed by its type, so they do not depend on creation order; all GlobalIds are unique."""
        f = ifcopenshell.open(str(built("nc")))
        ids = [e.GlobalId for e in f.by_type("IfcRoot")]
        self.assertEqual(len(ids), len(set(ids)))
        from estate import guids
        base = f"{config.load()['estate']['code']}/NC_514"
        for r in f.by_type("IfcRelConnectsPathElements"):
            self.assertEqual(r.GlobalId, guids.from_text(
                f"{base}/IfcRelConnectsPathElements/{r.RelatingElement.GlobalId}/{r.RelatingConnectionType}"))
        for w in f.by_type("IfcWall"):
            rel = uel.get_material(w, should_inherit=False).AssociatedTo[0]
            self.assertEqual(rel.GlobalId, guids.from_text(f"{base}/IfcRelAssociatesMaterial/{w.GlobalId}"))
        n = 0
        for t in f.by_type("IfcDoorType") + f.by_type("IfcWindowType"):
            for d in t.HasPropertySets or ():
                if d.Name in ("BBIM_Door", "BBIM_Window"):
                    self.assertEqual(d.GlobalId, guids.from_text(f"{base}/pset/{t.GlobalId}/{d.Name}"))
                    n += 1
        self.assertGreaterEqual(n, 5)

    def test_ifc4_copy_keeps_parametric_data(self):
        """The IFC4 engine copy (ifcpatch Migrate, as pipeline/runner.migrate_ifc4) keeps joins, usages, reference
        lines, BBIM psets and layer priorities, with the same GlobalIds."""
        import ifcpatch
        for which, name in (("nc", "NC_514"), ("pt4", "t_pt4")):
            src = built(which)
            g = ifcpatch.execute({"input": str(src), "file": ifcopenshell.open(str(src)), "recipe": "Migrate",
                                  "arguments": ["IFC4"]})
            out = OUT / "ifc4" / f"{name}_ifc4.ifc"
            out.parent.mkdir(parents=True, exist_ok=True)
            ifcpatch.write(g, str(out))
            f4 = ifcopenshell.open(str(out))
            self.assertEqual(f4.schema, "IFC4")
            a, b = parametric_summary(ifcopenshell.open(str(src))), parametric_summary(f4)
            for k in a:
                self.assertTrue(a[k], (which, k))
                self.assertEqual(a[k], b[k], (which, k))

    def test_validation_no_worse(self):
        """estate validate's per-file checks (ifcqa + IDS, programme, geometry): the parametric PT4 block fails no
        check the baseline passes and has no more failing items; ifcqa (schema, IDS) has no errors at all."""
        from estate.validate.cli import FAMILIES, validate_one

        def failing(path):
            rec = validate_one(path)
            return rec, {(fam, r["check"]): (r["level"], r["count"]) for fam in FAMILIES
                         for r in rec.get(fam, {}).get("results", []) if r["level"] in ("error", "warn")}
        rec_new, new = failing(built("pt4"))
        rec_old, old = failing(built("pt4_flat"))
        self.assertEqual([k for k in new if k[0] == "ifcqa"], [])
        for k, (lv, n) in new.items():
            self.assertIn(k, old, f"new failure {k}")
            self.assertLessEqual(n, old[k][1], k)
        self.assertEqual(rec_new["summary"]["failed_errors"], rec_old["summary"]["failed_errors"])
        self.assertLessEqual(rec_new["summary"]["warning_items"], rec_old["summary"]["warning_items"])
        rec_nc, nc = failing(built("nc"))
        self.assertEqual(nc, {})

    def test_tessellation_unchanged(self):
        """The export tessellation (export/meshcache) of every element: same triangles and vertices as the baseline."""
        from estate.export import meshcache

        def by_name(path):
            f = ifcopenshell.open(str(path))
            m = meshcache.tessellate(f)
            out, seen = {}, Counter()
            for g in sorted(m, key=lambda g: f.by_guid(g).id()):
                r = m[g]
                k = (r["cls"], r["name"], r["storey"])
                out[k + (seen[k],)] = r
                seen[k] += 1
            return out
        new, old = by_name(built("pt4")), by_name(built("pt4_flat"))
        self.assertEqual(set(new), set(old))
        self.assertEqual(sum(len(r["faces"]) for r in new.values()), sum(len(r["faces"]) for r in old.values()))
        for k, r in new.items():
            o = old[k]
            self.assertEqual(len(r["faces"]), len(o["faces"]), k)
            self.assertTrue(np.allclose(np.sort(r["verts"], axis=0), np.sort(o["verts"], axis=0), atol=TOL), k)
            self.assertTrue(np.array_equal(r["colours"], o["colours"]), k)


# ----------------------------------------------------------------------------- 5. Bonsai in Blender
JOB = r'''"""Test job (written by tests/test_parametric.py): Bonsai's door editing and wall recalculation on a .blend."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, ROOT)
from estate.blender import _boot  # noqa: E402


def sections(f, els):
    """[(plan footprint, bottom z, top z)] of the bodies (as tests/test_parametric.sections)."""
    import numpy as np
    import shapely
    import ifcopenshell.geom
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    s.set("disable-opening-subtractions", True)
    out = []
    for el in els:
        sh = ifcopenshell.geom.create_shape(s, el)
        v = np.array(sh.geometry.verts).reshape(-1, 3)
        tris = (shapely.Polygon(v[t][:, :2]) for t in np.array(sh.geometry.faces).reshape(-1, 3))
        out.append((shapely.union_all([p for p in tris if p.area > 1e-10]), float(v[:, 2].min()), float(v[:, 2].max())))
    return out


def footprint(f, els):
    import shapely
    return shapely.union_all([s[0] for s in sections(f, els)])


def band_differences(before, after):
    """[(z, missing m², extra m²)] of the height bands whose wall section union differs (> 1e-6 m²)."""
    import shapely
    zs = sorted({round(z, 6) for sec in (before, after) for _, z0, z1 in sec for z in (z0, z1)})
    out = []
    for a, b in zip(zs, zs[1:]):
        z = (a + b) / 2
        ub = shapely.union_all([fp for fp, z0, z1 in before if z0 < z < z1])
        ua = shapely.union_all([fp for fp, z0, z1 in after if z0 < z < z1])
        if ub.difference(ua).area > 1e-6 or ua.difference(ub).area > 1e-6:
            out.append((z, ub.difference(ua).area, ua.difference(ub).area))
    return out


def world_bounds(obj):
    import numpy as np
    m = np.array(obj.matrix_world)
    v = np.array([(m @ np.array([*c, 1.0]))[:3] for c in obj.bound_box])
    return [v.min(axis=0).tolist(), v.max(axis=0).tolist()]


def select_only(obj):
    import bpy
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def run(a):
    import bpy
    import shapely
    _boot.boot()
    import bonsai.tool as tool
    import ifcopenshell.util.element as uel
    bpy.ops.wm.open_mainfile(filepath=a["blend"])
    f = tool.Ifc.get()
    assert f is not None, "Bonsai has no IFC"
    door = next(d for d in f.by_type("IfcDoor") if d.Name == a["door"])
    win = next(w for w in f.by_type("IfcWindow") if tool.Ifc.get_object(w) is not None)
    out = {"is_door": tool.Parametric.is_door(door), "is_window": tool.Parametric.is_window(win),
           "door_is_window": tool.Parametric.is_window(door)}
    obj = tool.Ifc.get_object(door)
    b0, n0 = world_bounds(obj), len(obj.data.vertices)
    select_only(obj)
    r1 = bpy.ops.bim.enable_editing_door()
    editing = tool.Model.get_door_props(obj).is_editing
    r2 = bpy.ops.bim.finish_editing_door()
    obj = tool.Ifc.get_object(door)
    out["door"] = {"enable": sorted(r1), "editing": editing, "finish": sorted(r2), "bounds": [b0, world_bounds(obj)],
                   "verts": [n0, len(obj.data.vertices)], "material": getattr(uel.get_material(door), "Name", None),
                   "still_editing": tool.Model.get_door_props(obj).is_editing}
    wall = next(w for w in f.by_type("IfcWall") if w.Name == a["wall"])
    storey = uel.get_container(wall)
    walls = [w for w in f.by_type("IfcWall") if uel.get_container(w) == storey]
    rebuilt = [wall] + [r.RelatedElement for r in wall.ConnectedTo] + [r.RelatingElement for r in wall.ConnectedFrom]
    sec0, own0 = sections(f, walls), footprint(f, [wall])
    select_only(tool.Ifc.get_object(wall))
    r3 = bpy.ops.bim.recalculate_wall()
    sec1, own1 = sections(f, walls), footprint(f, [wall])
    fp0, fp1 = (shapely.union_all([s[0] for s in sec]) for sec in (sec0, sec1))
    out["wall"] = {"recalculate": sorted(r3), "rebuilt": len(rebuilt), "union": [fp0.area, fp1.area],
                   "symdiff": fp0.symmetric_difference(fp1).area, "bands": band_differences(sec0, sec1),
                   "own_change": own0.symmetric_difference(own1).area,
                   "mesh_bounds": world_bounds(tool.Ifc.get_object(wall)), "ifc_bounds": list(own1.bounds)}
    return out


if __name__ == "__main__":
    _boot.main(run)
'''


@unittest.skipIf(SKIP_BLENDER, "Blender tests disabled")
class TestBonsai(unittest.TestCase):
    def test_bonsai_edits_doors_and_walls(self):
        """In Bonsai: a door and a window are parametric (BBIM data inherited from their types); enable -> finish
        editing a door rebuilds it from its data within 1 mm; bim.recalculate_wall on a wall (Bonsai regenerates it
        and the walls joined to it, here at a lobby corner where a parapet continues the facade wall) keeps every
        wall section of the storey at every height, and the wall's mesh matches its new IFC body."""
        from estate.blender.run import run_blender
        d = TESTS / "BLK_P"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        ifc = d / "BLK_P.ifc"
        shutil.copyfile(built("pt4"), ifc)
        made = run_blender("make_blend", {"ifc": str(ifc)}, timeout=900)
        job = TESTS / "bonsai_parametric_job.py"
        job.write_text(JOB.replace("sys.path.insert(0, ROOT)", f"sys.path.insert(0, {str(env.ROOT)!r})"),
                       encoding="utf-8")
        res = run_blender(str(job), {"blend": made["blend"], "door": "#05-101 Main door", "wall": "L5 EXT LOBBY|U107:SY"},
                          timeout=900)
        self.assertEqual((res["is_door"], res["is_window"], res["door_is_window"]), (True, True, False))
        dr = res["door"]
        self.assertEqual((dr["enable"], dr["editing"], dr["finish"], dr["still_editing"]),
                         (["FINISHED"], True, ["FINISHED"], False))
        self.assertLess(np.abs(np.array(dr["bounds"][0]) - np.array(dr["bounds"][1])).max(), 1e-3)
        self.assertEqual(dr["verts"][0], dr["verts"][1])
        self.assertEqual(dr["material"], "Timber door")        # inherited from the type after Bonsai's edit
        wr = res["wall"]
        self.assertEqual(wr["recalculate"], ["FINISHED"])
        self.assertGreater(wr["rebuilt"], 1)
        self.assertLess(wr["symdiff"], 0.01 * wr["union"][0])
        self.assertEqual(wr["bands"], [])                      # and every wall section, at every height
        self.assertGreater(wr["own_change"], 1e-3)             # Bonsai rebuilt the wall's corners itself
        mb, ib = np.array(wr["mesh_bounds"]), np.array(wr["ifc_bounds"])
        self.assertLess(np.abs(mb[:, :2].ravel() - ib[[0, 1, 2, 3]].reshape(2, 2).ravel()).max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
