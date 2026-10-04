"""Drawing tests: the numpy mesh/plane cut (closed loops, holes, vertices on the plane), silhouettes, flight run
direction, the IFC door-swing convention, and the IFC-cut stair sections and plans of the PT4 test block
(build/t_pt4.ifc): one sheet per stair enclosure, door swings sweeping into rooms, deterministic output.

Run: "<blender python>" -I -B tests/test_drawings.py -v
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estate.env import bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402
import shapely  # noqa: E402
import shapely.affinity  # noqa: E402
from PIL import Image  # noqa: E402

from estate.env import BUILD  # noqa: E402
from estate.draw import iplans, sections  # noqa: E402

T_PT4 = BUILD / "t_pt4.ifc"
PLAN_AXES = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])


def box_mesh(x0, y0, z0, x1, y1, z1, inward=False):
    """A closed box as (verts, faces), outward-facing unless inward (the inner skin of a hollow solid)."""
    V = np.array([(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], float)
    F = np.array([(0, 2, 1), (1, 2, 3), (4, 5, 6), (5, 7, 6), (0, 1, 4), (1, 5, 4), (2, 6, 3), (3, 6, 7),
                  (0, 4, 2), (2, 4, 6), (1, 3, 5), (3, 7, 5)])
    return V, (F[:, ::-1] if inward else F)


def mesh(*parts):
    vs, fs, n = [], [], 0
    for V, F in parts:
        vs.append(V)
        fs.append(F + n)
        n += len(V)
    return dict(verts=np.vstack(vs), faces=np.vstack(fs), cls="IfcWall", name="t", storey="L1")


class TestCut(unittest.TestCase):
    def test_box_cut_is_closed_square(self):
        d = mesh(box_mesh(0, 0, 0, 2, 1, 3))
        g = sections.cut(d, (0, 0, 1.2), (0, 0, 1), PLAN_AXES)
        self.assertAlmostEqual(g.area, 2.0, places=6)
        self.assertEqual(g.geom_type, "Polygon")
        seg = sections.plane_segments(d["verts"], d["faces"], (0, 0, 1.2), (0, 0, 1))
        self.assertEqual(seg.shape, (8, 2, 3))          # 4 sides x 2 triangles, each crossing once

    def test_shared_edges_meet_exactly(self):
        d = mesh(box_mesh(0.1, 0.3, 0.0, 2.7, 1.9, 3.3))
        seg = sections.plane_segments(d["verts"], d["faces"], (0.37, 0.0, 0.0), (0.6, 0.8, 0.0))
        ends, counts = np.unique(seg.reshape(-1, 3), axis=0, return_counts=True)
        self.assertTrue((counts == 2).all(), "every cut point is shared by exactly two segments")

    def test_hollow_solid_keeps_its_hole(self):
        d = mesh(box_mesh(0, 0, 0, 2, 2, 1), box_mesh(0.5, 0.5, -1, 1.5, 1.5, 2, inward=True))
        g = sections.cut(d, (0, 0, 0.5), (0, 0, 1), PLAN_AXES)
        self.assertAlmostEqual(g.area, 4.0 - 1.0, places=6)

    def test_vertex_on_plane_and_miss(self):
        d = mesh(box_mesh(0, 0, 0, 1, 1, 1))
        self.assertIsNone(sections.cut(d, (0, 0, 2.0), (0, 0, 1), PLAN_AXES))
        g = sections.cut(d, (0, 0, 1.0), (0, 0, 1), PLAN_AXES)    # the top face lies on the plane
        self.assertTrue(g is None or abs(g.area - 1.0) < 1e-6)

    def test_vertical_cut_and_silhouettes(self):
        d = mesh(box_mesh(0, 0, 0, 2, 1, 3))
        axes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])       # look along x: horizontal = y, vertical = z
        g = sections.cut(d, (0.7, 0, 0), (1, 0, 0), axes)
        self.assertAlmostEqual(g.area, 1.0 * 3.0, places=6)
        self.assertAlmostEqual(sections.silhouette(d, PLAN_AXES).area, 2.0, places=6)
        self.assertAlmostEqual(sections.silhouette(d, axes).area, 3.0, places=6)

    def test_run_direction(self):
        V, F = box_mesh(0, 0, 0, 1.2, 2.5, 0.3)
        V = V.copy()
        V[:, 2] += 0.6 * (2.5 - V[:, 1])            # rises towards -y
        r = sections.run_direction(V)
        self.assertTrue(np.allclose(r, (0.0, -1.0), atol=1e-6), r)


class TestDoorConvention(unittest.TestCase):
    def test_hinge_side_and_opening_side(self):
        o, ux, uy = np.array([10.0, 5.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0])
        (h, c, r), = iplans.swing_leaves(o, ux, uy, 0.9, "SINGLE_SWING_LEFT", face=0.15)
        self.assertTrue(np.allclose(h, o + uy * 0.15) and np.allclose(c, ux) and r == 0.9)
        (h, c, r), = iplans.swing_leaves(o, ux, uy, 0.9, "SINGLE_SWING_RIGHT", face=0.15)
        self.assertTrue(np.allclose(h, o + ux * 0.9 + uy * 0.15) and np.allclose(c, -ux))
        two = iplans.swing_leaves(o, ux, uy, 1.8, "DOUBLE_DOOR_SINGLE_SWING")
        self.assertEqual([round(r, 3) for _, _, r in two], [0.9, 0.9])


@unittest.skipUnless(T_PT4.exists(), "build/t_pt4.ifc not built (python build/t_pt4.py)")
class TestIfcDrawings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from estate.export import meshcache
        cls.data = meshcache.load(T_PT4)
        cls.tmp = tempfile.TemporaryDirectory()
        t0 = time.time()
        cls.files = iplans.draw_all("t_pt4", T_PT4, cls.tmp.name, data=cls.data)
        cls.seconds = time.time() - t0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_sheets_written(self):
        f, meshes = self.data
        groups = sections.stair_groups(f, meshes)
        self.assertEqual(len(groups), 2)                                   # PT4: two stair cores
        self.assertTrue(all(len(g["stairs"]) == len({s.Name for s in g["stairs"]}) for g in groups))
        names = sorted(p.name for p in self.files)
        self.assertEqual(names, sorted(["t_pt4_stair_1.png", "t_pt4_stair_2.png", "t_pt4_L1_ifc.png",
                                        "t_pt4_L5_ifc.png", "t_pt4_RF_ifc.png"]))
        self.assertLess(self.seconds, 60)

    def test_section_shows_cut_flights_and_slabs(self):
        im = np.asarray(Image.open(Path(self.tmp.name) / "t_pt4_stair_1.png").convert("RGB")).reshape(-1, 3)
        colours = {tuple(c) for c in np.unique(im, axis=0)}
        for c in (sections.NEAR["IfcStairFlight"], sections.NEAR["IfcSlab"], sections.NEAR["IfcWall"],
                  sections.BEYOND["IfcStairFlight"][0]):
            self.assertIn(c, colours)

    def test_plan_shows_walls_glazing_doors(self):
        im = np.asarray(Image.open(Path(self.tmp.name) / "t_pt4_L5_ifc.png").convert("RGB")).reshape(-1, 3)
        colours = {tuple(c) for c in np.unique(im, axis=0)}
        for c in (iplans.DOOR, iplans.GLASS):
            self.assertIn(c, colours)
        self.assertGreater(int((im == np.array(iplans.DOOR)).all(1).sum()), 500)

    def test_swings_sweep_into_spaces(self):
        f, meshes = self.data
        st = next(s for s in f.by_type("IfcBuildingStorey") if s.Name == "L5")
        axes, off = iplans.building_frame(f, st)
        rooms = [sections.silhouette(meshes[s.GlobalId], axes, off) for s in f.by_type("IfcSpace")
                 if meshes.get(s.GlobalId, {}).get("storey") == "L5"]
        stairs = [shapely.affinity.affine_transform(g["footprint"], [*axes[:2].T.ravel(), *off])
                  for g in sections.stair_groups(f, meshes)]       # older fixtures have no stair IfcSpaces
        spaces = shapely.union_all(rooms + stairs)
        doors = [d for d in f.by_type("IfcDoor") if d.ContainedInStructure[0].RelatingStructure == st]
        swings = 0
        for door in doors:
            sym = iplans.door_symbol(door, meshes, axes[:2], off)
            for hinge, closed, r in sym["leaves"]:
                sector = shapely.Polygon([hinge] + iplans.arc_points(hinge, r, closed, sym["uy"]))
                self.assertGreater(sector.intersection(spaces).area / sector.area, 0.6, door.Name)
                swings += 1
        self.assertGreater(swings, 20)

    def test_deterministic(self):
        f, _ = self.data
        with tempfile.TemporaryDirectory() as td:
            p = iplans.ifc_plan(T_PT4, "L5", Path(td) / "again.png", title="t_pt4", data=self.data)
            self.assertEqual(p.read_bytes(), (Path(self.tmp.name) / "t_pt4_L5_ifc.png").read_bytes())
        self.assertEqual(iplans.plan_storeys(f), ["L1", "L5", "RF"])


if __name__ == "__main__":
    unittest.main()
