"""Engine export tests: glb writer/reader spec checks and mesh instancing, LOD export of the PT4 test block (door
leaves that swing open about their hinges, resolved portals, shared meshes, LOD1 smaller than LOD0 on disk), the
neighbourhood centre (shutters that roll, instanced hawker furniture, spawns outside the hall), site tiles, local
frames, spawn points on the pedestrian network and at the bus stops, the manifest against the masterplan, and
(when Blender is installed) the round trip through Blender's glTF importer."""
from __future__ import annotations

import json
import multiprocessing
import struct
import sys
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estate.env import bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402

from estate import config, env  # noqa: E402
from estate.export import glb  # noqa: E402

T_PT4 = env.BUILD / "t_pt4.ifc"
OUT = env.BUILD / "export_tests" / "unittest"
MODEL_SITE = env.MODEL / "SITE.ifc"


def cube(c=(0.0, 0.0, 0.0), s=1.0):
    """12 outward-facing triangles of an axis-aligned cube (Z up)."""
    v = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float) * s + c
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    out = []
    for a, b, c2, d in quads:
        out += [[v[a], v[b], v[c2]], [v[a], v[c2], v[d]]]
    return np.array(out)


class TestGlb(unittest.TestCase):
    def test_cube_round_trip(self):
        tris = cube()
        W = glb.GlbWriter()
        red, grey = W.material("red", (1, 0, 0, 1)), W.material("glass", (0.5, 0.7, 0.9, 0.4))
        self.assertEqual(W.material("red", (1, 0, 0, 1)), red)
        p1 = glb.soup(glb.to_yup(tris[:6]), glb.box_uvs(tris[:6]))
        p2 = glb.soup(glb.to_yup(tris[6:]), glb.box_uvs(tris[6:]))
        root = W.node("root")
        M = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 3], [0, 0, 0, 1]], float)   # 90 deg about z + move
        W.node("DOOR_test_0123456789abcdefghijkl", W.mesh("cube", [(p1, red), (p2, grey)]),
               matrix=glb.matrix_to_yup(M), parent=root)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cube.glb"
            stats = W.write(path)
            data = path.read_bytes()
            self.assertEqual(len(data) % 4, 0)
            magic, version, length = struct.unpack_from("<III", data, 0)
            self.assertEqual((magic, version, length), (glb.MAGIC, 2, len(data)))
            jlen, jtype = struct.unpack_from("<II", data, 12)
            self.assertEqual((jlen % 4, jtype), (0, glb.CHUNK_JSON))
            self.assertEqual(glb.validate(path), [])
            g, binary = glb.read_glb(path)
            s = glb.summary(path)
        self.assertEqual(stats["triangles"], 12)
        self.assertEqual(s["triangles"], 12)
        self.assertEqual((s["nodes"], s["mesh_nodes"], s["door_nodes"], s["materials"]), (2, 1, 1, 2))
        self.assertEqual(sum(g["accessors"][p["attributes"]["POSITION"]]["count"] for p in g["meshes"][0]["primitives"]), 24)
        self.assertEqual(g["materials"][1]["alphaMode"], "BLEND")
        self.assertEqual(g["materials"][0]["pbrMetallicRoughness"]["roughnessFactor"], 0.8)
        # world bounds (Z up) of the rotated, moved unit cube
        np.testing.assert_allclose(s["bounds"], [[9, 20, 3], [10, 21, 4]], atol=1e-6)
        for v in g["bufferViews"]:
            self.assertEqual(v["byteOffset"] % 4, 0)

    def test_soup_cleaning(self):
        t = cube()[:2]
        dup = np.concatenate([t, t[:1], [[[0, 0, 0], [1, 0, 0], [2, 0, 0]]]])   # duplicate + zero-area triangle
        p = glb.soup(dup)
        self.assertEqual(len(p["indices"]) // 3, 2)
        self.assertEqual(list(p["kept"]), [0, 1])
        self.assertEqual(len(p["positions"]), 4)
        np.testing.assert_allclose(np.linalg.norm(p["normals"], axis=1), 1.0, atol=1e-6)

    def test_frames(self):
        v = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(glb.to_yup(v), [[1, 3, -2]])
        np.testing.assert_allclose(glb.from_yup(glb.to_yup(v)), v)
        M = np.eye(4)
        M[:3, 3] = (1, 2, 3)
        np.testing.assert_allclose(glb.matrix_to_yup(M)[:3, 3], [1, 3, -2])
        self.assertAlmostEqual(np.linalg.det(glb.YUP[:3, :3]), 1.0)

    def test_empty_file(self):
        W = glb.GlbWriter()
        W.node("only")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "empty.glb"
            W.write(path)
            self.assertEqual(glb.validate(path), [])
            self.assertEqual(glb.summary(path)["triangles"], 0)

    def test_shared_mesh(self):
        """One mesh drawn by three nodes: counted per node, stored once."""
        tris = cube()
        W = glb.GlbWriter()
        mesh = W.mesh("cube", [(glb.soup(glb.to_yup(tris), glb.box_uvs(tris)), W.material("red", (1, 0, 0, 1)))])
        root = W.node("root")
        for k in range(3):
            M = np.eye(4)
            M[:3, 3] = (5.0 * k, 0.0, 0.0)
            W.node(f"DOOR_d{k}_x" if k else "DOOR_d0_x_leaf", mesh, matrix=glb.matrix_to_yup(M), parent=root)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "shared.glb"
            stats = W.write(path)
            self.assertEqual(glb.validate(path), [])
            s = glb.summary(path)
        self.assertEqual((stats["meshes"], stats["mesh_nodes"], stats["triangles"]), (1, 3, 36))
        self.assertEqual((s["meshes"], s["mesh_nodes"], s["triangles"]), (1, 3, 36))
        self.assertEqual((s["door_nodes"], s["door_leaves"]), (2, 1))
        np.testing.assert_allclose(s["bounds"], [[0, 0, 0], [11, 1, 1]], atol=1e-6)

    def test_node_counts(self):
        c = glb.node_counts(["DOOR_a_g1", "DOOR_a_g1_leaf", "DOOR_b_g2", "DOOR_b_g2_leaf_L", "DOOR_b_g2_leaf_R",
                             "WIN_w_g3", "LIFT_l_g4", "TREE_t_g5", "FURN_f_g6", "L1_static", "L1DOOR_a_g1", "L1WIN_w_g3"])
        self.assertEqual(c, dict(door_nodes=2, door_leaves=3, window_nodes=1, lift_nodes=1, tree_nodes=1,
                                 furniture_nodes=1))


class TestGeometryHelpers(unittest.TestCase):
    def test_split_grid(self):
        """A triangle across two tile lines is cut into pieces that each lie in one tile, keeping area and normal."""
        from estate.export.engine import split_grid
        tri = np.array([[[10.0, 10.0, 0.0], [250.0, 30.0, 0.0], [40.0, 180.0, 0.0]]])
        pieces, src = split_grid(tri, 100.0)
        self.assertGreater(len(pieces), 3)
        self.assertTrue((src == 0).all())
        n, ln = glb.face_normals(pieces)
        n0, l0 = glb.face_normals(tri)
        self.assertAlmostEqual(float(ln.sum() / 2), float(l0.sum() / 2), places=6)
        self.assertTrue((n[:, 2] > 0).all() and n0[0, 2] > 0)
        for p in pieces:
            i, j = np.floor(p.mean(axis=0)[:2] / 100.0)
            self.assertTrue((p[:, 0] >= i * 100 - 1e-6).all() and (p[:, 0] <= (i + 1) * 100 + 1e-6).all())
            self.assertTrue((p[:, 1] >= j * 100 - 1e-6).all() and (p[:, 1] <= (j + 1) * 100 + 1e-6).all())
        same, src = split_grid(cube(c=(5.0, 5.0, 0.0)), 100.0)
        self.assertEqual(len(same), 12)

    def test_octahedron(self):
        from estate.export.engine import octahedron
        t = octahedron((0, 0, 2), (4, 2, 6))
        n, _ = glb.face_normals(t)
        out = ((t.mean(axis=1) - (2, 1, 4)) * n).sum(axis=1)
        self.assertEqual(len(t), 8)
        self.assertTrue((out > 0).all())
        np.testing.assert_allclose([t.reshape(-1, 3).min(0), t.reshape(-1, 3).max(0)], [[0, 0, 2], [4, 2, 6]])

    def test_resolve_sides(self):
        """Link ends in no room: outside the outline -> outside; inside -> the FromSpace / ToSpace not yet used."""
        from shapely.geometry import box
        from estate.export.engine import BuildingExport
        stair = dict(guid="S", name="L2-STAIR1", storey="L2", _poly=box(0, 0, 3, 5))
        lobby = dict(guid="C", name="L2-CORR", storey="L2", _poly=box(3, 0, 10, 5))
        by_name = {r["name"]: [r] for r in (stair, lobby)}
        name_of = {r["guid"]: r["name"] for r in (stair, lobby)}
        dr = dict(storey="L2", from_face="L2-STAIR1", to_face="L2-CORR")
        ends = ((2.7, 2.0), (3.3, 2.0))
        res = BuildingExport._resolve_sides(dr, [None, "C"], ends, [True, True], by_name, name_of)
        self.assertEqual(res, (["S", "C"], [False, False], ["L2-STAIR1", "L2-CORR"]))
        res = BuildingExport._resolve_sides(dr, [None, None], ends, [True, True], by_name, name_of)
        self.assertEqual(res[0], ["S", "C"])
        res = BuildingExport._resolve_sides(dict(dr, from_face="STAIR1", to_face="CORR"), [None, None], ends[::-1],
                                            [True, True], by_name, name_of)
        self.assertEqual(res[0], ["C", "S"])                  # face ids + storey prefix, ends swapped
        res = BuildingExport._resolve_sides(dict(dr, to_face="@out"), ["S", None], ends, [True, False], by_name,
                                            name_of)
        self.assertEqual(res, (["S", None], [False, True], ["L2-STAIR1", "outside"]))
        res = BuildingExport._resolve_sides(dict(dr, from_face="L2-PLANT"), [None, "C"], ends, [True, True], by_name,
                                            name_of)
        self.assertEqual(res[0], [None, "C"])                 # a room without an IfcSpace stays unresolved

    def test_entrance_spawns(self):
        """Spawns step out along the path that leaves the entrance, clear of a driveway that runs up to the door."""
        from shapely.geometry import box
        from estate.export.manifest import entrance_spawns
        fp = box(0, 0, 40, 12)
        graph = {"nodes": [dict(id="e1", xy=[20.0, -0.1], kind="entrance", ref="BLK_1"),
                           dict(id="l1", xy=[20.0, 4.0], kind="lift_lobby", ref="BLK_1"),
                           dict(id="j1", xy=[20.0, -8.0], kind="junction"),
                           dict(id="e2", xy=[40.1, 6.0], kind="entrance", ref="BLK_1"),
                           dict(id="j2", xy=[48.0, 6.0], kind="junction")],
                 "edges": [dict(a="e1", b="l1", kind="footpath"), dict(a="e1", b="j1", kind="footpath"),
                           dict(a="e2", b="j2", kind="linkway")]}
        drive = box(14, -30, 26, -1.0)                       # ends 1 m short of the south face
        sp = entrance_spawns("BLK_1", "block", [(20.0, -0.1), (40.1, 6.0)], fp, graph, paving=drive)
        self.assertEqual([s["name"] for s in sp], ["Void deck entrance S", "Void deck entrance E"])
        from shapely.geometry import Point
        for s in sp:
            self.assertGreaterEqual(drive.distance(Point(*s["position"])), 0.3 - 1e-9)
            self.assertFalse(fp.contains(Point(*s["position"])))
        np.testing.assert_allclose(sp[1]["position"], [41.6, 6.0])
        np.testing.assert_allclose(sp[1]["facing"], [-1.0, 0.0])
        self.assertLess(sp[0]["position"][1], -0.1)
        self.assertEqual(entrance_spawns("NC_1", "nc", [(40.1, 6.0)], fp, graph)[0]["name"], "Entrance E")

    def test_entrance_spawns_leave_outdoors(self):
        """WE-R5: an open side of a hall inside a precinct footprint. The spawn follows the outdoor path (through a
        5 cm graph stub) away from the hall, not the indoor route into it, and the entrance is matched through the
        graph meta although a junction lies nearer to the planned point than the entrance node."""
        from shapely.geometry import box
        from estate.export.manifest import entrance_spawns
        fp = box(0, 0, 40, 12)                             # precinct: a hall (x < 10) and a forecourt beyond it
        graph = {"nodes": [dict(id="n0", xy=[10.0, 5.0], kind="nc", ref="NC_1"),
                           dict(id="jx", xy=[10.0, 4.45], kind="junction"),
                           dict(id="j1", xy=[10.0, 7.7], kind="junction"),
                           dict(id="j2", xy=[10.05, 5.0], kind="junction"),
                           dict(id="j3", xy=[13.0, 5.0], kind="junction")],
                 "edges": [dict(a="n0", b="j1", kind="footpath", via="indoor"), dict(a="n0", b="j2", kind="footpath"),
                           dict(a="j2", b="j3", kind="footpath"), dict(a="jx", b="j1", kind="footpath", via="indoor")],
                 "meta": {"entrances": [dict(site="NC_1", k=0, node=[10.0, 5.0], planned=[10.0, 4.4])]}}
        sp = entrance_spawns("NC_1", "nc", [(10.0, 4.4)], fp, graph)
        self.assertEqual((sp[0]["name"], sp[0]["source"]), ("Entrance E", "graph"))
        np.testing.assert_allclose(sp[0]["position"], [11.5, 5.0], atol=1e-6)
        np.testing.assert_allclose(sp[0]["entrance"], [10.0, 5.0])
        np.testing.assert_allclose(sp[0]["facing"], [-1.0, 0.0])
        del graph["edges"][1]                              # only the indoor route: taken rather than no spawn
        self.assertEqual(entrance_spawns("NC_1", "nc", [(10.0, 4.4)], fp, graph)[0]["name"], "Entrance N")

    def test_bus_stop_spawns(self):
        """WE-R5: a bus stop whose masterplan point lies on the kerb line spawns on its platform node, AGENT_RADIUS
        clear of the carriageway and facing its road; without a graph it steps back off the kerb."""
        from shapely.geometry import Point, box
        from estate.export.manifest import bus_stop_spawns
        from estate.rules import AGENT_RADIUS
        road = box(0, 377, 400, 397)                       # carriageway north of the kerb line y = 377
        stops = [dict(id="BS1", road="AVE5", position=[100.0, 377.0, 0.0])]
        roads = {"AVE5": [[0.0, 387.0], [400.0, 387.0]]}
        graph = {"nodes": [dict(id="n1", xy=[100.0, 375.3], z=0.0, kind="bus_stop", ref="BS1")], "edges": [],
                 "meta": {"bus_stops": [dict(id="BS1", road="AVE5", node=[100.0, 375.3])]}}
        sp = bus_stop_spawns(stops, graph, road, roads)[0]
        self.assertEqual((sp["name"], sp["source"]), ("Bus stop BS1", "graph"))
        np.testing.assert_allclose(sp["position"], [100.0, 375.3, 0.0])
        np.testing.assert_allclose(sp["facing"], [0.0, 1.0, 0.0])
        del graph["meta"]                                  # an older graph: the bus_stop node of that ref
        np.testing.assert_allclose(bus_stop_spawns(stops, graph, road, roads)[0]["position"], [100.0, 375.3, 0.0])
        sp = bus_stop_spawns(stops, None, road, roads)[0]
        self.assertEqual(sp["source"], "masterplan+clear")
        self.assertGreaterEqual(road.distance(Point(*sp["position"][:2])), AGENT_RADIUS)
        self.assertLess(road.distance(Point(*sp["position"][:2])), AGENT_RADIUS + 0.25)


@unittest.skipUnless(T_PT4.exists(), "build/t_pt4.ifc not built (run build/t_pt4.py)")
class TestEngineExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from estate.export.engine import export_building
        import ifcopenshell
        cls.r = export_building(T_PT4, OUT, "t_pt4", local_matrix="building")
        cls.info = json.loads(Path(cls.r["files"]["engine"]).read_text(encoding="utf-8"))
        cls.f = ifcopenshell.open(str(T_PT4))

    def test_glbs_valid(self):
        for k in ("lod0", "lod1", "lod2"):
            self.assertEqual(glb.validate(self.r["files"][k]), [], k)
        s0, s1, s2 = (glb.summary(self.r["files"][k]) for k in ("lod0", "lod1", "lod2"))
        self.assertEqual(s0["door_nodes"], len(self.f.by_type("IfcDoor")))
        self.assertEqual(s0["lift_nodes"], 2)
        self.assertEqual((s1["door_nodes"], s2["door_nodes"]), (0, 0))
        self.assertGreater(s0["triangles"], s1["triangles"])
        self.assertGreater(s1["triangles"], s2["triangles"])
        self.assertTrue(set(n for n in s0["names"] if n.endswith("_static")) >= {"L1_static", "L2_static", "RF_static"})

    def test_lod1_smaller_than_lod0(self):
        """WE-R6: LOD1 instances the doors and windows it keeps (one mesh per type, as LOD0 does), so it is smaller
        on disk than LOD0 as well as drawing fewer triangles."""
        assert_lod1_smaller(self, self.r)

    def test_engine_json(self):
        d = self.info
        for k in ("frame", "ifc_sha256", "transform", "spawns", "verified_agent", "storeys", "doors", "lifts", "rooms",
                  "flats", "portals", "climbable_edges", "budget"):
            self.assertIn(k, d)
        np.testing.assert_allclose(d["transform"], np.eye(4))
        self.assertEqual(len(d["doors"]), len(self.f.by_type("IfcDoor")))
        self.assertEqual(len(d["rooms"]), len(self.f.by_type("IfcSpace")))
        self.assertEqual(len(d["flats"]), 44)
        self.assertTrue(all(fl["main_door"] for fl in d["flats"]))
        for lift in d["lifts"]:
            self.assertEqual(len(lift["landing_doors"]), len(lift["served_levels"]))
        self.assertGreaterEqual(len(d["spawns"]), 1)
        self.assertEqual(d["verified_agent"]["radius"], 0.3)
        self.assertLessEqual(d["budget"]["max"], d["budget"]["limit"])
        nodes = set(glb.summary(self.r["files"]["lod0"])["names"])
        self.assertTrue(all(dr["node"] in nodes for dr in d["doors"]))
        flat_rooms = [p for p in d["portals"] if p["kind"] in ("internal", "bath", "main")]
        self.assertTrue(all(p["spaces"][0] and p["spaces"][1] for p in flat_rooms))

    def test_lod1_drops_interior(self):
        from estate.export.engine import BuildingExport
        X = BuildingExport(T_PT4, "building")
        kept = {X.meshes[g]["cls"] for g in X.meshes if X.lod1_keep(g)}
        self.assertNotIn("IfcSpace", kept)
        self.assertNotIn("IfcStairFlight", kept)
        int_walls = [g for g, t in X.wall_type.items() if t.startswith(("INT", "WET"))]
        self.assertTrue(int_walls and not any(X.lod1_keep(g) for g in int_walls))

    def test_door_leaves(self):
        """Every swing door has a leaf child whose origin is its hinge: turned by open_sign * 90 deg about its up
        axis it stands on the swing side, as wide as the leaf, beside the hinge jamb."""
        g, binary = glb.read_glb(self.r["files"]["lod0"])
        W = glb.world_matrices(g)
        index = {n["name"]: i for i, n in enumerate(g["nodes"])}
        checked = 0
        for d in self.info["doors"]:
            if "SWING" in (d["operation"] or ""):
                self.assertTrue(d["leaves"] and d["leaf_node"], d["name"])
                self.assertEqual(d["hinge"], d["leaves"][0]["pivot"])
                self.assertAlmostEqual(d["leaf_thickness"], 0.035, places=3)
            for lf in d["leaves"]:
                self.assertLessEqual(len(lf["node"].encode()), 63)
                i = index[lf["node"]]
                self.assertIn(i, g["nodes"][index[d["node"]]]["children"])
                Wz = glb.ZUP @ W[i] @ glb.YUP
                np.testing.assert_allclose(Wz[:3, 3], lf["pivot"], atol=2e-3)
                if lf["motion"] != "swing":
                    continue
                v = np.concatenate([glb.from_yup(glb.accessor(g, binary, p["attributes"]["POSITION"]).astype(float))
                                    for p in g["meshes"][g["nodes"][i]["mesh"]]["primitives"]])
                a = np.radians(90.0 * lf["open_sign"])
                R = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
                opened = (v @ Wz[:3, :3].T) @ R.T
                y, x = opened @ np.array(d["swing_side"]), opened @ np.array(d["along_wall"])
                self.assertGreater(y.min(), -0.03, lf["node"])
                self.assertAlmostEqual(float(y.max()), lf["width"], delta=0.08)
                self.assertLess(float(np.abs(x).max()), lf["thickness"] + 0.06)
                checked += 1
        self.assertGreater(checked, 100)

    def test_instancing(self):
        """Doors and windows of one type share their glTF meshes."""
        s0 = glb.summary(self.r["files"]["lod0"])
        self.assertEqual(s0["window_nodes"], len(self.f.by_type("IfcWindow")))
        self.assertGreater(s0["door_leaves"], 0)
        self.assertLess(s0["meshes"], (s0["door_nodes"] + s0["window_nodes"]) / 10)
        self.assertEqual(s0["triangles"], self.r["lod0"]["triangles"])

    @unittest.skipUnless(env.BLENDER_EXE.exists(), "Blender not installed")
    def test_blender_round_trip(self):
        from estate.export.engine import blender_roundtrip
        rep = blender_roundtrip([self.r["files"][k] for k in ("lod0", "lod1", "lod2")])
        for p, r in rep.items():
            self.assertTrue(r["ok"], f"{p}: {r['diffs']}")


def build_pt4(path, at=None, rot=0, storeys=4):
    """Author a small PT4 block (optionally placed in the estate). Runs in a spawned process: IfcWriter swaps
    ifcopenshell's entity hash while authoring, and the test process must keep the native one."""
    from estate.blocks.builder import build_building, derive_plan
    from estate.blocks.point import plan_point
    from estate.ifc.writer import IfcWriter
    spec = dict(blk="900", storeys=storeys, stacks={s: {"t": "4R-PT-A"} for s in ("101", "103", "105", "107")},
                void_deck=["rc_centre"])
    plan = plan_point(spec)
    W = IfcWriter(guid_key=f"test/export/{Path(path).stem}", typed_openings=True)
    build_building(W, plan, derive_plan(plan), placement=None if at is None else config.placement_matrix(at, rot))
    W.write(path)
    W.close()
    return str(path)


class TestFreshBlock(unittest.TestCase):
    """A PT4 authored by the current writer: stair enclosures and refuse rooms are IfcSpaces, so every passable
    portal has a room or the outside at both ends, and every door gets its leaves."""

    @classmethod
    def setUpClass(cls):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from estate.export.engine import export_building
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "fresh_pt4.ifc"
        with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
            ex.submit(build_pt4, path).result()
        cls.r = export_building(path, OUT, "fresh_pt4", local_matrix="building", cache=False, site_kind="block")
        cls.info = json.loads(Path(cls.r["files"]["engine"]).read_text(encoding="utf-8"))

    def test_portals_resolved(self):
        self.assertEqual(self.r["problems"], [])
        passable = [p for p in self.info["portals"] if p["passable"]]
        self.assertTrue(passable)
        for p in passable:
            for room, out in zip(p["spaces"], p["outside"]):
                self.assertTrue(room or out, p)
        stairs = [p for p in passable if p["kind"] == "stair"]
        self.assertTrue(stairs and all(any("STAIR" in (n or "") for n in p["space_names"]) for p in stairs))

    def test_leaves_and_labels(self):
        doors = self.info["doors"]
        self.assertTrue(all(d["leaves"] for d in doors))
        self.assertTrue(all(sp["name"].startswith("Void deck entrance") for sp in self.info["spawns"]))
        self.assertTrue(all("room" in sp for sp in self.info["spawns"]))
        self.assertLessEqual(self.info["budget"]["max"], self.info["budget"]["limit"])

    def test_lod1_smaller_than_lod0(self):
        assert_lod1_smaller(self, self.r)


def assert_lod1_smaller(test, r):
    """LOD1 bytes < LOD0 bytes, LOD1 draws fewer triangles, and its kept doors / windows are instanced nodes under
    <stem>_openings that share their meshes (and are not DOOR_ / WIN_ nodes an engine would swap)."""
    s0, s1 = glb.summary(r["files"]["lod0"]), glb.summary(r["files"]["lod1"])
    test.assertLess(Path(r["files"]["lod1"]).stat().st_size, Path(r["files"]["lod0"]).stat().st_size)
    test.assertLess(s1["triangles"], s0["triangles"])
    test.assertEqual((s1["door_nodes"], s1["window_nodes"]), (0, 0))
    inst = [n for n in s1["names"] if n.startswith(("L1DOOR_", "L1WIN_"))]
    test.assertEqual(len(inst), r["lod1"]["opening_nodes"])
    test.assertGreater(len(inst), 0)
    test.assertLess(s1["meshes"], len(inst) / 4)


@unittest.skipUnless(MODEL_SITE.exists(), "model/SITE.ifc not built")
class TestSiteTiles(unittest.TestCase):
    """The external works are cut into tiles with instanced trees, and LOD1 is a real reduction."""

    @classmethod
    def setUpClass(cls):
        from estate.export.engine import export_building
        import ifcopenshell
        cls.r = export_building(MODEL_SITE, OUT, "SITE", local_matrix="building")
        cls.info = json.loads(Path(cls.r["files"]["engine"]).read_text(encoding="utf-8"))
        f = ifcopenshell.open(str(MODEL_SITE))
        cls.trees = [e for e in f.by_type("IfcGeographicElement") if e.PredefinedType != "TERRAIN"]

    def test_tiles(self):
        from estate.export.engine import TILE
        self.assertNotIn("lod2", self.r["files"])
        s0 = glb.summary(self.r["files"]["lod0"])
        tiles = self.info["tiles"]
        self.assertGreater(len(tiles), 4)
        self.assertEqual(s0["tree_nodes"], len(self.trees))
        self.assertEqual(sum(t["trees"] for t in tiles), len(self.trees))
        self.assertLess(s0["meshes"], len(tiles) * 3 + 20)
        for t in tiles:
            i, j = t["index"]
            for cat in ("ground", "structures", "furniture"):
                b = s0["node_bounds"].get(f"{t['node']}_{cat}")
                if b:
                    self.assertGreaterEqual(b[0][0], i * TILE - 1e-3)
                    self.assertLessEqual(b[1][0], (i + 1) * TILE + 1e-3)
                    self.assertGreaterEqual(b[0][1], j * TILE - 1e-3)
                    self.assertLessEqual(b[1][1], (j + 1) * TILE + 1e-3)

    def test_lod1_reduced(self):
        s0, s1 = glb.summary(self.r["files"]["lod0"]), glb.summary(self.r["files"]["lod1"])
        self.assertLess(s1["triangles"], 0.35 * s0["triangles"])
        self.assertEqual(s1["tree_nodes"], 0)
        self.assertEqual(self.r["lod1"]["tree_proxies"], len(self.trees))

    @unittest.skipUnless(env.BLENDER_EXE.exists(), "Blender not installed")
    def test_blender_round_trip(self):
        from estate.export.engine import blender_roundtrip
        for p, r in blender_roundtrip([self.r["files"][k] for k in ("lod0", "lod1")]).items():
            self.assertTrue(r["ok"], f"{p}: {r['diffs']}")


def build_site_fixture(folder):
    """SITE.ifc, its pedestrian graph and masterplan.json written together (as the build does) into ``folder``.
    Runs in a spawned process (IfcWriter swaps the entity hash)."""
    from estate import masterplan
    from estate.site.builder import build_site
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    mp = masterplan.resolve()
    build_site(mp, folder / "SITE.ifc", graph_path=folder / "SITE_graph.json")
    masterplan.write_json(mp, folder / "masterplan.json")
    return str(folder)


def build_nc(path):
    """NC 514 authored by the current code into ``path`` (spawned process, as build_pt4)."""
    from estate.site.centre import build_nc_ifc
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    build_nc_ifc(config.load(), path)
    return str(path)


_FIXTURES = {}


def fixtures(nc=False) -> dict:
    """Build the site fixture (OUT/site_fixture) and, with ``nc``, a fresh NC 514 (OUT/fresh_nc/NC_514.ifc) once per
    test run, in parallel spawned processes. Returns {"site": folder, "nc": ifc path}."""
    jobs = {"site": (build_site_fixture, OUT / "site_fixture"), "nc": (build_nc, OUT / "fresh_nc" / "NC_514.ifc")}
    todo = [k for k in jobs if k not in _FIXTURES and (nc or k == "site")]
    if todo:
        with ProcessPoolExecutor(len(todo), mp_context=multiprocessing.get_context("spawn")) as ex:
            futs = {k: ex.submit(*jobs[k]) for k in todo}
            for k, fut in futs.items():
                _FIXTURES[k] = Path(fut.result())
    return _FIXTURES


class TestSpawnsOnFreshSite(unittest.TestCase):
    """ENG-5 / WE-R5 on a consistent site: every spawn of every estate site, and every bus stop spawn, stands on the
    pedestrian side, at least AGENT_RADIUS from carriageways, junctions, driveways, crossovers and zebra crossings;
    entrance spawns stand on the outdoor path that leaves the entrance."""

    @classmethod
    def setUpClass(cls):
        from estate.export.manifest import vehicle_paving
        cls.folder = fixtures()["site"]
        cls.paving = vehicle_paving(cls.folder)
        cls.mp = json.loads((cls.folder / "masterplan.json").read_text(encoding="utf-8"))

    def test_spawns_clear_of_vehicle_paving(self):
        from shapely.geometry import Point
        from estate.export.manifest import site_spawns
        from estate.rules import AGENT_RADIUS
        n = 0
        for s in self.mp["sites"]:
            kind, spawns = site_spawns(s["id"], self.folder)
            self.assertEqual(len(spawns or []), len(s["entrances"]))
            for sp in spawns or []:
                self.assertGreaterEqual(self.paving.distance(Point(*sp["position"])), AGENT_RADIUS - 1e-6,
                                        f"{s['id']} {sp}")
                self.assertFalse(s["kind"] != "block" and sp["name"].startswith("Void deck"), sp)
                self.assertEqual(sp["source"], "graph", f"{s['id']} {sp}")    # every entrance is on the graph
                n += 1
        self.assertGreater(n, 2 * len(self.mp["sites"]))

    def test_spawns_not_on_indoor_routes(self):
        """No entrance spawn stands on an indoor or void-deck route unless it is also on an outdoor path."""
        from shapely.geometry import LineString, Point
        from estate.export.manifest import THROUGH_BUILDING, site_spawns
        g = json.loads((self.folder / "SITE_graph.json").read_text(encoding="utf-8"))
        xy = {nd["id"]: nd["xy"] for nd in g["nodes"]}
        lines = {k: [LineString([xy[e["a"]], xy[e["b"]]]) for e in g["edges"]
                     if (e.get("via") in THROUGH_BUILDING) == k and e["kind"] != "crossing"] for k in (True, False)}
        n = 0
        for s in self.mp["sites"]:
            for sp in site_spawns(s["id"], self.folder)[1] or []:
                p = Point(*sp["position"])
                if min(ln.distance(p) for ln in lines[True]) < 1e-3:
                    self.assertLess(min(ln.distance(p) for ln in lines[False]), 1e-3, f"{s['id']} {sp}")
                n += 1
        self.assertGreater(n, 0)

    def test_bus_stop_spawns(self):
        from shapely.geometry import Point
        from estate import masterplan
        from estate.export.manifest import write_manifest
        from estate.rules import AGENT_RADIUS
        with tempfile.TemporaryDirectory() as td:
            d = json.loads(write_manifest(masterplan.resolve(), out=Path(td) / "m.json", model_dir=self.folder)
                           .read_text(encoding="utf-8"))
        bus = [sp for sp in d["spawn_points"] if sp["site"] == "SITE"]
        self.assertEqual(sorted(sp["stop"] for sp in bus), sorted(b["id"] for b in d["bus_stops"]))
        for sp in d["spawn_points"]:
            self.assertGreaterEqual(self.paving.distance(Point(*sp["position"][:2])), AGENT_RADIUS - 1e-6, sp)
        for sp in bus:
            self.assertEqual(sp["source"], "graph", sp)
            self.assertAlmostEqual(float(np.hypot(*sp["facing"][:2])), 1.0, places=3)


class TestFreshCentre(unittest.TestCase):
    """NC 514 authored by the current code and exported with its masterplan spawns: every passable door but the lift
    doors has a leaf (WE-R2: the 48 stall shutters roll), the hawker tables and stools are instanced so LOD0 stays
    small and LOD1 smaller still (WE-R6), and no spawn stands inside the hawker hall (WE-R5)."""

    @classmethod
    def setUpClass(cls):
        import ifcopenshell
        from estate.export.engine import export_building
        from estate.export.manifest import site_spawns
        fx = fixtures(nc=True)
        kind, entrances = site_spawns("NC_514", fx["site"])
        cls.r = export_building(fx["nc"], OUT, "fresh_nc", local_matrix="building", cache=False, entrances=entrances,
                                site_kind=kind)
        cls.info = json.loads(Path(cls.r["files"]["engine"]).read_text(encoding="utf-8"))
        cls.f = ifcopenshell.open(str(fx["nc"]))

    def test_passable_doors_have_leaves(self):
        names = set(glb.summary(self.r["files"]["lod0"])["names"])
        doors = [d for d in self.info["doors"] if d["passable"] and d["kind"] != "lift"]
        self.assertTrue(doors)
        for d in doors:
            self.assertTrue(d["leaves"], d["name"])
            self.assertTrue(all(lf["node"] in names for lf in d["leaves"]), d["name"])
        shutters = [d for d in self.info["doors"] if "ROLLING" in (d["operation"] or "")]
        self.assertEqual(len(shutters), 48)
        for d in shutters:
            self.assertEqual([lf["motion"] for lf in d["leaves"]], ["roll"], d["name"])
            lf = d["leaves"][0]
            np.testing.assert_allclose(lf["travel"], [0.0, 0.0, lf["height"]], atol=1e-3)
            self.assertGreater(lf["height"], 0.5 * d["height"])

    def test_furniture_instanced(self):
        s0 = glb.summary(self.r["files"]["lod0"])
        self.assertGreater(len(self.f.by_type("IfcFurniture")), 700)
        self.assertGreaterEqual(s0["furniture_nodes"], 750)                 # 150 tables, 600 stools
        self.assertEqual(s0["furniture_nodes"], self.r["lod0"]["furniture_nodes"])
        self.assertLess(s0["meshes"], 60)
        self.assertEqual(s0["triangles"], self.r["lod0"]["triangles"])
        self.assertLess(Path(self.r["files"]["lod0"]).stat().st_size, 2.5e6)
        assert_lod1_smaller(self, self.r)

    def test_spawns_outside_the_hall(self):
        rooms = {r["name"]: r for r in self.info["rooms"]}
        self.assertGreaterEqual(len(self.info["spawns"]), 4)
        for sp in self.info["spawns"]:
            self.assertTrue(sp["name"].startswith("Entrance"), sp)
            self.assertTrue(sp["room"] is None or rooms[sp["room"]]["external"], sp)

    @unittest.skipUnless(env.BLENDER_EXE.exists(), "Blender not installed")
    def test_blender_round_trip(self):
        from estate.export.engine import blender_roundtrip
        for p, r in blender_roundtrip([self.r["files"][k] for k in ("lod0", "lod1")]).items():
            self.assertTrue(r["ok"], f"{p}: {r['diffs']}")


class TestPlacedFrame(unittest.TestCase):
    def test_rotated_block_exports_block_local(self):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from estate.export.engine import export_building
        at, rot = (255.0, 345.0), 1
        out = {}
        with tempfile.TemporaryDirectory() as td:
            paths = {"local": Path(td) / "local.ifc", "placed": Path(td) / "placed.ifc"}
            with ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as ex:
                list(ex.map(build_pt4, [paths["local"], paths["placed"]], [None, at], [0, rot]))
            for tag, path in paths.items():
                r = export_building(path, td, tag, local_matrix="building", cache=False)
                out[tag] = (glb.summary(r["files"]["lod0"]), json.loads(Path(r["files"]["engine"]).read_text(encoding="utf-8")))
        (sa, ja), (sb, jb) = out["local"], out["placed"]
        self.assertEqual(sa["triangles"], sb["triangles"])
        np.testing.assert_allclose(sa["bounds"], sb["bounds"], atol=1e-3)
        np.testing.assert_allclose(jb["transform"], config.placement_matrix(at, rot), atol=1e-9)
        np.testing.assert_allclose(ja["transform"], np.eye(4), atol=1e-12)
        self.assertEqual(sorted(s["position"] for s in ja["spawns"]), sorted(s["position"] for s in jb["spawns"]))
        self.assertEqual(sorted(r["polygon"] for r in ja["rooms"]), sorted(r["polygon"] for r in jb["rooms"]))


class TestManifest(unittest.TestCase):
    def test_manifest_without_outputs(self):
        from estate import masterplan
        from estate.export.manifest import write_manifest
        mp = masterplan.resolve(plans=False)
        with tempfile.TemporaryDirectory() as td:
            p = write_manifest(mp, out=Path(td) / "estate_manifest.json", model_dir=Path(td))
            d = json.loads(p.read_text(encoding="utf-8"))
        cfg = mp["cfg"]
        self.assertEqual(len(d["sites"]), len(mp["sites"]))
        self.assertEqual(d["crs"]["epsg"], cfg["georef"]["epsg"])
        self.assertEqual(len(d["roads"]), len(cfg["road"]))
        self.assertEqual(len(d["bus_stops"]), len(cfg["bus_stop"]))
        b501 = next(s for s in d["sites"] if s["id"] == "BLK_501")
        np.testing.assert_allclose(b501["transform"], config.placement_matrix(b501["at"], b501["rot"]))
        self.assertEqual(b501["files"]["glb_lod0"], {"path": "BLK_501/BLK_501_lod0.glb", "exists": False})
        self.assertIsNone(d["pedestrian_graph"])
        self.assertNotIn("glb_lod2", d["site"]["files"])
        self.assertTrue(all(s["approximate"] for s in d["sites"] if s["kind"] == "block"))
        self.assertTrue(any("approximate" in w for w in d["warnings"]))

    def test_lift_lobbies_inside_footprints(self):
        """CP-2 / R9: the manifest lists the planned lift lobbies and entrances, inside / on the real footprint."""
        from shapely.geometry import Point
        from estate import masterplan
        from estate.export.manifest import write_manifest
        mp = masterplan.resolve()
        with tempfile.TemporaryDirectory() as td:
            d = json.loads(write_manifest(mp, out=Path(td) / "m.json").read_text(encoding="utf-8"))
        by_id = {s.id: s for s in mp["sites"]}
        for site in d["sites"]:
            if site["kind"] != "block":
                continue
            s = by_id[site["id"]]
            self.assertFalse(site["approximate"], site["id"])
            self.assertEqual(len(site["lift_lobbies"]), len(s.plan.meta["lift_lobbies"]), site["id"])
            self.assertEqual(len(site["entrances"]), len(s.plan.meta["entrances"]), site["id"])
            fp = s.footprint.buffer(0.1)                    # the planned footprint the lobbies come from
            for p in site["lift_lobbies"]:
                self.assertTrue(fp.contains(Point(p)), f"{site['id']} lobby {p} outside its footprint")
            for p in site["entrances"]:
                self.assertLess(fp.distance(Point(p)), 1.5, f"{site['id']} entrance {p} away from its footprint")
        self.assertFalse([w for w in d["warnings"] if "approximate" in w])

    @unittest.skipUnless((env.MODEL / "masterplan.json").exists(), "model/masterplan.json not written")
    def test_matches_masterplan_json(self):
        from estate import masterplan
        from estate.export.manifest import write_manifest
        ref = {s["id"]: s for s in json.loads((env.MODEL / "masterplan.json").read_text(encoding="utf-8"))["sites"]}
        with tempfile.TemporaryDirectory() as td:
            d = json.loads(write_manifest(masterplan.resolve(), out=Path(td) / "m.json").read_text(encoding="utf-8"))
        for site in d["sites"]:
            r = ref[site["id"]]
            for k in ("lift_lobbies", "entrances"):
                self.assertEqual(len(site[k]), len(r[k]), f"{site['id']} {k}")
                if site[k]:
                    np.testing.assert_allclose(site[k], r[k], atol=0.05, err_msg=f"{site['id']} {k}")

    @unittest.skipUnless((env.MODEL / "masterplan.json").exists() and MODEL_SITE.exists(), "model not built")
    def test_spawns_off_vehicle_paving(self):
        """ENG-5 / WE-R5: every estate spawn, and every bus stop spawn, stands AGENT_RADIUS clear of carriageways,
        junctions, driveways and crossings."""
        from shapely.geometry import Point
        from estate.export.manifest import ENTRANCE_NODES, bus_stop_spawns, site_spawns, vehicle_paving
        from estate.rules import AGENT_RADIUS
        paving = vehicle_paving()
        sites = json.loads((env.MODEL / "masterplan.json").read_text(encoding="utf-8"))["sites"]
        graph_path = env.MODEL / "SITE_graph.json"
        nodes = json.loads(graph_path.read_text(encoding="utf-8"))["nodes"] if graph_path.exists() else []
        known = np.array([n["xy"] for n in nodes if n["kind"] in ENTRANCE_NODES]).reshape(-1, 2)
        for s in sites:
            for e in s["entrances"] if s["kind"] == "block" else []:
                if not len(known) or np.hypot(*(known - e).T).min() > 0.6:
                    self.skipTest("model/SITE.ifc and SITE_graph.json are older than model/masterplan.json")
        n = 0
        for s in sites:
            kind, spawns = site_spawns(s["id"])
            for sp in spawns or []:
                self.assertGreaterEqual(paving.distance(Point(*sp["position"])), AGENT_RADIUS - 1e-6, f"{s['id']} {sp}")
                self.assertTrue(sp["name"].startswith("Void deck entrance" if kind == "block" else "Entrance"))
                n += 1
        self.assertGreater(n, len(sites))
        mp = json.loads((env.MODEL / "masterplan.json").read_text(encoding="utf-8"))
        bus = [dict(id=b["id"], road=b.get("road"), position=[*b["at"], 0.0]) for b in mp["bus_stops"]]
        graph = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.exists() else None
        for sp in bus_stop_spawns(bus, graph, paving, {r["id"]: r["centreline"] for r in mp["roads"]}):
            self.assertGreaterEqual(paving.distance(Point(*sp["position"][:2])), AGENT_RADIUS - 1e-6, sp)


if __name__ == "__main__":
    unittest.main()
