"""Walk tests: union-find, voxel walkability on synthetic scenes, door linings and the door-band test, the
main-door threshold, site routing and the graph-vs-SITE.ifc checks (zebras and kerb ramps, also on the real
SITE.ifc and on a copy without its kerb ramps), the estate graph, and the legacy regression (r = 0.30: every room
and the roof reachable; r = 0.40: rooms behind 0.8-0.9 m doors cut off).

Run: "<blender python>" -I -B tests/test_nav.py -v
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estate.env import bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402
import shapely  # noqa: E402

from estate import env  # noqa: E402
from estate.validate import nav2d, nav3d, navgraph  # noqa: E402

OUT = env.BUILD / "nav_tests" / "unittest"
LEGACY = env.BUILD / "legacy" / "hdb_block_legacy.ifc"
PT4 = env.BUILD / "t_pt4.ifc"


def box_tris(x0, y0, z0, x1, y1, z1):
    return nav3d._box_tris((x0, y0, z0), (x1, y1, z1))


def scene(boxes, r, lo=(-0.5, -0.5, -0.5), hi=(6.5, 3.5, 3.0), extra=None):
    T = np.concatenate([box_tris(*b) for b in boxes] + ([extra] if extra is not None else []))
    elem = np.repeat(np.arange(len(boxes) + (extra is not None)), [12] * len(boxes) + ([len(extra)] if extra is not None
                                                                                        else []))
    g = nav3d.Grid.around(lo, hi, 0.1)
    S, up, _ = nav3d.voxelise(T, elem, g)
    k, ix, iy = nav3d.walkable_cells(S, g, nav3d.voxel_radius(r, 0.1), 18, 4)
    cells = nav3d.connect(k, ix, iy, g, 4, up)
    lab = nav3d.components(len(cells.key), cells.ea, cells.eb)
    return g, cells, lab, T, elem


def cell_at(g, cells, x, y, z=None):
    m = (cells.ix == g.ix(x)) & (cells.iy == g.iy(y))
    idx = np.nonzero(m)[0]
    if z is not None:
        idx = idx[np.abs(cells.fz[idx] - z) < 0.2]
    return int(idx[0]) if len(idx) else None


def label_at(g, cells, lab, x, y, z=None):
    c = cell_at(g, cells, x, y, z)
    return int(lab[c]) if c is not None else None


def door(x0, w, wall_y=1.5, h=2.1, jamb=0.05):
    """A door record as read_model builds it: frame at x0 along a wall on y = wall_y, opening 0..w."""
    M = np.eye(4)
    M[:3, 3] = (x0, wall_y - 0.05, 0.0)
    d = dict(guid="d", name="test door", kind="internal", M=M, w=w, clear=round(w - 2 * jamb, 3), jamb=jamb,
             jamb_src="pset", y0=-0.015, y1=0.1, yc=0.0425, h=h, z=0.0, storey="L1", sides=())
    d["bbox"] = (np.array([x0, wall_y - 0.065, 0.0]), np.array([x0 + w, wall_y + 0.05, h]))
    return d


def door_walls(x0, w, wall_y=1.5, t=0.2):
    """A wall across the 6 x 3 m floor with an opening of width w at x0 (head at 2.1 m)."""
    a, b = wall_y - t / 2, wall_y + t / 2
    return [(0, a, 0, x0, b, 2.4), (x0 + w, a, 0, 6, b, 2.4), (x0, a, 2.1, x0 + w, b, 2.4)]


class UnionFind(unittest.TestCase):
    def test_matches_networkx(self):
        import networkx as nx
        rng = np.random.default_rng(7)
        n = 3000
        a, b = rng.integers(0, n, 2500), rng.integers(0, n, 2500)
        lab = nav3d.components(n, a, b)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(zip(a.tolist(), b.tolist()))
        for comp in nx.connected_components(G):
            comp = sorted(comp)
            self.assertTrue((lab[comp] == comp[0]).all())

    def test_run_all_and_disk(self):
        rng = np.random.default_rng(3)
        F = rng.random((40, 9, 9)) > 0.2
        for n in (1, 3, 4, 7, 14):
            G = nav3d._run_all(F, n)
            ref = np.stack([F[k:k + n].all(axis=0) for k in range(len(F) - n + 1)])
            self.assertTrue(np.array_equal(G, ref), n)
        A = np.zeros((1, 15, 15), bool)
        A[0, 7, 7] = True
        D = nav3d._dilate_disk(A, 3)[0]
        yy, xx = np.mgrid[-7:8, -7:8]
        self.assertTrue(np.array_equal(D, (xx ** 2 + yy ** 2) <= 9))

    def test_radius_rounds_up(self):
        # 0.45 m used to be tested as 0.40 m (round half to even) and 0.35 m as 0.30 m
        self.assertEqual(nav3d.voxel_radius(0.45, 0.1), 5)
        self.assertEqual(nav3d.voxel_radius(0.35, 0.1), 4)
        self.assertEqual(nav3d.voxel_radius(0.30, 0.1), 3)
        self.assertEqual(nav3d.voxel_radius(0.40, 0.1), 4)
        self.assertEqual(nav3d.voxel_radius(0.30, 0.025), 12)


class Walkable(unittest.TestCase):
    floor = (0, 0, -0.2, 6, 3, 0)

    def door_scene(self, y0, w, r):
        wall = [(2.9, 0, 0, 3.1, y0, 2.4), (2.9, y0 + w, 0, 3.1, 3, 2.4), (2.9, y0, 2.1, 3.1, y0 + w, 2.4)]
        g, cells, lab, _, _ = scene([self.floor] + wall, r)
        return label_at(g, cells, lab, 1.0, 1.5) == label_at(g, cells, lab, 5.0, 1.5) is not None

    def test_door_widths(self):
        self.assertTrue(self.door_scene(1.1, 0.8, 0.30))     # 0.8 m bare opening: passable at r = 0.30
        self.assertFalse(self.door_scene(1.1, 0.6, 0.30))    # 0.6 m: not
        self.assertFalse(self.door_scene(1.1, 0.8, 0.40))    # 0.8 m at r = 0.40: cut off (legacy behaviour)
        self.assertTrue(self.door_scene(1.0, 1.0, 0.40))     # 1.0 m main door at r = 0.40: passable

    def test_steps_and_floor_height(self):
        for h, linked in ((0.3, True), (0.6, False)):
            g, cells, lab, _, _ = scene([self.floor, (3.5, 0, 0, 6, 3, h)], 0.30)
            a, b = label_at(g, cells, lab, 1.0, 1.5), label_at(g, cells, lab, 5.0, 1.5, h)
            self.assertIsNotNone(b)
            self.assertEqual(a == b, linked, h)
            m = (cells.ix == g.ix(5.0)) & (cells.iy == g.iy(1.5))
            self.assertAlmostEqual(float(cells.fz[m].max()), h, places=3)

    def test_closed_solids_are_filled(self):
        g = nav3d.Grid.around((-0.5, -0.5, -0.5), (3.5, 3.5, 3.0), 0.1)
        T = np.concatenate([box_tris(0, 0, -0.2, 3, 3, 0), box_tris(0.5, 0.5, 0, 2.5, 2.5, 2.2)])
        S, _, _ = nav3d.voxelise(T, np.repeat([0, 1], 12), g)
        self.assertTrue(S[g.kz(1.0), g.ix(1.5), g.iy(1.5)])  # inside the 2.2 m "tank"


class DoorLinings(unittest.TestCase):
    """A framed 0.8 m door (0.70 m clear between its linings) against a 0.60 m and a 0.80 m agent."""
    floor = (0, 0, -0.2, 6, 3, 0)

    def run_door(self, x0, w, radii=(0.30,), side_b=None):
        d = door(x0, w)
        boxes = [self.floor] + door_walls(x0, w)
        if side_b is not None:                    # a raised floor beyond the door (a threshold step)
            boxes.append((0, 1.6, 0, 6, 3, side_b))
        out = {}
        for r in radii:
            g, cells, lab, T, elem = scene(boxes, r, extra=nav3d.jamb_tris(d))
            band, _ = nav3d.door_band(d, nav3d.Tris.of(T, elem), list(radii), 1.8, 0.4, 0.015)
            b = band[f"{r:.2f}"]
            ia, ib = cell_at(g, cells, x0 + w / 2, 0.6), cell_at(g, cells, x0 + w / 2, 2.4, side_b)
            coarse = lab[ia] == lab[ib]
            index = nav3d.CellIndex.of(cells, g)
            linked = coarse
            if b["passable"]:
                p = nav3d._door_portal(cells, g, d, b["zA"], b["zB"], 0.05, index)
                if p is not None:
                    lab2 = nav3d.components(len(cells.key), np.concatenate([cells.ea, p[0]]),
                                            np.concatenate([cells.eb, p[1]]))
                    linked = lab2[ia] == lab2[ib]
            out[r] = dict(coarse=bool(coarse), band=b, linked=bool(linked))
        return out

    def test_framed_08_door(self):
        for x0 in (2.0, 2.02, 2.05):             # jambs on, beside and across the 0.1 m cell boundaries
            res = self.run_door(x0, 0.8, (0.30, 0.40))
            self.assertTrue(res[0.30]["band"]["passable"], x0)
            self.assertTrue(res[0.30]["band"]["step_free"], x0)
            self.assertTrue(res[0.30]["linked"], x0)            # the portal joins both sides
            self.assertFalse(res[0.40]["band"]["passable"], x0)
            self.assertFalse(res[0.40]["linked"], x0)
        # the 0.1 m grid alone decides a 0.70 m clear gap by where the jambs fall: this is why doors get a band
        self.assertTrue(self.run_door(2.0, 0.8)[0.30]["coarse"])
        self.assertFalse(self.run_door(2.02, 0.8)[0.30]["coarse"])

    def test_other_widths(self):
        self.assertFalse(self.run_door(2.0, 0.6)[0.30]["band"]["passable"])             # 0.50 m clear
        res = self.run_door(2.03, 1.0, (0.40,))                                          # 0.90 m clear main door
        self.assertTrue(res[0.40]["band"]["passable"])
        self.assertTrue(res[0.40]["linked"])

    def test_threshold_step(self):
        res = self.run_door(2.0, 0.8, side_b=0.05)[0.30]
        self.assertTrue(res["band"]["passable"])
        self.assertFalse(res["band"]["step_free"])            # a 50 mm threshold is a step for a wheelchair
        self.assertAlmostEqual(res["band"]["zB"], 0.05, places=3)

    def test_lining_boxes_follow_the_placement(self):
        d = door(1.0, 0.8)
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # rotated 90 degrees
        d["M"][:3, :3] = R
        T = nav3d.jamb_tris(d)
        lo, hi = T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0)
        self.assertAlmostEqual(hi[1] - lo[1], 0.8)                           # the opening now runs along y
        self.assertAlmostEqual(hi[2], 2.1)
        for B in (T[:12], T[12:]):                                            # still outward-facing
            n = np.cross(B[:, 1] - B[:, 0], B[:, 2] - B[:, 0])
            c = B.mean(axis=1) - B.reshape(-1, 3).mean(0)
            self.assertGreater(float((n * c).sum(axis=1).min()), 0.0)


class MainDoorThreshold(unittest.TestCase):
    """The main door counts as step-free only if the step-free street component continues into the entry room."""

    def test_step_into_flat(self):
        floor = (0, 0, -0.2, 6, 3, 0)
        d = door(2.0, 1.0)
        boxes = [floor] + door_walls(2.0, 1.0) + [(0, 1.6, 0, 6, 3, 0.05)]       # the flat floor is 50 mm up
        g, cells, lab, T, elem = scene(boxes, 0.30, extra=nav3d.jamb_tris(d))
        sf = np.abs(cells.fz[cells.ea] - cells.fz[cells.eb]) <= 0.015
        index = nav3d.CellIndex.of(cells, g)
        band, _ = nav3d.door_band(d, nav3d.Tris.of(T, elem), [0.30], 1.8, 0.4, 0.015)
        self.assertFalse(band["0.30"]["step_free"])
        lab_sf = nav3d.components(len(cells.key), cells.ea[sf], cells.eb[sf])
        street_sf = lab_sf == lab_sf[cell_at(g, cells, 3.0, 0.4)]
        main = dict(d, unit="#02-101", sides=("#02-101 F", "L2-CORR"))
        foyer = dict(name="#02-101 F", poly=shapely.box(0, 1.6, 6, 3), storey="L1")
        corridor = dict(name="L2-CORR", poly=shapely.box(0, 0, 6, 1.4), storey="L1")
        by_ref = {("L1", "#02-101 F"): [foyer], ("L1", "L2-CORR"): [corridor]}
        near = nav3d._cells_near(cells, g, d["bbox"], 0.0, 0.8, index=index)
        self.assertTrue(street_sf[near].any())                # the old test: corridor cells near the door
        c, how = nav3d._flat_side_cells(cells, g, main, by_ref, {}, index)
        self.assertEqual(how, "entry room")
        self.assertTrue(len(c))
        self.assertFalse(street_sf[c].any())                  # the new test: the foyer is not step-free
        c2, how2 = nav3d._flat_side_cells(cells, g, dict(main, sides=()), {}, {("L1", "#02-101"): [foyer["poly"]]},
                                          index)
        self.assertEqual(how2, "flat spaces")
        self.assertEqual(sorted(c2.tolist()), sorted(c.tolist()))


class SiteRouting(unittest.TestCase):
    def test_dial_goes_round_a_wall(self):
        walk = np.ones((20, 20), bool)
        walk[10, :15] = False
        d = nav2d.dial(walk, np.ones((20, 20), np.int32), (0, 0))
        self.assertGreater(d[19, 0], 19 * 5)
        path = nav2d.trace(d, walk, np.ones((20, 20), np.int32), (19, 0))
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (19, 0))
        self.assertTrue(all(walk[p] for p in path))
        self.assertTrue(all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1 for p, q in zip(path, path[1:])))

    def test_graph_routes(self):
        OUT.mkdir(parents=True, exist_ok=True)
        mp = {"bus_stops": [{"id": "BS1", "at": [0.0, 0.0]}]}
        for slope, step_free in ((0.05, True), (0.2, False)):
            g = {"nodes": [{"id": "s", "xy": [0, 0], "kind": "bus_stop", "ref": "BS1"},
                           {"id": "a", "xy": [10, 0], "kind": "junction"},
                           {"id": "l", "xy": [20, 0], "kind": "lift_lobby", "ref": "BLK_1"}],
                 "edges": [{"a": "s", "b": "a", "length": 10, "covered": True, "kind": "linkway", "slope": 0},
                           {"a": "a", "b": "l", "length": 10, "covered": False, "kind": "crossing", "slope": slope}]}
            p = OUT / "graph.json"
            p.write_text(json.dumps(g))
            routes, _, _, _ = nav2d._graph_routes(p, mp, [dict(id="BLK_1", lobbies=[(20.0, 0.0)])])
            r = routes[0]
            self.assertTrue(r["reachable"])
            self.assertAlmostEqual(r["length_m"], 20.0)
            self.assertAlmostEqual(r["covered_share"], 0.5)
            self.assertEqual(r["step_free"], step_free)


class GraphAgainstSiteIfc(unittest.TestCase):
    """The pedestrian graph is used only for the SITE.ifc it was written with, and is checked against it."""

    def graph(self, sha=None):
        g = {"nodes": [{"id": "s", "xy": [0, 0], "kind": "bus_stop", "ref": "BS1"},
                       {"id": "k1", "xy": [10, 0], "kind": "crossing"},
                       {"id": "k2", "xy": [10, 12], "kind": "crossing"},
                       {"id": "m1", "xy": [30, 0], "kind": "crossing"},
                       {"id": "m2", "xy": [30, 12], "kind": "crossing"},
                       {"id": "l", "xy": [20, 20], "kind": "lift_lobby", "ref": "BLK_1"}],
             "edges": [{"a": "s", "b": "k1", "length": 10, "covered": True, "kind": "linkway", "slope": 0},
                       {"a": "k1", "b": "k2", "length": 12, "covered": False, "kind": "crossing", "slope": 0.08},
                       {"a": "k2", "b": "l", "length": 12.8, "covered": True, "kind": "linkway", "slope": 0},
                       {"a": "k1", "b": "m1", "length": 20, "covered": False, "kind": "sidewalk", "slope": 0},
                       {"a": "m1", "b": "m2", "length": 12, "covered": False, "kind": "crossing", "slope": 0.08},
                       {"a": "m2", "b": "l", "length": 12.8, "covered": False, "kind": "footpath", "slope": 0}],
             "meta": {"crossings": [{"id": "X01", "kind": "bus"}]}}
        if sha:
            g["meta"]["ifc_sha256"] = sha
        return g

    @staticmethod
    def site_ifc():
        road = shapely.box(-50, 2, 80, 10)                                   # carriageway y 2..10
        return {"covered": [shapely.box(-1, -1.5, 10.5, 1.5)],             # canopy over s-k1 only
                "crossing": [shapely.box(8, 2, 12, 10)], "crossing_names": ["Zebra crossing X01"],
                "carriageway": [road], "kerb": [],
                "ramps": [dict(name="Crossing X01 kerb ramp 1", kind="kerb_ramp", poly=shapely.box(8, 0.2, 12, 2),
                               slope=0.079),
                          dict(name="Crossing X01 kerb ramp 2", kind="kerb_ramp", poly=shapely.box(8, 10, 12, 11.8),
                               slope=0.079)],
                "paving": [shapely.box(-50, -2, 80, 2), shapely.box(-50, 10, 80, 30)]}

    def test_sha_gate(self):
        OUT.mkdir(parents=True, exist_ok=True)
        ifc = OUT / "SITE_fake.ifc"
        ifc.write_bytes(b"ISO-10303-21; fake site model for the sha256 gate")
        gp = OUT / "graph_sha.json"
        for sha, used in ((None, False), ("0" * 64, False), (nav2d._sha256(ifc), True)):
            gp.write_text(json.dumps(self.graph(sha)))
            notes = []
            data = nav2d.usable_graph(gp, ifc, notes)
            self.assertEqual(data is not None, used, sha)
            self.assertEqual(len(notes), 0 if used else 1)
        notes = []
        self.assertIsNone(nav2d.usable_graph(gp, OUT / "missing.ifc", notes))
        self.assertTrue(notes)

    def test_graph_checks(self):
        fixes, rep = nav2d.graph_vs_ifc(self.graph(), self.site_ifc(), footprints=[shapely.box(18, 18, 25, 25)])
        self.assertEqual(rep["edges_over_carriageway"], 2)
        self.assertEqual(len(rep["kerb_drops"]), 1)                       # m1-m2 has no zebra and no kerb ramps
        self.assertIs(fixes[("m1", "m2")]["step_free"], False)
        self.assertNotIn(("k1", "k2"), fixes)                             # on the zebra, ramps at both kerbs
        self.assertIn(("k2", "l"), fixes)                                 # 'covered' but in the open until the block
        self.assertLess(fixes[("k2", "l")]["covered"], 0.5)
        self.assertNotIn(("s", "k1"), fixes)
        self.assertEqual(rep["zebras_missing"], [])
        self.assertFalse(rep["ok"])
        mp = {"bus_stops": [{"id": "BS1", "at": [0.0, 0.0]}]}
        tg = [dict(id="BLK_1", lobbies=[(20.0, 20.0)])]
        raw, _, _, _ = nav2d._graph_routes(self.graph(), mp, tg)
        fixed, _, G, _ = nav2d._graph_routes(self.graph(), mp, tg, edge_fix=fixes)
        self.assertTrue(raw[0]["step_free"] and fixed[0]["step_free"])  # still step-free over the zebra
        self.assertFalse(G.edges["m1", "m2"]["step_free"])
        self.assertLess(fixed[0]["covered_share"], raw[0]["covered_share"])
        _, rep2 = nav2d.graph_vs_ifc(dict(self.graph(), meta={"crossings": [{"id": "X02", "kind": "junction"}]}),
                                     self.site_ifc(), footprints=[])
        self.assertEqual(rep2["zebras_missing"], ["X02"])

    def test_carriageway_cut_around_the_zebra(self):
        """The real SITE.ifc topology: the carriageway has a hole under the zebra, so a crossing edge runs over no
        carriageway at all. It must still be checked for its zebra and both kerb ramps."""
        ifc = self.site_ifc()
        ifc["carriageway"] = [shapely.box(-50, 2, 80, 10).difference(ifc["crossing"][0])]
        self.assertLess(ifc["carriageway"][0].intersection(ifc["crossing"][0]).area, 1e-9)
        g = self.graph()
        g["nodes"] += [{"id": "p", "xy": [40, -1], "kind": "path"}, {"id": "q", "xy": [40, 2], "kind": "path"}]
        g["edges"].append({"a": "p", "b": "q", "length": 3, "covered": False, "kind": "sidewalk", "slope": 0})
        fps = [shapely.box(18, 18, 25, 25)]
        fixes, rep = nav2d.graph_vs_ifc(g, ifc, footprints=fps)
        self.assertEqual(rep["edges_over_carriageway"], 2)            # k1-k2 on the zebra, m1-m2 on the carriageway
        self.assertEqual(rep["zebras_crossed"], ["Zebra crossing X01"])
        self.assertNotIn("step_free", fixes.get(("k1", "k2"), {}))    # zebra and a 1:12.7 kerb ramp at both kerbs
        self.assertNotIn(("p", "q"), fixes)                            # ends on the kerb line: not a road crossing
        self.assertEqual(len(rep["kerb_drops"]), 1)
        self.assertIn("off a zebra crossing", rep["kerb_drops"][0])     # m1-m2
        r1, r2 = ifc["ramps"]
        for ramps, n in (([], 0), ([r1], 1), ([dict(r1, slope=0.12), r2], 1)):   # no ramps, one, one too steep
            fixes, rep = nav2d.graph_vs_ifc(g, dict(ifc, ramps=ramps), footprints=fps)
            self.assertIs(fixes[("k1", "k2")]["step_free"], False, n)
            drop = [d for d in rep["kerb_drops"] if d.startswith("k1-k2")]
            self.assertEqual(len(drop), 1, rep["kerb_drops"])
            self.assertIn("over Zebra crossing X01", drop[0])
            self.assertIn(f"at {n} of 2 kerbs", drop[0])
            self.assertFalse(rep["ok"])
        mp = {"bus_stops": [{"id": "BS1", "at": [0.0, 0.0]}]}
        routes, _, _, _ = nav2d._graph_routes(g, mp, [dict(id="BLK_1", lobbies=[(20.0, 20.0)])], edge_fix=fixes)
        self.assertTrue(routes[0]["reachable"])
        self.assertFalse(routes[0]["step_free"])                      # both ways over the road now have a kerb drop

    def test_zebra_names(self):
        """Only 'Zebra crossing <id>' is a zebra; the 'Crossing approaches' footpath stays paving."""
        c = nav2d.classify
        self.assertEqual(c("Zebra crossing X04 FLEXIBLE IfcPavement", "IfcPavement", -0.3, -0.15), "crossing")
        self.assertEqual(c("Zebra crossing X04 Pedestrian crossing USERDEFINED IfcSlab", "IfcSlab", -0.3, -0.15),
                         "crossing")                                     # the IFC4 copy
        self.assertEqual(c("Crossing approaches Footpath PAVING IfcSlab", "IfcSlab", -0.1, 0.0), "paving")
        self.assertEqual(c("Crossing X01 kerb ramp 2 STRAIGHT_RUN_RAMP IfcRamp", "IfcRamp", -0.3, 0.0), "ramp")
        self.assertEqual(c("Junction J1 FLEXIBLE IfcPavement", "IfcPavement", -0.3, -0.15), "carriageway")
        self.assertEqual(c("Sample Avenue 5 carriageway FLEXIBLE IfcPavement", "IfcPavement", -0.3, -0.15),
                         "carriageway")
        self.assertEqual(c("Sample Avenue 5 kerbs NOTDEFINED IfcKerb", "IfcKerb", -0.45, 0.0), "kerb")
        self.assertEqual(c("Driveway crossover 1 FLEXIBLE IfcPavement", "IfcPavement", -0.3, 0.0), "paving")
        self.assertEqual(c("Covered linkway 3 roof ROOF IfcSlab", "IfcSlab", 2.4, 2.6), "covered")


def _real_site():
    """(SITE.ifc, SITE_graph.json) written together: the model's, else the scratch build of the nav fixes
    (estate.site.builder.build_site(masterplan.resolve(), 'build/fix3_nav/SITE.ifc', graph_path=...))."""
    for d in (env.MODEL, env.BUILD / "fix3_nav"):
        ifc, gj = d / "SITE.ifc", d / "SITE_graph.json"
        if ifc.exists() and gj.exists() and nav2d.usable_graph(gj, ifc, []) is not None:
            return ifc, gj
    return None


class RealSiteCrossings(unittest.TestCase):
    """graph_vs_ifc on a real SITE.ifc and the pedestrian graph written with it: the zebra / kerb-ramp check
    inspects every crossing (the carriageways are cut around the zebras), passes as built, and catches kerb ramps
    removed from a copy of the IFC."""

    @classmethod
    def setUpClass(cls):
        found = _real_site()
        if found is None:
            raise unittest.SkipTest("no SITE.ifc with its own SITE_graph.json (./estate.sh site)")
        cls.ifc_path, cls.graph_path = found
        cls.data = json.loads(cls.graph_path.read_text(encoding="utf-8"))
        cls.ifc = nav2d.read_site_ifc(cls.ifc_path)
        cls.listed = sorted(f"Zebra crossing {c['id']}" for c in cls.data["meta"]["crossings"]
                            if c.get("kind") in ("bus", "junction"))

    def test_as_built(self):
        names = self.ifc["crossing_names"]
        self.assertTrue(names)
        self.assertTrue(all(re.match(r"Zebra crossing X\d+$", n) for n in names), names)
        cw, zb = shapely.union_all(self.ifc["carriageway"]), shapely.union_all(self.ifc["crossing"])
        self.assertLess(cw.intersection(zb).area, 1e-3)               # the topology the old check missed
        fixes, rep = nav2d.graph_vs_ifc(self.data, self.ifc, footprints=[])
        self.assertGreater(rep["edges_over_carriageway"], 0)
        self.assertEqual(rep["zebras_crossed"], self.listed)          # every bus / junction crossing inspected
        self.assertEqual(rep["kerb_drops"], [])
        self.assertEqual(rep["zebras_missing"], [])
        self.assertFalse(any("step_free" in f for f in fixes.values()))
        xid = self.listed[0].split()[-1]
        one = [r for r in self.ifc["ramps"] if r["name"] != f"Crossing {xid} kerb ramp 2"]
        self.assertEqual(len(one), len(self.ifc["ramps"]) - 1)
        _, rep1 = nav2d.graph_vs_ifc(self.data, dict(self.ifc, ramps=one), footprints=[])
        self.assertEqual(len(rep1["kerb_drops"]), 1, rep1["kerb_drops"])
        self.assertIn(f"over Zebra crossing {xid}: kerb ramps (<= 1:10) at 1 of 2 kerbs", rep1["kerb_drops"][0])

    def test_kerb_ramps_removed_from_a_copy(self):
        import ifcopenshell
        import ifcopenshell.api.root
        from estate import masterplan
        OUT.mkdir(parents=True, exist_ok=True)
        f = ifcopenshell.open(str(self.ifc_path))
        gone = [e for e in f.by_type("IfcRamp") if " kerb ramp " in (e.Name or "")]
        self.assertEqual(len(gone), 2 * len(self.listed))
        for e in gone:
            ifcopenshell.api.root.remove_product(f, product=e)
        copy = OUT / "SITE_no_kerb_ramps.ifc"
        f.write(str(copy))
        ifc = nav2d.read_site_ifc(copy, cache=False)
        self.assertFalse([r for r in ifc["ramps"] if r["kind"] == "kerb_ramp"])
        self.assertEqual(sorted(ifc["crossing_names"]), sorted(self.ifc["crossing_names"]))
        _, rep = nav2d.graph_vs_ifc(self.data, ifc, footprints=[])
        self.assertEqual(rep["zebras_crossed"], self.listed)
        self.assertEqual(len(rep["kerb_drops"]), rep["edges_over_carriageway"])
        self.assertTrue(all("at 0 of 2 kerbs" in d for d in rep["kerb_drops"]), rep["kerb_drops"])
        self.assertFalse(rep["ok"])
        # the whole site walk test, with the graph re-stamped for the copy so that it is used
        g = json.loads(json.dumps(self.data))
        g["meta"]["ifc_sha256"] = nav2d._sha256(copy)
        gp = OUT / "SITE_no_kerb_ramps_graph.json"
        gp.write_text(json.dumps(g), encoding="utf-8")
        res = nav2d.check_site(masterplan.resolve(), site_graph_json=gp, site_ifc=copy, out_dir=None, png=False)
        self.assertEqual(res["mode"], "graph", res["assumptions"])
        self.assertFalse(res["graph_checks"]["ok"])
        bad = {frozenset(e.split()[0].split("-")) for e in res["graph"]["not_step_free_edges"]}
        drops = {frozenset(d.split()[0].split("-")) for d in res["graph_checks"]["kerb_drops"]}
        self.assertEqual(len(drops), rep["edges_over_carriageway"])
        self.assertLessEqual(drops, bad)                              # every crossing edge is no longer step-free
        self.assertTrue(any("kerb ramps (<= 1:10) at both kerbs" in a for a in res["assumptions"]), res["assumptions"])


class EstateGraph(unittest.TestCase):
    def test_flats_from_bus_stops(self):
        OUT.mkdir(parents=True, exist_ok=True)
        mp = {"bus_stops": [{"id": "BS1", "at": [0, 0]}, {"id": "BS2", "at": [99, 0]}],
              "sites": [SimpleNamespace(id="BLK_1", kind="block", lift_lobbies=[(5, 5)], storeys=3, plan=None)]}
        nav = {"stem": "BLK_1", "results": {"0.30": {
            "summary": {"ok": False, "step_free_ok": True, "start_cells": 10},
            "lifts": [{"step_free_from_street": True}],
            "flats": [{"unit": "#02-101", "reachable": True, "main_door_step_free": True},
                      {"unit": "#02-103", "reachable": False, "main_door_step_free": True}]}}}
        site = {"mode": "test", "routes": [
            {"bus_stop": "BS1", "target": "BLK_1", "lobby": 0, "reachable": True, "length_m": 120.0, "step_free": True},
            {"bus_stop": "BS2", "target": "BLK_1", "lobby": 0, "reachable": True, "length_m": 80.0,
             "step_free": False}]}
        rep = navgraph.estate_report(mp, [nav], site, out_path=OUT / "nav_estate.json")
        f1, f2 = rep["flats"]
        self.assertTrue(f1["reachable_from_all"])
        self.assertFalse(f1["step_free_from_all"])            # BS2's route has a step
        self.assertTrue(f1["from"]["BS1"]["step_free"])
        self.assertEqual(f1["from"]["BS2"]["walk_m"], 80.0)
        self.assertFalse(f2["reachable_from_all"])            # a room of #02-103 is cut off
        self.assertEqual(rep["summary"]["reachable_from_every_bus_stop"], 1)


@unittest.skipUnless(LEGACY.exists(), "build/legacy/hdb_block_legacy.ifc missing (run estate.cmd legacy)")
class LegacyRegression(unittest.TestCase):
    def test_legacy_block(self):
        rep = nav3d.check_building(LEGACY, OUT / "legacy", [0.30, 0.40], png=False)
        r30, r40 = rep["results"]["0.30"]["summary"], rep["results"]["0.40"]["summary"]
        self.assertEqual((r30["spaces"], r30["reachable"]), (497, 497))   # 496 IfcSpace + the roof
        self.assertTrue(r30["roof_reachable"])
        self.assertEqual(r30["flats_reachable"], 44)
        self.assertEqual(r30["doors_impassable"], 0)                     # 0.70 m clear between linings fits r 0.30
        cut = {"Bedroom 2", "Bedroom 3", "Common Bathroom", "Household Shelter", "Kitchen", "Master Bathroom",
               "Master Bedroom", "Service Yard"}                       # rooms behind 0.8-0.9 m doors
        self.assertEqual(set(r40["unreachable_by_room"]), cut)
        self.assertTrue(all(v["count"] == 44 for v in r40["unreachable_by_room"].values()))
        self.assertTrue(r40["roof_reachable"])
        self.assertEqual(r30["flats_main_door_step_free"], 0)            # the 0.12-0.15 m void deck step
        self.assertEqual(rep["agent"]["effective_radius"], {"0.30": 0.3, "0.40": 0.4})
        self.assertEqual(rep["warnings"], [])
        self.assertEqual(rep["doors"]["narrowest_clear_m"], 0.7)


@unittest.skipUnless(PT4.exists(), "build/t_pt4.ifc missing")
class PointBlock(unittest.TestCase):
    def test_t_pt4(self):
        rep = nav3d.check_building(PT4, OUT / "t_pt4", 0.30, png=False)
        s = rep["results"]["0.30"]["summary"]
        self.assertTrue(s["ok"])
        self.assertTrue(s["step_free_ok"])
        self.assertEqual(s["flats_main_door_step_free"], s["flats"])
        self.assertEqual(s["main_doors_checked_in"], {"entry room": s["flats"]})
        lifts = rep["results"]["0.30"]["lifts"]
        self.assertTrue(lifts and all(not lf["missing_levels"] and lf["step_free_from_street"] for lf in lifts))


if __name__ == "__main__":
    unittest.main()
