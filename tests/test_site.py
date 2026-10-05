"""Site works: free-space routing through ground floors, cover splitting, plan sections of building meshes, the bus
shelter opening, escape paths from stair discharges, and the pedestrian graph of the real masterplan checked against
the building IFCs and the SITE.ifc it is written with (no edge through a wall, column or table; crossings over zebras
with kerb ramps; covered edges under a roof; graph tied to the IFC by sha256; site ground meeting every building
floor; every published stair discharge on paving; the IFC4 copy keeping the GlobalIds of the IFC4X3 master), and the
SVG site plan drawn beside the PNG from the same layout (themes as groups, footprints as paths, real text, stable
bytes).

Run: "<blender python>" -I -B tests/test_site.py -v
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estate.env import bootstrap  # noqa: E402

bootstrap()

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import LineString, Point, box  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from estate import env  # noqa: E402
from estate.site import builder  # noqa: E402
from estate.site.linkways import FreeSpace, split_cover  # noqa: E402



def box_mesh(x0, y0, z0, x1, y1, z1):
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    f = np.array([(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
                  (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)])
    return v, f


class FreeSpaceRouting(unittest.TestCase):
    def setUp(self):
        self.walk = box(0, 0, 20, 10)
        self.obs = unary_union([box(9.8, 0, 10.2, 8), box(4.7, 4.7, 5.3, 5.3)])   # a wall with a 2 m gap, a column
        self.fs = FreeSpace(self.walk, self.obs, clearance=0.35, res=0.2)

    def test_route_goes_round_the_wall_with_clearance(self):
        r = self.fs.routes((1.0, 5.0), [(19.0, 5.0)])[0]
        self.assertGreaterEqual(r.distance(self.obs), 0.35 - 1e-6)
        self.assertTrue(self.walk.covers(r))
        best = 2 * math.hypot(8.45, 3.35) + 1.1                     # round the wall end, 0.35 clear
        self.assertLess(r.length, best * 1.02)
        self.assertGreaterEqual(r.length, best - 1e-6)

    def test_terminal_inside_an_obstacle_is_unreachable(self):
        self.assertEqual(self.fs.routes((1.0, 5.0), [(5.0, 5.0)]), {})

    def test_terminal_on_the_edge_next_to_an_obstacle(self):
        r = self.fs.routes((10.0, 10.0), [(1.0, 1.0)])[0]           # on the region edge, 2 m above the wall end
        self.assertEqual(tuple(r.coords[0]), (10.0, 10.0))
        self.assertAlmostEqual(r.intersection(self.obs).length, 0.0)

    def test_deterministic(self):
        a = self.fs.routes((1.0, 5.0), [(19.0, 5.0), (0.5, 9.5)])
        b = FreeSpace(self.walk, self.obs, 0.35, 0.2).routes((1.0, 5.0), [(19.0, 5.0), (0.5, 9.5)])
        self.assertEqual({k: v.wkt for k, v in a.items()}, {k: v.wkt for k, v in b.items()})

    def test_route_keeps_under_cover(self):
        """A void deck with its slab over y 0-6 and an open strip beyond: the leg between two points under the
        slab stays under it (and its string pull does not cut across the open strip), unless told otherwise."""
        walk = box(0, 0, 20, 10)
        obs = box(9.0, 3.0, 11.0, 9.0)                               # a core: round it under the slab or outside
        cover = box(-1, -1, 21, 6)
        a, b = (2.0, 7.0), (18.0, 7.0)                               # both just outside the slab line
        plain = FreeSpace(walk, obs, 0.35, 0.2).routes(a, [b])[0]
        dry = FreeSpace(walk, obs, 0.35, 0.2, cover=cover, penalty=2.0).routes(a, [b])[0]
        self.assertGreater(plain.difference(cover).length, 8.0)     # shortest: round the top of the core, in the open
        self.assertLess(dry.difference(cover).length, plain.difference(cover).length - 8.0)
        self.assertGreaterEqual(dry.distance(obs), 0.35 - 1e-6)
        self.assertTrue(walk.covers(dry))


class CoverAndSections(unittest.TestCase):
    def test_split_cover(self):
        ln = LineString([(0, 0), (10, 0), (10, 10)])
        parts = split_cover(ln, box(-1, -1, 4, 1))
        self.assertAlmostEqual(sum(p.length for p, _ in parts), ln.length, places=6)
        self.assertAlmostEqual(sum(p.length for p, c in parts if c), 4.0, places=6)
        self.assertEqual(split_cover(ln, box(-1, -1, 11, 11)), [(ln, True)])

    def test_sections_catch_table_tops_and_stool_seats(self):
        ped = box_mesh(-0.06, -0.06, 0.0, 0.06, 0.06, 0.71)
        top = box_mesh(-0.4, -0.4, 0.71, 0.4, 0.4, 0.75)
        v = np.vstack([ped[0], top[0]])
        f = np.vstack([ped[1], top[1] + len(ped[0])])
        g = builder._sections(v, f, builder.GROUND_CUTS)
        self.assertAlmostEqual(g.area, 0.64, places=3)              # the 0.73 m cut finds the table top
        seat = box_mesh(-0.17, -0.17, 0.42, 0.17, 0.17, 0.46)
        self.assertAlmostEqual(builder._sections(*seat, builder.GROUND_CUTS).area, 0.34 ** 2, places=4)
        self.assertIsNone(builder._sections(*box_mesh(0, 0, -0.2, 5, 5, 0.0), [0.5]))

    def test_edge_hits(self):
        G = nx.Graph()
        G.add_edge((0.0, 0.0), (10.0, 0.0), kind="footpath", via="void_deck")
        G.add_edge((0.0, 2.0), (10.0, 2.0), kind="footpath", via=None)
        hits = builder.edge_hits(G, [("L1 column C01", box(4.7, -0.3, 5.3, 0.3)), ("bench", box(0, 1.98, 10, 1.99))])
        self.assertEqual([(h["obstacle"], h["length"]) for h in hits], [("L1 column C01", 0.6)])


class BusShelter(unittest.TestCase):
    def test_shelter_centre_line_is_open(self):
        from estate import masterplan
        from estate.site import amenities
        from estate.site import roads as R
        net = R.layout_roads(masterplan.resolve(plans=False))
        self.assertTrue(net.bus_stops)
        for b in net.bus_stops:
            obs = builder._piece_obstacles(amenities.bus_shelter_pieces(b, "part"))
            walk = LineString([b.node, b.stub])
            self.assertEqual(len(obs), 7)                              # 4 posts, the split back panel, 2 benches
            for name, poly in obs:
                self.assertGreater(walk.distance(poly), 0.5, name)


class EscapePaths(unittest.TestCase):
    """A stair discharge on the south facade of a 20 x 20 m building, a sidewalk 8 m south, a driveway 0.6 m east of
    the door: the path is 1.8 m wide, shifted west to keep 0.3 m off the driveway with the door still on it."""

    def setUp(self):
        self.site = SimpleNamespace(id="X", footprint=box(0, 10, 20, 30))
        self.walk = box(-10, 0, 30, 2)
        self.drive = box(4.6, 2, 10, 10)

    def plan(self, pts, blocked=None, floors=None):
        pub = {"X": dict(discharge=pts)}
        blocked = unary_union([self.site.footprint] + ([blocked] if blocked is not None else []))
        return builder.escape_paths([self.site], pub, self.walk, blocked, self.drive, floors=floors,
                                    extent=box(-20, -20, 40, 40))

    def test_path_keeps_clear_of_the_driveway(self):
        (e,), warn = self.plan([(4.0, 10.0)])
        self.assertEqual(warn, [])
        self.assertEqual(e["width"], 1.8)
        self.assertAlmostEqual(e["offset"], -0.6)
        self.assertAlmostEqual(e["length"], 8.0)
        self.assertEqual(e["out"], (0.0, -1.0))
        self.assertGreaterEqual(e["poly"].distance(self.drive), builder.DRIVE_CLEAR - 1e-6)
        x0, _, x1, _ = e["poly"].bounds                                  # the door well inside the path's sides
        self.assertGreaterEqual(min(4.0 - x0, x1 - 4.0), builder.ESCAPE_DOOR - 1e-6)
        self.assertTrue(e["poly"].buffer(1e-6).contains(Point(4.0, 10.0)))
        self.assertAlmostEqual(e["poly"].distance(self.walk), 0.0)
        self.assertLess(e["poly"].intersection(self.site.footprint).area, 1e-9)

    def test_narrower_path_where_the_wide_one_does_not_fit(self):
        (e,), warn = self.plan([(4.0, 10.0)], blocked=box(-5, 2, 2.7, 10))
        self.assertEqual(warn, [])
        self.assertEqual(e["width"], 1.5)
        self.assertLess(e["poly"].intersection(box(-5, 2, 2.7, 10)).area, 1e-9)
        self.assertGreaterEqual(e["poly"].distance(self.drive), builder.DRIVE_CLEAR - 1e-6)

    def test_discharge_on_its_own_floor_or_blocked(self):
        (e,), warn = self.plan([(10.0, 20.0)], floors={"X": self.site.footprint})
        self.assertTrue(e.get("on_floor"))
        self.assertIsNone(e["poly"])
        self.assertEqual(warn, [])
        (e,), warn = self.plan([(4.0, 10.0)], blocked=box(-5, 4, 25, 5))        # a wall between door and sidewalk
        self.assertTrue(e.get("failed"))
        self.assertEqual(len(warn), 1)

    def test_graph_line_runs_to_the_footway_centreline(self):
        (e,), _ = self.plan([(4.0, 10.0)])
        sidewalk = (LineString([(-10, 1), (30, 1)]), dict(kind="sidewalk", width=2.0))
        lines, points = builder._escape_lines([e], [sidewalk])
        self.assertEqual(len(lines), 1)
        ln, a = lines[0]
        self.assertEqual(list(ln.coords), [(4.0, 10.0), (4.0, 1.0)])
        self.assertEqual((a["kind"], a["via"]), ("escape", "stair_discharge"))
        self.assertEqual(points, [dict(xy=(4.0, 10.0), kind="stair_discharge", ref="X")])
        self.assertEqual(builder._with_cover(lines, box(-50, -50, 50, 50))[0][1]["covered"], False)


SVG = "{http://www.w3.org/2000/svg}"


class SitePlanBackends(unittest.TestCase):
    """The SVG backend's primitives on a 100 x 100 m frame: holes, translucency, dashes, multi-line text and the
    bottom-to-top theme order that keeps the SVG stacked as the PNG is painted."""

    def svg(self, draw):
        import xml.etree.ElementTree as ET
        from estate.site.siteplan import SvgPlan
        pl = SvgPlan((0, 0, 100, 100), width=500)
        draw(pl)
        p = Path(tempfile.mkdtemp(prefix="unittest-", dir=env.BUILD)) / "t.svg"
        try:
            pl.save(p)
            return pl, ET.parse(p).getroot()
        finally:
            shutil.rmtree(p.parent, ignore_errors=True)

    def test_polygon_with_hole_is_one_evenodd_path(self):
        ring = box(10, 10, 90, 90).difference(box(40, 40, 60, 60))
        two = unary_union([box(0, 0, 5, 5), box(20, 0, 25, 5)])

        def draw(pl):
            pl.layer("roads")
            pl.fill(ring, (226, 112, 36, 110), outline=(226, 112, 36), width=2, name="ring")
            pl.fill(two, (78, 80, 86))
        pl, root = self.svg(draw)
        paths = root.find(f"{SVG}g[@id='roads']").findall(SVG + "path")
        self.assertEqual(len(paths), 2)
        r = paths[0]
        self.assertEqual((r.get("id"), r.get("fill-rule"), r.get("fill"), r.get("fill-opacity")),
                         ("ring", "evenodd", "#e27024", "0.431"))
        self.assertEqual((r.get("stroke"), r.get("stroke-width")), ("#e27024", "2"))
        self.assertEqual(r.get("d").count("M"), 2)
        self.assertEqual(paths[1].get("d").count("M"), 2)                  # a MultiPolygon is still one path
        self.assertIsNone(paths[1].get("fill-opacity"))
        self.assertEqual(root.get("viewBox"), f"0 0 {pl.W} {pl.H}")

    def test_dashes_text_and_layer_order(self):
        def draw(pl):
            pl.layer("graph")
            pl.dashed((10, 10), (10, 50), (70, 70, 78), 2)
            pl.layer("labels")
            pl.text((50, 50), "501\n20 st", 22)
            pl.text((50, 20), "BS1", 15, (25, 70, 170), stroke=2)
        pl, root = self.svg(draw)
        (ln,) = root.find(f"{SVG}g[@id='graph']").findall(SVG + "line")
        dash, gap = (float(v) for v in ln.get("stroke-dasharray").split())
        self.assertAlmostEqual(dash, 1.2 * pl.s, delta=0.006)
        self.assertAlmostEqual(gap, 0.8 * pl.s, delta=0.006)
        a, b = root.find(f"{SVG}g[@id='labels']").findall(SVG + "text")
        lines = a.findall(SVG + "tspan")
        self.assertEqual([t.text for t in lines], ["501", "20 st"])
        for t in lines:                                                      # both lines centred on the anchor
            self.assertAlmostEqual(float(t.get("x")), pl.P((50, 50))[0], delta=0.006)
        self.assertGreater(float(lines[1].get("y")) - float(lines[0].get("y")), 20)
        self.assertEqual((a.get("text-anchor"), b.text, b.get("stroke"), b.get("stroke-width"), b.get("paint-order")),
                         ("middle", "BS1", "#ffffff", "4", "stroke"))
        with self.assertRaises(ValueError):
            self.svg(lambda pl: (pl.layer("labels"), pl.layer("roads")))


class SiteGraph(unittest.TestCase):
    """The real masterplan against the building IFCs in model/ (footprints where an IFC does not exist yet)."""

    @classmethod
    def setUpClass(cls):
        from estate import masterplan
        global OUT
        (env.BUILD / "site_tests").mkdir(parents=True, exist_ok=True)
        OUT = Path(tempfile.mkdtemp(prefix="unittest-", dir=env.BUILD / "site_tests"))   # test runs may overlap
        cls.mp = masterplan.resolve()
        cls.L = builder.plan_site(cls.mp)
        cls.info = builder.build_site(cls.mp, OUT / "SITE.ifc", graph_path=OUT / "SITE_graph.json", layout=cls.L,
                                      plan_png=OUT / "site_plan.png", plan_svg=OUT / "site_plan.svg")
        cls.info4 = builder.write_site(cls.L, OUT / "SITE_ifc4.ifc", "IFC4")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(OUT, ignore_errors=True)

    def test_no_edge_through_an_obstacle(self):
        self.assertEqual(self.L.graph_check["obstacle_hits"], 0, self.L.graph_check["obstacle_hit_examples"][:5])
        self.assertEqual(self.info["graph_check"]["issues"], [])

    def test_ground_floor_legs_are_routed(self):
        G = self.L.graph
        legs = [(u, v) for u, v, d in G.edges(data=True) if d.get("via") in ("void_deck", "indoor")]
        self.assertTrue(legs)
        for s in self.mp["sites"]:
            if s.kind == "block":
                for q in s.lift_lobbies:
                    u = (round(float(q[0]), 2), round(float(q[1]), 2))
                    self.assertIn(u, G, f"{s.id} lobby {q} not in the graph")
                    self.assertEqual(G.nodes[u]["kind"], "lift_lobby")
        self.assertEqual(nx.number_connected_components(G), 1)
        self.assertEqual(self.L.graph_stats["n_unreachable"], 0)

    def test_covered_edges_under_cover(self):
        cb = self.L.cover.buffer(0.1)
        for u, v, d in self.L.graph.edges(data=True):
            if d["covered"]:
                self.assertLess(LineString([u, v]).difference(cb).length, 0.05, (u, v, d))
            if d["kind"] in ("crossing", "sidewalk"):
                self.assertFalse(d["covered"])

    def test_graph_tied_to_its_ifc(self):
        from estate.export.meshcache import sha256
        g = json.loads((OUT / "SITE_graph.json").read_text(encoding="utf-8"))
        self.assertEqual(g["meta"]["source"], "SITE.ifc")
        self.assertEqual(g["meta"]["ifc_sha256"], sha256(OUT / "SITE.ifc"))
        self.assertEqual(g["meta"]["checks"]["issues"], 0)
        self.assertEqual(set(g["meta"]["buildings"]), {s.id for s in self.mp["sites"]})

    def test_crossings_have_zebra_and_ramps(self):
        import ifcopenshell
        f = ifcopenshell.open(str(OUT / "SITE.ifc"))
        names = {e.Name for e in f.by_type("IfcElement")}
        for c in self.L.net.crossings:
            self.assertIn(f"Zebra crossing {c.id}", names)
            self.assertIn(f"Crossing {c.id} kerb ramp 1", names)
            self.assertIn(f"Crossing {c.id} kerb ramp 2", names)
        # a crossing edge with no zebra under it is reported
        L = self.L
        G0 = L.graph
        try:
            G = G0.copy()
            G.add_edge((1.0, 1.0), (1.0, 9.0), kind="crossing", covered=False, length=8.0, width=4.0, slope=0.08)
            L.graph = G
            issues = builder.check_graph(L, OUT / "SITE.ifc")
        finally:
            L.graph = G0
        self.assertTrue(any("not over a zebra" in i for i in issues))

    def test_site_ground_meets_every_building_floor(self):
        """Every point of the estate is ground: a site surface or a building's floor slab, and the site never
        paves over a building's floor (F1: the centre's forecourt)."""
        ground_kinds = ("carriageway", "crossing", "driveway", "crossover", "kerb", "sidewalk", "paving", "ramp",
                        "terrain")
        site = unary_union([p.geom for p in self.L.pieces if p.kind in ground_kinds and p.geom is not None])
        floors = unary_union([g.floor for g in self.L.grounds.values() if g.floor is not None])
        x0, y0, x1, y1 = self.mp["extent"].bounds
        X, Y = np.meshgrid(np.arange(x0 + 0.37, x1, 1.0), np.arange(y0 + 0.37, y1, 1.0))
        on_site = shapely.contains_xy(site, X, Y)
        on_floor = shapely.contains_xy(floors, X, Y)
        holes = np.count_nonzero(~(on_site | on_floor))
        self.assertLessEqual(holes, 2, f"{holes} sample points with no ground")
        self.assertLess(site.intersection(floors).area, 0.5)

    def test_open_forecourt_carries_linkways(self):
        """The centre's forecourt (its own floor slab open to the sky) is open ground: the centre's entrances sit
        on the hawker hall and shop block, a linkway canopy (with columns) crosses the forecourt, and the site never
        paves it. Car parks and blocks have no open ground (ramps and decks above count as overhead)."""
        nc = [s for s in self.mp["sites"] if s.kind == "nc"]
        if not nc or self.L.grounds[nc[0].id].source == "footprint":
            self.skipTest("no centre IFC in model/")
        s = nc[0]
        og = self.L.open_ground.get(s.id)
        self.assertIsNotNone(og)
        self.assertGreater(og.area, 0.2 * s.footprint.area)
        self.assertEqual(set(self.L.open_ground), {s.id})
        roofed = s.footprint.difference(og)
        ents = [e for e in self.L.entrances if e.site is s]
        self.assertTrue(ents)
        for e in ents:
            self.assertLess(roofed.boundary.distance(Point(e.E)), 0.05, e.E)        # on the hall or shop block
        ring = s.footprint.exterior
        self.assertTrue(any(ring.distance(Point(e.E)) > 1.0 for e in ents))      # not all pushed to the rect edge
        roofs = unary_union([p.geom for p in self.L.pieces if p.kind == "roof" and (p.assembly or "").startswith("lw:")])
        self.assertGreater(roofs.intersection(og).area, 20.0)
        ground_kinds = ("carriageway", "crossing", "driveway", "crossover", "kerb", "sidewalk", "paving", "ramp")
        paved = unary_union([p.geom for p in self.L.pieces if p.kind in ground_kinds and p.geom is not None])
        self.assertLess(paved.intersection(og).area, 0.5)

    def test_nearest_stop_routes_covered(self):
        """nav2d on the graph and SITE.ifc just written: every lift lobby reached from its nearest bus stop with at
        least the target covered share, within the walking limit, step-free, the graph agreeing with the IFC."""
        from estate.validate import nav2d
        res = nav2d.check_site(self.mp, site_graph_json=OUT / "SITE_graph.json", site_ifc=OUT / "SITE.ifc",
                               out_dir=None, png=False)
        self.assertEqual(res["mode"], "graph", res["assumptions"])
        self.assertTrue(res["graph_checks"]["ok"], res["graph_checks"])
        s = res["summary"]
        self.assertTrue(s["reachability_ok"] and s["step_free_ok"] and s["walk_ok"], s)
        self.assertTrue(s["covered_ok"], s["nearest_stop_covered_below_target"])

    def test_stair_discharges_paved(self):
        """HDB-N1: every stair discharge a building publishes (build.json) lets out within 0.5 m of a paved walkable
        surface of SITE.ifc (or onto the building's own floor), on a path at grade 0.3 m clear of every driveway,
        which the pedestrian graph joins to the network as an uncovered 'escape' edge."""
        import ifcopenshell
        pts = [(s, q) for s in self.mp["sites"] if s.kind != "block"
               for q in ((builder.published(s) or {}).get("discharge") or [])]
        if not pts:
            self.skipTest("no building in model/ publishes a stair discharge")
        plans = builder._plans_from_ifc(OUT / "SITE.ifc", lambda e: e.is_a("IfcSlab") and e.PredefinedType in
                                        ("PAVING", "SIDEWALK"))
        paved = unary_union([g for g, _ in plans.values()])
        slabs = {e.Name for e in ifcopenshell.open(str(OUT / "SITE.ifc")).by_type("IfcSlab")}
        drives = unary_union([p.geom for p in self.L.pieces if p.kind in ("driveway", "crossover")])
        G = self.L.graph
        stops = [u for u in G.nodes if G.nodes[u]["kind"] == "bus_stop"]
        for s, q in pts:
            if s.footprint.buffer(-0.05).contains(Point(q)):
                self.assertLess(self.L.grounds[s.id].floor.distance(Point(q)), 0.05, (s.id, q))
                continue
            self.assertLessEqual(paved.distance(Point(q)), 0.5, (s.id, q))
            e = next(e for e in self.L.escapes if e["site"] is s and e["at"] == (round(q[0], 3), round(q[1], 3)))
            self.assertFalse(e.get("failed"), e)
            if e["poly"] is not None:
                p = next(p for p in self.L.pieces if p.name == f"{s.name} escape path {e['k'] + 1}")
                self.assertIn(p.name, slabs)
                self.assertAlmostEqual(p.z0 + p.h, 0.0)
                self.assertGreaterEqual(p.geom.distance(drives), builder.DRIVE_CLEAR - 1e-3)
            u = (round(e["start"][0], 2), round(e["start"][1], 2))
            self.assertEqual(G.nodes[u]["kind"], "stair_discharge")
            self.assertTrue(any(nx.has_path(G, u, b) for b in stops))
            for v in G.neighbors(u):
                self.assertEqual(G.edges[u, v]["kind"], "escape")
                self.assertFalse(G.edges[u, v]["covered"])

    def test_ifc4_copy_keeps_the_guids(self):
        """CP2-1: every IfcProduct of SITE_ifc4.ifc has the GlobalId and Name it has in SITE.ifc; only the IFC4X3
        road facilities (IfcRoad, IfcRoadPart) have no IFC4 counterpart; neither file repeats a GlobalId."""
        import ifcopenshell
        a, b = ifcopenshell.open(str(OUT / "SITE.ifc")), ifcopenshell.open(str(OUT / "SITE_ifc4.ifc"))
        for f in (a, b):
            ids = [e.GlobalId for e in f.by_type("IfcRoot")]
            self.assertEqual(len(ids), len(set(ids)))
        pa = {e.GlobalId: e for e in a.by_type("IfcProduct")}
        pb = {e.GlobalId: e for e in b.by_type("IfcProduct")}
        self.assertEqual([(e.is_a(), e.Name) for g, e in sorted(pb.items()) if g not in pa or pa[g].Name != e.Name], [])
        self.assertEqual({pa[g].is_a() for g in set(pa) - set(pb)}, {"IfcRoad", "IfcRoadPart"})
        types_a = {(e.GlobalId, e.Name) for e in a.by_type("IfcTypeObject")}
        self.assertEqual({(e.GlobalId, e.Name) for e in b.by_type("IfcTypeObject")}, types_a)

    def test_guids_survive_road_part_edits(self):
        """Adding a road part and renaming another (IFC4X3-only entities, created before everything else) keeps
        every other GlobalId: assemblies, pavements and kerbs take theirs from what they are, not from a counter."""
        import ifcopenshell
        L, parts = self.L, dict(self.L.parts)
        first = sorted(parts)[0]
        road = parts[first]["road"]
        try:
            edited = dict(parts)
            edited[first] = dict(parts[first], name=parts[first]["name"] + " (renamed)")
            L.parts = {f"{road}:SIDEWALK:EXTRA": dict(road=road, predefined="SIDEWALK", name="Extra sidewalk",
                                                     usage="LONGITUDINAL"), **edited}
            builder.write_site(L, OUT / "edited" / "SITE.ifc", "IFC4X3")
        finally:
            L.parts = parts
        a = {e.GlobalId: e for e in ifcopenshell.open(str(OUT / "SITE.ifc")).by_type("IfcProduct")}
        b = {e.GlobalId: e for e in ifcopenshell.open(str(OUT / "edited" / "SITE.ifc")).by_type("IfcProduct")}
        self.assertEqual(sorted((e.is_a(), e.Name) for g, e in a.items() if g not in b),
                         [("IfcRoadPart", parts[first]["name"])])
        self.assertEqual(sorted((e.is_a(), e.Name) for g, e in b.items() if g not in a),
                         sorted([("IfcRoadPart", "Extra sidewalk"), ("IfcRoadPart", parts[first]["name"] + " (renamed)")]))
        self.assertTrue(any(e.is_a("IfcElementAssembly") for e in a.values()))

    def test_site_build_json_paths_relative(self):
        """reports/site_build.json as the build and `estate.cmd site` write it (both through write_site_report), read
        back from disk: every file path project-relative, so the published report names no folder of the machine that
        built it; the pedestrian graph holds no absolute path either."""
        from estate import leaks
        written = builder.write_site_report(self.info, OUT / "site_build.json")
        rep = json.loads((OUT / "site_build.json").read_text(encoding="utf-8"))
        self.assertEqual(rep, json.loads(json.dumps(written, default=str)))
        self.assertEqual([rep[k] for k in builder.REPORT_PATHS],
                         [env.rel(OUT / n) for n in ("SITE.ifc", "SITE_graph.json", "site_plan.png", "site_plan.svg")])
        self.assertTrue(rep["path"].startswith("build/site_tests/unittest-"), rep["path"])
        self.assertEqual(leaks.absolute_strings(rep), [])
        self.assertEqual(leaks.scan_json(json.dumps(rep)), [])
        graph = json.loads((OUT / "SITE_graph.json").read_text(encoding="utf-8"))
        self.assertEqual(leaks.absolute_strings(graph), [])

    def test_site_plan_svg(self):
        """reports/site_plan.svg as build_site writes it beside the PNG from the same layout: XML in the PNG's pixel
        frame, one group per theme, one footprint path per site, block numbers and bus stop names as real text,
        coordinates to 0.01 px, and the same bytes on a second render."""
        import re
        import xml.etree.ElementTree as ET
        from PIL import Image
        from estate.site import siteplan
        png, svg = OUT / "site_plan.png", OUT / "site_plan.svg"
        self.assertEqual((self.info["site_plan"], self.info["site_plan_svg"]), (str(png), str(svg)))
        root = ET.parse(svg).getroot()
        with Image.open(png) as im:
            W, H = im.size
        self.assertEqual((root.get("viewBox"), root.get("width"), root.get("height")), (f"0 0 {W} {H}", str(W), str(H)))
        self.assertEqual([g.get("id") for g in root.findall(SVG + "g")], list(siteplan.LAYERS))
        fps = root.find(f"{SVG}g[@id='footprints']").findall(SVG + "path")
        self.assertEqual([p.get("id") for p in fps if (p.get("id") or "").startswith("footprint-")],
                         [f"footprint-{s.id}" for s, _ in self.L.draw["footprints"]])
        self.assertTrue(all(p.get("fill-rule") == "evenodd" for p in root.iter(SVG + "path")))
        texts = [[t for t in e.itertext() if t.strip()] for e in root.iter(SVG + "text")]
        self.assertTrue(self.L.net.bus_stops)
        for b in self.L.net.bus_stops:
            self.assertIn([b.id], texts)
        for s in self.mp["sites"]:
            if s.kind == "block":
                self.assertIn([str(s.blk), f"{s.storeys} st"], texts)
        self.assertEqual(len(root.find(f"{SVG}g[@id='trees']").findall(f".//{SVG}circle")), len(self.L.trees))
        coords = " ".join(e.get(k) or "" for e in root.iter() for k in ("d", "points", "x", "y", "cx", "cy"))
        self.assertIsNone(re.search(r"\d\.\d{3}", coords))
        siteplan.render(self.L, OUT / "again_plan" / "site_plan.svg")
        self.assertEqual(svg.read_bytes(), (OUT / "again_plan" / "site_plan.svg").read_bytes())

    def test_deterministic(self):
        again = builder.build_site(self.mp, OUT / "again" / "SITE.ifc", graph_path=OUT / "again" / "SITE_graph.json")
        self.assertEqual(again["graph_check"]["issues"], [])
        self.assertEqual((OUT / "SITE.ifc").read_bytes(), (OUT / "again" / "SITE.ifc").read_bytes())
        a = json.loads((OUT / "SITE_graph.json").read_text(encoding="utf-8"))
        b = json.loads((OUT / "again" / "SITE_graph.json").read_text(encoding="utf-8"))
        for g in (a, b):
            g["meta"].pop("path", None)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
