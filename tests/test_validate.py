"""Validator tests: vertical-ray barrier heights, stair re-measurement on a generated stair tower, the IDS, flat-type
aliases, the IFC-level programme checks on the legacy Blk 123 (exactly its known layout defects), the ifcqa /
programme / geometry checks on the representative PT4 block (build/t_pt4.ifc) and a subset of the mutation suite.

Run: "<blender python>" -I -B tests/test_validate.py -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estate.env import bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402
from shapely.geometry import box  # noqa: E402

from estate.env import BUILD  # noqa: E402
from estate.validate import geometry as geom  # noqa: E402
from estate.validate import ifcqa  # noqa: E402
from estate.validate import programme  # noqa: E402

T_PT4 = BUILD / "t_pt4.ifc"
_BLK = BUILD.parent / "model" / "BLK_501" / "BLK_501.ifc"
if _BLK.exists():    # prefer a real estate point block (georeferenced, shared identity)
    T_PT4 = _BLK
# keep the real hash descriptor in case a test leaves an IfcWriter's deterministic context open
import ifcopenshell  # noqa: E402

_ENTITY_HASH = ifcopenshell.ifcopenshell_wrapper.entity_instance.__dict__["__hash__"]


def box_el(name, x0, y0, z0, x1, y1, z1, cls="IfcWall"):
    """A synthetic closed box element (12 triangles) for ray tests."""
    V = np.array([(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], float)
    F = [(0, 2, 1), (1, 2, 3), (4, 5, 6), (5, 7, 6), (0, 1, 4), (1, 5, 4), (2, 6, 3), (3, 6, 7),
         (0, 4, 2), (2, 4, 6), (1, 3, 5), (3, 7, 5)]
    return geom.El(name, None, {"cls": cls, "name": name, "verts": V, "faces": np.array(F)}, None)


class TestRays(unittest.TestCase):
    def test_barrier_heights(self):
        els = [box_el("below sill", 0, 0, 0.0, 2, 0.2, 1.0), box_el("above window", 0, 0, 2.2, 2, 0.2, 2.6),
               box_el("over door", 3, 0, 2.1, 4, 0.2, 2.6), box_el("full", 5, 0, 0.0, 6, 0.2, 2.6),
               box_el("wall below", 5, 0, -2.8, 6, 0.2, -0.2), box_el("parapet", 7, 0, 0.0, 8, 0.2, 1.1),
               box_el("rail on parapet", 9, 0, 0.0, 10, 0.2, 0.6), box_el("rail top", 9, 0, 0.6, 10, 0.2, 1.05)]
        rays = geom.Rays(els)
        pts = [(1.0, 0.1), (3.5, 0.1), (5.5, 0.1), (7.5, 0.1), (9.5, 0.1), (11.0, 0.1)]
        h = geom.barrier_heights(rays, pts, np.zeros(len(pts)))
        np.testing.assert_allclose(h, [1.0, 0.0, 2.6, 1.1, 1.05, 0.0], atol=1e-6)

    def test_footprint_and_box(self):
        e = box_el("w", 0, 0, 0, 3, 0.2, 2.6)
        self.assertAlmostEqual(e.footprint.area, 0.6, places=6)
        self.assertAlmostEqual(geom._thickness(e.footprint), 0.2, places=6)
        b = geom.solid_box(e)
        self.assertEqual(len(b.T), 4)


class TestStairTower(unittest.TestCase):
    """A two-storey stair tower authored with the estate stair code, re-measured from its tessellation."""

    @classmethod
    def setUpClass(cls):
        import ifcopenshell
        from estate.export import meshcache
        from estate.geom.walls import ring_walls
        from estate.ifc.stairs import flight_run, stair
        from estate.ifc.writer import IfcWriter
        W = IfcWriter(guid_key="test/validate-stair")
        try:
            b = W.building("Stair tower")
            st = W.storeys(b, ["L1", "L2", "RF"], [0.0, 2.8, 5.6])
            inner = box(0.1, 0.1, 2.7, 5.5)
            W.slab(box(-0.1, -0.1, 2.9, 5.7), 0.0, "L1 slab", st[0], "BASESLAB")
            for s, z in ((st[1], 2.8), (st[2], 5.6)):
                W.slab(box(-3.0, -0.1, 2.9, 8.0).difference(inner), z, f"{s.Name} slab", s)
            for s, z in ((st[0], 0.0), (st[1], 2.8)):          # enclosure walls (the entry wall is left out)
                for w in ring_walls(box(-0.1, -0.1, 2.9, 5.7), 0.2, z, 2.6, f"{s.Name} core", "CORE", False, s,
                                    skip=lambda p, q: p[1] > 5.6 and q[1] > 5.6):
                    W.build_wall(w)
            stair(W, "L1 stair", (0.1, 2.7, 5.5, 0.1), 0.0, 2.8, st[0], flight_run(2.8), entry="N")
            stair(W, "L2 stair", (0.1, 2.7, 5.5, 0.1), 2.8, 5.6, st[1], None, entry="N")
            W.flush()
            cls.tmp = tempfile.TemporaryDirectory()
            path = Path(cls.tmp.name) / "stair.ifc"
            W.write(path)
        finally:
            W.close()
            ifcopenshell.ifcopenshell_wrapper.entity_instance.__hash__ = _ENTITY_HASH
        cls.f = ifcopenshell.open(str(path))
        cls.meshes = meshcache.tessellate(cls.f, num_threads=2)
        cls.res = {r["check"]: r for r in geom.check(cls.f, cls.meshes)}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_stair_rules_hold(self):
        for c in ("stair_riser", "stair_riser_variation", "stair_going", "stair_risers_per_flight", "stair_width",
                  "stair_headroom", "stair_landings", "stair_rise", "stair_open_sides"):
            self.assertEqual(self.res[c]["count"], 0, (c, self.res[c]["examples"]))
        self.assertAlmostEqual(self.res["stair_riser"]["value"]["max"], 0.175, places=4)
        self.assertAlmostEqual(self.res["stair_going"]["value"]["min"], 0.28, places=4)
        self.assertGreaterEqual(self.res["stair_width"]["value"]["min"], 1.0)
        self.assertGreaterEqual(self.res["stair_headroom"]["value"]["min"], 2.0)

    def test_balustrade_measured_above_pitch_line(self):
        # rules.BALUSTRADE_H: the balustrade sits 1.0 m above the pitch line (Approved Document barrier height)
        self.assertAlmostEqual(self.res["stair_balustrade_height"]["value"]["min"], 1.0, places=3)
        self.assertEqual(self.res["stair_balustrade_height"]["count"], 0)

    def test_raised_riser_is_caught(self):
        import ifcopenshell
        from estate.export import meshcache
        f = ifcopenshell.file.from_string(self.f.to_string())
        fl = sorted(f.by_type("IfcStairFlight"), key=lambda e: e.Name)[0]
        r = fl.RiserHeight
        for p in fl.Representation.Representations[0].Items[0].SweptArea.OuterCurve.Points:
            if abs(p.Coordinates[1] - 2 * r) < 1e-6:
                p.Coordinates = (p.Coordinates[0], p.Coordinates[1] + 0.03)
        meshes = dict(self.meshes)
        meshes.update(meshcache.tessellate(f, num_threads=1, include=[fl]))
        res = {x["check"]: x for x in geom.check(f, meshes)}
        self.assertGreater(res["stair_riser"]["count"], 0)
        self.assertGreater(res["stair_riser_variation"]["count"], 0)


class TestIds(unittest.TestCase):
    def test_ids_is_schema_valid(self):
        from ifctester import ids
        spec = ids.open(str(ifcqa.IDS_PATH), validate=True)
        self.assertGreaterEqual(len(spec.specifications), 8)

    def test_room_codes_match_templates(self):
        import xml.etree.ElementTree as ET
        from estate.flats.template import ROOM_CODES
        ids_ns, xs_ns = "{http://standards.buildingsmart.org/IDS}", "{http://www.w3.org/2001/XMLSchema}"
        spec = next(s for s in ET.parse(ifcqa.IDS_PATH).getroot().iter(ids_ns + "specification")
                    if s.get("identifier") == "SN5-ROOM-CODE")
        codes = {e.get("value") for e in spec.iter(xs_ns + "enumeration")}
        self.assertEqual(codes, set(ROOM_CODES) | {"STALL", "SHOP"})


class TestFlatTypes(unittest.TestCase):
    def test_aliases(self):
        for label, t in [("4R", "4R"), ("4-Room", "4R"), ("4-room flat", "4R"), ("3-Room", "3R"), ("5 room", "5R"),
                         ("2-Room Flexi", "2RF"), ("3Gen", "3G"), ("Executive", "EA"), ("Executive apartment", "EA")]:
            self.assertEqual(ifcqa.canonical_flat_type(label), t, label)
        for label in (None, "", "Four-room", "6R", 4, "Shop"):
            self.assertIsNone(ifcqa.canonical_flat_type(label), label)


class TestLegacyProgramme(unittest.TestCase):
    """The original Blk 123 rebuilt by estate/blocks/legacy.py: the IFC-level programme checks must flag exactly its
    known layout defects (1.8 m2 shelters, 9.9 % glazing in bedroom 3, bathrooms without a vent window: the master
    bath of the flats whose vent window faces the stair core, and the internal common bath) and nothing else."""

    @classmethod
    def setUpClass(cls):
        import ifcopenshell
        from estate.blocks.legacy import build_legacy
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "legacy.ifc"
        W = None
        try:
            W, _, _ = build_legacy()
            W.write(path)
        finally:
            if W is not None:
                W.close()
            ifcopenshell.ifcopenshell_wrapper.entity_instance.__hash__ = _ENTITY_HASH
        cls.f = ifcopenshell.open(str(path))
        cls.X = programme.Index(cls.f)
        cls.res = {r["check"]: r for r in programme.check(cls.f, index=cls.X)}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exactly_the_known_defects(self):
        failing = {c: r["count"] for c, r in self.res.items() if r["count"]}
        self.assertEqual(failing, {"shelter_area": 44, "glazing": 44, "bath_vent": 66})
        self.assertEqual(self.res["flat_type"]["value"], 44)          # '4-Room' read as 4R
        self.assertIn("1.82 m2 < 2.8 m2", self.res["shelter_area"]["examples"][0])
        self.assertTrue(all(" B3: " in x and "9.9%" in x for x in self.res["glazing"]["examples"]))
        self.assertAlmostEqual(self.res["glazing"]["value"]["min_ratio"], 0.0989, places=4)

    def test_bath_vents_by_room(self):
        from collections import Counter
        missing = Counter()
        for fl in self.X.flats:
            for code in ("MBa", "CB"):
                s = fl.room(code)
                if sum(o.width * o.height for o in self.X.served.get(s, ()) if o.kind == "window") < 0.2:
                    missing[code] += 1
        self.assertEqual(missing, {"CB": 44, "MBa": 22})

    def test_doors_found_without_navigation_spaces(self):
        # the legacy doors carry no FromSpace / ToSpace: every flat door is placed by probing across its wall
        self.assertFalse(any(d.declared for d in self.X.doors))
        flat_doors = [d for d in self.X.doors if d.door_kind not in ("lift", "stair", "roof")]
        self.assertEqual(len(flat_doors), 44 * 9)
        self.assertTrue(all(all(isinstance(x, programme.Space) for x in d.sides) for d in flat_doors))

    def test_zone_ifa_reads_legacy_labels(self):
        qa = {r["check"]: r for r in ifcqa.check(self.f, schema=False, ids=False)}
        self.assertEqual(qa["zone_ifa"]["count"], 0, qa["zone_ifa"]["examples"])




@unittest.skipUnless(T_PT4.exists(), "build/t_pt4.ifc not built (python build/t_pt4.py)")
class TestPT4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ifcopenshell
        from estate.export import meshcache
        cls.f, cls.meshes = meshcache.load(T_PT4)
        cls.qa = {r["check"]: r for r in ifcqa.check(ifcopenshell.open(str(T_PT4)), express_rules=False)}
        cls.geo = {r["check"]: r for r in geom.check(cls.f, cls.meshes)}

    def test_ifcqa_structure_clean(self):
        for c in ("schema", "unique_guids", "containment", "opening_voids", "filling", "openings_per_wall",
                  "space_in_zone", "required_psets", "zone_ifa", "element_budget", "placements_right_handed"):
            self.assertEqual(self.qa[c]["count"], 0, (c, self.qa[c]["examples"]))
        ids_rows = [r for c, r in self.qa.items() if c.startswith("ids:")]
        self.assertTrue(ids_rows and all(r["count"] == 0 for r in ids_rows))
        self.assertEqual(self.qa["georef"]["count"], 0)       # every estate file carries the shared EPSG:3414 map conversion

    def test_geometry_clean_except_known(self):
        for c in ("stair_riser", "stair_going", "stair_width", "stair_headroom", "stair_rise", "wall_overlap",
                  "opening_clash", "lift_shaft_slab", "space_overlap", "wet_stack", "wet_over_dry", "barrier_heights"):
            self.assertEqual(self.geo[c]["count"], 0, (c, self.geo[c]["examples"]))
        self.assertAlmostEqual(self.geo["wet_stack"]["value"]["min_iou"], 1.0, places=4)

    def test_programme_clean(self):
        import ifcopenshell
        res = programme.check(ifcopenshell.open(str(T_PT4)))
        self.assertTrue(res)
        for r in res:
            self.assertEqual(r["count"], 0, (r["check"], r["examples"]))
        self.assertGreater(next(r for r in res if r["check"] == "flat_type")["value"], 0)

    def test_programme_reads_tessellated_space(self):
        # a room remodelled in Bonsai as a mesh (no extruded profile) is measured from its tessellation
        import ifcopenshell
        import ifcopenshell.api.geometry as geometry
        import ifcopenshell.util.element as uel
        import ifcopenshell.util.placement as upl
        from estate.export import meshcache
        f = ifcopenshell.open(str(T_PT4))
        sp = next(s for s in sorted(f.by_type("IfcSpace"), key=lambda s: s.Name)
                  if (uel.get_pset(s, "SampleCity_Room") or {}).get("RoomCode") == "B2")
        area = programme.solid_footprint(sp)[0].area
        mesh = meshcache.tessellate(f, num_threads=1, include=[sp])[sp.GlobalId]
        V = np.asarray(mesh["verts"], float).reshape(-1, 3)
        local = (np.c_[V, np.ones(len(V))] @ np.linalg.inv(upl.get_local_placement(sp.ObjectPlacement)).T)[:, :3]
        body = next(c for c in f.by_type("IfcGeometricRepresentationSubContext") if c.ContextIdentifier == "Body")
        rep = geometry.add_mesh_representation(f, context=body, vertices=[local.tolist()],
                                               faces=[np.asarray(mesh["faces"]).reshape(-1, 3).tolist()])
        geometry.unassign_representation(f, product=sp, representation=sp.Representation.Representations[0])
        geometry.assign_representation(f, product=sp, representation=rep)
        self.assertIsNone(programme.solid_footprint(sp))
        X = programme.Index(f)
        self.assertAlmostEqual(X.by_name[sp.Name].poly.area, area, places=4)
        for r in programme.check(f, index=X):
            self.assertEqual(r["count"], 0, (r["check"], r["examples"]))

    def test_common_spaces_outside_zones_not_flagged(self):
        # lobbies, corridors, void deck, stairs and refuse rooms belong to no flat; with or without a common-property
        # zone (no SampleCity_Flat, or one without FlatType / UnitNumber) no check may flag them
        import ifcopenshell
        import ifcopenshell.api.group as group
        import ifcopenshell.api.pset as pset
        import ifcopenshell.api.root as root
        f = ifcopenshell.open(str(T_PT4))
        common = [s for s in f.by_type("IfcSpace") if ifcqa._zone_of(s) is None]
        self.assertTrue(common)
        for variant in ("none", "zone", "zone_with_flat_pset"):
            if variant != "none":
                z = root.create_entity(f, ifc_class="IfcZone", name=f"Common property {variant}")
                z.ObjectType = "COMMON"
                group.assign_group(f, products=common, group=z)
                if variant == "zone_with_flat_pset":
                    pset.edit_pset(f, pset=pset.add_pset(f, product=z, name="SampleCity_Flat"), properties={"Block": "501"})
            qa = {r["check"]: r for r in ifcqa.check(f, schema=False, ids=False)}
            for c in ("space_in_zone", "required_psets", "zone_ifa"):
                self.assertEqual(qa[c]["count"], 0, (variant, c, qa[c]["examples"]))
            for r in programme.check(f):
                self.assertEqual(r["count"], 0, (variant, r["check"], r["examples"]))

    def test_validate_one_has_programme_family(self):
        from estate.validate import cli
        rec = cli.validate_one(T_PT4, schema=False, ids=False, geometry=False)
        self.assertIn("programme", rec)
        names = {r["check"] for r in rec["programme"]["results"]}
        self.assertTrue({"required_rooms", "shelter_area", "glazing", "bath_vent", "main_door"} <= names)
        self.assertEqual(rec["summary"]["checks"], len(rec["ifcqa"]["results"]) + len(rec["programme"]["results"]))
        self.assertEqual(rec["summary"]["failed_errors"], 0)

    def test_mutation_subset_caught(self):
        from estate.validate import mutations
        names = ["duplicate_guid", "mirror_mapped_item", "zone_ifa_out_of_band", "raise_riser", "slab_in_lift_shaft",
                 "door_clash", "georef_eastings", "shelter_out_of_zone", "remove_bath_vent", "bad_room_code",
                 "stale_door_spaces"]
        r = mutations.run(T_PT4, only=names)
        self.assertEqual(r["total"], len(names))
        self.assertEqual(r["detected"], r["total"], [x for x in r["rows"] if not x["detected"]])

    def test_federation_of_two_buildings(self):
        from estate.validate import mutations
        a, b = mutations.federation_pair(T_PT4)
        res = {x["check"]: x for x in ifcqa.check_federation([a, b])["results"]}
        self.assertEqual(res["federation_guids"]["count"], 0)
        self.assertEqual(res["federation_identity"]["count"], 0)
        self.assertEqual(res["federation_georef"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
