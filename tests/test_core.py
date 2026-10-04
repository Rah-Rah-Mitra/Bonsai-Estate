"""Core regression tests: wall derivation, templates, the PT4 typology and deterministic IFC output."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estate.env import bootstrap  # noqa: E402

bootstrap()

from shapely.geometry import box  # noqa: E402

from estate.blocks.builder import derive, derive_plan, plan_errors  # noqa: E402
from estate.blocks.plate import CORRIDOR, ROOM, Face, Plate  # noqa: E402
from estate.flats.template import Template  # noqa: E402
from estate.env import CONFIG  # noqa: E402
from estate.geom.arrangement import runs_from_faces  # noqa: E402

LEGACY_4R = CONFIG / "flats" / "4R-PT-A.toml"


def pt4_spec(t="4R-PT-A"):
    return dict(blk="900", storeys=6, stacks={s: {"t": t} for s in ("101", "103", "105", "107")}, void_deck=["rc_centre"])


class TestArrangement(unittest.TestCase):
    def test_runs_label_both_sides(self):
        faces = {"A": box(0, 0, 4, 3), "B": box(4, 0, 7, 3), "C": box(0, 3, 7, 5)}
        runs = runs_from_faces(faces)
        pairs = {frozenset((r.left, r.right)) for r in runs}
        self.assertIn(frozenset(("A", "B")), pairs)
        self.assertIn(frozenset(("A", "C")), pairs)
        self.assertIn(frozenset(("B", "C")), pairs)
        self.assertIn(frozenset(("A", "@out")), pairs)
        # the A|C and B|C boundaries are split at x=4 (T junction)
        ac = [r for r in runs if r.faces() == frozenset(("A", "C"))][0]
        self.assertAlmostEqual(ac.length, 4.0)

    def test_overlap_detected(self):
        with self.assertRaises(ValueError):
            runs_from_faces({"A": box(0, 0, 4, 3), "B": box(3, 0, 7, 3)})


class TestTemplate(unittest.TestCase):
    def test_legacy_template_areas(self):
        t = Template.load(LEGACY_4R)
        lf = t.realize(t.pick_variant(1))
        self.assertAlmostEqual(lf.width, 11.0)
        self.assertAlmostEqual(lf.envelope.area, 11.0 * 8.6, places=4)
        self.assertEqual(len(lf.doors), 9)

    def test_signature_stable_and_mirror_canonical(self):
        t = Template.load(LEGACY_4R)
        a = t.realize({"mirror": False, "kitchen": "closed", "outdoor": "ac_ledge"})
        self.assertEqual(a.signature(), t.realize({"mirror": False, "kitchen": "closed", "outdoor": "ac_ledge"}).signature())
        from estate.flats.template import mirror_x
        self.assertEqual(a.signature(canonical=True), mirror_x(a).signature(canonical=True))


class TestPointBlock(unittest.TestCase):
    def setUp(self):
        from estate.flats.template import catalogue
        self.cat = catalogue()

    def test_plan_has_no_errors(self):
        from estate.blocks.point import plan_point
        dp = derive_plan(plan_point(pt4_spec(), 0, self.cat))
        self.assertEqual(plan_errors(dp), [])
        self.assertEqual(len(dp["typical"].plate.flats), 4)

    def test_ifc_deterministic(self):
        from estate.blocks.builder import build_building
        from estate.blocks.point import plan_point
        from estate.ifc.writer import IfcWriter
        shas = []
        with tempfile.TemporaryDirectory() as d:
            for i in range(2):
                plan = plan_point(pt4_spec(), 0, self.cat)
                W = IfcWriter(guid_key="test/900", typed_openings=True)
                build_building(W, plan)
                p = Path(d) / f"run{i}" / "b.ifc"      # same file name: it is recorded in the header
                p.parent.mkdir()
                W.write(p)
                W.close()
                shas.append(hashlib.sha256(p.read_bytes()).hexdigest())
        self.assertEqual(shas[0], shas[1])


class TestOpenings(unittest.TestCase):
    def test_door_must_fit(self):
        p = Plate()
        p.add(Face("A", box(0, 0, 1.0, 3), ROOM, flat="1", room="CB", meta={"category": "wet"}))
        p.add(Face("C", box(0, -2, 1.0, 0), CORRIDOR))
        from estate.blocks.plate import DoorSpec
        p.doors.append(DoorSpec("A", "C", (0.5, 0.0), 0.95, "bath", "A", (0.0, 0.0), "too wide"))
        dp = derive(p, "typical")
        self.assertTrue(any("too wide" in e for e in dp.errors))


if __name__ == "__main__":
    unittest.main()


class TestLegacyBaseline(unittest.TestCase):
    """estate.cmd legacy must reproduce hdb_block.ifc's element counts exactly (the regression baseline)."""

    def test_legacy_counts(self):
        import ifcopenshell
        from estate.blocks.legacy import build_legacy
        from estate.cli import BASELINE_CLASSES, count_classes
        from estate.env import ROOT
        with tempfile.TemporaryDirectory() as d:
            W, _, _ = build_legacy()
            p = Path(d) / "legacy.ifc"
            W.write(p)
            W.close()
            new = count_classes(ifcopenshell.open(str(p)))
        base = count_classes(ifcopenshell.open(str(ROOT / "hdb_block.ifc")))
        self.assertEqual(new, base)
        self.assertEqual(len(BASELINE_CLASSES), len(base))
