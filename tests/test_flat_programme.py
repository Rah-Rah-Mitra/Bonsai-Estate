"""Flat templates (config/flats): household shelter doors open outwards without clashing, service yards on the outside.

Proposed for tests/ (review findings F5, F6, F7, F10). Uses the estate catalogue, the harness cases of
estate/flats/check.py and the door-swing geometry of build/fix_flats/swing_check.py (to move into check.py).
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estate.env import bootstrap  # noqa: E402

bootstrap()

import tomllib  # noqa: E402

from shapely.geometry import box  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from estate.blocks.builder import derive  # noqa: E402
from estate.flats.check import check_template, harness, harness_cases, rules  # noqa: E402
from estate.flats.template import Template, catalogue  # noqa: E402
from estate.flats.check import swing_issues  # noqa: E402

YARD_TYPES = ("3R", "4R", "5R", "3G", "EA")


def estate_templates():
    return {k: t for k, t in catalogue().items() if t.estate}


class TestShelterDoors(unittest.TestCase):
    def test_shelter_doors_open_outwards(self):
        for tid, t in estate_templates().items():
            doors = [d for d in t.doors if d.get("kind") == "shelter"]
            self.assertEqual(len(doors), 1, tid)
            d = doors[0]
            self.assertIn("HS", d["between"], tid)
            self.assertNotEqual(d.get("into", d["between"][0]), "HS", f"{tid}: the shelter door swings into the shelter")

    def test_no_swing_clashes_in_any_case(self):
        for tid, t in estate_templates().items():
            for r in check_template(t):
                if r.get("dp") is None:
                    continue
                self.assertEqual(swing_issues(r["dp"], "T"), [], f"{tid} {r['case']} {r['variant']}")

    def test_clash_is_detected(self):
        """The shelter door of 3R-PT-A swung out into the 1.4 m foyer (instead of the living) hits the main door."""
        src = (ROOT / "config" / "flats" / "3R-PT-A.toml").read_text(encoding="utf-8")
        good = 'between = ["HS", "LD"]\nat = [4.2, 2.0]\nwidth = 0.8\nkind = "shelter"\ninto = "LD"\nhinge = "hi"\n'
        self.assertIn(good, src)
        bad = src.replace(good, 'between = ["HS", "F"]\nat = [4.8, 1.0]\nwidth = 0.8\nkind = "shelter"\ninto = "F"\n')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "3R-PT-A.toml"
            p.write_text(bad, encoding="utf-8")
            t = Template.load(p)
            lf = t.realize({"mirror": False, "kitchen": "closed", "outdoor": "ac_ledge"})
            issues = swing_issues(derive(harness(lf, "PT", "N"), "typical"), "T")
        self.assertIn("door-swing", {c for c, _, _ in issues})


class TestServiceYards(unittest.TestCase):
    def test_rules_require_a_yard(self):
        req = rules()["flat"]["required"]
        for ft in YARD_TYPES:
            self.assertIn("SY", req[ft], ft)
        self.assertNotIn("SY", req["2RF"])

    def test_yards_open_to_the_outside(self):
        R = rules()["room"]
        for tid, t in estate_templates().items():
            if t.flat_type not in YARD_TYPES:
                continue
            self.assertIn("SY", t.rooms, tid)
            ops = [w for w in t.windows if w["room"] == "SY" and w.get("kind") == "opening"]
            self.assertEqual(len(ops), 1, f"{tid}: one 'opening' window in the yard")
            # never on the access side (y = 0, corridor / lobby)
            self.assertGreater(ops[0]["at"][1], 0.0, f"{tid}: yard opening on the corridor / lobby side")
            k = unary_union([box(*r) for r in t.rooms["K"]])
            sy = unary_union([box(*r) for r in t.rooms["SY"]])
            self.assertGreater(k.intersection(sy).length, 0.7, f"{tid}: yard not next to the kitchen")
            for r in check_template(t):
                if r.get("dp") is None:
                    continue
                net = r["dp"].spaces.get("UT:SY")
                self.assertIsNotNone(net, tid)
                self.assertGreaterEqual(net.area, R["min_area"]["SY"] - 1e-6, f"{tid} {r['case']}")
                if set(t.fits) & {"SL", "LC"}:     # corridor / L-block units (point-block yards predate this rule)
                    self.assertFalse(net.buffer(-0.59, join_style="mitre").is_empty, f"{tid}: yard narrower than 1.2 m")
                codes = {i.code for i in r["issues"] if i.level == "error"}
                self.assertFalse(codes & {"yard-on-corridor", "missing-room", "kitchen-vent"}, f"{tid} {r['case']}")


class TestFourRoomVariety(unittest.TestCase):
    def test_l_block_four_room_is_its_own_layout(self):
        """4R-L must not share most of its plan with 4R-S-A (same room code on the same cells, either handing)."""
        cat = catalogue()
        a, b = cat["4R-L"], cat["4R-S-A"]

        def rooms(t, mirror):
            out = {}
            for c, rects in t.rooms.items():
                g = unary_union([box(*r) for r in rects])
                if mirror:
                    from shapely.affinity import affine_transform
                    g = affine_transform(g, [-1, 0, 0, 1, t.width, 0])
                out[c] = g
            return out
        ra = rooms(a, False)
        for m in (False, True):
            rb = rooms(b, m)
            shared = sum(ra[c].intersection(rb[c]).area for c in ra if c in rb and c not in ("BAL",))
            self.assertLess(shared / (a.width * a.depth), 0.5, f"mirror={m}: {shared:.1f} m2 shared")


def estate_templates():
    return {k: t for k, t in catalogue().items() if t.estate}


class TestMasterBedrooms(unittest.TestCase):
    def test_master_bedroom_has_a_normal_window_off_the_access_side(self):
        """HDB-N2: every master bedroom gets a window at a normal sill (<= 1.0 m) that is not on the access side
        (y = 0, corridor or lift lobby), so it has an outlook and is not lit only by a high corridor window."""
        for tid, t in estate_templates().items():
            if "MB" not in t.rooms:          # 2-room Flexi: a single bedroom (BED)
                continue
            wins = [w for w in t.windows if w["room"] == "MB" and w.get("kind", "window") == "window"]
            ok = [w for w in wins if float(w.get("sill", 1.0)) <= 1.0 + 1e-9 and w["at"][1] > 1e-6]
            self.assertTrue(ok, f"{tid}: master bedroom has no normal window off the access side")

    def test_l_block_four_room_master_suite_at_the_rear(self):
        t = catalogue()["4R-L"]
        mb = unary_union([box(*r) for r in t.rooms["MB"]])
        mba = unary_union([box(*r) for r in t.rooms["MBa"]])
        self.assertAlmostEqual(mb.bounds[3], t.depth, 6, "MB does not reach the rear facade")
        self.assertGreater(mb.bounds[1], 0.0, "MB still on the corridor")
        self.assertGreater(mb.intersection(mba).length, 0.7, "en-suite not next to the master bedroom")
        corner = [c for c, rects in t.rooms.items() for r in rects if r[0] == 0.0 and r[1] == 0.0]
        self.assertEqual(corner, ["B2"], "a smaller bedroom takes the corridor corner")
        for r in check_template(t):
            if r.get("dp") is None:
                continue
            area = {c: r["dp"].spaces[f"UT:{c}"].area for c in ("MB", "B2", "B3")}
            self.assertGreater(area["MB"], max(area["B2"], area["B3"]), f"{r['case']} {r['variant']}: MB not the largest")


class TestThreeGenVariety(unittest.TestCase):
    def test_corner_templates_are_different_plans_on_the_same_envelope(self):
        """F11: the L-block corner has two 3Gen plans (topology, not just dimensions) on the same 12 x 10 m
        envelope, so either can take the corner without changing the L-block geometry."""
        cat = catalogue()
        corner = {k: t for k, t in cat.items() if t.flat_type == "3G" and "LC" in t.fits and t.estate}
        self.assertGreaterEqual(len(corner), 2)
        ref = cat["3G-C"]
        topo = {}
        for tid, t in corner.items():
            self.assertEqual((t.width, t.depth), (ref.width, ref.depth), tid)
            self.assertFalse(t.variants.get("mirror", False), f"{tid}: the corner unit is handed by the block")
            topo[tid] = {t.realize({"mirror": False, "kitchen": k, "outdoor": o}).topology()
                         for k in t.variant_space()["kitchen"] for o in t.variant_space()["outdoor"]}
        ids = sorted(topo)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                self.assertFalse(topo[a] & topo[b], f"{a} and {b} share a plan")

    def test_grandparents_suite_and_kitchen_move_in_3g_c2(self):
        t = catalogue()["3G-C2"]
        b4 = unary_union([box(*r) for r in t.rooms["B4"]])
        self.assertAlmostEqual(b4.bounds[3], t.depth, 6, "B4 not at the rear")
        self.assertTrue(any(w["room"] == "B4" and abs(w["at"][1] - t.depth) < 1e-6 for w in t.windows))
        k = unary_union([box(*r) for r in t.rooms["K"]])
        self.assertGreater(k.bounds[1], 0.0, "kitchen on the corridor (as in 3G-C)")
        hs = next(d for d in t.doors if d.get("kind") == "shelter")
        self.assertNotEqual(hs["into"], "HS")

    def test_l_blocks_use_both_corner_plans(self):
        cfg = tomllib.loads((ROOT / "config" / "estate.toml").read_text(encoding="utf-8"))
        corners = {b["blk"]: b["corner"] for b in cfg["block"] if b.get("typology") == "LB"}
        self.assertEqual(len(set(corners.values())), len(corners), corners)

    def test_plan_has_two_3gen_plans(self):
        """The plan-level variety check (estate/commands.py cmd_plan) counts 3Gen plans by topology."""
        from estate import masterplan
        mp = masterplan.resolve()
        topo = {fi.variant["topology"] for s in mp["sites"] if s.kind == "block" and s.plan is not None
                for fi in s.plan.typical.flats.values() if fi.flat_type == "3G"}
        self.assertGreaterEqual(len(topo), rules()["variety"]["min_per_type_small"])


if __name__ == "__main__":
    unittest.main()
