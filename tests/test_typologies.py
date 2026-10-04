"""PT4 point block, SL corridor slab and LB L-block typologies on the fixture templates (tests/fixtures/flats).

Run: "<blender python>" -I -B tests/test_typologies.py [-v]
"""
from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from estate.env import bootstrap  # noqa: E402

bootstrap()

from shapely.geometry import LineString, Point, box  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from estate import config  # noqa: E402
from estate.blocks import plate_ops  # noqa: E402
from estate.blocks.builder import derive, derive_plan, plan_errors  # noqa: E402
from estate.blocks.lblock import LIGHTWELL, plan_lblock  # noqa: E402
from estate.blocks.plate import AMENITY, CORRIDOR, DECK, LIFT, LOBBY, REFUSE, VOID  # noqa: E402
from estate.blocks.point import (AMENITY_USES, C, CORE_GAP, DOOR_APPROACH, DOOR_CLEAR,  # noqa: E402
                                 LETTERBOX_D, LIFT_W, LOBBY_HALF, QUADRANT_ORDER, REF_W, assign_quadrants,
                                 letterbox_rules, plan_point)
from estate.blocks.slab import CORR_D, LOBBY_D, MODULE_W, plan_slab  # noqa: E402
from estate.flats.template import Template  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "flats"
PT_BLOCKS = ("501", "502", "503", "504", "505", "506")
SL_BLOCKS = ("507", "508", "509", "512")
LB_BLOCKS = ("510", "511")
PLANNERS = {"PT4": plan_point, "SL": plan_slab, "LB": plan_lblock}


def fixture_catalogue() -> dict:
    # loaded directly (not through flats.template.catalogue, which caches one folder process-wide)
    return {t.id: t for t in (Template.load(p) for p in sorted(FIXTURES.glob("*.toml")))}


def door_line(d):
    return LineString([d.hinge, (2 * d.at[0] - d.hinge[0], 2 * d.at[1] - d.hinge[1])])


def walk_path(plan):
    return unary_union([f.poly for f in plan.typical.faces.values() if f.kind in (CORRIDOR, LOBBY)])


def quadrant_of(plan, face):
    """The point-block quadrant (stack) whose flat envelope an amenity room sits under."""
    return next(st for st, fi in plan.typical.flats.items() if fi.envelope.within(face.poly.buffer(1e-6)))


class TypologyCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = fixture_catalogue()
        cls.cfg = config.load()
        cls.seed = int(cls.cfg["estate"]["seed"])
        cls.plans, cls.derived = {}, {}

    def spec(self, blk, **over):
        return dict(config.block(self.cfg, blk), street=self.cfg["estate"]["street"], **over)

    def plan(self, blk):
        if blk not in self.plans:
            sp = self.spec(blk)
            self.plans[blk] = PLANNERS[sp["typology"]](sp, self.seed, self.cat)
            self.derived[blk] = derive_plan(self.plans[blk])
        return self.plans[blk], self.derived[blk]

    def fixture_checks(self, blk):
        """Void deck fixtures: a bin centre under every refuse chute room, letterbox banks for every flat
        (letterbox_checks); every glazed window has an outlook."""
        plan, dp = self.plan(blk)
        uses = plan.void_deck_uses
        ref_t = sorted(f.id for f in plan.typical.faces.values() if f.kind == REFUSE)
        ref_g = sorted(f.id for f in plan.ground.faces.values() if f.kind == REFUSE)
        self.assertTrue(ref_t, f"{blk}: no refuse chute room")
        self.assertEqual(ref_t, ref_g, f"{blk}: a bin centre under every refuse chute room")
        for fid in ref_g:
            self.assertTrue(plan.ground.faces[fid].name.startswith("Bin centre"))
            self.assertEqual(plan.ground.faces[fid].meta["use"], "bin")
        self.assertTrue(all(plan.typical.faces[f].name.startswith("Refuse chute room") for f in ref_t))
        self.assertFalse(any(f.kind == REFUSE for f in plan.roof.faces.values()))
        for pl, access in ((plan.typical, None), (plan.ground, "VOID")):
            for fid in ref_t:
                doors = [d for d in pl.doors if fid in (d.a, d.b)]
                self.assertEqual(len(doors), 1, f"{blk} {fid}")
                self.assertEqual(doors[0].kind, "service")
                self.assertEqual(doors[0].into, fid)
                self.assertGreaterEqual(doors[0].width, 0.8)
                if access:
                    self.assertIn(access, (doors[0].a, doors[0].b))
                else:
                    self.assertIn(pl.kind(doors[0].b if doors[0].a == fid else doors[0].a), ("corridor", "lobby"))
        furn = plan.meta["furniture"]
        banks = [it for it in furn if it["object_type"] == "Letterbox bank"]
        self.assertEqual(bool(banks), "letterbox" in uses, f"{blk}: letterbox banks iff configured")
        if banks:
            self.letterbox_checks(plan, dp, blk)
        self.assertEqual(len({it["name"] for it in furn}), len(furn))
        self.assertEqual(plate_ops.window_outlook(dp["typical"]), [], f"{blk}: windows facing a wall")
        return plan, dp, banks

    def letterbox_checks(self, plan, dp, blk, on_path=False):
        """HDB-N3: a box for every flat, in banks of whole box columns standing against a core / amenity wall or
        the void deck edge, on the open void deck, clear of walls, columns, doors and (unless on_path) the walk
        path, with clear_front free in front and near a lift lobby; every lift lobby has a bank."""
        R = letterbox_rules()
        banks = [it for it in plan.meta["furniture"] if it["object_type"] == "Letterbox bank"]
        flats = len(plan.typical.flats) * (plan.storeys - 1)
        summary = plan.meta["letterboxes"]
        self.assertEqual((summary["flats"], summary["banks"], summary["per_bank"]), (flats, len(banks), 48), blk)
        self.assertEqual(sum(it["boxes"] for it in banks), summary["boxes"], blk)
        self.assertGreaterEqual(summary["boxes"], flats, f"{blk}: {summary['boxes']} letterboxes for {flats} flats")
        void = plan.ground.faces["VOID"].poly
        walls = dp["ground"].wall_union
        cols = unary_union([box(x - 0.3, y - 0.3, x + 0.3, y + 0.3)
                            for x, y in plate_ops.column_points(dp["typical"], plan.ground)])
        lines = [door_line(d) for d in plan.ground.doors]
        approach = unary_union([g.buffer(DOOR_APPROACH, cap_style="flat") for g in lines])
        walk = walk_path(plan)
        polys = [it["poly"] for it in banks]
        for it in banks:
            p, nm = it["poly"], f"{blk} {it['name']}"
            x0, y0, x1, y1 = p.bounds
            length, depth = max(x1 - x0, y1 - y0), min(x1 - x0, y1 - y0)
            self.assertEqual(it["level"], "ground")
            self.assertAlmostEqual(depth, LETTERBOX_D, 6, nm)
            n_cols = round(length / R["col_w"])
            self.assertAlmostEqual(length, n_cols * R["col_w"], 6, f"{nm}: whole box columns")
            self.assertGreaterEqual(length, R["bank_min_length"] - 1e-6, nm)
            self.assertEqual(it["boxes"], n_cols * R["per_col"], nm)
            self.assertTrue(void.contains(p), f"{nm} not on the open void deck")
            self.assertAlmostEqual(p.intersection(walls).area, 0.0, 9, f"{nm} clashes with a wall")
            self.assertAlmostEqual(p.intersection(cols).area, 0.0, 9, f"{nm} clashes with a void deck column")
            self.assertLessEqual(p.distance(void.boundary), CORE_GAP + 1e-6, f"{nm} stands free on the void deck")
            for d, g in zip(plan.ground.doors, lines):
                self.assertGreaterEqual(p.distance(g), DOOR_CLEAR - 1e-6, f"{nm} '{d.name}'")
            self.assertAlmostEqual(p.intersection(approach).area, 0.0, 9, f"{nm} in front of a door")
            if not on_path:
                self.assertAlmostEqual(p.intersection(walk).area, 0.0, 9, f"{nm} on the walk path")
            d_lobby = min(p.distance(Point(q)) for q in plan.meta["lift_lobbies"])
            self.assertLessEqual(d_lobby, R["near_lobby"] + 1e-6, f"{nm}: {d_lobby:.1f} m from a lift lobby")
            # clear_front free in front: on the void deck, off walls, columns and the other banks
            k = R["clear_front"]
            if (x1 - x0) < (y1 - y0):
                fronts = [box(x0 - k, y0, x0, y1), box(x1, y0, x1 + k, y1)]
            else:
                fronts = [box(x0, y0 - k, x1, y0), box(x0, y1, x1, y1 + k)]
            rest = unary_union([q for q in polys if q is not p] + [walls, cols])
            self.assertTrue(any(void.buffer(1e-6).contains(f) and f.intersection(rest).area < 1e-9 for f in fronts),
                            f"{nm}: no {k} m clear in front")
        self.assertEqual(plate_ops.overlaps(polys), [], blk)
        for q in plan.meta["lift_lobbies"]:
            self.assertTrue(any(p.distance(Point(q)) <= R["near_lobby"] + 1e-6 for p in polys),
                            f"{blk}: no letterbox bank at the lift lobby {q}")
        return banks

    def common_checks(self, blk):
        plan, dp = self.plan(blk)
        self.assertEqual(plan_errors(dp), [], f"Blk {blk}")
        for lvl, d in dp.items():
            self.assertEqual(plate_ops.check_plate(d.plate), [], f"Blk {blk} {lvl}")
        self.assertEqual(plate_ops.access_issues(plan.typical), [], f"Blk {blk}")
        walk, worst = plate_ops.stair_walk(plan.typical)
        self.assertLessEqual(walk, 30.0, f"Blk {blk}: {walk:.1f} m to a stair door at {worst}")
        # stair enclosures: 2.8 x 5.6 with the 5.6 m side along the entry axis, on every level
        for s in plan.stairs:
            for pl in (plan.ground, plan.typical, plan.roof):
                x0, y0, x1, y1 = pl.faces[s.face].poly.bounds
                along, across = (y1 - y0, x1 - x0) if s.entry in "NS" else (x1 - x0, y1 - y0)
                self.assertAlmostEqual(along, 5.6, 6, f"{blk} {s.face}")
                self.assertAlmostEqual(across, 2.8, 6, f"{blk} {s.face}")
        # every core has two lifts and a refuse room; lift doors on all levels but the roof
        lifts = [f for f in plan.typical.faces.values() if f.kind == LIFT]
        refuse = [f for f in plan.typical.faces.values() if f.kind == REFUSE]
        self.assertEqual(len(lifts), 2 * len(refuse))
        self.assertEqual(sorted(plan.lifts), sorted(f.id for f in lifts))
        for lvl, pl in (("ground", plan.ground), ("typical", plan.typical), ("roof", plan.roof)):
            n = sum(1 for d in pl.doors if d.kind == "lift")
            self.assertEqual(n, 0 if lvl == "roof" else len(lifts), f"{blk} {lvl}")
        roof_doors = [d for d in plan.roof.doors if d.kind == "roof"]
        self.assertEqual(len(roof_doors), len(plan.stairs))
        # void deck / roof deck, amenities, tank
        self.assertEqual(sum(1 for f in plan.ground.faces.values() if f.kind == VOID), 1)
        deck = plan.roof.faces["DECK"]
        self.assertEqual(deck.kind, DECK)
        uses = [u for u in plan.void_deck_uses if u in AMENITY_USES]
        amen = [f for f in plan.ground.faces.values() if f.kind == AMENITY]
        self.assertEqual(sorted(f.meta["use"] for f in amen), sorted(uses))
        for _, poly, _ in plan.roof_items:
            self.assertTrue(deck.poly.contains(poly))
        fp = plan.footprint
        for p in plan.meta["entrances"]:
            self.assertGreater(fp.exterior.distance(box(p[0], p[1], p[0], p[1])), -1e-9)
            self.assertFalse(fp.buffer(-0.05).contains(box(*p, *p)), f"{blk} entrance {p} inside the footprint")
        if plan.typology != "PT4":
            # stacks numbered 101.. along the corridor
            self.assertEqual(plan.meta["stacks"], [f"1{i:02d}" for i in range(1, len(plan.typical.flats) + 1)])
            self.assertEqual(set(plan.meta["stacks"]), set(plan.typical.flats))
        self.fixture_checks(blk)
        return plan, dp


class PointTests(TypologyCase):
    def test_point_blocks(self):
        for blk in PT_BLOCKS:
            with self.subTest(blk=blk):
                plan, dp = self.common_checks(blk)
                t = plan.typical
                # two 2.2 m lift shafts and a 1.2 m refuse chute room in the 5.6 m lift strip
                xs = [t.faces[f].poly.bounds for f in ("LIFT1", "LIFT2", "REF1")]
                self.assertEqual([round(b[2] - b[0], 4) for b in xs], [LIFT_W, LIFT_W, REF_W])
                self.assertAlmostEqual(xs[0][0], -C, 6)
                self.assertAlmostEqual(xs[2][2], C, 6)
                self.assertTrue(all(abs(b[1] - LOBBY_HALF) < 1e-6 for b in xs))
                # the flats still start at the core strip (template x = 0 at x = +-C)
                for st, fi in t.flats.items():
                    x0, _, x1, _ = fi.envelope.bounds
                    self.assertAlmostEqual(x0 if st in ("103", "105") else x1, C if st in ("103", "105") else -C, 6)
                # the letterbox banks stay out of the lift lobby band (the walk path from both entrances)
                banks = [it for it in plan.meta["furniture"] if it["object_type"] == "Letterbox bank"]
                self.assertTrue(banks)
                lobby_band = box(-100, -LOBBY_HALF, 100, LOBBY_HALF)
                for it in banks:
                    self.assertAlmostEqual(it["poly"].intersection(lobby_band).area, 0.0, 9)
                ref_door = next(d for d in t.doors if "REF1" in (d.a, d.b))
                self.assertEqual(ref_door.width, 0.8)

    def test_void_deck_uses(self):
        sp = self.spec("501", void_deck=["rc_centre"])
        p = plan_point(sp, self.seed, self.cat)
        self.assertEqual(p.meta["furniture"], [])
        self.assertIsNone(p.meta["letterboxes"])
        self.assertIn("REF1", p.ground.faces)               # the chute always discharges into a bin centre
        with self.assertRaises(ValueError):
            plan_point(self.spec("501", void_deck=["letterbox", "swimming_pool"]), self.seed, self.cat)
        with self.assertRaises(ValueError):
            plan_slab(self.spec("507", void_deck=["letterbox", "bowling_alley"]), self.seed, self.cat)

    def test_amenity_quadrants(self):
        """CP-11(d): pinned amenities first, the others take the free quadrants in QUADRANT_ORDER; the fixture uses
        (letterbox, bin) take no quadrant, so five uses with four amenities do not collide."""
        uses = ["rc_centre", "kindergarten", "letterbox", "senior_centre", "eldercare"]
        sp = self.spec("501", void_deck=uses)
        self.assertEqual(assign_quadrants(sp, uses), [("AM1", "rc_centre", "105"), ("AM2", "kindergarten", "107"),
                                                      ("AM4", "senior_centre", "101"), ("AM5", "eldercare", "103")])
        p = plan_point(sp, self.seed, self.cat)
        dp = derive_plan(p)
        self.assertEqual(plan_errors(dp), [])
        amen = {f.meta["use"]: quadrant_of(p, f) for f in p.ground.faces.values() if f.kind == AMENITY}
        self.assertEqual(amen, {"rc_centre": "105", "kindergarten": "107", "senior_centre": "101", "eldercare": "103"})
        # four amenity rooms leave only the lift lobby: the banks stand there, still with clear_front in front
        self.letterbox_checks(p, dp, "501 (4 amenities)", on_path=True)
        # a pin is honoured and the unpinned uses flow around it
        q = assign_quadrants(self.spec("501", void_deck_at={"kindergarten": "105"}), ["rc_centre", "kindergarten"])
        self.assertEqual(q, [("AM1", "rc_centre", "107"), ("AM2", "kindergarten", "105")])
        # the configured blocks keep their amenity quadrant ('letterbox' first, one amenity -> 105)
        self.assertEqual(QUADRANT_ORDER[0], "105")
        self.assertEqual(assign_quadrants(self.spec("501"), ["letterbox", "rc_centre", "bin"]),
                         [("AM2", "rc_centre", "105")])
        for bad in ({"void_deck": ["rc_centre", "kindergarten"],
                     "void_deck_at": {"rc_centre": "103", "kindergarten": "103"}},
                    {"void_deck": ["rc_centre"], "void_deck_at": {"rc_centre": "109"}},
                    {"void_deck": uses + ["study_corner"]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                plan_point(self.spec("501", **bad), self.seed, self.cat)

    def test_letterbox_capacity(self):
        """HDB-N3: the boxes follow the flat count (a taller block gets more) and an impossible count raises."""
        base, _ = self.plan("501")
        tall = plan_point(self.spec("501", storeys=40), self.seed, self.cat)
        self.assertEqual(tall.meta["letterboxes"]["flats"], 4 * 39)
        self.assertGreater(tall.meta["letterboxes"]["boxes"], base.meta["letterboxes"]["boxes"])
        self.letterbox_checks(tall, derive_plan(tall), "501 (40 storeys)")
        slab = plan_slab(self.spec("507", storeys=30), self.seed, self.cat)
        self.letterbox_checks(slab, derive_plan(slab), "507 (30 storeys)")
        uses = ["rc_centre", "kindergarten", "letterbox", "senior_centre", "eldercare"]
        with self.assertRaisesRegex(ValueError, "letterboxes for"):
            plan_point(self.spec("501", void_deck=uses, storeys=400), self.seed, self.cat)


class SlabTests(TypologyCase):
    def test_slab_blocks(self):
        for blk in SL_BLOCKS:
            with self.subTest(blk=blk):
                plan, dp = self.common_checks(blk)
                seq = self.spec(blk)["sequence"]
                fp = plan.footprint.bounds
                self.assertAlmostEqual((fp[0] + fp[2]) / 2, 0.0, 6)             # centred on x = 0
                self.assertTrue(all(s.entry == "S" for s in plan.stairs))
                n_stair = seq.count("stair") + seq.count("core")
                self.assertEqual(len(plan.stairs), n_stair)
                self.assertEqual(len(plan.typical.flats), sum(1 for s in seq if s not in MODULE_W))
                corr = plan.typical.faces["CORR"]
                self.assertEqual(corr.kind, CORRIDOR)
                cx0, cy0, cx1, cy1 = corr.poly.bounds
                self.assertAlmostEqual(cy1, 0.0)
                self.assertAlmostEqual(cy0, -LOBBY_D)                          # lift lobby bays
                self.assertAlmostEqual(cx1 - cx0, plan.meta["length"])
                # 2.0 m clear corridor along the whole block, a 3.0 m bay in front of each core's two lifts
                self.assertTrue(corr.poly.buffer(1e-6).contains(box(cx0, -CORR_D, cx1, 0.0)))
                n_core = seq.count("core")
                self.assertAlmostEqual(corr.poly.area, (cx1 - cx0) * CORR_D + n_core * 2 * 2.8 * (LOBBY_D - CORR_D), 6)
                self.assertEqual(len(plan.meta["lift_lobbies"]), n_core)
                for p in plan.meta["lift_lobbies"]:
                    self.assertTrue(corr.poly.contains(box(*p, *p).buffer(0.1)))
                    bay = LineString([(p[0], 0.0), (p[0], -LOBBY_D)])                # in front of the lifts
                    self.assertTrue(corr.poly.buffer(1e-6).contains(bay))
                # templates in sequence order west -> east
                xs = [plan.typical.flats[st].envelope.centroid.x for st in plan.meta["stacks"]]
                self.assertEqual(xs, sorted(xs))
                self.assertEqual([plan.typical.flats[st].template for st in plan.meta["stacks"]],
                                 [s for s in seq if s not in MODULE_W])
                # letterbox banks at every core's lift lobby (letterbox_checks: one within near_lobby of each)
                banks = [it for it in plan.meta["furniture"] if it["object_type"] == "Letterbox bank"]
                self.assertGreaterEqual(len(banks), n_core)

    def test_corridor_side_rotation(self):
        plan = plan_slab(self.spec("512", corridor="N"), self.seed, self.cat)
        self.assertTrue(all(s.entry == "N" for s in plan.stairs))
        corr = plan.typical.faces["CORR"].poly
        flats_y = max(fi.envelope.bounds[3] for fi in plan.typical.flats.values())
        self.assertGreaterEqual(corr.bounds[1], flats_y - 1e-6)
        self.assertEqual(plan_errors(derive_plan(plan)), [])
        # the letterbox banks turn with the plan, still on the void deck and clear of the turned plan's columns
        self.assertTrue(plan.meta["furniture"])
        self.letterbox_checks(plan, derive_plan(plan), "512 (corridor N)")

    def test_amenity_pin_and_determinism(self):
        sp = self.spec("507", void_deck=["letterbox", "rc_centre", "kindergarten", "bin"],
                       void_deck_at={"kindergarten": "101"})
        a = plan_slab(sp, self.seed, self.cat)
        b = plan_slab(sp, self.seed, self.cat)
        am = {f.meta["use"]: f.meta["under"] for f in a.ground.faces.values() if f.kind == AMENITY}
        self.assertEqual(am["kindergarten"], "101")
        self.assertEqual(len(am), 2)
        self.assertEqual(plan_errors(derive_plan(a)), [])
        key = lambda p: sorted((f.id, tuple(round(v, 4) for v in f.poly.bounds)) for f in p.typical.faces.values())  # noqa: E731
        self.assertEqual(key(a), key(b))
        self.assertEqual([fi.signature for fi in a.typical.flats.values()],
                         [fi.signature for fi in b.typical.flats.values()])
        self.assertEqual([it["poly"].wkt for it in a.meta["furniture"]], [it["poly"].wkt for it in b.meta["furniture"]])


class LBlockTests(TypologyCase):
    def test_lblocks(self):
        for blk in LB_BLOCKS:
            with self.subTest(blk=blk):
                plan, dp = self.common_checks(blk)
                sp = self.spec(blk)
                mirrored = bool(sp.get("mirror"))
                x0, y0, x1, y1 = plate_ops.bounds(plan.typical)
                # origin at the outer corner of the L
                self.assertAlmostEqual(y1, 0.0, 6)
                self.assertAlmostEqual(x1 if mirrored else x0, 0.0, 6)
                self.assertEqual(len(plan.typical.flats), len(sp["wing_a"]) + len(sp["wing_b"]) + 1
                                 - sum(1 for s in sp["wing_a"] + sp["wing_b"] if s in MODULE_W))
                corner = plan.typical.flats[plan.meta["corner_stack"]]
                self.assertEqual(corner.template, sp["corner"])
                cb = corner.envelope.bounds
                self.assertAlmostEqual(cb[3], 0.0, 6)                        # corner unit at the outer corner
                self.assertAlmostEqual(cb[2] if mirrored else cb[0], 0.0, 6)
                # wing B stairs entered from the courtyard-side corridor
                wing_b = {"W" if mirrored else "E"}
                entries = {s.entry for s in plan.stairs}
                self.assertEqual(entries, {"S"} | wing_b)
                # the two corridors meet edge to edge (no wall between them)
                a, b = plan.typical.faces["CORR_A"].poly, plan.typical.faces["CORR_B"].poly
                self.assertAlmostEqual(a.intersection(b).area, 0.0, 6)
                self.assertGreaterEqual(a.boundary.intersection(b.boundary).length, CORR_D - 1e-6)
                pair = frozenset(("CORR_A", "CORR_B"))
                for r in dp["typical"].runs:                            # derived runs are the walls (and railings)
                    parts = [frozenset(p[2:]) for p in r.meta.get("parts", [])] or [r.faces()]
                    self.assertNotIn(pair, parts, f"{blk}: wall {r.wall} between the corridors")
                # corridors on the inner side: every flat lies on the outer side of its corridor
                wing_b_flats = []
                for st, fi in plan.typical.flats.items():
                    c = fi.envelope.centroid
                    if a.bounds[0] - 1e-6 <= c.x <= a.bounds[2] + 1e-6 and c.y > a.bounds[3] - 1e-6:
                        continue                                        # wing A (or corner): north of corridor A
                    side = (c.x < b.bounds[0]) if not mirrored else (c.x > b.bounds[2])
                    self.assertTrue(side, f"{blk} flat {st} is not on the outer side of corridor B")
                    wing_b_flats.append(fi)
                # wing B is set back from corridor A by the light well, crossed by corridor B as an open bridge
                ya = a.bounds[3] - CORR_D                               # corridor A's outer edge (bays aside)
                band_a = box(a.bounds[0], ya, a.bounds[2], a.bounds[3])
                self.assertAlmostEqual(min(fi.envelope.distance(band_a) for fi in wing_b_flats), LIGHTWELL, 6)
                self.assertEqual(plan.meta["lightwell"], LIGHTWELL)
                bx0, bx1 = (b.bounds[0], b.bounds[0] + CORR_D) if not mirrored else (b.bounds[2] - CORR_D, b.bounds[2])
                self.assertTrue(b.buffer(1e-6).contains(box(bx0, ya - LIGHTWELL, bx1, ya)))
                well = box(*((x0, ya - LIGHTWELL, bx0, ya) if not mirrored else (bx1, ya - LIGHTWELL, x1, ya)))
                self.assertAlmostEqual(sum(f.poly.intersection(well).area for f in plan.typical.faces.values()), 0.0, 6)
                # letterbox banks at both lift lobbies (corner core and wing B core)
                self.assertGreaterEqual(len(plan.meta["furniture"]), 2)

    def test_corner_outlook(self):
        """Without the light well the corner unit's corridor windows (bedroom 4, kitchen) face wing B's flank."""
        plan, dp = self.plan("510")
        corner = plan.meta["corner_stack"]
        self.assertEqual(plate_ops.window_outlook(dp["typical"]), [])
        tight = plan_lblock(self.spec("510", lightwell=0.0), self.seed, self.cat)
        self.assertEqual(plan_errors(derive_plan(tight)), [])
        issues = plate_ops.window_outlook(derive(tight.typical, "typical"))
        rooms = {fid.split(":")[1] for _, fid, _ in issues if fid.startswith(f"U{corner}:")}
        self.assertTrue({"B4", "K"} <= rooms, issues)
        self.assertTrue(all(d < 2.1 for _, _, d in issues))

    def test_mirror_is_exact(self):
        p0, _ = self.plan("510")
        sp = self.spec("510", mirror=True, void_deck=self.spec("511")["void_deck"])
        p1 = plan_lblock(sp, self.seed, self.cat)
        self.letterbox_checks(p1, derive_plan(p1), "510 (mirrored)")
        self.assertEqual(sorted(round(it["poly"].area, 6) for it in p1.meta["furniture"]),
                         sorted(round(it["poly"].area, 6) for it in p0.meta["furniture"]))
        b = lambda g: tuple(round(v, 4) for v in g.bounds)  # noqa: E731
        m = {f.id: b(f.poly) for f in p1.typical.faces.values()}
        for f in p0.typical.faces.values():
            x0, y0, x1, y1 = b(f.poly)
            self.assertEqual(m[f.id], (round(-x1, 4) + 0.0, y0, round(-x0, 4) + 0.0, y1), f.id)


class PlateOpsTests(TypologyCase):
    def test_transforms_roundtrip(self):
        plan, _ = self.plan("507")
        pl = plan.typical
        r4 = pl
        for _ in range(4):
            r4 = plate_ops.rotate(r4, 1, about=(3.0, -1.0))
        mm = plate_ops.mirror(plate_ops.mirror(pl, "x", 2.5), "x", 2.5)
        for other in (r4, mm):
            for fid, f in pl.faces.items():
                self.assertTrue(f.poly.equals(other.faces[fid].poly), fid)
            self.assertEqual([d.at for d in pl.doors], [d.at for d in other.doors])
            self.assertEqual([d.hinge for d in pl.doors], [d.hinge for d in other.doors])
            self.assertEqual([w.at for w in pl.windows], [w.at for w in other.windows])
        rot = plate_ops.rotate(pl, 1)
        self.assertEqual(plan_errors({"typical": derive(rot, "typical")}), [])

    def test_map_entry(self):
        r = plate_ops.rotation(1)
        self.assertEqual([plate_ops.map_entry(e, r) for e in "NESW"], ["W", "N", "E", "S"])
        m = plate_ops.mirror_x()
        self.assertEqual([plate_ops.map_entry(e, m) for e in "NESW"], ["N", "W", "S", "E"])
        c = plate_ops.compose(plate_ops.translation(5, 0), plate_ops.rotation(2))
        self.assertEqual(c((1.0, 1.0)), (4.0, -1.0))

    def test_transform_plan_moves_everything(self):
        plan, _ = self.plan("512")
        xf = plate_ops.compose(plate_ops.translation(10.0, 20.0), plate_ops.rotation(1))
        q = plate_ops.transform_plan(plan, xf)
        self.assertTrue(all(s.entry == "E" for s in q.stairs))
        self.assertEqual(q.meta["corridor_side"], "E")
        self.assertEqual(q.meta["entrances"][0], plate_ops.snap_pt(xf(plan.meta["entrances"][0])))
        self.assertEqual(plan_errors(derive_plan(q)), [])
        self.assertEqual([(round(l.poly.area, 6), l.kind) for l in q.typical.ledges],
                         [(round(l.poly.area, 6), l.kind) for l in plan.typical.ledges])
        self.assertIn("bay", {l.kind for l in q.typical.ledges})
        for a, b in zip(plan.meta["furniture"], q.meta["furniture"]):
            self.assertTrue(b["poly"].equals(plate_ops.snap_geom(xf.poly(a["poly"]))))

    def test_column_points_match_builder(self):
        """plate_ops.column_points (what the letterbox banks keep clear of) is where builder._columns stands the
        void deck columns, for each typology and a turned slab."""
        from estate.blocks.builder import _columns
        from estate.ifc.writer import IfcWriter
        plans = [self.plan(b)[0] for b in ("501", "507", "510")]
        plans.append(plan_slab(self.spec("512", corridor="E"), self.seed, self.cat))
        for plan in plans:
            with self.subTest(blk=plan.blk):
                dp = derive_plan(plan)
                W = IfcWriter(guid_key=f"test/columns/{plan.blk}")
                st = W.storeys(W.building(plan.name), ["L1"], [0.0])[0]
                kept = _columns(W, plan, dp, st, 3.4)
                W.close()
                pts = plate_ops.column_points(dp["typical"], plan.ground)
                self.assertEqual([tuple(map(float, q)) for q in kept], pts)
                self.assertTrue(set(pts) <= set(plate_ops.column_candidates(dp["typical"], plan.ground)))

    def test_ledge_kind_survives_rotation(self):
        """transform_plate keeps Ledge.kind: bay windows stay glazed (not climbable AC ledges) in a turned slab."""
        kinds = {}
        for side in "SNE":
            plan = plan_slab(self.spec("512", corridor=side), self.seed, self.cat)
            kinds[side] = Counter(l.kind for l in plan.typical.ledges)
        self.assertGreater(kinds["S"]["bay"], 0)
        self.assertEqual(kinds["N"], kinds["S"])
        self.assertEqual(kinds["E"], kinds["S"])
        plan, _ = self.plan("512")
        turned = plate_ops.transform_plate(plan.typical, plate_ops.rotation(1))
        back = plate_ops.transform_plate(turned, plate_ops.rotation(3))
        self.assertEqual([(l.name, l.kind) for l in back.ledges], [(l.name, l.kind) for l in plan.typical.ledges])


if __name__ == "__main__":
    unittest.main()
