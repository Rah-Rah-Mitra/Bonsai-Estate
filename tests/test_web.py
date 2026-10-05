"""Web export (estate/web, the web stage): the SN5W walk-grid format against its golden file, storey bands, synthetic
scenes (an unreachable platform, a stamped doorway, an opened leaf blocked or let through), and the PT4 test block
built through the stage's own code: rooms, doors, spawns and lift landings on the exported grid, stairs against the
IFC, opened leaves against the LOD0 leaves, no side effect on nav3d, determinism, the manifest and export_info.json,
and the leak scan of the walk grid.

Run: "<blender python>" -I -B tests/test_web.py -v      (SN5W_REGEN=1 rewrites tests/fixtures/web/sn5w_sample.bin)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estate.env import bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402
import shapely  # noqa: E402

from estate import config, env, leaks  # noqa: E402
from estate.validate import nav3d  # noqa: E402
from estate.validate import nav_doorpose as doorpose  # noqa: E402
from estate.web import sn5w, walk  # noqa: E402

OUT = env.BUILD / "web_tests" / "unittest"
PT4 = env.BUILD / "t_pt4.ifc"
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "web" / "sn5w_sample.bin"
B = sn5w.BLOCKED


# ----------------------------------------------------------------------------- the golden file
def sample_grid() -> dict:
    """The synthetic grid of tests/fixtures/web/sn5w_sample.bin (240 bytes), the contract the portfolio's decoder is
    tested against. 4 x 3 cells of 0.1 m for a 0.20 m agent and a 0.40 m step, corner of cell (0, 0) at block-local
    (-1.25, 2.5); three storeys, the reference L2 (index 1). Rasters are absolute mm above each storey's FFL, row iy
    = 0 first, B = 0x7FFF (no floor):

        L1  ffl 0.0   delta   [   0,  10,   5, -100 ]   overflow (ix 0, iy 0, 2800), (ix 2, iy 1, 2400)
                              [-120,   B, 175,    B ]
                              [   B,   B, 350, 1200 ]
        L2  ffl 3.5   raw     [   0,   0,   5,    B ]   overflow (ix 1, iy 0, 2600)
                              [-120,   0, 175,    B ]
                              [   B,   B, 350, 1200 ]
        RF  ffl 6.25  same    (L2's raster)            overflow (ix 0, iy 2, 2000)

    L1 is stored as int16 (L1 - L2) mod 2^16: [0, 10, 0, 32669 | 0, 32767, 0, 0 | 0, 0, 0, 0]; the 32669 is the
    wrap of -100 - 32767, the 32767 the wrap of 32767 - 0. Byte layout: header 0..63 (flags 1: a delta layer),
    table 64..159, L1 raster 160..183, L1 overflow 184..199 (sorted by iy, ix, mm), L2 raster 200..223, L2 overflow
    224..231, RF overflow 232..239 (RF has no raster: offset and length 0)."""
    ref = [0, 0, 5, B, -120, 0, 175, B, B, B, 350, 1200]
    return dict(nx=4, ny=3, cell=0.1, radius=0.2, step=0.4, origin=(-1.25, 2.5), ref=1,
                modes=["delta", "raw", "same"],
                layers=[dict(tag="L1", ffl=0.0, raster=[0, 10, 5, -100, -120, B, 175, B, B, B, 350, 1200],
                             overflow=[(2, 1, 2400), (0, 0, 2800)]),
                        dict(tag="L2", ffl=3.5, raster=ref, overflow=[(1, 0, 2600)]),
                        dict(tag="RF", ffl=6.25, raster=list(ref), overflow=[(0, 2, 2000)])])


def sample_bytes() -> bytes:
    s = sample_grid()
    return sn5w.encode(s["nx"], s["ny"], s["cell"], s["radius"], s["step"], s["origin"], s["layers"], s["ref"],
                       modes=s["modes"])


class Sn5wFormat(unittest.TestCase):
    def test_golden_file(self):
        data = sample_bytes()
        if os.environ.get("SN5W_REGEN"):
            GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN.write_bytes(data)
        self.assertEqual(data, GOLDEN.read_bytes())
        self.assertEqual(len(data), 240)

    def test_layout_byte_for_byte(self):
        """The golden file read with struct alone, field by field, as the format's description in sn5w.py says."""
        d = GOLDEN.read_bytes()
        self.assertEqual(d[:4], b"SN5W")
        self.assertEqual(struct.unpack_from("<HH", d, 4), (1, 1))                    # version 1, flags: delta
        cell, radius, step, ox, oy = struct.unpack_from("<5f", d, 8)
        self.assertEqual((ox, oy), (-1.25, 2.5))
        self.assertEqual([round(v, 6) for v in (cell, radius, step)], [0.1, 0.2, 0.4])
        self.assertEqual(struct.unpack_from("<4H", d, 28), (4, 3, 3, 1))           # nx, ny, layers, ref
        self.assertEqual(d[36:64], b"\0" * 28)
        table = [struct.unpack_from("<4sfBBHIIIII", d, 64 + 32 * i) for i in range(3)]
        self.assertEqual(table, [(b"L1\0\0", 0.0, 1, 0, 0, 160, 24, 2, 184, 0),
                                 (b"L2\0\0", 3.5, 0, 0, 0, 200, 24, 1, 224, 0),
                                 (b"RF\0\0", 6.25, 2, 0, 0, 0, 0, 1, 232, 0)])
        self.assertEqual(list(struct.unpack_from("<12h", d, 160)), [0, 10, 0, 32669, 0, 32767, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(struct.unpack_from("<12h", d, 200)), [0, 0, 5, B, -120, 0, 175, B, B, B, 350, 1200])
        self.assertEqual([struct.unpack_from("<HHhH", d, o) for o in (184, 192, 224, 232)],
                         [(0, 0, 2800, 0), (2, 1, 2400, 0), (1, 0, 2600, 0), (0, 2, 2000, 0)])

    def test_round_trip(self):
        t = sn5w.decode(GOLDEN.read_bytes())
        s = sample_grid()
        self.assertEqual((t["nx"], t["ny"], t["ref"], t["delta"], t["coarse"]), (4, 3, 1, True, False))
        for got, want, mode in zip(t["layers"], s["layers"], s["modes"]):
            self.assertEqual((got["tag"], got["ffl"], got["mode"]), (want["tag"], want["ffl"], mode))
            self.assertEqual(list(got["raster"]), want["raster"])
            self.assertEqual(got["overflow"], sorted(want["overflow"], key=lambda r: (r[1], r[0], r[2])))

    def test_modes_chosen(self):
        """Without explicit modes: the reference raw, an identical layer 'same', a layer with few differences delta,
        a different one raw; a numpy raster gives the same bytes as a list."""
        ref = np.full(400, B, np.int16)
        ref[:300] = 0
        few = ref.copy()
        few[5] = 40
        other = np.full(400, B, np.int16)
        other[100:] = 175
        lays = [dict(tag=t, ffl=f, raster=r) for t, f, r in (("L1", 0, other), ("L2", 3, ref), ("L3", 6, few),
                                                               ("RF", 9, ref.copy()))]
        t = sn5w.decode(sn5w.encode(20, 20, 0.1, 0.2, 0.4, (0, 0), lays, 1))
        self.assertEqual([lay["mode"] for lay in t["layers"]], ["raw", "raw", "delta", "same"])
        self.assertEqual(list(t["layers"][2]["raster"]), few.tolist())
        as_list = [dict(lay, raster=lay["raster"].tolist()) for lay in lays]
        self.assertEqual(sn5w.encode(20, 20, 0.1, 0.2, 0.4, (0, 0), as_list, 1),
                         sn5w.encode(20, 20, 0.1, 0.2, 0.4, (0, 0), lays, 1))

    def test_strict_reader(self):
        """Anything the format does not account for is refused: a trailing byte, a pad, a reserved byte, a NUL inside a
        tag, a section moved, a 'same' layer with a raster."""
        d = GOLDEN.read_bytes()
        self.assertEqual(sn5w.layer_tags(d), ["L1", "L2", "RF"])

        def poke(off, b):
            return d[:off] + b + d[off + len(b):]
        for bad in (d + b"\0", poke(40, b"x"), poke(64 + 9, b"\1"), poke(64, b"\0"), poke(232 + 6, b"\1"),
                    poke(64 + 32 + 12, struct.pack("<I", 202)), poke(64 + 64 + 16, struct.pack("<I", 24)), d[:-1],
                    poke(4, struct.pack("<H", 2))):
            with self.assertRaises(ValueError):
                sn5w.read_table(bad)

    def test_leak_scan_reads_the_tags(self):
        """estate/leaks.py reads an SN5W file as one: its tags are its text (a planted drive path in a tag is found),
        its heights are not, and a file that does not parse is not scanned."""
        self.assertEqual(leaks.scan_bytes(GOLDEN.read_bytes(), "BLK_1_walk.bin"), [])
        s = sample_grid()
        s["layers"][2]["tag"] = "Q:" + chr(92)
        planted = sn5w.encode(4, 3, 0.1, 0.2, 0.4, (0, 0), s["layers"], 1, modes=s["modes"])
        self.assertEqual([f.where for f in leaks.scan_bytes(planted, "x_walk.bin")], ["x_walk.bin layer 2"])
        broken = leaks.scan_bytes(GOLDEN.read_bytes() + b"C:" + chr(92).encode(), "x_walk.bin")
        self.assertEqual([f.text for f in broken], [leaks.UNSCANNABLE])


class CoarseFallback(unittest.TestCase):
    def test_fallback_when_over_the_cap(self):
        """Over max_walk_gz the grid is written at 0.2 m (flags bit 1): a coarse cell is walkable only where all four
        0.1 m cells are, at the highest of their floors; an overflow floor survives where all four cells have a floor
        within 0.2 m of it. Under the cap nothing changes."""
        rng = np.random.default_rng(5)
        nx, ny = 61, 40                                          # odd nx: the last coarse column has two cells
        lays = []
        for li in range(3):
            r = np.where(rng.random(nx * ny) < 0.8, rng.integers(-20, 20, nx * ny), B).astype(np.int16)
            ov = [(ix, iy, 2600) for iy in range(0, 4) for ix in range(0, 4)] if li == 1 else []
            lays.append(dict(tag=f"L{li + 1}", ffl=3.0 * li, raster=r, overflow=ov))
        fine, _, coarse = walk.encode(nx, ny, (1.0, 2.0), lays, walk.WalkConfig())
        self.assertFalse(coarse)
        self.assertEqual(sn5w.read_table(fine)["nx"], nx)
        data, ref, coarse = walk.encode(nx, ny, (1.0, 2.0), lays, walk.WalkConfig(max_walk_gz=3000))
        with self.assertRaises(ValueError):                    # too big even at 0.2 m: the stage fails
            walk.encode(nx, ny, (1.0, 2.0), lays, walk.WalkConfig(max_walk_gz=100))
        t = sn5w.decode(data)
        self.assertTrue(coarse and t["coarse"] and t["flags"] & 2)
        self.assertEqual((t["nx"], t["ny"], round(t["cell"], 6), t["origin"]), (31, 20, 0.2, (1.0, 2.0)))
        for lay, src in zip(t["layers"], lays):
            R = np.full((40, 62), B, np.int64)
            R[:, :nx] = np.asarray(src["raster"], np.int64).reshape(ny, nx)
            quads = np.stack([R[0::2, 0::2], R[0::2, 1::2], R[1::2, 0::2], R[1::2, 1::2]])
            want = np.where((quads != B).all(0), quads.max(0), B)
            self.assertEqual(np.asarray(lay["raster"]).reshape(20, 31).tolist(), want.tolist())
        self.assertEqual(sorted(t["layers"][1]["overflow"]), [(0, 0, 2600), (0, 1, 2600), (1, 0, 2600), (1, 1, 2600)])


class Bands(unittest.TestCase):
    def test_band_edges(self):
        """A floor exactly at FFL_s - 0.25 belongs to storey s, a hair lower to s - 1; below the lowest storey is the
        lowest storey's, above the top one the top one's."""
        ffl = [0.0, 3.6, 6.4, 9.2]
        for s in range(1, 4):
            self.assertEqual(int(walk.band_of(ffl[s] - 0.25, ffl, 0.25)), s)
            self.assertEqual(int(walk.band_of(ffl[s] - 0.25 - 1e-6, ffl, 0.25)), s - 1)
        self.assertEqual(int(walk.band_of(-2.0, ffl, 0.25)), 0)
        self.assertEqual(int(walk.band_of(30.0, ffl, 0.25)), 3)


# ----------------------------------------------------------------------------- synthetic scenes
def _box(x0, y0, z0, x1, y1, z1):
    return nav3d._box_tris((x0, y0, z0), (x1, y1, z1))


def _door(x0, w, wall_y=1.5, h=2.1, jamb=0.05):
    """A door record as nav3d.read_model builds it (tests/test_nav.py's door())."""
    M = np.eye(4)
    M[:3, 3] = (x0, wall_y - 0.05, 0.0)
    d = dict(guid=f"door{x0}", name="test door", kind="internal", M=M, w=w, clear=round(w - 2 * jamb, 3), jamb=jamb,
             jamb_src="pset", y0=-0.015, y1=0.1, yc=0.0425, h=h, z=0.0, storey="L1", sides=())
    d["bbox"] = (np.array([x0, wall_y - 0.065, 0.0]), np.array([x0 + w, wall_y + 0.05, h]))
    return d


def _scene(boxes, doors=(), lo=(-0.5, -0.5, -0.5), hi=(6.5, 3.5, 3.0)):
    tris = [_box(*b) for b in boxes] + [nav3d.jamb_tris(d) for d in doors]
    T = np.concatenate(tris)
    elem = np.repeat(np.arange(len(tris)), [len(t) for t in tris])
    g = nav3d.Grid.around(lo, hi, 0.1)
    return walk.solve(T, elem, g, 0.0, list(doors), walk.WalkConfig())


def _wall_with_door(x0, w, a=1.4, b=1.6):
    """A floor 6 x 3 m and a wall across it (and past the grid, so the door is the only way through)."""
    return [(0, 0, -0.2, 6, 3, 0), (-1, a, 0, x0, b, 2.4), (x0 + w, a, 0, 7, b, 2.4), (x0, a, 2.1, x0 + w, b, 2.4)]


def _leaf(x0, w, jamb=0.05, wall_y=1.5):
    """The swing leaf of _door(x0, w), hinged on its left jamb on the +y face, as the engine JSON records it."""
    rec = dict(node="DOOR_test_leaf", motion="swing", pivot=[x0 + jamb, wall_y - 0.05 + 0.1, 0.0], thickness=0.035,
               width=round(w - 2 * jamb, 4), height=2.05, axis=[0.0, 0.0, 1.0], open_sign=1, both_ways=False,
               max_angle_deg=90.0)
    return dict(door=0, guid="d", storey="L1", rec=rec, along=[1.0, 0.0, 0.0], across=[0.0, 1.0, 0.0],
                offset=np.zeros(3), z=0.0, obstructs=True, angle=None)


class Scenes(unittest.TestCase):
    def test_unreachable_platform_is_absent(self):
        """A 1 m platform with no way up has a walkable top, but it is not reachable from the street, so no layer
        holds it; the floor around it is there."""
        w = _scene([(0, 0, -0.2, 6, 3, 0), (2, 1, 0, 4, 2, 1.0)])
        top = np.abs(w.fz - 1.0) < 0.01
        self.assertTrue(top.any())
        keep = walk.reachable(w, np.zeros(len(w.ix), bool))
        self.assertFalse((keep & top).any())
        nx, ny, origin, lays = walk.layers(w, keep, [("L1", 0.0), ("L2", 3.6)], np.zeros(3), 0.25)
        r = np.asarray(lays[0]["raster"]).reshape(ny, nx)
        self.assertTrue(set(np.unique(r[r != B]).tolist()) <= {0})          # only the floor, at FFL
        self.assertFalse(lays[0]["overflow"])
        i, j = int(round((3.0 - origin[0]) / 0.1 - 0.5)), int(round((1.5 - origin[1]) / 0.1 - 0.5))
        self.assertEqual(int(r[j, i]), B)                                   # the platform's footprint: no floor

    def test_narrow_doorway_is_stamped(self):
        """A 0.5 m clear opening passes the 0.025 m door test at r = 0.20 but not the 0.1 m grid where its jambs fall
        off the cell lines: the line of cells across it is stamped walkable, so its two sides connect on the grid;
        on the cell lines the grid passes it on its own and nothing is stamped."""
        for x0, stamped in ((2.02, True), (2.0, False)):
            d = _door(x0, 0.6)
            w = _scene(_wall_with_door(x0, 0.6), [d])
            a, b = w.sides[0]
            none = np.zeros(len(w.ix), bool)
            region = np.arange(len(w.ix))
            self.assertTrue(walk.connected(w, region, a, b, none), x0)
            self.assertEqual(bool(w.stamped.any()), stamped, x0)
            self.assertEqual(walk.connected(w, region, a, b, w.stamped.copy()), not stamped, x0)

    def test_opened_leaf_blocks_or_passes(self):
        """An opened leaf is blocked (dilated by r) when its door still connects; a leaf that would cut its narrow
        door is let through and counted. A rolled-up shutter clears the agent's head."""
        for x0, w_, want in ((2.0, 0.9, "blocked"), (2.02, 0.6, "passthrough")):
            d = _door(x0, w_)
            w = _scene(_wall_with_door(x0, w_), [d])
            lf = _leaf(x0, w_)
            blocked = walk.block_leaves(w, [d], [lf], walk.WalkConfig())
            self.assertEqual(lf["grid"], want, (x0, w_))
            self.assertEqual(w.stats["leaf_passthrough"], int(want == "passthrough"))
            self.assertEqual(bool(blocked.any()), want == "blocked")
            self.assertTrue(walk.connected(w, np.arange(len(w.ix)), *w.sides[0], blocked))
            if want == "blocked":                  # cells within r of the opened leaf are gone, cells beyond are not
                cx, cy = w.cx(), w.cy()
                poly = walk.leaf_polygon(lf)[0]
                inside = shapely.contains_xy(poly.buffer(0.19), cx, cy) & (np.abs(w.fz) < 0.1)
                self.assertTrue(inside.any() and blocked[inside].all())
                self.assertFalse(blocked[shapely.contains_xy(poly.buffer(0.3), cx, cy) == 0].any())
        roll = dict(motion="roll", pivot=[0.0, 0.0, 0.0], thickness=0.05, width=2.0, height=2.09, travel=[0, 0, 2.09])
        self.assertGreaterEqual(doorpose.opened_z(roll, (1, 0, 0), (0, 1, 0))[0], 1.8)

    def test_leaf_narrowed_in_a_shallow_room(self):
        """Opened to 90 degrees in a room 1.1 m deep, the leaf meets the far wall and cuts off the part of the room
        behind its hinge, which the street reached before: the leaf opens less (and stays blocked) so that part is
        reached round its edge."""
        x0, w_ = 2.0, 0.8
        d = _door(x0, w_)
        room = [(0.5, 1.6, 0, 0.6, 2.9, 2.4), (5.4, 1.6, 0, 5.5, 2.9, 2.4), (-1, 2.7, 0, 7, 2.9, 2.4)]
        w = _scene(_wall_with_door(x0, w_) + room, [d])
        lf = _leaf(x0, w_)
        blocked = walk.block_leaves(w, [d], [lf], walk.WalkConfig())
        self.assertEqual(lf["grid"], "blocked")
        self.assertIn(lf["angle"], walk.NARROW)
        self.assertEqual((w.stats["leaf_narrowed"], w.stats["leaf_passthrough"]), (1, 0))
        keep = walk.reachable(w, blocked)
        behind = (w.cx() < x0 - 0.3) & (w.cy() > 1.8) & (w.cy() < 2.5) & (w.cx() > 0.8)
        self.assertTrue(behind.any() and keep[behind].any())
        at90 = np.zeros(len(w.ix), bool)
        at90[walk._leaf_cells(w, walk.ColumnIndex(w.ix, w.iy, w.g.ny), lf, 90.0, walk.WalkConfig())] = True
        self.assertFalse(walk.reachable(w, at90)[behind].any())          # at 90 degrees the part behind is cut off


class DoorPose(unittest.TestCase):
    def test_spec_frame_and_record(self):
        """leaf_spec, leaf_frame and leaf_record, the formulas engine.py imports: a right-hinged swing leaf pivots at
        x1 on the swing face and opens with sign -1; a sliding pair slides apart; a shutter rolls up its height."""
        lo, hi = np.array([0.05, -0.02, 0.01]), np.array([0.85, 0.015, 2.06])
        s = doorpose.leaf_spec("SINGLE_SWING_RIGHT", lo, hi, 0, 1)
        self.assertEqual((s["motion"], s["pivot"], s["open_sign"]), ("swing", (0.85, 0.015), -1))
        self.assertEqual((s["width"], s["thickness"], s["height"], s["z0"]), (0.8, 0.035, 2.05, 0.01))
        left, right = (doorpose.leaf_spec("DOUBLE_DOOR_SLIDING", lo, hi, k, 2) for k in (0, 1))
        self.assertAlmostEqual(left["travel"][0], -0.8)
        self.assertAlmostEqual(right["travel"][0], 0.8)
        self.assertAlmostEqual(doorpose.leaf_spec("ROLLING", lo, hi, 0, 1)["travel"][2], 2.05)
        np.testing.assert_allclose(doorpose.leaf_frame((0.85, 0.015))[:3, 3], [0.85, 0.015, 0.0])
        M = np.eye(4)
        M[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        M[:3, 3] = (10, 20, 3)
        rec = doorpose.leaf_record(M, dict(s, node="n"))
        self.assertEqual((rec["pivot"], rec["axis"]), ([9.985, 20.85, 3.0], [0.0, 0.0, 1.0]))

    def test_open_matrix(self):
        """open_matrix turns a swing leaf about its hinge (the pivot stays put) and moves a sliding leaf by travel."""
        rec = dict(motion="swing", pivot=[1.0, 2.0, 0.0], axis=[0, 0, 1], open_sign=-1, max_angle_deg=90.0,
                   width=0.8, thickness=0.04, height=2.0)
        O = doorpose.open_matrix(rec)
        np.testing.assert_allclose(O @ [1.0, 2.0, 0.0, 1.0], [1.0, 2.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(O @ [0.2, 2.0, 0.0, 1.0], [1.0, 2.8, 0.0, 1.0], atol=1e-12)
        sl = dict(motion="slide", pivot=[0, 0, 0], travel=[-0.8, 0, 0], width=0.8, thickness=0.04, height=2.0)
        np.testing.assert_allclose(doorpose.open_matrix(sl)[:3, 3], [-0.8, 0, 0])
        ring = doorpose.opened_footprint(rec, (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        self.assertAlmostEqual(shapely.Polygon(ring).area, 0.8 * 0.04, places=9)


# ----------------------------------------------------------------------------- the PT4 test block
def _export_pt4(out):
    from estate.export.engine import export_building
    return export_building(PT4, out, "t_pt4", local_matrix="building")


@unittest.skipUnless(PT4.exists(), "build/t_pt4.ifc missing")
class PointBlockWeb(unittest.TestCase):
    """build/t_pt4.ifc exported (engine JSON), then the web stage's build() twice, between two runs of nav3d."""

    @classmethod
    def setUpClass(cls):
        from estate.web import export
        shutil.rmtree(OUT, ignore_errors=True)
        cls.nav_before = nav3d.check_building(PT4, OUT / "nav_a", 0.30, png=False)
        cls.r = _export_pt4(OUT)
        cls.eng = json.loads(Path(cls.r["files"]["engine"]).read_text(encoding="utf-8"))
        cfg = config.load()
        cls.web = [export.build(PT4, cls.r["files"]["engine"], OUT / run, "t_pt4", cfg) for run in ("a", "b")]
        cls.nav_after = nav3d.check_building(PT4, OUT / "nav_b", 0.30, png=False)
        cls.bin = (OUT / "a" / "t_pt4_walk.bin").read_bytes()
        cls.doc = json.loads((OUT / "a" / "t_pt4_web.json").read_text(encoding="utf-8"))
        cls.grid = sn5w.decode(cls.bin)
        cls.floors = cls._floors(cls.grid)

    @staticmethod
    def _floors(t):
        """Every floor of the decoded grid: arrays x, y (cell centres, block-local), z (absolute), layer index."""
        xs, ys, zs, ls = [], [], [], []
        c, (ox, oy), nx = t["cell"], t["origin"], t["nx"]
        for li, lay in enumerate(t["layers"]):
            r = np.asarray(lay["raster"], np.int64)
            idx = np.nonzero(r != B)[0]
            pts = [(i % nx, i // nx, int(r[i])) for i in idx] + list(lay["overflow"])
            for ix, iy, mm in pts:
                xs.append(ox + (ix + 0.5) * c)
                ys.append(oy + (iy + 0.5) * c)
                zs.append(lay["ffl"] + mm / 1000.0)
                ls.append(li)
        return dict(x=np.array(xs), y=np.array(ys), z=np.array(zs), layer=np.array(ls))

    def near(self, x, y, z, reach, dz=0.3):
        f = self.floors
        return np.nonzero((np.hypot(f["x"] - x, f["y"] - y) <= reach) & (np.abs(f["z"] - z) <= dz))[0]

    def test_layers(self):
        t = self.grid
        storeys = sorted(self.eng["storeys"].items(), key=lambda kv: kv[1])
        self.assertEqual([lay["tag"] for lay in t["layers"]], [k for k, _ in storeys])
        self.assertEqual([round(lay["ffl"], 4) for lay in t["layers"]], [round(v, 4) for _, v in storeys])
        self.assertNotIn(t["ref"], (0, len(storeys) - 1))                 # a typical storey
        self.assertTrue(t["delta"] or any(lay["mode"] == "same" for lay in t["layers"]))
        self.assertEqual((round(t["cell"], 6), round(t["radius"], 6), t["coarse"]), (0.1, 0.2, False))
        self.assertLessEqual(self.web[0]["gz_bytes"], walk.WalkConfig.of(config.load()).max_walk_gz)
        self.assertGreater(self.doc["walk"]["cells"], 100000)

    def test_rooms_have_walkable_cells(self):
        """Every room nav3d reaches (lift shafts and AC ledges are not rooms to walk) has a walkable cell inside its
        outline shrunk by 0.05 m, at its floor."""
        f, missing, n = self.floors, [], 0
        for room in self.eng["rooms"]:
            label = f"{room['name']} {room.get('room') or ''}"
            if nav3d.EXCLUDED_SPACES.search(label) or len(room["polygon"]) < 4:
                continue
            n += 1
            inner = shapely.Polygon([p[:2] for p in room["polygon"]]).buffer(-0.05)
            sel = np.abs(f["z"] - room["floor_z"]) <= 0.3
            if not shapely.contains_xy(inner, f["x"][sel], f["y"][sel]).any():
                missing.append(room["name"])
        self.assertGreater(n, 500)
        self.assertEqual(missing, [])

    def test_doors_connect_with_leaves_blocked(self):
        """Every passable door joins its two sides within 1.5 m of its opening on the exported grid, with the opened
        leaves blocked; neighbours are 4-adjacent cells (main or overflow) whose floors differ by at most a step."""
        t, f = self.grid, self.floors
        c = t["cell"]
        bad, n = [], 0
        for p in self.eng["portals"]:
            if not p["passable"]:
                continue
            n += 1
            (ax, ay, az), (bx, by, bz) = p["link"]
            mx, my = (ax + bx) / 2, (ay + by) / 2
            idx = np.nonzero((np.hypot(f["x"] - mx, f["y"] - my) <= 1.5) & (np.abs(f["z"] - p["floor_z"]) <= 1.0))[0]
            ii = np.round((f["x"][idx] - t["origin"][0]) / c - 0.5).astype(int)
            jj = np.round((f["y"][idx] - t["origin"][1]) / c - 0.5).astype(int)
            pos = {}
            for k, (i, j) in enumerate(zip(ii, jj)):
                pos.setdefault((int(i), int(j)), []).append(k)
            ea, eb = [], []
            for (i, j), ks in pos.items():
                for di, dj in ((1, 0), (0, 1)):
                    for k in ks:
                        for k2 in pos.get((i + di, j + dj), ()):
                            if abs(f["z"][idx[k]] - f["z"][idx[k2]]) <= 0.4 + 1e-6:
                                ea.append(k)
                                eb.append(k2)
            lab = nav3d.components(len(idx), np.array(ea, np.int64), np.array(eb, np.int64))
            sa = [k for k in range(len(idx)) if np.hypot(f["x"][idx[k]] - ax, f["y"][idx[k]] - ay) <= 0.3]
            sb = [k for k in range(len(idx)) if np.hypot(f["x"][idx[k]] - bx, f["y"][idx[k]] - by) <= 0.3]
            if not (sa and sb and set(lab[sa].tolist()) & set(lab[sb].tolist())):
                bad.append(p["node"])
        self.assertGreater(n, 400)
        self.assertEqual(bad, [])

    def test_spawns_and_lift_landings(self):
        """Every spawn point is within 0.3 m of a walkable L1 cell; every lift landing has a walkable cell within
        0.8 m on every level it serves."""
        for sp in self.eng["spawns"]:
            x, y, z = sp["position"]
            self.assertTrue(len(self.near(x, y, z, 0.3)), sp["name"])
        by_node = {d["node"]: d for d in self.eng["doors"]}
        n = 0
        for lift in self.eng["lifts"]:
            for level, node in lift["landing_door_by_level"].items():
                d = by_node[node]
                cx, cy, _ = d["centre"]
                self.assertTrue(len(self.near(cx, cy, lift["levels"][level], 0.8)), f"{lift['name']} {level}")
                n += 1
        self.assertGreater(n, 20)

    def test_stairs(self):
        """Every IfcStair: its flights in order, risers and riser heights as the IFC says (and their sum is the
        stair's NumberOfRiser), a path rising no more than 0.40 m between points, from the stair's floor to within
        0.1 m of the next one, standing on the walk grid along the way."""
        import ifcopenshell
        import ifcopenshell.util.element as uel
        f = ifcopenshell.open(str(PT4))
        by_name = {s.Name: s for s in f.by_type("IfcStair")}
        self.assertEqual(sorted(s["name"] for s in self.doc["stairs"]), sorted(by_name))
        off_grid = total = 0
        for s in self.doc["stairs"]:
            st = by_name[s["name"]]
            flights = [p for p in uel.get_decomposition(st) if p.is_a("IfcStairFlight")]
            self.assertEqual(len(s["flights"]), len(flights))
            self.assertEqual(sorted((x["risers"], x["riser"]) for x in s["flights"]),
                             sorted((p.NumberOfRisers, round(p.RiserHeight, 4)) for p in flights))
            self.assertEqual(sum(x["risers"] for x in s["flights"]),
                             uel.get_psets(st)["Pset_StairCommon"]["NumberOfRiser"])
            self.assertTrue(all(x["going"] == 0.28 for x in s["flights"]))
            path = np.array(s["path"])
            self.assertLessEqual(float(np.diff(path[:, 2]).max()), 0.40 + 1e-9, s["name"])
            self.assertGreaterEqual(float(np.diff(path[:, 2]).min()), -1e-9, s["name"])
            self.assertAlmostEqual(path[0, 2], s["from_ffl"], delta=0.01)
            self.assertLess(abs(path[-1, 2] - s["to_ffl"]), 0.1)
            self.assertEqual(s["to_ffl"], self.eng["storeys"][s["to"]])
            self.assertTrue(s["room"] and "STAIR" in s["room"], s)
            self.assertEqual([round(lg["z"], 3) for lg in s["landings"]][-1], s["to_ffl"])
            for x, y, z in path:
                total += 1
                off_grid += not len(self.near(x, y, z, 0.25, 0.25))
        self.assertLessEqual(off_grid, 0.05 * total)

    def test_opened_leaves_match_lod0(self):
        """The closed box nav_doorpose reads from a leaf record holds that leaf's LOD0 mesh (handles stick out a few
        cm), and open_matrix puts the mesh on the swing side beside the hinge, inside the opened footprint."""
        from estate.export import glb
        g, binary = glb.read_glb(self.r["files"]["lod0"])
        W = glb.world_matrices(g)
        index = {n["name"]: i for i, n in enumerate(g["nodes"])}
        by_leaf = {lf["node"]: (d, lf) for d in self.eng["doors"] if d["passable"] for lf in d["leaves"]}
        checked = 0
        for node, (d, rec) in sorted(by_leaf.items()):
            i = index[node]
            v = np.concatenate([glb.from_yup(glb.accessor(g, binary, p["attributes"]["POSITION"]).astype(float))
                                for p in g["meshes"][g["nodes"][i]["mesh"]]["primitives"]])
            Wz = glb.ZUP @ W[i] @ glb.YUP
            closed = v @ Wz[:3, :3].T + Wz[:3, 3]
            box = doorpose.leaf_corners(rec, d["along_wall"], d["swing_side"], opened=False)
            lo, hi = box.min(0) - 0.07, box.max(0) + 0.07
            self.assertTrue(((closed >= lo) & (closed <= hi)).all(), node)
            O = doorpose.open_matrix(rec)
            opened = closed @ O[:3, :3].T + O[:3, 3]
            ring = shapely.Polygon(doorpose.opened_footprint(rec, d["along_wall"], d["swing_side"])).buffer(0.07)
            self.assertTrue(shapely.contains_xy(ring, opened[:, 0], opened[:, 1]).all(), node)
            side = (opened[:, :2] - np.array(rec["pivot"][:2])) @ np.array(d["swing_side"][:2])
            self.assertGreater(float(side.min()), -0.03, node)                  # on the swing side
            checked += 1
        self.assertGreater(checked, 400)

    def test_web_json(self):
        doc, eng = self.doc, self.eng
        self.assertEqual((doc["schema"], doc["site"], doc["frame"]), ("sample-town-n5/web/1", "t_pt4",
                                                                       "block-local, Z up, m"))
        leaves = [lf["node"] for d in eng["doors"] if d["passable"] for lf in d["leaves"]]
        lift = {lf["node"] for d in eng["doors"] if not d["passable"] for lf in d["leaves"]}
        self.assertTrue(lift)
        self.assertEqual(sorted(x["leaf_node"] for x in doc["doors"]), sorted(leaves))
        self.assertFalse(lift & {x["leaf_node"] for x in doc["doors"]})       # lift landing doors stay closed
        for x in doc["doors"]:
            self.assertEqual(len(x["open"]), 12)
            R = np.array(x["open"]).reshape(3, 4)[:, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
            if x["motion"] == "swing":               # a turn about the vertical by the leaf's angle
                self.assertTrue(45.0 <= x["angle"] <= 90.0)
                self.assertAlmostEqual(abs(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), x["angle"], places=3)
            self.assertEqual(x["blocked"][0], x["blocked"][-1])
            self.assertIn(x["grid"], ("blocked", "passthrough"))
        self.assertEqual(leaks.absolute_strings(doc), [])

    def test_leaf_passthrough_share(self):
        w = self.doc["walk"]
        self.assertEqual(w["leaves"], len(self.doc["doors"]))
        self.assertLessEqual(w["leaf_passthrough"], 0.01 * w["leaves"])
        self.assertEqual(w["leaf_blocked"] + w["leaf_passthrough"] + w["leaf_overhead"], w["leaves"])
        self.assertEqual((w["doors_impassable"], w["doors_unlinked"], w["doors_cut"]), (0, 0, 0))
        # t_pt4's flats: the main door's full swing covers the shelter door, and the service yard door's cuts the
        # yard in two, so both open less instead of going through
        narrowed = [x for x in self.doc["doors"] if x.get("angle", 90.0) < 90.0]
        self.assertEqual(len(narrowed), w["leaf_narrowed"])
        self.assertTrue(narrowed and all(x["grid"] == "blocked" for x in narrowed))

    def test_no_side_effects_on_nav(self):
        """nav3d's report is the same before and after the web stage ran in this process (timing and memory
        aside), and the walk grids leave nav3d's own results alone."""
        a, b = (json.loads(json.dumps(r)) for r in (self.nav_before, self.nav_after))
        for r in (a, b):
            r.pop("timing_s")
            r.pop("memory")
        self.assertEqual(a, b)
        self.assertEqual((OUT / "nav_a" / "t_pt4_nav.json").read_bytes().split(b'"timing_s"')[0],
                         (OUT / "nav_b" / "t_pt4_nav.json").read_bytes().split(b'"timing_s"')[0])

    def test_deterministic(self):
        for name in ("t_pt4_walk.bin", "t_pt4_web.json"):
            self.assertEqual(hashlib.sha256((OUT / "a" / name).read_bytes()).hexdigest(),
                             hashlib.sha256((OUT / "b" / name).read_bytes()).hexdigest(), name)

    def test_leak_clean(self):
        for name in ("t_pt4_walk.bin", "t_pt4_web.json"):
            self.assertEqual(leaks.scan_file(OUT / "a" / name), [], name)

    def test_manifest_and_export_info(self):
        """After the stage: the manifest lists each building's walk grid and web JSON (files.walk, files.web), and
        export_info.json hashes that manifest and those files, sums the walk figures, and is the same bytes when
        written again (no clock)."""
        from estate import masterplan
        from estate.export.manifest import write_manifest
        from estate.web.info import write_export_info
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "BLK_501"
            folder.mkdir()
            for ext in ("walk.bin", "web.json"):
                shutil.copy(OUT / "a" / f"t_pt4_{ext}", folder / f"BLK_501_{ext}")
            man = write_manifest(masterplan.resolve(plans=False), out=Path(td) / "estate_manifest.json",
                                 model_dir=Path(td))
            man_sha = hashlib.sha256(man.read_bytes()).hexdigest()
            cfg = config.load()
            p = write_export_info(cfg, model_dir=Path(td))
            first = p.read_bytes()
            self.assertEqual(write_export_info(cfg, model_dir=Path(td)).read_bytes(), first)
            info = json.loads(first)
            d = json.loads(man.read_text(encoding="utf-8"))
        self.assertEqual(info["manifest_sha256"], man_sha)
        b501 = next(s for s in d["sites"] if s["id"] == "BLK_501")
        self.assertEqual(b501["files"]["walk"]["path"], "BLK_501/BLK_501_walk.bin")
        self.assertEqual(b501["files"]["walk"]["sha256"], hashlib.sha256(self.bin).hexdigest())
        self.assertEqual(b501["files"]["web"]["path"], "BLK_501/BLK_501_web.json")
        self.assertEqual(b501["files"]["glb_lod0"], {"path": "BLK_501/BLK_501_lod0.glb", "exists": False})
        self.assertNotIn("walk", d["site"]["files"])
        self.assertEqual(info["files"]["BLK_501/BLK_501_walk.bin"],
                         {"sha256": hashlib.sha256(self.bin).hexdigest(), "bytes": len(self.bin)})
        self.assertEqual(set(info["files"]), {"BLK_501/BLK_501_walk.bin", "BLK_501/BLK_501_web.json"})
        self.assertEqual(info["walk"]["leaves"], self.doc["walk"]["leaves"])
        self.assertEqual((info["walk"]["radius"], info["walk"]["voxel"], info["walk"]["band_pad"]), (0.2, 0.1, 0.25))
        self.assertEqual((info["schema"], info["seed"]), ("sample-town-n5/export-info/1", 20261004))
        self.assertEqual(sorted(info), ["commit", "dirty", "dirty_scope", "files", "manifest_sha256", "schema", "seed",
                                        "tools", "walk"])                  # no date or time of any kind
        self.assertEqual(sorted(info["tools"]), ["blender", "bonsai", "ifcopenshell", "python"])


class StageWiring(unittest.TestCase):
    def test_stage_registered(self):
        """web runs after estate, is switched by [outputs] web, is keyed on its code with nav3d, the door poses and
        the mesh cache, and its records are writers of the manifest and export_info (cmd_build's tail)."""
        from estate import commands
        from estate.pipeline import hooks, state
        cfg = config.load()
        stages = commands.default_stages(cfg)
        self.assertEqual(stages[-2:], ["estate", "web"])
        self.assertNotIn("web", commands.default_stages(dict(cfg, outputs=dict(cfg["outputs"], web=False))))
        self.assertIn("web", hooks.STAGES)
        self.assertEqual(state.STAGE_DEPS["web"], ("estate/web/*.py", "estate/validate/nav3d.py",
                                                   "estate/validate/nav_doorpose.py", "estate/export/meshcache.py"))
        self.assertIn("estate/validate/nav_doorpose.py", state.STAGE_DEPS["glb"])
        nav = {p.resolve() for g in state.STAGE_DEPS["nav"] for p in env.ROOT.glob(g)}
        self.assertIn((env.ROOT / "estate" / "validate" / "nav_doorpose.py").resolve(), nav)
        self.assertEqual(cfg["web"], {"radius": 0.2, "cell": 0.1, "band_pad": 0.25, "max_walk_gz": 131072})

    def test_provenance_order(self):
        """cmd_build's tail writes the manifest, then export_info (which hashes it), when glb, blend, estate or web
        ran; nothing for an ifc-only build."""
        from unittest import mock
        from estate import commands
        calls = []
        with mock.patch("estate.export.manifest.write_manifest", lambda *a, **k: calls.append("manifest")), \
                mock.patch("estate.web.info.write_export_info", lambda *a, **k: calls.append("export_info")):
            failed = []
            commands.write_provenance(["nav", "web"], failed)
            commands.write_provenance(["ifc", "check"], failed)
        self.assertEqual((calls, failed), (["manifest", "export_info"], []))


if __name__ == "__main__":
    unittest.main()
