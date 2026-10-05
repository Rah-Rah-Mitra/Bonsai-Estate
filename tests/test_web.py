"""Web export (estate/web, the web stage): the SN5W walk-grid format against its golden file, storey bands, the grid
as a viewer reads it and the rules stair paths keep on it (walkcheck), synthetic scenes (an unreachable platform, a
stamped doorway, an opened leaf blocked or let through), and the PT4 test block built through the stage's own code:
rooms, doors, spawns and lift landings on the exported grid, stairs against the IFC and on the grid, opened leaves
against the LOD0 leaves, no side effect on nav3d, determinism, the manifest and export_info.json, and the leak scan
of the walk grid. Then the estate's own buildings, where the web stage's outputs under model/ are current: stair
paths walkable, stair feet joined to their doors, doors joined across.

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
from estate.web import sn5w, walk, walkcheck  # noqa: E402

OUT = env.BUILD / "web_tests" / "unittest"
PT4 = env.BUILD / "t_pt4.ifc"
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "web" / "sn5w_sample.bin"
B = sn5w.BLOCKED


# ----------------------------------------------------------------------------- the golden file
def sample_grid() -> dict:
    """The synthetic grid of tests/fixtures/web/sn5w_sample.bin (304 bytes), the contract the portfolio's decoder is
    tested against. 4 x 3 cells for a 0.20 m agent and a 0.40 m step, corner of cell (0, 0) at block-local
    (-1.25, 2.5), flagged as the coarse fallback (cell 0.2 m: a decoder reads the cell from the header, never assumes
    0.1); four storeys, the reference L2 (index 1). The tags are labels (a three-character one, as real files have
    L10..L25). Rasters are absolute mm above each storey's FFL, row iy = 0 first, B = 0x7FFF (no floor):

        L1   ffl 0.0   delta   [   0,  10,   5, -100 ]   overflow (ix 0, iy 0, 2800), (ix 2, iy 1, 2400)
                               [-120,   B, 175,    B ]
                               [   B,   B, 350, 1200 ]
        L2   ffl 3.5   raw     [   0,   0,   5,    B ]   overflow (ix 1, iy 0, -250), (ix 1, iy 0, 2400)
                               [-120,   0, 175,    B ]
                               [   B,   B, 350, 1200 ]
        L12  ffl 6.25  raw     every cell B             no overflow: count and offset 0
        RF   ffl 9.0   same    (L2's raster)            overflow (ix 0, iy 2, 2000)

    Every floor keeps the band rule (whole mm): storey s owns FFL_s - 250 <= ffl + mm < FFL_s+1 - 250, the lowest
    everything below, RF everything above; L2's -250 sits exactly on its lower edge. L1 is stored as int16 (L1 - L2)
    mod 2^16: [0, 10, 0, 32669 | 0, 32767, 0, 0 | 0, 0, 0, 0]; the 32669 is the wrap of -100 - 32767, the 32767 the
    wrap of 32767 - 0. Byte layout: header 0..63 (flags 3: a delta layer, coarse), table 64..191, L1 raster
    192..215, L1 overflow 216..231 (sorted by iy, ix, mm), L2 raster 232..255, L2 overflow 256..271, L12 raster
    272..295, RF overflow 296..303 (RF has no raster: offset and length 0)."""
    ref = [0, 0, 5, B, -120, 0, 175, B, B, B, 350, 1200]
    return dict(nx=4, ny=3, cell=0.2, radius=0.2, step=0.4, origin=(-1.25, 2.5), ref=1, coarse=True,
                modes=["delta", "raw", "raw", "same"],
                layers=[dict(tag="L1", ffl=0.0, raster=[0, 10, 5, -100, -120, B, 175, B, B, B, 350, 1200],
                             overflow=[(2, 1, 2400), (0, 0, 2800)]),
                        dict(tag="L2", ffl=3.5, raster=ref, overflow=[(1, 0, 2400), (1, 0, -250)]),
                        dict(tag="L12", ffl=6.25, raster=[B] * 12, overflow=[]),
                        dict(tag="RF", ffl=9.0, raster=list(ref), overflow=[(0, 2, 2000)])])


def sample_bytes() -> bytes:
    s = sample_grid()
    return sn5w.encode(s["nx"], s["ny"], s["cell"], s["radius"], s["step"], s["origin"], s["layers"], s["ref"],
                       modes=s["modes"], coarse=s["coarse"])


def band_errors(t) -> list:
    """Floors of a decoded grid (main and overflow) outside their layer's band, in whole mm: FFL_s - 250 <= FFL_s +
    mm < FFL_s+1 - 250 (no lower edge for the lowest layer, no upper one for the top)."""
    ffl = [int(round(lay["ffl"] * 1000)) for lay in t["layers"]]
    bad = []
    for li, lay in enumerate(t["layers"]):
        lo = ffl[li] - 250 if li else -(1 << 40)
        hi = ffl[li + 1] - 250 if li + 1 < len(ffl) else 1 << 40
        mms = [v for v in lay["raster"] if v != B] + [mm for _, _, mm in lay["overflow"]]
        bad += [(lay["tag"], mm) for mm in mms if not lo <= ffl[li] + mm < hi]
    return bad


class Sn5wFormat(unittest.TestCase):
    def test_golden_file(self):
        data = sample_bytes()
        if os.environ.get("SN5W_REGEN"):
            GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN.write_bytes(data)
        self.assertEqual(data, GOLDEN.read_bytes())
        self.assertEqual(len(data), 304)

    def test_layout_byte_for_byte(self):
        """The golden file read with struct alone, field by field, as the format's description in sn5w.py says."""
        d = GOLDEN.read_bytes()
        self.assertEqual(d[:4], b"SN5W")
        self.assertEqual(struct.unpack_from("<HH", d, 4), (1, 3))                    # version 1, flags: delta, coarse
        cell, radius, step, ox, oy = struct.unpack_from("<5f", d, 8)
        self.assertEqual((ox, oy), (-1.25, 2.5))
        self.assertEqual([round(v, 6) for v in (cell, radius, step)], [0.2, 0.2, 0.4])
        self.assertEqual(struct.unpack_from("<4H", d, 28), (4, 3, 4, 1))           # nx, ny, layers, ref
        self.assertEqual(d[36:64], b"\0" * 28)
        table = [struct.unpack_from("<4sfBBHIIIII", d, 64 + 32 * i) for i in range(4)]
        self.assertEqual(table, [(b"L1\0\0", 0.0, 1, 0, 0, 192, 24, 2, 216, 0),
                                 (b"L2\0\0", 3.5, 0, 0, 0, 232, 24, 2, 256, 0),
                                 (b"L12\0", 6.25, 0, 0, 0, 272, 24, 0, 0, 0),
                                 (b"RF\0\0", 9.0, 2, 0, 0, 0, 0, 1, 296, 0)])
        self.assertEqual(list(struct.unpack_from("<12h", d, 192)), [0, 10, 0, 32669, 0, 32767, 0, 0, 0, 0, 0, 0])
        self.assertEqual(list(struct.unpack_from("<12h", d, 232)), [0, 0, 5, B, -120, 0, 175, B, B, B, 350, 1200])
        self.assertEqual(list(struct.unpack_from("<12h", d, 272)), [B] * 12)
        self.assertEqual([struct.unpack_from("<HHhH", d, o) for o in (216, 224, 256, 264, 296)],
                         [(0, 0, 2800, 0), (2, 1, 2400, 0), (1, 0, -250, 0), (1, 0, 2400, 0), (0, 2, 2000, 0)])

    def test_round_trip(self):
        """The golden file decodes to the sample grid, and every floor in it keeps the band rule it documents."""
        t = sn5w.decode(GOLDEN.read_bytes())
        s = sample_grid()
        self.assertEqual((t["nx"], t["ny"], t["ref"], t["delta"], t["coarse"]), (4, 3, 1, True, True))
        for got, want, mode in zip(t["layers"], s["layers"], s["modes"]):
            self.assertEqual((got["tag"], got["ffl"], got["mode"]), (want["tag"], want["ffl"], mode))
            self.assertEqual(list(got["raster"]), want["raster"])
            self.assertEqual(got["overflow"], sorted(want["overflow"], key=lambda r: (r[1], r[0], r[2])))
        self.assertEqual(len(t["layers"]), len(s["layers"]))
        self.assertEqual(band_errors(t), [])

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
        tag, a section moved, a 'same' layer with a raster, an overflow offset with no records."""
        d = GOLDEN.read_bytes()
        self.assertEqual(sn5w.layer_tags(d), ["L1", "L2", "L12", "RF"])

        def poke(off, b):
            return d[:off] + b + d[off + len(b):]
        for bad in (d + b"\0", poke(40, b"x"), poke(64 + 9, b"\1"), poke(64, b"\0"), poke(296 + 6, b"\1"),
                    poke(64 + 32 + 12, struct.pack("<I", 234)), poke(64 + 96 + 16, struct.pack("<I", 24)), d[:-1],
                    poke(4, struct.pack("<H", 2)), poke(64 + 64 + 24, struct.pack("<I", 296))):
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
        """Bands are decided in whole millimetres, the file's unit: a floor exactly at FFL_s - 0.25 belongs to storey
        s, and so does one a micron either side of it (float noise: MSCP_513's ramps meet three band edges at
        edge - 1.7e-6 and edge + 1.7e-6); a millimetre lower belongs to s - 1. Below the lowest storey is the lowest
        storey's, above the top one the top one's."""
        ffl = [0.0, 3.6, 6.4, 9.2]
        for s in range(1, 4):
            edge = ffl[s] - 0.25
            for z in (edge, edge - 1e-6, edge + 1e-6, edge - 1.713e-06, edge + 1.712e-06):
                self.assertEqual(int(walk.band_of(z, ffl, 0.25)), s, (s, z))
            self.assertEqual(int(walk.band_of(edge - 0.001, ffl, 0.25)), s - 1)
        self.assertEqual(int(walk.band_of(-2.0, ffl, 0.25)), 0)
        self.assertEqual(int(walk.band_of(30.0, ffl, 0.25)), 3)

    def test_layers_at_a_band_edge(self):
        """A floor a micron under a band edge (as MSCP_513's ramps meet theirs) is stored in the upper storey at
        -250 mm, as one a micron over it is, never in the lower one at its band's top (+2750 there)."""
        for z in (-1.713e-06, 1.712e-06):
            w = _scene([(0, 0, z - 0.2, 6, 3, z)])
            self.assertTrue(len(w.fz) and np.abs(w.fz - z).max() < 1e-5)
            nx, ny, origin, lays = walk.layers(w, np.ones(len(w.ix), bool), [("L2", -2.75), ("L3", 0.25)],
                                               np.zeros(3), 0.25)
            self.assertFalse((np.asarray(lays[0]["raster"]) != B).any(), z)
            got = set(np.asarray(lays[1]["raster"])[np.asarray(lays[1]["raster"]) != B].tolist())
            self.assertEqual(got, {-250}, z)


# ----------------------------------------------------------------------------- the grid as a viewer reads it
def ramp(blocked=(), out_of_band=False) -> bytes:
    """A one-flight stair on 12 x 3 cells of 0.1 m, corner of cell (0, 0) at block-local (0, 0): every row rises
    from 0.0 (ix 0, 1) by 0.1 m a cell to 1.0 (ix 11). L1 at FFL 0.0, L2 at 1.0, so the band edge is 0.75: ix 0..8
    are L1 floors (0 .. 700 mm), ix 9..11 L2 floors (-200 .. 0 mm). ``blocked``: cells (ix, iy) with no floor;
    ``out_of_band``: ix 9 is filed in L1 at +800 instead."""
    h = [max(0, ix - 1) * 100 for ix in range(12)]
    l1, l2 = [B] * 36, [B] * 36
    for iy in range(3):
        for ix in range(12):
            if (ix, iy) in blocked:
                continue
            if h[ix] < 750 or (out_of_band and ix == 9):
                l1[iy * 12 + ix] = h[ix]
            else:
                l2[iy * 12 + ix] = h[ix] - 1000
    return sn5w.encode(12, 3, 0.1, 0.2, 0.4, (0.0, 0.0), [dict(tag="L1", ffl=0.0, raster=l1),
                                                         dict(tag="L2", ffl=1.0, raster=l2)], 0)


def ramp_stair(**kw) -> dict:
    """The stair of ramp(): up the middle row, one point per cell centre."""
    path = [[ix * 0.1 + 0.05, 0.15, max(0, ix - 1) * 0.1] for ix in range(12)]
    return dict(dict(name="S", storey="L1", to="L2", from_ffl=0.0, to_ffl=1.0, path=path), **kw)


class WalkCheck(unittest.TestCase):
    """estate/web/walkcheck.py: the viewer's floor search and snap on the golden file, and each rule a stair path
    is held to, one by one, on a ramp."""

    def test_grid_matches_decode(self):
        """Grid (numpy) holds exactly the floors sn5w.decode (stdlib) finds, cell by cell."""
        t = sn5w.decode(GOLDEN.read_bytes())
        g = walkcheck.Grid(GOLDEN.read_bytes())
        for iy in range(t["ny"]):
            for ix in range(t["nx"]):
                want = []
                for li, lay in enumerate(t["layers"]):
                    v = lay["raster"][iy * t["nx"] + ix]
                    want += [(round(lay["ffl"] + v / 1000.0, 6), li)] if v != B else []
                    want += [(round(lay["ffl"] + mm / 1000.0, 6), li) for x, y, mm in lay["overflow"]
                             if (x, y) == (ix, iy)]
                self.assertEqual(sorted((round(f, 6), li) for f, li in g.floors(ix, iy)), sorted(want), (ix, iy))
        s = g.spans()
        self.assertEqual(len(s["ix"]), sum(1 for lay in t["layers"] for v in lay["raster"] if v != B)
                         + sum(len(lay["overflow"]) for lay in t["layers"]))

    def test_floor_at_and_nearest(self):
        """floorAt: the floor closest to the feet within the step, ties to the higher, none beyond the step; the cell
        is floor((x - origin) / cell). nearestWalkable: the point itself on a floor, else the nearest cell centre with
        one (ties to the lower iy), none beyond the reach."""
        g = walkcheck.Grid(GOLDEN.read_bytes())          # cell (1, 0): L1 0.01; L2 3.5, 3.25 and 5.9; RF 9.0
        x, y = -0.95, 2.6
        self.assertEqual(g.cell_of(x, y), (1, 0))
        self.assertEqual(g.cell_of(-1.25 + 0.2 - 1e-9, 2.5), (0, 0))
        self.assertEqual(g.floor_at(x, y, 0.3), (0.01, 0))
        f, layer = g.floor_at(x, y, 3.4)
        self.assertEqual((round(f, 6), layer), (3.5, 1))
        f, layer = g.floor_at(x, y, 3.375)                # 3.25 and 3.5 equally far: the higher
        self.assertEqual((round(f, 6), layer), (3.5, 1))
        self.assertIsNone(g.floor_at(x, y, 1.0))
        self.assertIsNone(g.floor_at(-2.0, y, 0.0))      # off the grid
        self.assertEqual(g.nearest(x, y, 0.0, 0.25)[:2], (x, y))
        nx_, ny_, f, layer = g.nearest(-0.95, 2.78, 0.0, 0.25)  # cell (1, 1) has no L1 floor; (1, 0) is nearest
        self.assertEqual((round(nx_, 6), round(ny_, 6), round(f, 6), layer), (-0.95, 2.6, 0.01, 0))
        self.assertIsNone(g.nearest(-0.95, 2.78, 0.0, 0.15))
        self.assertEqual([g.band(z) for z in (-1.0, 3.249, 3.25, 3.2499996, 5.999, 6.0, 8.75, 20.0)],
                         [0, 0, 1, 1, 1, 2, 3, 3])

    def test_stair_rules(self):
        """A ramp walked cell by cell passes; each rule fails on its own: a first point on a blocked cell (which
        still snaps within 0.1 m: the snap alone would not see it), a blocked cell under the line, a point far off
        the grid, a last point short of the next storey, a rise of more than a step, a storey the grid lacks."""
        g = walkcheck.Grid(ramp())
        self.assertEqual(walkcheck.stair_errors(g, [ramp_stair()]), [])
        errs = walkcheck.stair_errors(walkcheck.Grid(ramp(blocked={(0, 1)})), [ramp_stair()])
        self.assertEqual(len(errs), 2, errs)
        self.assertIn("sample(s) every 0.05 m along the path with no floor under them, first (0.050, 0.150, 0.000)",
                      errs[0])
        self.assertIn("the first point (0.050, 0.150, 0.000) stands on no floor, not L1 at 0.000", errs[1])
        errs = walkcheck.stair_errors(walkcheck.Grid(ramp(blocked={(5, 1)})), [ramp_stair()])
        self.assertEqual(len(errs), 1, errs)
        self.assertIn("with no floor under them", errs[0])
        stair = ramp_stair()
        stair["path"][5] = [0.55, 0.6, 0.4]
        errs = walkcheck.stair_errors(g, [stair])
        self.assertTrue(any("no walkable cell of their band within 0.1 m, first path[5] (0.550, 0.600, 0.400) (L1)"
                            in e for e in errs), errs)
        errs = walkcheck.stair_errors(g, [ramp_stair(path=ramp_stair()["path"][:9])])
        self.assertEqual(errs, ["S: the last point (0.850, 0.150, 0.700) stands on 0.700 on L1, not L2 at 1.000"])
        stair = ramp_stair()
        stair["path"] = stair["path"][:2] + stair["path"][7:]
        self.assertTrue(any("rises more than a step" in e for e in walkcheck.stair_errors(g, [stair])))
        self.assertTrue(any("'L9' is not a layer" in e for e in walkcheck.stair_errors(g, [ramp_stair(to="L9")])))
        self.assertEqual(walkcheck.stair_errors(g, [ramp_stair(path=[[0.05, 0.15, 0.0]])]),
                         ["S: a path of 1 point(s)"])

    def test_bands(self):
        """A floor filed in a layer outside that layer's band is reported (a viewer prunes layers by band); the
        golden file and the ramp keep the rule."""
        self.assertEqual(walkcheck.band_errors(walkcheck.Grid(GOLDEN.read_bytes())), [])
        self.assertEqual(walkcheck.band_errors(walkcheck.Grid(ramp())), [])
        self.assertEqual(walkcheck.band_errors(walkcheck.Grid(ramp(out_of_band=True))),
                         ["layer L1: 3 floor(s) outside its band [-inf, 750) mm, e.g. 800 mm"])
        self.assertEqual(walkcheck.site_errors(ramp(), {"stairs": [ramp_stair()]}), [])


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

    def test_door_reach_seeds_from_the_doorway(self):
        """door_reach measures what the way through the opening reaches: a side cell cut off from it (the corner
        between an opened leaf and its jamb) seeds nothing, so the area cut off with it is lost, not reached. Here
        held cells cut the part of side b right of x = 2.6 off from the opening; side cells of b lie in it, and when
        every side cell seeded it passed as reached (BLK_509's void-deck stairs, whose leaves at 75 degrees cut the
        first flight off from the door and still counted as whole)."""
        d = _door(2.0, 0.9)
        w = _scene(_wall_with_door(2.0, 0.9), [d])
        a, b = w.sides[0]
        x, y = np.round(w.cx(), 3), np.round(w.cy(), 3)
        held = (x >= 2.6) & (((y >= 1.75) & (y <= 2.05)) | ((x == 2.6) & (y >= 1.75)))
        pocket = (x >= 2.7) & (y >= 2.1) & (np.abs(w.fz) < 0.1)
        self.assertTrue(pocket[b].any())                                   # side cells of b lie in the pocket
        region = np.arange(len(w.ix))
        self.assertTrue(walk.connected(w, region, a, b, held))
        joined, reach = walk.door_reach(w, region, a, b), walk.door_reach(w, region, a, b, held)
        self.assertTrue(np.isin(np.nonzero(pocket)[0], joined).all() and pocket.sum() > 100)
        self.assertFalse(pocket[reach].any())
        self.assertTrue(((x < 2.5) & (y > 1.9))[reach].any())             # the rest of side b is still reached

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


# ----------------------------------------------------------------------------- floors of an exported grid
def grid_floors(data: bytes) -> dict:
    """Every floor of an SN5W file (walkcheck.Grid.spans, raster and overflow): arrays ix, iy, x, y (cell centres,
    block-local), z (absolute) and layer; ``order``/``xs`` index them by x for disc()."""
    g = walkcheck.Grid(data)
    s = g.spans()
    x = g.ox + (s["ix"] + 0.5) * g.cell
    order = np.argsort(x, kind="stable")
    return dict(ix=s["ix"], iy=s["iy"], x=x, y=g.oy + (s["iy"] + 0.5) * g.cell,
                z=np.asarray(g.ffl)[s["layer"]] + s["mm"] / 1000.0, layer=s["layer"], order=order, xs=x[order])


def disc(f, x, y, r):
    """The floors whose cell centre lies within r (plan) of (x, y), in index order."""
    a, b = np.searchsorted(f["xs"], [x - r - 1e-9, x + r + 1e-9])
    idx = np.sort(f["order"][a:b])
    return idx[np.hypot(f["x"][idx] - x, f["y"][idx] - y) <= r]


def components(f, idx, step=0.4):
    """Component labels of the floors ``idx``: 4-adjacent cells (main or overflow) whose floors differ by at most a
    step."""
    n = len(idx)
    ix, iy, z = f["ix"][idx].astype(np.int64), f["iy"][idx].astype(np.int64), f["z"][idx]
    ny = int(iy.max()) + 2 if n else 1
    key = ix * ny + iy                                    # (ix + 1, iy) is key + ny, (ix, iy + 1) is key + 1
    order = np.argsort(key, kind="stable")
    skey = key[order]
    ea, eb = [np.zeros(0, np.int64)], [np.zeros(0, np.int64)]
    for d in (ny, 1):
        lo, hi = np.searchsorted(skey, key + d, "left"), np.searchsorted(skey, key + d, "right")
        for j in range(int((hi - lo).max()) if n else 0):
            sel = np.nonzero(hi - lo > j)[0]
            other = order[lo[sel] + j]
            ok = np.abs(z[sel] - z[other]) <= step + 1e-6
            ea.append(sel[ok])
            eb.append(other[ok])
    return nav3d.components(n, np.concatenate(ea), np.concatenate(eb))


def door_front(doors, p, f, idx, reach=(0.1, 0.8)):
    """(along, across, the floors of idx in front of portal p's opening on each side): within its clear width
    (``doors``: the engine JSON's doors by node), ``reach`` m off its threshold, at its floor. A door's sides are
    probed here, not at the portal's ``link`` points: those sit at the middle of the rooms' edges, which a counter can
    fill (NC_514's stall shutters)."""
    (t0x, t0y, _), (t1x, t1y, _) = p["threshold"]
    along = np.array([t1x - t0x, t1y - t0y])
    wl = float(np.linalg.norm(along))
    along /= wl
    mid = np.array([(t0x + t1x) / 2, (t0y + t1y) / 2])
    rel = np.stack([f["x"][idx] - mid[0], f["y"][idx] - mid[1]], 1)
    u, v = rel @ along, rel @ np.array([-along[1], along[0]])
    ok = (np.abs(u) <= (doors[p["node"]].get("clear_width") or wl) / 2) & (np.abs(f["z"][idx] - p["floor_z"]) <= 0.3)
    return u, v, (np.nonzero(ok & (v < -reach[0]) & (v >= -reach[1]))[0],
                  np.nonzero(ok & (v > reach[0]) & (v <= reach[1]))[0])


def door_cuts(eng, f) -> tuple[int, list]:
    """(passable doors, those whose two sides do not join within 1.5 m of the opening on the grid ``f``): the floors
    in front of the opening (door_front) on one side and the other fall in one component of the floors within 1.5 m
    (plan) and 1 m (height) of it, with the opened leaves blocked as exported."""
    doors = {d["node"]: d for d in eng["doors"]}
    bad, n = [], 0
    for p in eng["portals"]:
        if not p["passable"]:
            continue
        n += 1
        (t0x, t0y, _), (t1x, t1y, _) = p["threshold"]
        idx = disc(f, (t0x + t1x) / 2, (t0y + t1y) / 2, 1.5)
        idx = idx[np.abs(f["z"][idx] - p["floor_z"]) <= 1.0]
        lab = components(f, idx)
        _, _, (sa, sb) = door_front(doors, p, f, idx)
        if not (len(sa) and len(sb) and set(lab[sa].tolist()) & set(lab[sb].tolist())):
            bad.append(p["node"])
    return n, bad


def stair_door_cuts(doc, eng, f) -> tuple[int, list]:
    """(stair doors, cuts): for every stair of the web JSON ``doc`` and every passable door into its room on its
    storey, the start of its path (the landing point and the foot of the first flight) must be joined to the stair
    side of the door's opening within 4 m on the grid ``f``: an opened leaf never cuts a flight off from its own
    door (the void-deck fire doors of BLK_507..512 did, at 75 degrees)."""
    doors = {d["node"]: d for d in eng["doors"]}
    bad, n = [], 0
    for s in doc["stairs"]:
        for p in eng["portals"]:
            if not (p["passable"] and p["storey"] == s["storey"] and s["room"] in p["space_names"]):
                continue
            n += 1
            (t0x, t0y, _), (t1x, t1y, _) = p["threshold"]
            mx, my = (t0x + t1x) / 2, (t0y + t1y) / 2
            idx = disc(f, mx, my, 4.0)
            idx = idx[(f["z"][idx] > p["floor_z"] - 1.0) & (f["z"][idx] < p["floor_z"] + 2.0)]
            lab = components(f, idx)
            u, _, sides = door_front(doors, p, f, idx)
            lx, ly, _ = p["link"][p["space_names"].index(s["room"])]
            side = sides[int(np.dot([lx - mx, ly - my], [-(t1y - t0y), t1x - t0x]) > 0)]
            doorway = set(lab[side[np.argsort(np.abs(u[side]), kind="stable")[:3]]].tolist())
            for q in s["path"][:2]:
                near = np.nonzero((np.hypot(f["x"][idx] - q[0], f["y"][idx] - q[1]) <= 0.3)
                                  & (np.abs(f["z"][idx] - q[2]) <= 0.25))[0]
                if not (len(side) and len(near) and doorway & set(lab[near].tolist())):
                    bad.append((s["name"], p["node"], q))
    return n, bad


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
        cls.floors = grid_floors(cls.bin)

    @staticmethod
    def _floors(t):
        """Every floor of a grid sn5w.decode read (the stdlib reader): arrays x, y (cell centres, block-local), z
        (absolute), layer index; what grid_floors (numpy) must agree with."""
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

    def test_grid_floors_match_decode(self):
        """The numpy reader the checks below and estate/web/walkcheck.py use (grid_floors, walkcheck.Grid) finds
        exactly the floors the stdlib reader does (sn5w.decode), delta and same layers included; and the components
        of the floors around a door are the same, built by vector or by loop."""
        ref, f = self._floors(self.grid), self.floors
        key = lambda d: sorted(zip(np.round(d["x"], 4).tolist(), np.round(d["y"], 4).tolist(),  # noqa: E731
                                   np.round(d["z"], 4).tolist(), d["layer"].tolist()))
        self.assertEqual(key(ref), key(f))
        p = next(p for p in self.eng["portals"] if p["passable"])
        (t0x, t0y, _), (t1x, t1y, _) = p["threshold"]
        idx = disc(f, (t0x + t1x) / 2, (t0y + t1y) / 2, 1.5)
        lab, pos = components(f, idx), {}
        for k, c in enumerate(zip(f["ix"][idx].tolist(), f["iy"][idx].tolist())):
            pos.setdefault(c, []).append(k)
        for (i, j), ks in pos.items():                    # every step between neighbours stays in one component
            for k in ks:
                for k2 in pos.get((i + 1, j), []) + pos.get((i, j + 1), []):
                    if abs(f["z"][idx[k]] - f["z"][idx[k2]]) <= 0.4:
                        self.assertEqual(lab[k], lab[k2])
        self.assertGreater(len(set(lab.tolist())), 1)       # and floors a storey apart do not join

    def test_doors_connect_with_leaves_blocked(self):
        """Every passable door joins its two sides within 1.5 m of its opening on the exported grid, with the opened
        leaves blocked (door_cuts)."""
        n, bad = door_cuts(self.eng, self.floors)
        self.assertGreater(n, 400)
        self.assertEqual(bad, [])

    def test_stair_paths_meet_their_doors(self):
        """For every stair and every passable door into its room on its storey, the start of its path is joined to
        the stair side of the door's opening within 4 m (stair_door_cuts)."""
        n, bad = stair_door_cuts(self.doc, self.eng, self.floors)
        self.assertGreater(n, 10)
        self.assertEqual(bad, [])

    def test_walkcheck(self):
        """The written files keep what the stage checked before writing them (estate/web/walkcheck.py, as the release
        checks them again): every stair path walkable on the grid as a viewer decodes it, every floor in its band."""
        self.assertEqual(walkcheck.site_errors(self.bin, self.doc), [])
        self.assertGreater(len(self.doc["stairs"]), 20)

    def test_bands(self):
        """Every floor of the exported grid keeps the band rule, in whole mm."""
        self.assertEqual(band_errors(self.grid), [])

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
        0.1 m of the next one, standing on the walk grid all the way: every 0.05 m of it has a floor within 0.4 m (a
        viewer's floorAt) in its own cell."""
        import ifcopenshell
        import ifcopenshell.util.element as uel
        f = ifcopenshell.open(str(PT4))
        by_name = {s.Name: s for s in f.by_type("IfcStair")}
        self.assertEqual(sorted(s["name"] for s in self.doc["stairs"]), sorted(by_name))
        t = self.grid
        (ox, oy), c, cols = t["origin"], t["cell"], {}
        for x, y, z in zip(self.floors["x"], self.floors["y"], self.floors["z"]):
            cols.setdefault((int(np.floor((x - ox) / c)), int(np.floor((y - oy) / c))), []).append(z)
        off_grid, total = [], 0
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
            for p, q in zip(path[:-1], path[1:]):
                for u in np.linspace(0.0, 1.0, max(2, int(np.ceil(np.hypot(*(q[:2] - p[:2])) / 0.05)) + 1)):
                    x, y, z = p + u * (q - p)
                    total += 1
                    here = cols.get((int(np.floor((x - ox) / c)), int(np.floor((y - oy) / c))), ())
                    if not any(abs(h - z) <= 0.4 for h in here):
                        off_grid.append((s["name"], round(float(x), 3), round(float(y), 3), round(float(z), 3)))
        self.assertGreater(total, 1000)
        self.assertEqual(off_grid, [])
        self.assertEqual(self.doc["walk"]["stair_off_grid"], 0)

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
        # a .blend by path alone: Blender saves one scene as different bytes each time, and none is released
        self.assertEqual(b501["files"]["blend"], {"path": "BLK_501/BLK_501.blend", "exists": False})
        self.assertEqual(d["site"]["files"]["estate_blend"], {"path": "ESTATE.blend", "exists": False})
        self.assertNotIn("walk", d["site"]["files"])
        self.assertEqual(info["files"]["BLK_501/BLK_501_walk.bin"],
                         {"sha256": hashlib.sha256(self.bin).hexdigest(), "bytes": len(self.bin)})
        self.assertEqual(set(info["files"]), {"BLK_501/BLK_501_walk.bin", "BLK_501/BLK_501_web.json"})
        self.assertEqual(info["walk"]["leaves"], self.doc["walk"]["leaves"])
        self.assertEqual(info["walk"]["stair_off_grid"], 0)
        self.assertEqual((info["walk"]["radius"], info["walk"]["voxel"], info["walk"]["band_pad"]), (0.2, 0.1, 0.25))
        self.assertEqual((info["schema"], info["seed"]), ("sample-town-n5/export-info/1", 20261004))
        self.assertEqual(sorted(info), ["commit", "dirty", "dirty_scope", "files", "manifest_sha256", "schema", "seed",
                                        "tools", "walk"])                  # no date or time of any kind
        self.assertEqual(sorted(info["tools"]), ["blender", "bonsai", "ifcopenshell", "python"])


# ----------------------------------------------------------------------------- the estate's own buildings
def current_buildings() -> list:
    """[(id, folder)] of the buildings under model/ whose walk grid and web JSON are current: their web stage
    record in model/.state.json carries the key the stage would compute now (its code, [web], the agent, and the
    IFC and engine JSON on disk), and the files are there."""
    from estate.pipeline import hooks, runner
    from estate.pipeline import state as st
    cfg = config.load()
    state = st.load()
    out = []
    for t in runner.targets(cfg, None, include_site=False):
        eng = t.folder / f"{t.id}_engine.json"
        if t.kind == "site" or not (t.ifc.exists() and eng.exists()):
            continue
        if st.fresh(state, t.id, "web", hooks.web_key(cfg, t.id, t.ifc, eng)):
            out.append((t.id, t.folder))
    return out


class EstateBuildings(unittest.TestCase):
    """The walk grids and web JSONs the web stage wrote under model/ for the estate's own buildings, those still
    current (current_buildings; skipped when there are none: ./estate.sh build --stages nav,glb,web): the slab,
    L-block and EA void-deck stairs, the car park's ramps and the hawker centre's shop-block stairs that the test block
    does not have. Every stair path walkable on the grid as a viewer decodes it and every floor in its band
    (walkcheck, as the release checks them); every stair's foot joined to each door into its room (an opened fire
    door never cuts a flight off); every passable door joined across."""

    @classmethod
    def setUpClass(cls):
        cls.sites = current_buildings()
        cls.cache = {}

    def setUp(self):
        if not self.sites:
            self.skipTest("no current walk grid under model/ (./estate.sh build --stages nav,glb,web)")

    def load(self, sid, folder):
        if sid not in self.cache:
            data = (folder / f"{sid}_walk.bin").read_bytes()
            doc = json.loads((folder / f"{sid}_web.json").read_text(encoding="utf-8"))
            eng = json.loads((folder / f"{sid}_engine.json").read_text(encoding="utf-8"))
            self.cache[sid] = (data, doc, eng, grid_floors(data))
        return self.cache[sid]

    def test_stair_paths_walkable(self):
        bad, stairs = {}, 0
        for sid, folder in self.sites:
            data, doc, _, _ = self.load(sid, folder)
            stairs += len(doc["stairs"])
            errs = walkcheck.site_errors(data, doc)
            if errs or doc["walk"]["stair_off_grid"]:
                bad[sid] = errs or [f"stair_off_grid {doc['walk']['stair_off_grid']}"]
        self.assertGreater(stairs, 0)
        self.assertEqual(bad, {})

    def test_stair_feet_meet_their_doors(self):
        bad, n = {}, 0
        for sid, folder in self.sites:
            _, doc, eng, f = self.load(sid, folder)
            k, cut = stair_door_cuts(doc, eng, f)
            n += k
            if cut:
                bad[sid] = cut[:6]
        self.assertGreater(n, 0)
        self.assertEqual(bad, {})

    def test_doors_connect(self):
        bad, n = {}, 0
        for sid, folder in self.sites:
            _, doc, eng, f = self.load(sid, folder)
            k, cut = door_cuts(eng, f)
            n += k
            if cut:
                bad[sid] = cut[:6]
            w = doc["walk"]
            self.assertLessEqual(w["leaf_passthrough"], 0.01 * w["leaves"], sid)
            self.assertEqual((w["doors_impassable"], w["doors_unlinked"], w["doors_cut"]), (0, 0, 0), sid)
        self.assertGreater(n, 0)
        self.assertEqual(bad, {})


class StageWiring(unittest.TestCase):
    def test_stage_registered(self):
        """web runs after estate, is switched by [outputs] web, is keyed on its code (not export_info's or the
        release's, which shape none of its outputs) with nav3d, the door poses and the mesh cache, and its records
        are writers of the manifest and export_info (cmd_build's tail)."""
        from estate import commands
        from estate.pipeline import hooks, state
        cfg = config.load()
        stages = commands.default_stages(cfg)
        self.assertEqual(stages[-2:], ["estate", "web"])
        self.assertNotIn("web", commands.default_stages(dict(cfg, outputs=dict(cfg["outputs"], web=False))))
        self.assertIn("web", hooks.STAGES)
        self.assertEqual(state.STAGE_DEPS["web"], ("estate/web/__init__.py", "estate/web/export.py",
                                                   "estate/web/walk.py", "estate/web/sn5w.py", "estate/web/stairs.py",
                                                   "estate/web/walkcheck.py", "estate/validate/nav3d.py",
                                                   "estate/validate/nav_doorpose.py", "estate/export/meshcache.py"))
        web = {p.name for g in state.STAGE_DEPS["web"] for p in env.ROOT.glob(g)}
        self.assertEqual(web & {"info.py", "release.py"}, set())
        self.assertEqual({p.name for p in (env.ROOT / "estate" / "web").glob("*.py")} - web, {"info.py", "release.py"})
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
