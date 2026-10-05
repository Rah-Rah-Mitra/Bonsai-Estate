"""Blender stage tests: LINKED_MODEL references, the blender.exe job driver, .blend round trips, the estate link
and the eye-level street camera (estate/report/street.py on model/masterplan.json + SITE_graph.json).

The fast tests need only ifcopenshell. The Blender tests launch blender.exe (Bonsai 0.9) in the background and
take about a minute in all; set ESTATE_SKIP_BLENDER=1 to skip them. They work on copies under
build/blender_tests/ so relative paths are exercised and nothing in model/ is touched.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estate.env import bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402
import ifcopenshell  # noqa: E402
from shapely.geometry import Point, box, shape  # noqa: E402

from estate import config, env  # noqa: E402
from estate.geom.walls import rotate_z, translate  # noqa: E402
from estate.ifc.links import add_linked_models, decode_matrix, encode_matrix, linked_models  # noqa: E402
from estate.ifc.writer import IfcWriter  # noqa: E402

TESTS = env.BUILD / "blender_tests"
PT4 = env.BUILD / "t_pt4.ifc"
LEGACY = env.BUILD / "legacy" / "hdb_block_legacy.ifc"
CACHE_SUFFIXES = (".ifc.cache.blend", ".ifc.cache.json", ".ifc.cache.sqlite")
SKIP_BLENDER = os.environ.get("ESTATE_SKIP_BLENDER") == "1" or not env.BLENDER_EXE.exists()


def write_site(path: Path, entries: list, guid_key="test/SITE") -> Path:
    """A host SITE.ifc (shared project/site identity, a ground slab) with LINKED_MODEL references."""
    W = IfcWriter(guid_key=guid_key, **config.identity(config.load()))
    W.slab(box(-40.0, -40.0, 90.0, 40.0), 0.0, "Ground", W.site, predefined="BASESLAB", mat="slab", depth=0.3)
    W.flush()
    add_linked_models(W, entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    W.write(path)
    W.close()
    return path


def georeference_copy(path: Path) -> None:
    """Give a building IFC the estate's IfcMapConversion (as every federated file has; t_pt4/legacy have none)."""
    import ifcopenshell.api.georeference as georeference
    g = config.identity(config.load())["georef"]
    f = ifcopenshell.open(str(path))
    if not f.by_type("IfcMapConversion"):
        georeference.add_georeferencing(f, ifc_class="IfcMapConversion", name=f"EPSG:{g['epsg']}")
        georeference.edit_georeferencing(f, projected_crs={"Name": f"EPSG:{g['epsg']}"}, coordinate_operation={
            "Eastings": float(g["eastings"]), "Northings": float(g["northings"]), "OrthogonalHeight": float(g["height"])})
        f.write(str(path))


def build_spike_model(folder: Path, sources: dict, matrices: dict | None = None, georef: bool = True) -> Path:
    """Copy building IFCs to <folder>/<NAME>/<NAME>.ifc (dropping stale Bonsai caches) and write <folder>/SITE.ifc.

    With georef (default) the copies get the host's IfcMapConversion: Bonsai places a link at
    inv(host model origin) @ Identification @ (link model origin), so a building without the shared map
    conversion lands at (-E, -N, -H) from the site.
    """
    entries = []
    for name, src in sources.items():
        dst = folder / name / f"{name}.ifc"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if georef:
            georeference_copy(dst)
        for suf in CACHE_SUFFIXES:
            dst.with_suffix(suf).unlink(missing_ok=True)
        entries.append(dict(path=f"{name}/{name}.ifc", name=name, matrix=(matrices or {}).get(name)))
    return write_site(folder / "SITE.ifc", entries)


class TestLinks(unittest.TestCase):
    def test_encoding_matches_bonsai(self):
        m = translate(45.0, 0.0, 0.0) @ rotate_z(1)
        s = encode_matrix(m)
        # Bonsai: np.fromstring(reference[1], sep=",").reshape(4, 4)
        self.assertTrue(np.allclose(np.fromstring(s, sep=",", dtype=np.float64).reshape(4, 4), m))
        self.assertEqual(encode_matrix(None), ",".join(str(v) for v in np.eye(4).flatten().tolist()))
        self.assertTrue(np.allclose(decode_matrix(s), m))

    def test_round_trip_and_shared_documents(self):
        out = TESTS / "unit" / "SITE_links.ifc"
        m = translate(10.0, 5.0, 0.0) @ rotate_z(1)
        write_site(out, [dict(path="BLK_A/BLK_A.ifc", name="BLK_A"),
                         dict(path="BLK_B\\BLK_B.ifc", name="BLK_B", matrix=m),
                         dict(path="BLK_A/BLK_A.ifc", name="BLK_A again")])
        f = ifcopenshell.open(str(out))
        docs = [d for d in f.by_type("IfcDocumentInformation") if d.Scope == "LINKED_MODEL"]
        self.assertEqual(len(docs), 2)                       # one document per linked file, like Bonsai
        self.assertEqual(sorted(d.Name for d in docs), ["BLK_A.ifc", "BLK_B.ifc"])
        refs = linked_models(f)
        self.assertEqual([r["path"] for r in refs], ["BLK_A/BLK_A.ifc", "BLK_A/BLK_A.ifc", "BLK_B/BLK_B.ifc"])
        self.assertTrue(np.allclose(refs[2]["matrix"], m))
        # Bonsai applies Identification in map coordinates: a rotation is stored conjugated by the map frame
        G = translate(30000.0, 36000.0, 15.0)
        self.assertTrue(np.allclose(refs[2]["identification"], G @ m @ np.linalg.inv(G)))
        self.assertTrue(np.allclose(refs[0]["matrix"], np.eye(4)))
        # documents are associated with the project (rooted), GUIDs deterministic
        rels = f.by_type("IfcRelAssociatesDocument")
        self.assertEqual(len(rels), 2)
        self.assertTrue(all(r.RelatedObjects[0].is_a("IfcProject") for r in rels))
        again = out.parent / "again" / out.name          # same file name: it is recorded in the header
        again.parent.mkdir(exist_ok=True)
        write_site(again, [dict(path="BLK_A/BLK_A.ifc", name="BLK_A"),
                           dict(path="BLK_B\\BLK_B.ifc", name="BLK_B", matrix=m),
                           dict(path="BLK_A/BLK_A.ifc", name="BLK_A again")])
        self.assertEqual(out.read_bytes(), again.read_bytes())

    def test_absolute_paths_rejected(self):
        W = IfcWriter(guid_key="test/abs")
        with self.assertRaises(ValueError):
            add_linked_models(W, [dict(path="C:/model/BLK_A.ifc", name="A")])
        with self.assertRaises(ValueError):
            add_linked_models(W, [dict(path="/model/BLK_A.ifc", name="A")])
        W.close()


class TestDriver(unittest.TestCase):
    def test_parse_marker(self):
        from estate.blender.run import MARKER, parse_marker
        self.assertEqual(parse_marker(f"noise\n{MARKER}\n{{\"ok\": true, \"n\": 3}}\nBlender quit\n"), {"ok": True, "n": 3})
        self.assertIsNone(parse_marker("no marker here"))

    def test_cli_registers(self):
        import argparse
        from estate.blender import cli
        ap = argparse.ArgumentParser()
        sub = ap.add_subparsers(dest="cmd")
        cli.register(sub)
        a = ap.parse_args(["blend", "x.ifc", "--jobs", "3"])
        self.assertEqual((a.ifc, a.jobs), (["x.ifc"], 3))
        a = ap.parse_args(["estate-blend", "--site", "model/SITE.ifc"])
        self.assertEqual(a.site, "model/SITE.ifc")
        a = ap.parse_args(["render", "x.blend", "--out", "o"])
        self.assertEqual((a.blend, a.out), ("x.blend", "o"))

    @unittest.skipIf(SKIP_BLENDER, "Blender tests disabled")
    def test_failure_raises_with_log_tail(self):
        from estate.blender.run import BlenderError, run_blender
        with self.assertRaises(BlenderError) as cm:
            run_blender("make_blend", {"ifc": str(TESTS / "does_not_exist" / "X.ifc")}, timeout=300)
        self.assertIn("missing IFC", str(cm.exception))
        self.assertFalse(cm.exception.result.get("ok"))
        Path(cm.exception.result["args_file"]).unlink()      # failed jobs keep their args for a manual rerun


@unittest.skipIf(SKIP_BLENDER or not PT4.exists(), "Blender tests disabled or build/t_pt4.ifc missing")
class TestBlendRoundTrip(unittest.TestCase):
    def test_make_verify_and_stale_detection(self):
        from estate.blender.run import run_blender
        d = TESTS / "unit" / "BLK_U"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        ifc = d / "BLK_U.ifc"
        shutil.copyfile(PT4, ifc)
        made = run_blender("make_blend", {"ifc": str(ifc)}, timeout=900)
        self.assertEqual(made["ifc_file_stored"], "BLK_U.ifc")
        self.assertEqual(made["counts"]["element_objects"], made["expected"]["non_feature_elements"])
        self.assertFalse((d / "BLK_U.blend1").exists())
        ok = run_blender("verify_blend", {"blend": made["blend"], "expected": made["counts"]["by_class"]}, timeout=900)
        self.assertEqual(ok["diff"], {})
        # move the folder: the relative IFC path must still resolve
        moved = TESTS / "unit" / "BLK_U_moved"
        shutil.rmtree(moved, ignore_errors=True)
        shutil.copytree(d, moved)
        ok2 = run_blender("verify_blend", {"blend": str(moved / "BLK_U.blend")}, timeout=900)
        self.assertTrue(ok2["ok"])
        self.assertTrue(ok2["ifc_resolved"].replace("\\", "/").endswith("BLK_U_moved/BLK_U.ifc"))
        # regenerate the IFC under the .blend: verify must flag it as stale
        if LEGACY.exists():
            shutil.copyfile(LEGACY, moved / "BLK_U.ifc")
            bad = run_blender("verify_blend", {"blend": str(moved / "BLK_U.blend")}, timeout=900, check=False)
            self.assertFalse(bad["ok"])
            self.assertTrue(any("stale" in e for e in bad["errors"]))
            Path(bad["args_file"]).unlink()


@unittest.skipIf(SKIP_BLENDER or not PT4.exists() or not LEGACY.exists(), "Blender tests disabled or inputs missing")
class TestEstateLinks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from estate.blender.run import run_blender
        cls.folder = TESTS / "unit" / "model"
        shutil.rmtree(cls.folder, ignore_errors=True)
        site = build_spike_model(cls.folder, {"BLK_A": PT4, "BLK_B": LEGACY},
                                 {"BLK_B": translate(45.0, 0.0, 0.0) @ rotate_z(1)})
        cls.made = run_blender("make_estate", {"site": str(site), "use_cache": False}, timeout=1800)

    def test_host_with_two_links(self):
        from estate.blender.run import run_blender
        made = self.made
        self.assertEqual(made["mode"], "host")
        self.assertEqual([l["is_loaded"] for l in made["links"]], [True, True])
        self.assertTrue(all(p.startswith("//") for p in made["libraries"]))
        self.assertIsInstance(made["temp_py_removed"], list)
        by = {l["filepath"]: l for l in made["links"]}
        self.assertEqual(by["BLK_B/BLK_B.ifc"]["matrix_translation"], [45.0, 0.0, 0.0])
        ok = run_blender("verify_blend", {"blend": made["blend"], "links": True}, timeout=900)
        # Bonsai merges a linked model into a few chunk meshes: check polygons, not object counts
        self.assertTrue(all(l["is_loaded"] and l["instanced_polygons"] > 10000 for l in ok["links"]))

    def test_street_camera_render(self):
        """A placed eye-level camera over the linked host: views [] renders only it, upright, sky over ground."""
        from PIL import Image
        from estate.blender.run import run_blender
        from estate.report.street import SKY
        out = self.folder / "renders"
        cam = {"eye": [-39.0, -39.0, 1.6], "target": [25.0, 0.0, 1.6], "lens": 24.0, "clip": [0.1, 2000.0],
               "shift": [0.0, 0.15]}
        r = run_blender("render", {"blend": self.made["blend"], "out": str(out), "width": 800, "views": [],
                                   "cameras": {"street_T": cam}}, timeout=900)
        self.assertEqual((r["cameras"], r["skipped"]), (["street_T"], []))
        self.assertEqual([Path(f).name for f in r["files"]], ["ESTATE_street_T.png"])
        im = np.asarray(Image.open(r["files"][0]).convert("RGB")).astype(int)
        self.assertEqual(im.shape, (450, 800, 3))                           # 16:9 by default
        sky = np.array([round(255 * (1.055 * v ** (1 / 2.4) - 0.055)) for v in SKY])   # Standard view transform
        is_sky = np.abs(im - sky).max(axis=2) <= 6
        self.assertTrue(0.05 < is_sky.mean() < 0.8, is_sky.mean())        # the model fills most of a sky frame
        self.assertGreater(is_sky[0].mean(), 0.3)                          # level and upright: sky on top ...
        self.assertLess(is_sky[-1].mean(), 0.05)                           # ... the ground slab at the bottom
        from estate import leaks
        self.assertEqual(leaks.png_text(Path(r["files"][0]).read_bytes()), [])   # no File / Date text, no EXIF text
        # the camera of every shot, for a viewer that matches the render: <prefix>_views.json beside the images
        self.assertEqual(Path(r["views"]).name, "ESTATE_views.json")
        doc = json.loads(Path(r["views"]).read_text(encoding="utf-8"))
        self.assertEqual((doc["schema"], doc["blend"], sorted(doc["views"])),
                         ("sample-town-n5/render-views/1", "ESTATE.blend", ["street_T"]))
        v = doc["views"]["street_T"]
        self.assertEqual((v["type"], v["lens"], v["sensor_width"], v["sensor_fit"], v["shift_x"], v["shift_y"],
                          v["res"], v["clip"]), ("PERSP", 24.0, 36.0, "AUTO", 0.0, 0.15, [800, 450], [0.1, 2000.0]))
        M = np.array(v["matrix_world"])
        np.testing.assert_allclose(M[:3, 3], cam["eye"], atol=1e-5)
        look = np.subtract(cam["target"], cam["eye"])
        np.testing.assert_allclose(-M[:3, 2], look / np.linalg.norm(look), atol=1e-5)     # Blender looks along -Z
        self.assertEqual(leaks.scan_file(r["views"]), [])


MASTERPLAN, SITE_GRAPH = env.MODEL / "masterplan.json", env.MODEL / "SITE_graph.json"


@unittest.skipIf(not MASTERPLAN.exists() or not SITE_GRAPH.exists(), "model/masterplan.json or SITE_graph.json missing")
class TestStreetCamera(unittest.TestCase):
    """estate/report/street.py on the real masterplan and site graph (pure python, no Blender)."""

    @classmethod
    def setUpClass(cls):
        from estate.report import street
        cls.street = street
        cls.mp = json.loads(MASTERPLAN.read_text(encoding="utf-8"))
        cls.graph = json.loads(SITE_GRAPH.read_text(encoding="utf-8"))
        cls.cam = street.camera("BS1", cls.mp, cls.graph)
        cls.node = next(n for n in cls.graph["nodes"] if n.get("kind") == "bus_stop" and n.get("ref") == "BS1")
        cls.frame = street.stop_frame("BS1", cls.mp, cls.graph)

    def test_eye_at_the_stop_in_the_open(self):
        eye, out = self.cam["eye"], self.frame["out"]
        self.assertAlmostEqual(eye[2], self.node["z"] + 1.6, places=6)    # 1.6 m above the platform paving
        x0, y0, x1, y1 = self.mp["estate"]["extent"]
        self.assertTrue(x0 < eye[0] < x1 and y0 < eye[1] < y1, eye)
        P = Point(eye[:2])
        for s in self.mp["sites"]:
            self.assertGreater(shape(s["footprint"]).distance(P), 0.5, s["id"])
        rel = (eye[0] - self.node["xy"][0], eye[1] - self.node["xy"][1])
        self.assertGreater(rel[0] * out[0] + rel[1] * out[1], 1.3)         # behind the shelter's estate-side edge
        self.assertLess(math.dist(eye[:2], self.node["xy"]), 10.0)         # ... and still at the stop
        for roof in self.street._canopies(self.graph, self.frame):         # open sky over the eye
            self.assertGreaterEqual(roof.distance(P), self.street.CLEAR - 1e-9)

    def test_looks_level_into_the_estate_at_a_block(self):
        eye, target, out = self.cam["eye"], self.cam["target"], self.frame["out"]
        d = (target[0] - eye[0], target[1] - eye[1])
        self.assertGreater((d[0] * out[0] + d[1] * out[1]) / math.hypot(*d), 0.1)
        self.assertEqual(target[2], eye[2])                                 # level: verticals stay vertical
        site = next(s for s in self.mp["sites"] if s["id"] == self.cam["look_at"])
        self.assertEqual(site["kind"], "block")
        self.assertLess(shape(site["footprint"]).distance(Point(target[:2])), 5.0)
        self.assertEqual((self.cam["lens"], self.cam["clip"], self.cam["res"]), (24.0, [0.1, 2000.0], [1600, 900]))

    def test_tower_tops_in_frame(self):
        """Every building top inside the horizontal field of view projects into the frame, the target's high up."""
        c = self.cam
        t_h = 18.0 / c["lens"]
        t_v = t_h * c["res"][1] / c["res"][0]
        centre = c["shift"][1] * 2 * t_h
        eye = c["eye"]
        h = (c["target"][0] - eye[0], c["target"][1] - eye[1])
        h = (h[0] / math.hypot(*h), h[1] / math.hypot(*h))
        seen = {}
        for s in self.mp["sites"]:
            for v in shape(s["footprint"]).exterior.coords:
                rel = (v[0] - eye[0], v[1] - eye[1])
                depth = rel[0] * h[0] + rel[1] * h[1]
                if depth > 1.0 and abs(rel[0] * h[1] - rel[1] * h[0]) / depth <= t_h and s.get("height"):
                    frac = ((s["height"] - 1.6) / depth - (centre - t_v)) / (2 * t_v)
                    seen[s["id"]] = max(seen.get(s["id"], 0.0), frac)
        self.assertIn(c["look_at"], seen)
        self.assertTrue(all(f <= 0.9 for f in seen.values()), seen)
        self.assertGreater(seen[c["look_at"]], 0.7)
        self.assertTrue(0.15 <= (t_v - centre) / (2 * t_v) <= 0.5)          # horizon in the lower half

    def test_fallback_and_determinism(self):
        for b in self.graph["meta"]["bus_stops"]:      # masterplan-only shelter centre == the built graph node
            self.assertLess(math.dist(self.street.stop_frame(b["id"], self.mp)["node"], b["node"]), 0.01, b["id"])
        again = self.street.camera("BS1", MASTERPLAN, SITE_GRAPH)
        self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(self.cam, sort_keys=True))
        with self.assertRaises(KeyError):
            self.street.camera("BS9", self.mp, self.graph)

    def test_tuning_rerenders_only_estate(self):
        """street.py is in no stage's code hash (blend and render hash estate/blender/*.py, ifc estate/site/*.py),
        so a camera tweak reaches the build only through this dict in the ESTATE render key, which must carry
        everything render.py takes from it."""
        from estate.pipeline import state as st
        me = Path(self.street.__file__).resolve()
        for stage, globs in st.STAGE_DEPS.items():
            self.assertNotIn(me, {p.resolve() for g in globs for p in env.ROOT.glob(g)}, stage)
        self.assertEqual(sorted(self.cam), ["clip", "eye", "lens", "look_at", "res", "shift", "sky", "stop", "target"])
        self.assertEqual(self.cam["sky"], list(self.street.SKY))


if __name__ == "__main__":
    unittest.main()
