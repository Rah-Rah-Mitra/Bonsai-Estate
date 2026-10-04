"""Blender stage tests: LINKED_MODEL references, the blender.exe job driver, .blend round trips and the estate link.

The fast tests need only ifcopenshell. The Blender tests launch blender.exe (Bonsai 0.9) in the background and
take about a minute in all; set ESTATE_SKIP_BLENDER=1 to skip them. They work on copies under
build/blender_tests/ so relative paths are exercised and nothing in model/ is touched.
"""
from __future__ import annotations

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
from shapely.geometry import box  # noqa: E402

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
    def test_host_with_two_links(self):
        from estate.blender.run import run_blender
        folder = TESTS / "unit" / "model"
        shutil.rmtree(folder, ignore_errors=True)
        site = build_spike_model(folder, {"BLK_A": PT4, "BLK_B": LEGACY},
                                 {"BLK_B": translate(45.0, 0.0, 0.0) @ rotate_z(1)})
        made = run_blender("make_estate", {"site": str(site), "use_cache": False}, timeout=1800)
        self.assertEqual(made["mode"], "host")
        self.assertEqual([l["is_loaded"] for l in made["links"]], [True, True])
        self.assertTrue(all(p.startswith("//") for p in made["libraries"]))
        self.assertIsInstance(made["temp_py_removed"], list)
        by = {l["filepath"]: l for l in made["links"]}
        self.assertEqual(by["BLK_B/BLK_B.ifc"]["matrix_translation"], [45.0, 0.0, 0.0])
        ok = run_blender("verify_blend", {"blend": made["blend"], "links": True}, timeout=900)
        # Bonsai merges a linked model into a few chunk meshes: check polygons, not object counts
        self.assertTrue(all(l["is_loaded"] and l["instanced_polygons"] > 10000 for l in ok["links"]))


if __name__ == "__main__":
    unittest.main()
