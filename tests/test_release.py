"""Release zips (estate/web/release.py) on a small stand-in build: allowlisted entries only, byte-identical zips from
two runs (sorted entries, fixed time and mode, deflate 9), release_manifest.json, and every refusal: a leak in any
entry (a PNG text chunk naming a Users folder), a dirty generator, another commit than HEAD, an export changed after
export_info.json or the manifest hashed it, a required file missing.

Run: "<blender python>" -I -B tests/test_release.py -v
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estate.env import bootstrap  # noqa: E402

bootstrap()

from estate import env  # noqa: E402
from estate.web import release  # noqa: E402
from estate.web.info import git  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "web" / "sn5w_sample.bin"
BS = chr(92)
PLANTED = "C:" + BS + "Users" + BS + "x"           # built at run time: the tracked-file scan reads this file too


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def png(text: bytes = b"") -> bytes:
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
    extra = _chunk(b"tEXt", text) if text else b""
    return b"\x89PNG\r\n\x1a\n" + ihdr + extra + _chunk(b"IDAT", zlib.compress(b"\x00\x80")) + _chunk(b"IEND", b"")


def glb(name: str) -> bytes:
    js = json.dumps({"asset": {"version": "2.0", "extras": {"source": name}}}).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<I4s", len(js), b"JSON") + js
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stand_in(root: Path, commit=None, dirty=False) -> tuple[Path, Path]:
    """model/ and reports/ of a one-building build, with files the release must leave out beside the ones it takes;
    the manifest and export_info.json hash them as cmd_build would."""
    m, r = root / "model", root / "reports"
    b = m / "BLK_1"
    for d in (b / "nav", b / "plans", r / "renders" / "ESTATE", r / "plans"):
        d.mkdir(parents=True, exist_ok=True)
    ifc = "ISO-10303-21;\nHEADER;\nFILE_NAME('BLK_1.ifc','',(''),(''),'','','');\nENDSEC;\nEND-ISO-10303-21;\n"
    files = {m / "masterplan.json": b"{}", m / "SITE.ifc": ifc.encode(), m / "SITE_engine.json": b"{}",
             m / "SITE_graph.json": b'{"nodes": []}', m / "SITE_lod0.glb": glb("SITE"),
             m / "SITE_lod1.glb": glb("SITE"),
             b / "BLK_1.ifc": ifc.encode(), b / "BLK_1_engine.json": b'{"doors": []}', b / "BLK_1_nav.json": b"{}",
             b / "BLK_1_lod0.glb": glb("0"), b / "BLK_1_lod1.glb": glb("1"), b / "BLK_1_lod2.glb": glb("2"),
             b / "BLK_1_int_L01.glb": glb("L01"), b / "BLK_1_walk.bin": GOLDEN.read_bytes(),
             b / "BLK_1_web.json": b'{"schema": "sample-town-n5/web/1"}',
             # never released
             b / "BLK_1.blend": b"BLENDER-v502" + PLANTED.encode(), b / "BLK_1.ifc.cache.json": b"{}",
             m / ".state.json": b"{}", m / "ESTATE.blend": b"BLENDER", b / "nav" / "L1_r30.png": png(),
             b / "plans" / "BLK_1_L1_ifc.png": png(), b / "BLK_1.build.json": json.dumps({"path": PLANTED}).encode(),
             r / "report.json": b'{"sections": []}',
             r / "renders" / "ESTATE" / "ESTATE_aerial_NE.png": png(b"Software\0x"),
             r / "renders" / "ESTATE" / "ESTATE_views.json": b'{"views": {}}', r / "plans" / "x.png": png(),
             r / "site_plan.png": png()}
    for p, data in files.items():
        p.write_bytes(data)
    rec = {k: {"path": f"BLK_1/BLK_1_{k}.glb", "sha256": _sha(b / f"BLK_1_{k}.glb")} for k in ("lod0", "lod1", "lod2")}
    rec.update(ifc={"path": "BLK_1/BLK_1.ifc", "sha256": _sha(b / "BLK_1.ifc")},
               blend={"path": "BLK_1/BLK_1.blend", "sha256": "0" * 64})          # .blend: listed, never released
    man = {"sites": [{"id": "BLK_1", "files": rec}],
           "site": {"files": {"ifc": {"path": "SITE.ifc", "sha256": _sha(m / "SITE.ifc")}}},
           "pedestrian_graph_file": {"path": "SITE_graph.json", "sha256": _sha(m / "SITE_graph.json")}}
    (m / "estate_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    info = {"schema": "sample-town-n5/export-info/1", "commit": commit or git("rev-parse", "HEAD"), "dirty": dirty,
            "dirty_scope": "git status --porcelain -- estate config estate.py estate.sh estate.cmd",
            "manifest_sha256": _sha(m / "estate_manifest.json"),
            "files": {f"BLK_1/BLK_1_{k}": {"sha256": _sha(b / f"BLK_1_{k}"), "bytes": (b / f"BLK_1_{k}").stat().st_size}
                      for k in ("walk.bin", "web.json")}}
    (m / "export_info.json").write_text(json.dumps(info), encoding="utf-8")
    return m, r


@unittest.skipIf(git("rev-parse", "HEAD") is None, "not a git checkout")
class Release(unittest.TestCase):
    def setUp(self):
        env.BUILD.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="unittest-release-", dir=env.BUILD))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def run_release(self, model, reports, out="out"):
        return release.release("v1.2-rc", self.tmp / out, model, reports, log=lambda *a: None)

    def test_two_runs_identical(self):
        m, r = stand_in(self.tmp / "a")
        first = self.run_release(m, r, "out1")
        second = self.run_release(m, r, "out2")
        self.assertEqual(first, second)
        names = ["SampleTownN5_v1.2-rc_model.zip", "SampleTownN5_v1.2-rc_reports.zip"]
        self.assertEqual(sorted(first["zips"]), names)
        for name in names:
            a, b = (self.tmp / o / name for o in ("out1", "out2"))
            self.assertEqual(a.read_bytes(), b.read_bytes(), name)
            self.assertEqual(first["zips"][name], {"sha256": _sha(a), "bytes": a.stat().st_size})
        self.assertEqual(json.loads((self.tmp / "out1" / "release_manifest.json").read_text(encoding="utf-8")), first)
        self.assertEqual((first["tag"], first["commit"]), ("v1.2-rc", git("rev-parse", "HEAD")))
        model = ["model/BLK_1/BLK_1.ifc", "model/BLK_1/BLK_1_engine.json", "model/BLK_1/BLK_1_int_L01.glb",
                 "model/BLK_1/BLK_1_lod0.glb", "model/BLK_1/BLK_1_lod1.glb", "model/BLK_1/BLK_1_lod2.glb",
                 "model/BLK_1/BLK_1_nav.json", "model/BLK_1/BLK_1_walk.bin", "model/BLK_1/BLK_1_web.json",
                 "model/SITE.ifc", "model/SITE_engine.json", "model/SITE_graph.json", "model/SITE_lod0.glb",
                 "model/SITE_lod1.glb", "model/estate_manifest.json", "model/export_info.json",
                 "model/masterplan.json"]
        reports = ["reports/renders/ESTATE/ESTATE_aerial_NE.png", "reports/renders/ESTATE/ESTATE_views.json",
                   "reports/report.json"]
        from estate.report.bcf_out import ZIP_TIME
        for name, want in zip(names, (model, reports)):
            with zipfile.ZipFile(self.tmp / "out1" / name) as z:
                infos = z.infolist()
                self.assertEqual([i.filename for i in infos], want)             # sorted, allowlisted only
                for i in infos:
                    self.assertEqual((i.date_time, i.external_attr, i.create_system, i.compress_type),
                                     (ZIP_TIME, release.MODE, 3, zipfile.ZIP_DEFLATED), i.filename)
                    self.assertEqual(hashlib.sha256(z.read(i)).hexdigest(), first["entries"][i.filename])
        self.assertEqual(sorted(first["entries"]), sorted(model + reports))

    def test_leak_refused(self):
        """A render whose PNG text names a Users folder (what Blender wrote before U1) stops the release; no zip is
        written."""
        m, r = stand_in(self.tmp / "a")
        (r / "renders" / "ESTATE" / "ESTATE_iso.png").write_bytes(png(b"File\0" + PLANTED.encode("latin-1")))
        with self.assertRaises(release.Refused) as cm:
            self.run_release(m, r)
        self.assertIn("leak scan: 1 finding", str(cm.exception))
        self.assertFalse((self.tmp / "out").exists() and any((self.tmp / "out").iterdir()))

    def test_unreadable_entry_refused(self):
        """A walk grid that does not parse as SN5W is not scanned, and that refuses the release too."""
        m, r = stand_in(self.tmp / "a")
        b = m / "BLK_1" / "BLK_1_walk.bin"
        b.write_bytes(b.read_bytes() + b"\0")
        info = json.loads((m / "export_info.json").read_text(encoding="utf-8"))
        info["files"]["BLK_1/BLK_1_walk.bin"] = {"sha256": _sha(b), "bytes": b.stat().st_size}
        (m / "export_info.json").write_text(json.dumps(info), encoding="utf-8")
        with self.assertRaises(release.Refused) as cm:
            self.run_release(m, r)
        self.assertIn("leak scan", str(cm.exception))

    def test_provenance_refused(self):
        """A dirty generator, or an export of another commit than HEAD, is never released."""
        for kw, words in ((dict(dirty=True), "dirty"), (dict(commit="0" * 40), "HEAD is")):
            m, r = stand_in(self.tmp / words.split()[0], **kw)
            with self.assertRaises(release.Refused) as cm:
                self.run_release(m, r)
            self.assertIn(words, str(cm.exception))

    def test_stale_export_refused(self):
        """A walk grid / web JSON changed after export_info.json, or a file changed after the manifest hashed it, or
        a required file missing."""
        cases = (("BLK_1/BLK_1_web.json", b'{"schema": "edited"}', "differs from export_info"),
                 ("BLK_1/BLK_1_lod1.glb", glb("edited"), "differs from estate_manifest"),
                 ("BLK_1/BLK_1_lod2.glb", None, "missing from the build"))
        for i, (rel, data, words) in enumerate(cases):
            m, r = stand_in(self.tmp / f"c{i}")
            if data is None:
                (m / rel).unlink()
            else:
                (m / rel).write_bytes(data)
            with self.assertRaises(release.Refused) as cm:
                self.run_release(m, r)
            self.assertIn(words, str(cm.exception))
        with self.assertRaises(release.Refused):
            release.release("../v1", self.tmp / "out", m, r)

    def test_cli(self):
        ap = argparse.ArgumentParser()
        release.register(ap.add_subparsers(dest="cmd"))
        a = ap.parse_args(["release", "--tag", "v1.2", "--out", "dist"])
        self.assertEqual((a.tag, a.out, a.fn), ("v1.2", "dist", release.cmd_release))


if __name__ == "__main__":
    unittest.main()
