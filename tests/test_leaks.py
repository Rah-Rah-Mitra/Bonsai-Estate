"""Leak scan (estate/leaks.py): drive paths, Users folders and the author's username found in JSON strings, in PNG
text chunks (tEXt, and zTXt / iTXt once inflated), in a GLB's JSON chunk and in zip members, while pixel, vertex and
compressed bytes are left alone; and the stricter absolute-path test for the pipeline's own JSON.

Run: "<blender python>" -I -B tests/test_leaks.py -v
"""
from __future__ import annotations

import contextlib
import io
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

from estate import env, leaks  # noqa: E402

# Planted values are built at run time, so this file holds none of them: the tracked-file scan reads it too.
BS = chr(92)
DRIVE_PATH = "C:" + BS + "Users" + BS + "someone" + BS + "estate" + BS + "model" + BS + "ESTATE.blend"
POSIX_HOME = "/" + "Users" + "/someone/estate"
USER_UPPER = "RAPI" + "ULAR"            # the username, in the wrong case on purpose


def chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def png(*extra: bytes, pixels: bytes = b"\x00\x80") -> bytes:
    """A 1 x 1 grey PNG with the given chunks between its header and its pixel data."""
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
    return leaks.PNG_SIGNATURE + ihdr + b"".join(extra) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")


def glb(doc: dict, binary: bytes = b"") -> bytes:
    js = json.dumps(doc).encode("utf-8")
    js += b" " * (-len(js) % 4)
    body = struct.pack("<I4s", len(js), b"JSON") + js
    if binary:
        binary += b"\0" * (-len(binary) % 4)
        body += struct.pack("<I4s", len(binary), b"BIN\0") + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


class PngText(unittest.TestCase):
    def test_text_chunk_with_the_blend_path(self):
        """What Blender wrote into every render before U1: tEXt File=<the .blend's absolute path>."""
        from PIL import Image
        data = png(chunk(b"tEXt", b"File\0" + DRIVE_PATH.encode("latin-1")),
                   chunk(b"tEXt", b"Software\0matplotlib version3.10"))
        self.assertEqual(Image.open(io.BytesIO(data)).text["File"], DRIVE_PATH)     # a PNG any reader accepts
        self.assertEqual(leaks.png_text(data), [("tEXt", "File", DRIVE_PATH),
                                                ("tEXt", "Software", "matplotlib version3.10")])
        found = leaks.scan_bytes(data, "x.png")
        self.assertTrue(found)
        self.assertTrue(all(f.where == "x.png tEXt" and "File=" in f.text for f in found))

    def test_compressed_text_chunks_are_inflated(self):
        z = chunk(b"zTXt", b"Comment\0\0" + zlib.compress(f"saved in {DRIVE_PATH}".encode("latin-1")))
        i = chunk(b"iTXt", b"Author\0\x01\0en\0Autor\0" + zlib.compress(f"made by {USER_UPPER}".encode("utf-8")))
        data = png(z, i)
        self.assertEqual(leaks.find(data.decode("latin-1")), [])           # invisible in the raw bytes
        self.assertEqual([k for k, _, _ in leaks.png_text(data)], ["zTXt", "iTXt"])
        self.assertEqual(leaks.png_text(data)[1][2], f"en\nAutor\nmade by {USER_UPPER}")
        where = [f.where for f in leaks.scan_bytes(data, "x.png")]
        self.assertIn("x.png zTXt", where)
        self.assertIn("x.png iTXt", where)                                  # the username, in any case

    def test_uncompressed_itxt(self):
        data = png(chunk(b"iTXt", b"Source\0\0\0\0\0" + POSIX_HOME.encode("utf-8")))
        self.assertEqual(leaks.png_text(data), [("iTXt", "Source", POSIX_HOME)])
        self.assertEqual(len(leaks.scan_bytes(data, "x.png")), 1)

    def test_pixels_are_not_text(self):
        """The planted bytes inside pixel data (and a clean text chunk) are not a leak; read raw they would be."""
        data = png(chunk(b"tEXt", b"Software\0matplotlib"), pixels=b"\x00" + DRIVE_PATH.encode("latin-1"))
        raw = leaks.PNG_SIGNATURE + chunk(b"IDAT", DRIVE_PATH.encode("latin-1"))
        self.assertEqual(leaks.scan_bytes(data, "x.png"), [])
        self.assertEqual(leaks.scan_bytes(raw, "x.png"), [])
        self.assertTrue(leaks.find(raw.decode("latin-1")))


class JsonAndContainers(unittest.TestCase):
    def test_json_strings_and_keys(self):
        doc = {"file": "model/BLK_501/BLK_501.ifc", "runs": [{"note": "Warnings:\nnone"}, {"ifc": DRIVE_PATH}],
               USER_UPPER.lower(): 1}
        found = leaks.scan_json(json.dumps(doc), "BLK_501_nav.json")
        self.assertEqual({f.where for f in found},
                         {"BLK_501_nav.json $.runs[1].ifc", f"BLK_501_nav.json $.{USER_UPPER.lower()}"})
        self.assertEqual(leaks.scan_json(json.dumps({"note": "Warnings:\nnone", "a": "x: y"})), [])
        self.assertTrue(leaks.scan_json('{"broken": "' + DRIVE_PATH.replace(BS, BS + BS)))   # not JSON: read raw

    def test_absolute_strings(self):
        doc = {"path": "model/SITE.ifc", "graph": DRIVE_PATH, "posix": "/home/someone/x.ifc",
               "unc": BS + BS + "server" + BS + "share", "forward": "D:/estate/model", "note": "C: drive, 3 m"}
        self.assertEqual(leaks.absolute_strings(doc), [("$.graph", DRIVE_PATH), ("$.posix", "/home/someone/x.ifc"),
                                                       ("$.unc", BS + BS + "server" + BS + "share"),
                                                       ("$.forward", "D:/estate/model")])

    def test_glb_reads_the_json_chunk_only(self):
        self.assertTrue(leaks.scan_bytes(glb({"asset": {"version": "2.0", "generator": DRIVE_PATH}}), "a.glb"))
        clean = glb({"asset": {"version": "2.0"}}, binary=DRIVE_PATH.encode("latin-1") * 3)
        self.assertEqual(leaks.scan_bytes(clean, "a.glb"), [])

    def test_zip_members(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("model/BLK_501/BLK_501_nav.json", json.dumps({"file": DRIVE_PATH}))
            z.writestr("reports/renders/x.png", png(chunk(b"tEXt", b"File\0" + DRIVE_PATH.encode("latin-1"))))
            z.writestr("model/clean.json", json.dumps({"file": "model/BLK_501/BLK_501.ifc"}))
        data = buf.getvalue()
        where = {f.where for f in leaks.scan_bytes(data, "release.zip")}
        self.assertEqual(where, {"release.zip!model/BLK_501/BLK_501_nav.json $.file",
                                 "release.zip!reports/renders/x.png tEXt"})

    def test_allowance_is_narrow(self):
        line = 'set BLENDER="C:' + BS + "Program Files" + BS + "Blender Foundation" + BS + 'Blender 5.2"\n'
        self.assertEqual(len(leaks.find(line)), 1)
        self.assertEqual(leaks.find(line, allow=leaks.BLENDER_INSTALL), [])
        found = leaks.find(line + DRIVE_PATH, allow=leaks.BLENDER_INSTALL)      # any other drive path still counts
        self.assertEqual(len(found), 1)
        self.assertIn("Users", found[0].text)

    def test_cli(self):
        """estate/leaks.py PATH...: folders are walked; the exit status says whether anything was found."""
        env.BUILD.mkdir(parents=True, exist_ok=True)
        d = Path(tempfile.mkdtemp(prefix="unittest-leaks-", dir=env.BUILD))
        try:
            (d / "clean.json").write_text(json.dumps({"path": "model/SITE.ifc"}), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(leaks.main([str(d)]), 0)
                (d / "sub").mkdir()
                (d / "sub" / "bad.json").write_text(json.dumps({"path": DRIVE_PATH}), encoding="utf-8")
                self.assertEqual(leaks.main([str(d)]), 1)
            self.assertIn("2 files scanned, 1 leak(s)", out.getvalue())
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
