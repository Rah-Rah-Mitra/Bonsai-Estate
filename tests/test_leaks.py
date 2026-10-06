"""Leak scan (estate/leaks.py): drive paths, Users folders and the author's username found in JSON strings, in PNG
text chunks (tEXt, and zTXt / iTXt once inflated) and eXIf text entries, in a GLB's JSON chunk and in zip members,
while pixel, vertex and compressed bytes are left alone and binary data of no known format fails closed; the stricter
absolute-path test for the pipeline's own JSON; error text scrubbed of machine folders (env.scrub) before
reports/build_failures.json is written; every tracked file of the repository scanned clean; and every generated file
in model/ and reports/ that the current code wrote scanned clean (model/.state.json says which code wrote what).

Run: "<blender python>" -I -B tests/test_leaks.py -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import random
import shutil
import struct
import subprocess
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


def exif(ifd0: list, sub: list = ()) -> bytes:
    """A little-endian EXIF block: IFD0 (plus a pointer to the Exif sub-IFD when ``sub`` is given), the sub-IFD, then
    every value longer than four bytes. Entries are (tag, type, count, value bytes)."""
    ifd0 = list(ifd0) + ([(0x8769, 4, 1, None)] if sub else [])
    size0, size1 = 6 + 12 * len(ifd0), (6 + 12 * len(sub) if sub else 0)
    values = bytearray()

    def ifd(entries):
        out = struct.pack("<H", len(entries))
        for tag, kind, count, value in entries:
            value = struct.pack("<I", 8 + size0) if value is None else value     # the sub-IFD follows IFD0
            if len(value) <= 4:
                out += struct.pack("<HHI", tag, kind, count) + value.ljust(4, b"\0")
            else:
                out += struct.pack("<HHII", tag, kind, count, 8 + size0 + size1 + len(values))
                values.extend(value)
        return out + struct.pack("<I", 0)
    return b"II*\0" + struct.pack("<I", 8) + ifd(ifd0) + (ifd(list(sub)) if sub else b"") + bytes(values)


def tracked_files() -> list[str] | None:
    """Every file git tracks (project-relative), or None outside a git checkout."""
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=env.ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [f for f in out.decode("utf-8").split("\0") if f]


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

    def test_exif_text_entries(self):
        """eXIf: ASCII entries, UserComment and the Windows XP tags are text, in IFD0 and in the Exif sub-IFD; the
        resolutions are not."""
        block = exif([(0x011A, 5, 1, struct.pack("<II", 72, 1)),
                      (0x010E, 2, len(DRIVE_PATH) + 1, DRIVE_PATH.encode("latin-1") + b"\0"),     # ImageDescription
                      (0x013B, 2, 3, b"ab\0")],                                                    # Artist, inline
                     sub=[(0x9286, 7, 8 + len(USER_UPPER), b"ASCII\0\0\0" + USER_UPPER.encode("ascii")),
                          (0x9C9C, 1, 2 * len(POSIX_HOME) + 2, POSIX_HOME.encode("utf-16-le") + b"\0\0")])
        self.assertEqual(leaks.exif_text(block), [("0x010E", DRIVE_PATH), ("0x013B", "ab"), ("0x9286", USER_UPPER),
                                                  ("0x9C9C", POSIX_HOME)])
        data = png(chunk(b"eXIf", block))
        self.assertEqual([k for k, _, _ in leaks.png_text(data)], ["eXIf"] * 4)
        self.assertEqual([f.where for f in leaks.scan_bytes(data, "x.png")], ["x.png eXIf"] * 3)

    def test_blender_exif_has_no_text(self):
        """The eXIf chunk Blender 5.2 writes into a Workbench render (big-endian, X and Y resolution only) has no
        text; one that does not parse is read whole, so a path in it is still found."""
        blender = bytes.fromhex("4d4d002a000000180000004800000001000000480000000100020"
                                "11a00050000000100000008011b0005000000010000001000000000")
        self.assertEqual(leaks.exif_text(blender), [])
        self.assertEqual(leaks.png_text(png(chunk(b"eXIf", blender))), [])
        self.assertEqual(leaks.exif_text(blender[:30]), [("raw", blender[:30].decode("latin-1"))])    # truncated
        broken = png(chunk(b"eXIf", b"XX " + DRIVE_PATH.encode("latin-1")))                     # no byte order
        self.assertEqual([f.where for f in leaks.scan_bytes(broken, "x.png")], ["x.png eXIf"])


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

    def test_binary_of_no_known_format_fails_closed(self):
        """A binary the scanner has no reader for (here named like a walk grid, but not SN5W) is one "not scanned"
        finding rather than a raw read: 1 MB of random bytes matches the drive pattern by chance (once, for this
        seed). Text is still read. (A real walk grid is read as SN5W: tests/test_web.py.)"""
        noise = random.Random(20261005).randbytes(1 << 20)
        self.assertTrue(leaks.find(noise.decode("latin-1")))
        self.assertEqual(leaks.scan_bytes(noise, "BLK_509_walk.bin"),
                         [leaks.Leak("BLK_509_walk.bin", leaks.UNSCANNABLE)])
        for name, data in (("a.zip", b"PK\x03\x04" + noise[:64]), ("a.glb", b"glTF" + noise[:64]),     # broken
                           ("a.txt", "plain".encode("utf-16"))):                                       # not UTF-8
            self.assertEqual(leaks.scan_bytes(data, name), [leaks.Leak(name, leaks.UNSCANNABLE)])
        ifc = f"#1=IFCPROJECT('0x',$,'Estate',$,$,$,'{DRIVE_PATH}',$,$);\n".encode("utf-8")
        self.assertEqual([f.where for f in leaks.scan_bytes(ifc, "SITE.ifc")], ["SITE.ifc"])
        self.assertEqual(leaks.scan_bytes("Sample Town N5 – 14 blocks\n".encode("utf-8"), "a.md"), [])

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
                (d / "sub" / "a.glb").write_bytes(glb({"asset": {"generator": DRIVE_PATH}}, binary=b"\0" * 64))
                self.assertEqual(leaks.main([str(d / "sub" / "a.glb")]), 1)     # a GLB file: its JSON chunk is read
            self.assertIn("2 files scanned, 1 leak(s)", out.getvalue())
            self.assertIn("a.glb JSON chunk $.asset.generator", out.getvalue())
        finally:
            shutil.rmtree(d, ignore_errors=True)


class ScrubbedErrors(unittest.TestCase):
    def test_scrub(self):
        """env.scrub: the project folder becomes ".", Blender's folder "<blender>" and the home folder "~", with either
        slash (and any case on Windows), only where a whole folder name ends."""
        sep = os.sep
        self.assertEqual(env.scrub(f'File "{env.ROOT / "estate" / "commands.py"}", line 3'),
                         f'File ".{sep}estate{sep}commands.py", line 3')
        self.assertEqual(env.scrub((env.ROOT / "model" / "a.ifc").as_posix() + " missing"), "./model/a.ifc missing")
        self.assertEqual(env.scrub(f"({env.ROOT})"), "(.)")
        self.assertEqual(env.scrub(str(env.BLENDER_DIR / "5.2" / "python" / "Lib" / "x.py")),
                         sep.join(["<blender>", "5.2", "python", "Lib", "x.py"]))
        self.assertEqual(env.scrub(str(Path.home() / "AppData" / "x")), sep.join(["~", "AppData", "x"]))
        sibling = env.scrub(f"{env.ROOT}-release{sep}v1.2")                  # not the project: no "." in front
        self.assertFalse(sibling.startswith(".-"), sibling)
        self.assertTrue(sibling.endswith(f"{env.ROOT.name}-release{sep}v1.2"), sibling)
        if os.name == "nt":
            self.assertEqual(env.scrub(str(env.ROOT).lower() + BS + "x.py"), "." + BS + "x.py")
        doc = env.scrub({str(env.ROOT): [str(env.ROOT / "x"), 3, None, {"k": str(Path.home())}]})
        self.assertEqual(doc, {".": [f".{sep}x", 3, None, {"k": "~"}]})

    def test_build_failures_written_scrubbed(self):
        """reports/build_failures.json as cmd_build writes it (write_failures) after a failed IFC job, a failed
        render and a failed check: tracked, and copied into report.json and report.html, so its tracebacks and
        error text name no folder of this machine; with no failure it is "[]" as before."""
        from estate import commands
        trace = "".join(f'  File "{f}", line {i}, in job\n' for i, f in enumerate(
            (env.ROOT / "estate" / "pipeline" / "runner.py", env.BLENDER_DIR / "5.2" / "python" / "Lib" / "json.py",
             Path.home() / "AppData" / "Roaming" / "ifcopenshell" / "file.py")))
        failed = [dict(target="BLK_501", ok=False, error="ValueError: bad", trace="Traceback:\n" + trace, seconds=1.0),
                  dict(target="BLK_502", stage="render", error=f"log tail: saved {env.ROOT / 'model' / 'a.blend'}"),
                  dict(target="BLK_503", stage="check", errors=[("ifcqa.schema", 3)], ifc=env.ROOT / "model" / "x.ifc")]
        self.assertTrue(leaks.absolute_strings(json.loads(json.dumps(failed, default=str))))
        env.BUILD.mkdir(parents=True, exist_ok=True)
        d = Path(tempfile.mkdtemp(prefix="unittest-leaks-", dir=env.BUILD))
        try:
            text = commands.write_failures(failed, d / "build_failures.json").read_text(encoding="utf-8")
            doc = json.loads(text)
            self.assertEqual(leaks.scan_json(text), [])
            self.assertEqual(leaks.absolute_strings(doc), [])
            self.assertEqual((doc[2]["errors"], doc[2]["ifc"]), ([["ifcqa.schema", 3]], f".{os.sep}model{os.sep}x.ifc"))
            self.assertIn(f'File "<blender>{os.sep}5.2', doc[0]["trace"])
            self.assertEqual(commands.write_failures([], d / "none.json").read_text(encoding="utf-8"), "[]")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TrackedFiles(unittest.TestCase):
    def test_every_tracked_file_is_clean(self):
        """Every file git tracks: source, config, the tracked reports and the baseline model files. The only machine
        path allowed is the default Blender folder in estate.cmd and estate/env.py. No string of a tracked JSON file
        (the reports the build commits) starts like an absolute path either, on any machine."""
        names = tracked_files()
        if names is None:
            self.skipTest("not a git checkout")
        files = [env.ROOT / f for f in names]
        self.assertGreater(len(files), 100)
        found = [str(leak) for f in files if f.is_file()
                 for leak in leaks.scan_file(f, env.ROOT, allow=leaks.BLENDER_INSTALL)]
        found += [f"{env.rel(f)} {p}: {s!r}" for f in files if f.suffix == ".json" and f.is_file()
                  for p, s in leaks.absolute_strings(json.loads(f.read_text(encoding="utf-8")))]
        self.assertEqual(found, [])


GENERATED = (".json", ".png", ".glb", ".bin")


def generated_files(state: dict, tracked) -> tuple[list[Path], list[Path]]:
    """The generated JSON, PNG, GLB and walk-grid (SN5W, .bin) files under model/ and reports/ (not Bonsai's link
    caches, not the tracked reports: TrackedFiles scans those), split into (written by the current code, written by
    older code). A file is older only when every stage record listing it names other code than the stage has now
    (state.code_hash); a file no stage records (estate_manifest.json, the flat gallery, ...) counts as current."""
    from estate.pipeline import state as st
    current, older = set(), set()
    for stages in state.values():
        for stage, rec in (stages.items() if isinstance(stages, dict) else ()):
            if isinstance(rec, dict):
                (current if rec.get("code") == st.code_hash(stage) else older).update(rec.get("outputs", []))
    new, old = [], []
    for top in (env.MODEL, env.REPORTS):
        for p in sorted(top.rglob("*")):
            name = env.rel(p)
            if p.suffix in GENERATED and ".ifc.cache." not in p.name and name not in tracked and p.is_file():
                (old if name in older - current else new).append(p)
    return new, old


class GeneratedFiles(unittest.TestCase):
    """model/ and reports/ as the last build left them: untracked, and published in the release zips. Every JSON
    (keys and strings, and none that starts like an absolute path), every PNG's text chunks and eXIf text and every
    GLB's JSON chunk, of the files the current code wrote. Outputs of a stage whose code has changed since they were
    written are counted, not failed: the next build of that stage rewrites them (v1.2's forced build, all of them)."""

    @classmethod
    def setUpClass(cls):
        from estate.pipeline import state as st
        cls.new, cls.old = generated_files(st.load(), set(tracked_files() or ()))

    def test_outputs_of_the_current_code_are_clean(self):
        if not self.new:
            self.skipTest("no generated files: run ./estate.sh build")
        found = [str(leak) for p in self.new for leak in leaks.scan_file(p, env.ROOT)]
        found += [f"{env.rel(p)} {k}: {s!r}" for p in self.new if p.suffix == ".json"
                  for k, s in leaks.absolute_strings(json.loads(p.read_text(encoding="utf-8")))]
        self.assertEqual(found, [], f"{len(self.new)} files checked")

    def test_outputs_of_older_code(self):
        """Not a check: reports, as a skip, how many generated files the test above leaves to their next rebuild."""
        if self.old:
            self.skipTest(f"{len(self.old)} generated files written by older code not checked until rebuilt "
                          f"(./estate.sh build --force rebuilds them all); {len(self.new)} checked")


if __name__ == "__main__":
    unittest.main()
