"""Leak scan: machine paths and the author's username in files that get published (stdlib only, no bpy).

A published file must not name a folder of the machine that built it. The scan looks for a drive-letter path (a lone
letter, a colon and a backslash), a Users folder with either slash, or the author's username in any case.
``scan_file`` reads a file the way a reader of its format would:

- JSON: every key and string, decoded, so an escaped newline after a colon is not taken for a drive;
- PNG: the text chunks (tEXt, zTXt and iTXt, inflated); pixel data is never text;
- GLB: the JSON chunk, as JSON; the binary chunk holds vertex data;
- zip (release zips, BCF): every member name and, by type, every member;
- anything else: the raw bytes.

Compressed and binary data are not read raw because three random bytes match a drive pattern every few hundred
kilobytes. ``absolute_strings`` is the stricter test for the pipeline's own JSON: no string may start like an
absolute path.

Run: "<blender python>" -I -B estate/leaks.py PATH...    (files or folders; exit status 1 when anything is found)
"""
from __future__ import annotations

import glob
import io
import json
import re
import struct
import sys
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

# The brackets keep this file from matching itself: tests/test_leaks.py scans every tracked file.
PATTERN = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\|/User[s]/|\\User[s]\\|(?i:rapiula[r])")
# The default Blender folder in estate.cmd and estate/env.py: a machine path, but the same on every Windows machine
# and naming no user. Only the scan of tracked source files allows it; generated files never contain it.
BLENDER_INSTALL = re.compile(r"[A-Za-z]:\\Program Files\\Blender Foundation\\")
ABSOLUTE = re.compile(r"[A-Za-z]:[\\/]|[\\/]")      # a string that starts like this is an absolute path
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_INFLATE = 1 << 26        # compressed PNG text is never this large; a bigger one is scanned as far as this


@dataclass(frozen=True)
class Leak:
    where: str      # the file, then the chunk, zip member or JSON key path inside it
    text: str       # the match with up to 40 characters either side

    def __str__(self):
        return f"{self.where}: {self.text!r}"


def find(text: str, where: str = "", allow: re.Pattern | None = None) -> list[Leak]:
    """Every match in a string. A match where ``allow`` matches too (at the same place) is skipped."""
    return [Leak(where, text[max(0, m.start() - 40):m.end() + 40]) for m in PATTERN.finditer(text)
            if allow is None or not allow.match(text, m.start())]


def json_strings(obj, path: str = "$"):
    """(key path, string) for every key and every string value of a parsed JSON document, in document order."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}", str(k)
            yield from json_strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from json_strings(v, f"{path}[{i}]")


def absolute_strings(obj) -> list[tuple[str, str]]:
    """(key path, string) for every key or string of a parsed JSON document that starts like an absolute path."""
    return [(p, s) for p, s in json_strings(obj) if ABSOLUTE.match(s)]


def scan_json(text: str, where: str = "", allow=None) -> list[Leak]:
    try:
        doc = json.loads(text)
    except ValueError:
        return find(text, where, allow)
    return [leak for p, s in json_strings(doc) for leak in find(s, f"{where} {p}", allow)]


def _inflate(data: bytes) -> bytes:
    try:
        return zlib.decompressobj().decompress(data, MAX_INFLATE)
    except zlib.error:
        return data          # not zlib after all: scan the stored bytes


def png_text(data: bytes) -> list[tuple[str, str, str]]:
    """(chunk type, keyword, text) of every tEXt, zTXt and iTXt chunk of a PNG, in file order, compressed text
    inflated. An iTXt's language tag and translated keyword come first in its text, one per line."""
    out, i = [], len(PNG_SIGNATURE)
    while i + 8 <= len(data):
        n, kind = struct.unpack(">I4s", data[i:i + 8])
        body = data[i + 8:i + 8 + n]
        i += 12 + n                                      # length, type, data, CRC
        if kind == b"IEND":
            break
        if kind not in (b"tEXt", b"zTXt", b"iTXt"):
            continue
        key, _, rest = body.partition(b"\0")
        if kind == b"tEXt":
            text = rest.decode("latin-1")
        elif kind == b"zTXt":                            # compression method byte, then zlib data
            text = _inflate(rest[1:]).decode("latin-1")
        else:                                            # flag, method, language\0, translated keyword\0, text
            flag = rest[:1]
            lang, _, rest = rest[2:].partition(b"\0")
            translated, _, text = rest.partition(b"\0")
            parts = (lang, translated, _inflate(text) if flag == b"\x01" else text)
            text = "\n".join(p.decode("utf-8", "replace") for p in parts if p)
        out.append((kind.decode(), key.decode("latin-1"), text))
    return out


def scan_png(data: bytes, where: str = "", allow=None) -> list[Leak]:
    return [leak for kind, key, text in png_text(data) for leak in find(f"{key}={text}", f"{where} {kind}", allow)]


def glb_json(data: bytes) -> str | None:
    """The JSON chunk of a binary glTF (always its first chunk), or None when the data is not one."""
    if len(data) < 20 or data[:4] != b"glTF":
        return None
    n, kind = struct.unpack("<I4s", data[12:20])
    return data[20:20 + n].decode("utf-8", "replace") if kind == b"JSON" else None


def scan_zip(data: bytes, where: str = "", allow=None) -> list[Leak]:
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        out += find(z.comment.decode("latin-1"), f"{where} (zip comment)", allow)
        for info in z.infolist():
            name = f"{where}!{info.filename}"
            out += find(info.filename, name, allow) + find(info.comment.decode("latin-1"), name, allow)
            if not info.is_dir():
                out += scan_bytes(z.read(info), name, allow)
    return out


def scan_bytes(data: bytes, name: str = "", allow=None) -> list[Leak]:
    """Scan one file's bytes by format (sniffed; JSON by its name). ``name`` labels the findings."""
    if data.startswith(PNG_SIGNATURE):
        return scan_png(data, name, allow)
    js = glb_json(data)
    if js is not None:
        return scan_json(js, f"{name} JSON chunk", allow)
    if data[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        try:
            return scan_zip(data, name, allow)
        except zipfile.BadZipFile:
            pass
    if name.lower().endswith((".json", ".gltf")):
        return scan_json(data.decode("utf-8", "replace"), name, allow)
    return find(data.decode("latin-1"), name, allow)


def scan_file(path, root=None, allow=None) -> list[Leak]:
    """Scan one file; findings are labelled with its path relative to ``root`` when given."""
    path = Path(path)
    name = path.relative_to(root).as_posix() if root else path.as_posix()
    return scan_bytes(path.read_bytes(), name, allow)


def main(argv=None) -> int:
    files = []
    for arg in argv if argv is not None else sys.argv[1:]:
        for p in sorted(glob.glob(arg)) if any(c in arg for c in "*?[") else [arg]:
            p = Path(p)
            files += sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p]
    leaks = [leak for f in files for leak in scan_file(f)]
    for leak in leaks:
        print(leak)
    print(f"{len(files)} files scanned, {len(leaks)} leak(s)")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
