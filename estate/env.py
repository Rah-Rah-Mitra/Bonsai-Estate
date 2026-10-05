"""Paths and version pins for the estate pipeline (Windows, Blender 5.2 + Bonsai 0.9).

Everything runs on Blender's bundled CPython (isolated mode, ``-I -B``); Bonsai's site-packages provide
ifcopenshell, shapely, networkx, PIL, ifctester. ``bootstrap()`` makes those importable.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLENDER_VERSION = "5.2"
BLENDER_DIR = Path(os.environ.get("ESTATE_BLENDER_DIR", r"C:\Program Files\Blender Foundation\Blender 5.2"))
BLENDER_EXE = BLENDER_DIR / "blender.exe"
BLENDER_PY = BLENDER_DIR / BLENDER_VERSION / "python" / "bin" / "python.exe"
APPDATA = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
BONSAI_SITE = (APPDATA / "Blender Foundation" / "Blender" / BLENDER_VERSION / "extensions" / ".local" / "lib"
               / "python3.13" / "site-packages")
BONSAI_ADDON = "bl_ext.blender_org.bonsai"
VENDOR = ROOT / "vendor" / "py313"

CONFIG = ROOT / "config"
MODEL = ROOT / "model"
REPORTS = ROOT / "reports"
BUILD = ROOT / "build"
LEGACY = ROOT / "legacy"

PINS = {"python": (3, 13), "ifcopenshell": "0.9.0", "blender": "5.2"}


def bootstrap() -> None:
    """Put the project, Bonsai's site-packages and (last) the optional vendor dir on sys.path."""
    for i, p in enumerate((str(ROOT), str(BONSAI_SITE))):
        if p not in sys.path:
            sys.path.insert(i, p)
    if VENDOR.exists() and str(VENDOR) not in sys.path:
        sys.path.append(str(VENDOR))


def rel(path: Path | str) -> str:
    """Project-relative POSIX path for reports and logs."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _folder(path: Path) -> re.Pattern | None:
    """A folder as text spells it: either slash, any case on Windows, a whole name at its end (so the project folder
    does not match the start of "<project>-release"); None for a drive or filesystem root."""
    parts = [p for p in re.split(r"[\\/]+", str(path)) if p]
    if len(parts) < 2:
        return None
    return re.compile(r"[\\/]+".join(map(re.escape, parts)) + r"(?![^\\/\s\"'),;:\]>])", re.I if os.name == "nt" else 0)


SCRUB = [(_folder(p), token) for p, token in sorted(((ROOT, "."), (BLENDER_DIR, "<blender>"), (Path.home(), "~")),
                                                    key=lambda pt: -len(str(pt[0])))    # the project is in home
         if _folder(p)]


def scrub(doc):
    """Error text for a published report (a traceback, an exception message, a Blender log tail) with the project
    folder written ".", Blender's folder "<blender>" and the home folder "~", so it names no folder of the machine
    that built it. Takes a string, or a parsed JSON document whose keys and strings are all scrubbed."""
    if isinstance(doc, str):
        for folder, token in SCRUB:
            doc = folder.sub(lambda m: token, doc)
        return doc
    if isinstance(doc, dict):
        return {scrub(k): scrub(v) for k, v in doc.items()}
    if isinstance(doc, list):
        return [scrub(v) for v in doc]
    return doc
