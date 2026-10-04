"""Paths and version pins for the estate pipeline (Windows, Blender 5.2 + Bonsai 0.9).

Everything runs on Blender's bundled CPython (isolated mode, ``-I -B``); Bonsai's site-packages provide
ifcopenshell, shapely, networkx, PIL, ifctester. ``bootstrap()`` makes those importable.
"""
from __future__ import annotations

import os
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
