"""Boot code imported by every estate script that runs inside blender.exe (background or GUI).

Blender is launched by estate/blender/run.py as
``blender.exe -b --factory-startup --python-exit-code 1 --python estate/blender/<script>.py -- --json <args.json>``.
The args file is JSON; it carries the job's inputs plus ``result``, the path where this module writes the job's
JSON result (it is also printed after a marker line so a log alone is enough to recover it).

Order matters when booting Bonsai: a root logging handler must exist BEFORE the add-on is enabled, because
``bim.load_project_elements`` calls ``logging.basicConfig(filename=<bonsai data>/process.log, level=DEBUG)``,
which would otherwise make every parallel job append DEBUG lines to one shared log. The add-on must be enabled
with ``default_set=True`` (False raises KeyError while loading a project). ``save_version = 0`` stops Blender
writing .blend1 backups next to the deliverables.
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from estate import env  # noqa: E402  (pure paths, safe inside Blender)

MARKER = "@@ESTATE_RESULT@@"
PINS = {"blender": (5, 2), "ifcopenshell": "0.9.0"}
NON_RENDER_CLASSES = ("IfcSpace", "IfcOpeningElement", "IfcVirtualElement", "IfcAnnotation", "IfcGrid", "IfcGridAxis",
                      "IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcProject")
_T0 = time.time()


def args() -> dict:
    """Job arguments: the JSON file named after ``--json`` (or an inline JSON string) following ``--``."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--json" not in argv:
        return {}
    raw = argv[argv.index("--json") + 1]
    p = Path(raw)
    if p.suffix.lower() == ".json" and p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(raw)


def log(msg: str) -> None:
    print(f"[estate {time.time() - _T0:7.1f}s] {msg}", flush=True)


def boot(enable_bonsai: bool = True, log_file: str | None = None, set_prefs: bool = True) -> None:
    """Root logging handler, Bonsai, no .blend1 backups, version pins.

    ``set_prefs=False`` (interactive sessions) leaves the user's preferences alone: they may be auto-saved on exit.
    """
    import bpy
    import addon_utils

    rootlog = logging.getLogger()
    if not rootlog.handlers:
        h = logging.FileHandler(log_file, encoding="utf-8") if log_file else logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        rootlog.addHandler(h)
    rootlog.setLevel(logging.WARNING)
    if set_prefs:
        bpy.context.preferences.filepaths.save_version = 0
    assert tuple(bpy.app.version[:2]) == PINS["blender"], f"Blender {bpy.app.version_string}, expected 5.2"
    if enable_bonsai:
        addon_utils.enable(env.BONSAI_ADDON, default_set=True)
        import ifcopenshell
        assert ifcopenshell.version == PINS["ifcopenshell"], f"ifcopenshell {ifcopenshell.version}, expected 0.9.0"
        import bonsai.tool  # noqa: F401  (fails loudly if the add-on did not register)


def clear_scene() -> None:
    """Remove the factory startup objects (cube, camera, light) and their data."""
    import bpy
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights, bpy.data.materials):
        for d in list(coll):
            if d.users == 0:
                coll.remove(d)


def memory_mb() -> dict:
    """Working set and peak working set of this process (Windows), in MB."""
    if os.name != "nt":
        return {}

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return {}
    return {"working_set_mb": round(c.WorkingSetSize / 2**20, 1), "peak_working_set_mb": round(c.PeakWorkingSetSize / 2**20, 1)}


def process_peak_mb(handle) -> float | None:
    """Peak working set of another process given its Win32 handle (e.g. ``Popen._handle``)."""
    if os.name != "nt":
        return None

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t)] + [(f"f{i}", ctypes.c_size_t) for i in range(6)]

    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(ctypes.c_void_p(int(handle)), ctypes.byref(c), c.cb):
        return None
    return round(c.PeakWorkingSetSize / 2**20, 1)


def is_relative(path: str) -> bool:
    return bool(path) and not os.path.isabs(path) and not path.startswith("\\\\")


def ifc_object_counts() -> dict:
    """Objects bound to IFC entities, by IFC class, plus totals (needs Bonsai and a loaded IFC)."""
    import bpy
    import bonsai.tool as tool
    by_class, mismatched, elements, link_empties = {}, [], 0, 0
    for ob in bpy.data.objects:
        if ob.library:
            continue
        if ob.instance_type == "COLLECTION" and ob.instance_collection is not None:
            link_empties += 1      # a Bonsai link: bound to its IfcDocumentReference, named after the collection
            continue
        el = tool.Ifc.get_entity(ob)
        if el is None:
            continue
        cls = el.is_a()
        by_class[cls] = by_class.get(cls, 0) + 1
        elements += el.is_a("IfcElement")
        if "/" in ob.name and ob.name.split("/", 1)[0] != cls:
            mismatched.append(ob.name)
    return {"objects": len(bpy.data.objects), "ifc_objects": sum(by_class.values()), "element_objects": elements,
            "by_class": dict(sorted(by_class.items())), "mismatched": mismatched[:20], "n_mismatched": len(mismatched),
            "link_empties": link_empties}


def ifc_expected_counts(f) -> dict:
    """What a full load must produce: every non-feature IfcElement becomes an object."""
    els = f.by_type("IfcElement")
    feats = len(f.by_type("IfcFeatureElement"))
    return {"ifc_elements": len(els), "feature_elements": feats, "non_feature_elements": len(els) - feats,
            "spaces": len(f.by_type("IfcSpace")), "storeys": len(f.by_type("IfcBuildingStorey"))}


def sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def emit(result: dict, a: dict | None = None) -> None:
    """Write the result JSON to the job's result file and print it after the marker line."""
    result.setdefault("seconds", round(time.time() - _T0, 2))
    result.update({k: v for k, v in memory_mb().items() if k not in result})
    text = json.dumps(result, default=str)
    if a and a.get("result"):
        Path(a["result"]).write_text(text, encoding="utf-8")
    print(MARKER, flush=True)
    print(text, flush=True)


def main(fn) -> None:
    """Run ``fn(args) -> dict`` as a job: emit its result, or an error result and a non-zero exit code."""
    a = args()
    try:
        res = fn(a) or {}
        res.setdefault("ok", True)
        emit(res, a)
    except BaseException as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        emit({"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": tb[-4000:]}, a)
        raise
