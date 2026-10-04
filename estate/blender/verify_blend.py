"""Reopen a .blend in a fresh Blender and check it is a healthy Bonsai project. Runs inside blender.exe.

Args (JSON): blend, expected (optional {ifc class: object count} from make_blend), links (optional: require every
Bonsai link to be loaded and every linked library path to be relative), result.

Bonsai is enabled BEFORE ``wm.open_mainfile`` so its load_post handler re-reads the IFC (resolved from the
relative ``BIMProperties.ifc_file``) and relinks objects by STEP id. Checks: the IFC opened, the stored path is
relative, the IFC is the one the .blend was built from (sha256 in scene["estate"]), every non-feature IfcElement
has an object, objects are bound to entities of their own class (a stale .blend binds walls to doors), and the
counts per class equal the build's.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True         # Blender ignores -B; keep the project free of __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from estate.blender import _boot  # noqa: E402


def check_links(errors: list) -> list:
    import bpy
    import bonsai.tool as tool
    from estate.blender.make_estate import instanced_stats
    out = []
    props = tool.Project.get_project_props()
    libs = {Path(bpy.path.abspath(lib.filepath)).resolve(): lib for lib in bpy.data.libraries}
    for link in props.links:
        empty = tool.Project.get_link_empty_handle(link)
        inst = empty.instance_collection if empty is not None else None
        stats = instanced_stats(inst)
        out.append({"filepath": link.filepath, "is_loaded": bool(link.is_loaded), **stats,
                    "library": inst.library.filepath if inst is not None and inst.library else None})
        if not link.is_loaded:
            errors.append(f"link not loaded: {link.filepath}")
        elif stats["instanced_polygons"] == 0:
            errors.append(f"link loaded but its collection has no geometry: {link.filepath}")
    for path, lib in libs.items():
        if not lib.filepath.endswith(".cache.blend"):
            continue           # e.g. Bonsai's own workspace.blend
        if not lib.filepath.startswith("//"):
            errors.append(f"library path is absolute: {lib.filepath}")
        if not path.exists():
            errors.append(f"library file missing: {lib.filepath}")
    return out


def run(a: dict) -> dict:
    import bpy
    _boot.boot()
    import bonsai.tool as tool

    blend = Path(a["blend"]).resolve()
    assert blend.exists(), f"missing {blend}"
    t = time.time()
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    t_open = time.time() - t
    errors = []
    stored = tool.Blender.get_bim_props().ifc_file
    f = tool.Ifc.get()
    n_links = len(tool.Project.get_project_props().links)
    if f is None and a.get("links") and n_links:
        # an estate overview made by the link_ifc fallback: no host IFC, links stored in the .blend only
        links = check_links(errors)
        res = {"blend": str(blend), "ifc_file_stored": stored, "open_seconds": round(t_open, 2), "links": links,
               "libraries": [lib.filepath for lib in bpy.data.libraries], "errors": errors, "ok": not errors,
               "blend_bytes": os.path.getsize(blend), "host": False}
        if errors:
            res["error"] = "; ".join(errors)
        return res
    if f is None:
        raise AssertionError(f"Bonsai has no IFC after reopening {blend.name} (stored path {stored!r})")
    if not _boot.is_relative(stored):
        errors.append(f"IFC path stored absolute: {stored!r}")
    ifc_abs = Path(tool.Ifc.get_path()).resolve()
    meta = json.loads(bpy.context.scene.get("estate", "{}"))
    sha = _boot.sha256(ifc_abs)
    if meta.get("ifc_sha256") and meta["ifc_sha256"] != sha:
        errors.append(f"stale .blend: {ifc_abs.name} changed since the .blend was built (rebuild it)")
    expected = _boot.ifc_expected_counts(f)
    counts = _boot.ifc_object_counts()
    if counts["element_objects"] < expected["non_feature_elements"]:
        errors.append(f"{counts['element_objects']} element objects < {expected['non_feature_elements']} IfcElement")
    if counts["n_mismatched"]:
        errors.append(f"{counts['n_mismatched']} objects bound to entities of another class, e.g. {counts['mismatched'][:3]}")
    want = a.get("expected") or meta.get("counts")
    diff = {}
    if want:
        for cls in sorted(set(want) | set(counts["by_class"])):
            if want.get(cls, 0) != counts["by_class"].get(cls, 0):
                diff[cls] = [want.get(cls, 0), counts["by_class"].get(cls, 0)]
        if diff:
            errors.append(f"object counts differ from the build (class: [built, reopened]): {diff}")
    # links are only required for estate overviews; a SITE.blend lists its links without loading them
    links = check_links(errors if a.get("links") else [])
    if a.get("links") and not links:
        errors.append("no Bonsai links in the project")
    res = {"blend": str(blend), "ifc_file_stored": stored, "ifc_resolved": str(ifc_abs), "open_seconds": round(t_open, 2),
           "counts": counts, "expected": expected, "diff": diff, "links": links,
           "libraries": [lib.filepath for lib in bpy.data.libraries], "errors": errors, "ok": not errors,
           "blend_bytes": os.path.getsize(blend)}
    if errors:
        res["error"] = "; ".join(errors)
    return res


if __name__ == "__main__":
    _boot.main(run)
