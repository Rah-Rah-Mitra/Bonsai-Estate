"""One IFC -> one Bonsai .blend next to it (same stem), with a relative IFC path. Runs inside blender.exe.

Args (JSON): ifc (path), blend (default <ifc stem>.blend beside it), deflection_tolerance (0.05), cameras (true),
compress (true), result. The .blend is saved BEFORE the IFC is loaded because Bonsai can only store the IFC path
relative to a saved .blend (``use_relative_path=True`` forces ``should_start_fresh_session=False``). With
``is_advanced=True`` loading stops after opening the file so the project properties (no false origin, CPU
multiprocessing, no filter) can be set before ``bim.load_project_elements`` creates the objects.

Bonsai's load_post handler re-reads the IFC and relinks objects by STEP id when the .blend is reopened, so the
.blend must be rebuilt whenever its IFC is regenerated; the IFC's sha256 is stored in scene["estate"] to detect that.
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


def load_ifc(ifc: Path, deflection=0.05, elements=True) -> float:
    """Load ``ifc`` into the current (saved) .blend with a relative path; returns seconds spent creating objects."""
    import bpy
    import bonsai.tool as tool
    rel = ifc.resolve().relative_to(Path(bpy.path.abspath("//")).resolve())
    r = bpy.ops.bim.load_project(filepath=str(ifc.resolve()), is_advanced=True, use_relative_path=True,
                                 should_start_fresh_session=False, skip_recent=True, skip_autosave_recovery=True)
    assert r == {"FINISHED"}, f"bim.load_project returned {r}"
    assert tool.Ifc.get() is not None, f"Bonsai did not open {ifc}"
    props = tool.Project.get_project_props()
    props.false_origin_mode = "DISABLED"
    props.should_use_cpu_multiprocessing = True
    props.deflection_tolerance = float(deflection)
    props.filter_mode = "NONE"
    props.should_filter_spatial_elements = False
    props.element_limit_mode = "UNLIMITED"
    _boot.log(f"opened {rel.as_posix()} ({props.total_elements} IfcElement), creating objects")
    t = time.time()
    if elements:
        r = bpy.ops.bim.load_project_elements()
        assert r == {"FINISHED"}, f"bim.load_project_elements returned {r}"
    return time.time() - t


def run(a: dict) -> dict:
    import bpy
    _boot.boot()
    import bonsai.tool as tool
    from estate.blender import views

    ifc = Path(a["ifc"]).resolve()
    blend = Path(a.get("blend") or ifc.with_suffix(".blend")).resolve()
    assert ifc.exists(), f"missing IFC {ifc}"
    _boot.clear_scene()
    t0 = time.time()
    final = blend
    blend = blend.with_name(blend.stem + ".tmp.blend")     # same folder: Bonsai's relative IFC path stays valid
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    t_load0 = time.time()
    t_elements = load_ifc(ifc, a.get("deflection_tolerance", 0.05))
    t_load = time.time() - t_load0

    stored = tool.Blender.get_bim_props().ifc_file
    assert _boot.is_relative(stored), f"IFC path stored absolute: {stored!r}"
    expected = _boot.ifc_expected_counts(tool.Ifc.get())
    counts = _boot.ifc_object_counts()
    assert counts["element_objects"] >= expected["non_feature_elements"], \
        f"{counts['element_objects']} element objects < {expected['non_feature_elements']} non-feature IfcElement"
    assert counts["n_mismatched"] == 0, f"objects bound to the wrong entities: {counts['mismatched']}"

    views.hide_non_render()
    views.setup_workbench()
    bbox = views.world_bbox()
    if a.get("cameras", True) and bbox is not None:
        views.qa_cameras(bbox)
    scene = bpy.context.scene
    meta = {"ifc": stored, "ifc_sha256": _boot.sha256(ifc), "counts": counts["by_class"], "expected": expected,
            "bbox": [list(bbox[0]), list(bbox[1])] if bbox else None, "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    scene["estate"] = json.dumps(meta)
    t_save0 = time.time()
    types = bpy.data.collections.get(views.TYPE_COLLECTION)
    if types is not None:          # type geometry sits at the origin: keep it out of views and renders
        types.hide_render = True
        types.hide_viewport = True
    bpy.ops.wm.save_mainfile(compress=bool(a.get("compress", True)))
    os.replace(blend, final)                                # a killed job never leaves a half-written .blend
    for leftover in (blend.with_suffix(".blend1"),):
        if leftover.exists():
            leftover.unlink()
    blend = final
    t_save = time.time() - t_save0
    pending = len(tool.Project.get_project_props().pending_opening_recut)
    return {"blend": str(blend), "ifc": str(ifc), "ifc_file_stored": stored, "load_seconds": round(t_load, 2),
            "elements_seconds": round(t_elements, 2), "save_seconds": round(t_save, 2),
            "total_seconds": round(time.time() - t0, 2), "blend_bytes": blend.stat().st_size,
            "ifc_bytes": ifc.stat().st_size, "expected": expected, "counts": counts, "ifc_sha256": meta["ifc_sha256"],
            "bbox": meta["bbox"], "uncut_openings_elements": pending,
            "links": [l.filepath for l in tool.Project.get_project_props().links]}


if __name__ == "__main__":
    _boot.main(run)
