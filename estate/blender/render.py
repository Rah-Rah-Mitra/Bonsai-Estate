"""Workbench QA renders of a building or estate .blend. Runs inside blender.exe; the .blend is never saved.

Args (JSON): blend, out (folder, default <blend folder>/renders), views (default iso, top, aerial_NE, aerial_SW,
cutaway, cutplan; [] renders only the cameras), storey (name or index of the cutaway storey; default the sixth
storey from the ground, like the legacy L5 cutaway), cut_height (m above the storey's FFL, 1.5), width (1600),
prefix (blend stem), cameras ({view: {eye, target, lens (24), clip ([0.1, 2000]), shift ([0, 0]), res
([width, width * 9 / 16]), sky (views.SKY)}}, e.g. estate/report/street.py's bus-stop view), result.

Bonsai is enabled so the IFC opens with the .blend and the cutaway can use each element's storey: an element is
hidden when its storey is above the cut storey or its bounding box starts above the cut plane. Spaces, openings
and spatial containers are never rendered. Linked estates (ESTATE.blend) have no storeys, so their cutaway is
skipped. Views: iso = ortho from the south-east, top = ortho plan (north up), aerial_NE / aerial_SW = perspective
from 30 degrees up, cutaway = ortho iso of the cut storey, cutplan = ortho plan of the cut storey. Each placed
camera renders {prefix}_{view}.png in perspective from its eye (views.look_camera: not fitted to the model, clip
set explicitly) against a sky-coloured background, before the cut views hide anything.

Every job also writes {prefix}_views.json next to the images: each rendered view's camera (matrix_world row-major
in the estate frame, lens, sensor, shift, resolution, clip; rounded to 1e-6, no clock, no path), so a viewer can
match a render exactly (camera_info).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True         # Blender ignores -B; keep the project free of __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from estate.blender import _boot  # noqa: E402

VIEWS = ("iso", "top", "aerial_NE", "aerial_SW", "cutaway", "cutplan")
VIEWS_SCHEMA = "sample-town-n5/render-views/1"


def _n(v, nd=6):
    return round(float(v), nd) + 0.0


def camera_info(cam, res) -> dict:
    """The camera of one render: what a viewer needs to reproduce it. Blender cameras look along their local -Z
    with +Y up; lens and sensor in mm, shifts in units of the larger image side (sensor_fit AUTO)."""
    cd, M = cam.data, cam.matrix_world
    out = {"type": cd.type, "matrix_world": [[_n(M[i][j]) for j in range(4)] for i in range(4)],
           "lens": _n(cd.lens), "sensor_width": _n(cd.sensor_width), "sensor_fit": cd.sensor_fit,
           "shift_x": _n(cd.shift_x), "shift_y": _n(cd.shift_y), "res": [int(v) for v in res],
           "clip": [_n(cd.clip_start), _n(cd.clip_end)]}
    if cd.type == "ORTHO":
        out["ortho_scale"] = _n(cd.ortho_scale)
    return out


def storeys():
    """[(name, elevation, entity)] sorted by elevation, from the loaded IFC (empty without one)."""
    import bonsai.tool as tool
    f = tool.Ifc.get()
    if f is None:
        return []
    out = [(s.Name or f"#{s.id()}", float(s.Elevation or 0.0), s) for s in f.by_type("IfcBuildingStorey")]
    return sorted(out, key=lambda t: (t[1], t[0]))


def pick_storey(levels, want):
    if not levels:
        return None
    if want is None:
        return levels[min(5, len(levels) - 2)] if len(levels) > 1 else levels[0]
    if isinstance(want, int) or str(want).lstrip("-").isdigit():
        return levels[int(want)]
    for lv in levels:
        if lv[0] == want:
            return lv
    raise KeyError(f"no storey {want!r}; storeys: {[lv[0] for lv in levels]}")


def apply_cut(level, levels, cut_height) -> dict:
    """Hide everything above the cut; returns {object name: previous hide_render} to restore."""
    import bpy
    import bonsai.tool as tool
    import ifcopenshell.util.element as uel
    from mathutils import Vector
    name, elev, _ = level
    cut_z = elev + cut_height
    elev_of = {lv[2].id(): lv[1] for lv in levels}
    saved = {}
    for ob in bpy.data.objects:
        if ob.type != "MESH" or ob.hide_render:
            continue
        el = tool.Ifc.get_entity(ob)
        above = False
        if el is not None:
            cont = uel.get_container(el) or (uel.get_container(uel.get_aggregate(el)) if uel.get_aggregate(el) else None)
            if cont is not None and cont.id() in elev_of and elev_of[cont.id()] > elev + 1e-6:
                above = True
        if not above:
            zmin = min((ob.matrix_world @ Vector(c)).z for c in ob.bound_box)
            above = zmin >= cut_z - 1e-3
        if above:
            saved[ob.name] = ob.hide_render
            ob.hide_render = True
    return saved


def run(a: dict) -> dict:
    import bpy
    _boot.boot()
    from mathutils import Vector
    from estate.blender import views as V

    blend = Path(a["blend"]).resolve()
    out = Path(a.get("out") or (blend.parent / "renders")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    prefix = a.get("prefix") or blend.stem
    width = int(a.get("width", 1600))
    res = (width, int(width * 0.75))
    wanted = list(VIEWS if a.get("views") is None else a["views"])     # [] = only the placed cameras
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene = bpy.context.scene
    V.hide_non_render()
    V.setup_workbench(scene)
    bbox = V.world_bbox()
    assert bbox is not None, f"nothing to render in {blend.name}"
    files, timings, skipped, cams = [], {}, [], {}

    def shot(view, cam, r):
        t = time.time()
        path = out / f"{prefix}_{view}.png"
        V.render_still(cam, path, r, scene)
        files.append(str(path))
        cams[view] = camera_info(cam, r)
        timings[view] = round(time.time() - t, 2)

    for view in wanted:
        if view == "iso":
            shot(view, V.fit_camera("R_iso", bbox, V.direction(-45.0, 35.264), True, res), res)
        elif view == "top":
            shot(view, V.fit_camera("R_top", bbox, Vector((0.0, 0.0, 1.0)), True, (width, width)), (width, width))
        elif view in ("aerial_NE", "aerial_SW"):
            az = 45.0 if view == "aerial_NE" else -135.0
            shot(view, V.fit_camera(f"R_{view}", bbox, V.direction(az, 30.0), False, res), res)
    cameras = a.get("cameras") or {}
    for view in sorted(cameras):        # placed cameras, before the cut views hide anything
        c = cameras[view]
        r = tuple(int(v) for v in c.get("res") or (width, round(width * 9 / 16)))
        cam = V.look_camera(f"R_{view}", c["eye"], c["target"], c.get("lens", 24.0), c.get("clip", (0.1, 2000.0)),
                            c.get("shift", (0.0, 0.0)))
        restore = V.eye_level_look(scene, tuple(c.get("sky") or V.SKY))
        try:
            shot(view, cam, r)
        finally:
            restore()
    cut_views = [v for v in wanted if v in ("cutaway", "cutplan")]
    level = None
    if cut_views:
        levels = storeys()
        level = pick_storey(levels, a.get("storey"))
        if level is None:
            skipped += cut_views
        else:
            apply_cut(level, levels, float(a.get("cut_height", 1.5)))
            cb = V.world_bbox()
            tag = level[0].replace(" ", "")
            for view in cut_views:
                if view == "cutaway":
                    cam = V.fit_camera("R_cutaway", cb, V.direction(-45.0, 40.0), True, res)
                    shot(f"cutaway_{tag}", cam, res)
                else:
                    cam = V.fit_camera("R_cutplan", cb, Vector((0.0, 0.0, 1.0)), True, (width, width))
                    shot(f"cutplan_{tag}", cam, (width, width))
    unknown = [v for v in wanted if v not in VIEWS and v not in cameras]
    views = out / f"{prefix}_views.json"
    views.write_text(json.dumps({"schema": VIEWS_SCHEMA, "frame": "estate, metres, Z up", "blend": blend.name,
                                 "views": cams}, indent=1) + "\n", encoding="utf-8")
    return {"blend": str(blend), "out": str(out), "files": files, "views": str(views), "render_seconds": timings,
            "storey": level[0] if level else None, "skipped": skipped + unknown,
            "bbox": [list(bbox[0]), list(bbox[1])], "resolution": list(res), "cameras": sorted(cameras)}


if __name__ == "__main__":
    _boot.main(run)
