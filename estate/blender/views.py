"""Scene helpers for the Blender stages: world bounds, Workbench look, fitted QA cameras and still renders.

Bounds come from the evaluated depsgraph instances, so the same code frames a building .blend (local objects)
and ESTATE.blend (collection instances of linked cache .blend files). Cameras are fitted analytically: ortho
views project the 8 bounding-box corners into the camera plane, perspective views frame the bounding sphere.
"""
from __future__ import annotations

import math

import bpy
from mathutils import Matrix, Vector

from estate.blender._boot import NON_RENDER_CLASSES
from estate.report.street import SKY  # noqa: F401  (default eye-level background; bpy-free, so tests read it too)

BACKGROUND = (0.93, 0.94, 0.96)
# Render metadata Blender writes into every PNG as text chunks (File, Date, RenderTime, ...): all switched off, so a
# published render carries neither the .blend's absolute path nor a clock. Names this Blender lacks are skipped.
STAMP_FIELDS = ("date", "time", "render_time", "frame", "frame_range", "memory", "hostname", "camera", "lens", "scene",
                "marker", "filename", "sequencer_strip", "note")


def ifc_class(ob) -> str:
    return ob.name.split("/", 1)[0] if ob.name.startswith("Ifc") and "/" in ob.name else ""


TYPE_COLLECTION = "IfcTypeProduct"     # Bonsai puts type geometry (door / window types) here, at the origin


def is_type_object(ob) -> bool:
    return ifc_class(ob).endswith("Type") or any(
        c.name == TYPE_COLLECTION for c in ob.users_collection)


def hide_non_render() -> int:
    """Spaces, openings, spatial containers and annotations stay out of renders (they load as wireframes).

    Only geometry is touched: Bonsai's link empties are named after the linked IfcProject collection, and hiding
    an instancing empty would hide the whole linked building.
    """
    n = 0
    for ob in bpy.data.objects:
        if ob.type not in ("MESH", "CURVE") or ob.library is not None:
            continue
        if (ifc_class(ob) in NON_RENDER_CLASSES or is_type_object(ob)) and not ob.hide_render:
            ob.hide_render = True
            n += 1
    return n


def world_bbox(depsgraph=None, filter_fn=None):
    """(min Vector, max Vector) over every rendered mesh instance, or None when the scene is empty."""
    dg = depsgraph or bpy.context.evaluated_depsgraph_get()
    lo = Vector((math.inf,) * 3)
    hi = Vector((-math.inf,) * 3)
    found = False
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type != "MESH":
            continue
        src = ob.original
        if ifc_class(src) in NON_RENDER_CLASSES or src.hide_render or (src.library is None and is_type_object(src)):
            continue
        if filter_fn is not None and not filter_fn(src):
            continue
        mw = inst.matrix_world
        for c in ob.bound_box:
            p = mw @ Vector(c)
            lo = Vector(map(min, lo, p))
            hi = Vector(map(max, hi, p))
        found = True
    return (lo, hi) if found else None


def setup_workbench(scene=None, outline=True) -> None:
    """Workbench, material colours (Bonsai's IfcSurfaceStyle diffuse), cavity and shadows, true colours; no stamp
    and no render metadata."""
    scene = scene or bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.use_stamp = False
    for field in STAMP_FIELDS:
        if hasattr(scene.render, f"use_stamp_{field}"):
            setattr(scene.render, f"use_stamp_{field}", False)
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "MATERIAL"
    sh.show_cavity = True
    sh.cavity_type = "BOTH"
    sh.show_shadows = True
    sh.shadow_intensity = 0.35
    sh.show_object_outline = outline
    sh.show_backface_culling = False
    scene.display.light_direction = (0.45, -0.35, 0.82)
    scene.view_settings.view_transform = "Standard"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = BACKGROUND
    for screen in bpy.data.screens:     # the GUI viewport matches the renders
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "SOLID"
                    space.shading.color_type = "MATERIAL"
                    space.shading.show_cavity = True
                    space.clip_end = max(space.clip_end, 5000.0)


def _camera(name):
    cam = bpy.data.objects.get(name)
    if cam is None or cam.type != "CAMERA":
        cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
        bpy.context.scene.collection.objects.link(cam)
    return cam


def direction(azimuth_deg: float, elevation_deg: float) -> Vector:
    """Unit vector from the target towards the camera (azimuth from +x, counter-clockwise; +y is north)."""
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    return Vector((math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)))


def fit_camera(name, bbox, view_dir: Vector, ortho=True, res=(1600, 1200), margin=1.06, lens=35.0):
    """Create/update camera ``name`` looking along -view_dir at bbox, framed for resolution res."""
    lo, hi = bbox
    centre = (lo + hi) / 2
    d = view_dir.normalized()
    rot = d.to_track_quat("Z", "Y").to_matrix()
    corners = [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]
    radius = max((c - centre).length for c in corners) or 1.0
    cam = _camera(name)
    cd = cam.data
    rx, ry = res
    if ortho:
        local = [rot.transposed() @ (c - centre) for c in corners]
        xs, ys = [p.x for p in local], [p.y for p in local]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        shift = rot @ Vector(((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, 0.0))
        cd.type = "ORTHO"
        cd.shift_x = cd.shift_y = 0.0
        cd.ortho_scale = (max(w, h * rx / ry) if rx >= ry else max(h, w * ry / rx)) * margin
        dist = radius * 2.0
        loc = centre + shift + d * dist
    else:
        cd.type = "PERSP"
        cd.lens = lens
        cd.sensor_fit = "AUTO"
        t_big = cd.sensor_width / 2 / lens                    # tan(half angle) along the larger image side
        tx, ty = (t_big, t_big * ry / rx) if rx >= ry else (t_big * rx / ry, t_big)
        tx, ty = tx / margin, ty / margin
        # every corner inside the frustum: |x| <= (dist - z) tan(hx), |y| <= (dist - z) tan(hy) (upper bound) ...
        local = [rot.transposed() @ (c - centre) for c in corners]
        hi_d = max(max(p.z + abs(p.x) / tx, p.z + abs(p.y) / ty) for p in local)
        lo_d = max(p.z for p in local) + 1e-3

        def spans(dd):
            us = [p.x / (dd - p.z) for p in local]
            vs = [p.y / (dd - p.z) for p in local]
            return min(us), max(us), min(vs), max(vs)

        for _ in range(40):           # ... then the closest distance that fits once the frame is lens-shifted
            mid = (lo_d + hi_d) / 2
            u0, u1, v0, v1 = spans(mid)
            if (u1 - u0) / 2 <= tx and (v1 - v0) / 2 <= ty:
                hi_d = mid
            else:
                lo_d = mid
        dist = hi_d
        u0, u1, v0, v1 = spans(dist)
        cd.shift_x = (u0 + u1) / 2 / (2 * t_big)
        cd.shift_y = (v0 + v1) / 2 / (2 * t_big)
        loc = centre + d * dist
    cd.clip_start = max(0.05, (dist - radius * 1.5) * 0.5) if not ortho else 0.1
    cd.clip_end = dist + radius * 2.5
    cam.matrix_world = Matrix.Translation(loc) @ rot.to_4x4()
    cam["estate_view"] = {"ortho": ortho, "res": list(res)}
    return cam


def look_camera(name, eye, target, lens=24.0, clip=(0.1, 2000.0), shift=(0.0, 0.0)):
    """Create/update perspective camera ``name`` at ``eye`` looking at ``target`` (an eye-level street view).

    Not fitted to a bounding box: the eye is where a person stands. The orientation is a track-to through the
    direction quaternion (local -Z towards the target, local +Y towards world up), so a level eye -> target line
    keeps verticals vertical and ``shift`` (lens shift, in units of the larger image side) raises the frame the way
    an architectural shift lens does. Clip distances are set explicitly: fit_camera's bounding-sphere clipping
    would cut the shelter roof a metre above the eye.
    """
    eye, target = Vector(eye), Vector(target)
    d = eye - target
    assert d.length > 1e-6, f"camera {name}: eye and target coincide"
    rot = d.normalized().to_track_quat("Z", "Y").to_matrix()
    cam = _camera(name)
    cd = cam.data
    cd.type = "PERSP"
    cd.lens = float(lens)
    cd.sensor_fit = "AUTO"
    cd.sensor_width = 36.0
    cd.shift_x, cd.shift_y = (float(v) for v in shift)
    cd.clip_start, cd.clip_end = (float(v) for v in clip)
    cam.matrix_world = Matrix.Translation(eye) @ rot.to_4x4()
    cam["estate_view"] = {"ortho": False, "eye": list(eye), "target": list(target), "lens": float(lens)}
    return cam


def eye_level_look(scene=None, sky=SKY):
    """Sky-coloured background for an eye-level view; returns restore() for the QA views' pale studio background.

    Everything else stays as setup_workbench left it (studio light, cavity, outline, shadows on): a towers-on-sky
    street view reads well with it, and turning the Workbench shadow light towards the camera's back changed
    under a tenth of the pixels, imperceptibly, so there is no per-view sun.
    """
    scene = scene or bpy.context.scene
    saved = tuple(scene.world.color)
    scene.world.color = sky

    def restore():
        scene.world.color = saved
    return restore


def qa_cameras(bbox, res=(1600, 1200)):
    """QA_Iso (ortho from the south-east, 35 degrees up) and QA_Top (ortho plan, north up)."""
    iso = fit_camera("QA_Iso", bbox, direction(-45.0, 35.264), ortho=True, res=res)
    top = fit_camera("QA_Top", bbox, Vector((0.0, 0.0, 1.0)), ortho=True, res=(res[0], res[0]))
    bpy.context.scene.camera = iso
    return iso, top


def render_still(cam, path, res=(1600, 1200), scene=None) -> str:
    scene = scene or bpy.context.scene
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 90
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    return str(path)
