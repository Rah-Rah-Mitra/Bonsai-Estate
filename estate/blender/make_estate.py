"""Estate overview .blend: the host SITE.ifc plus every building linked through Bonsai. Runs inside blender.exe.

Args (JSON): site (host IFC with LINKED_MODEL references, see estate/ifc/links.py), blend (default ESTATE.blend
beside the site), query ("IfcElement, ! IfcFeatureElement"), use_cache (false: the building IFCs were just
regenerated), links (building IFCs for the fallbacks), glb (LOD glb files for the last fallback), result.

Primary path: load the host as make_blend does (saved .blend first, relative path), which fills
``BIMProjectProperties.links`` from the LINKED_MODEL references, then ``bim.load_link`` each link. load_link
spawns a child Blender that writes <ifc>.cache.blend/.cache.json/.cache.sqlite next to the linked IFC and links
that file's IfcProject collection into an instancing empty placed by the reference's Identification matrix.
Bonsai swallows failures (the operator always reports FINISHED), so ``link.is_loaded`` is asserted. Library
paths are made relative so the estate folder can move. The child's temp .py files are written to a job temp
dir (``tempfile.tempdir``) that is removed afterwards.

With ``warm`` and ``only`` (link locations) the job only builds those caches (see ``warm``); estate-blend --jobs
runs several warm jobs in parallel before the final one.

Fallbacks: (1) no or unloadable host -> ``bim.link_ifc`` per building IFC (links live in the .blend only);
(2) nothing linked -> import the LOD glb files (plain meshes, no Bonsai data).
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True         # Blender ignores -B; keep the project free of __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from estate.blender import _boot  # noqa: E402

QUERY = "IfcElement, ! IfcFeatureElement"
CHILD_PEAKS: list = []


@contextlib.contextmanager
def measured_children():
    """Record the peak working set of each child Blender that Bonsai starts with subprocess.run."""
    real = subprocess.run

    def run(cmd, *args, **kw):
        if kw.get("capture_output") or kw.get("input") is not None or args:
            return real(cmd, *args, **kw)
        p = subprocess.Popen(cmd, **kw)
        peak, t = 0.0, time.time()
        while p.poll() is None:
            peak = max(peak, _boot.process_peak_mb(p._handle) or 0.0)
            time.sleep(0.2)
        peak = max(peak, _boot.process_peak_mb(p._handle) or 0.0)
        CHILD_PEAKS.append({"peak_mb": peak, "seconds": round(time.time() - t, 2), "code": p.returncode})
        return subprocess.CompletedProcess(cmd, p.returncode)

    subprocess.run = run
    try:
        yield
    finally:
        subprocess.run = real


def cache_files(ifc: Path) -> dict:
    out = {}
    for suf in (".ifc.cache.blend", ".ifc.cache.json", ".ifc.cache.sqlite"):
        p = ifc.with_suffix(suf)
        out[suf] = p.stat().st_size if p.exists() else None
    return out


def instanced_stats(inst) -> dict:
    """Bonsai merges a linked model into a few chunk meshes, so count polygons as well as objects."""
    if inst is None:
        return {"instanced_objects": 0, "instanced_polygons": 0}
    obs = list(inst.all_objects)
    return {"instanced_objects": len(obs),
            "instanced_polygons": sum(len(o.data.polygons) for o in obs if o.type == "MESH" and o.data is not None)}


def link_info(link, seconds=None) -> dict:
    """State of one Bonsai link; with a host IFC, also whether the empty sits where the reference matrix says."""
    import numpy as np
    import bonsai.tool as tool
    from estate.ifc.links import decode_matrix, from_identification
    empty = tool.Project.get_link_empty_handle(link)
    inst = empty.instance_collection if empty is not None else None
    ifc = Path(tool.Ifc.resolve_uri(link.filepath)) if tool.Ifc.get() else Path(link.filepath)
    info = {"filepath": link.filepath, "is_loaded": bool(link.is_loaded), "seconds": seconds,
            "empty": empty.name if empty is not None else None,
            "matrix_translation": [round(v, 3) for v in empty.matrix_world.translation] if empty is not None else None,
            **instanced_stats(inst), "library": inst.library.filepath if inst is not None and inst.library else None,
            "cache": cache_files(ifc), "georeferenced": link.georeferenced}
    if tool.Ifc.get() is not None and empty is not None and link.ifc_definition_id:
        # where the reference says the building goes, in host coordinates (assumes a shared IfcMapConversion)
        want = from_identification(tool.Ifc.get(), decode_matrix(tool.Ifc.get().by_id(link.ifc_definition_id)[1]))
        got = np.array(empty.matrix_world)
        info["placement_ok"] = bool(np.allclose(got, want, atol=1e-3))
        if not info["placement_ok"]:
            info["placement_offset"] = [round(float(v), 3) for v in got[:3, 3] - want[:3, 3]]
    return info


def load_links(query: str, use_cache: bool, only=None) -> list:
    import bpy
    import bonsai.tool as tool
    out = []
    links = tool.Project.get_project_props().links
    for i in range(len(links)):
        link = links[i]
        if only is not None and link.filepath not in only:
            continue
        t = time.time()
        _boot.log(f"load_link {i + 1}/{len(links)}: {link.filepath}")
        bpy.ops.bim.load_link(link_index=i, use_cache=use_cache, query=query)
        out.append(link_info(link, round(time.time() - t, 2)))
        if CHILD_PEAKS:
            out[-1]["child"] = CHILD_PEAKS[-1]
    return out


def link_files(files, query: str, use_cache: bool) -> list:
    """Fallback 1: link building IFCs directly (no host IFC; the links are stored in the .blend only).

    ``files``: paths, or dicts with ``path`` and an optional local ``matrix`` applied to the link's empty.
    """
    import bpy
    import bonsai.tool as tool
    from mathutils import Matrix
    from estate.ifc.links import encode_matrix
    out = []
    for f in files:
        spec = f if isinstance(f, dict) else {"path": f}
        p = Path(spec["path"]).resolve()
        t = time.time()
        _boot.log(f"link_ifc {p.name}")
        bpy.ops.bim.link_ifc(filepath=str(p), directory="", use_relative_path=False, use_cache=use_cache, query=query)
        link = tool.Project.get_project_props().links[-1]
        empty = tool.Project.get_link_empty_handle(link)
        if spec.get("matrix") is not None and empty is not None:
            link.transformation = encode_matrix(spec["matrix"])
            empty.matrix_world = Matrix([list(r) for r in spec["matrix"]])
        out.append(link_info(link, round(time.time() - t, 2)))
    return out


def import_glbs(files) -> list:
    """Fallback 2: plain meshes from the LOD glb exports."""
    import bpy
    out = []
    for f in files:
        p = Path(f).resolve()
        if not p.exists():
            continue
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(p))
        new = [o for o in bpy.data.objects if o not in before]
        coll = bpy.data.collections.new(f"GLB/{p.stem}")
        bpy.context.scene.collection.children.link(coll)
        for o in new:
            for c in list(o.users_collection):
                c.objects.unlink(o)
            coll.objects.link(o)
        out.append({"glb": str(p), "objects": len(new)})
    return out


def host_links(site: Path) -> list:
    """LINKED_MODEL locations of a host IFC (read with ifcopenshell, no Bonsai)."""
    import ifcopenshell
    from estate.ifc.links import linked_models
    return list(dict.fromkeys(l["path"] for l in linked_models(ifcopenshell.open(str(site)))))


def warm(a: dict) -> dict:
    """Cache-only job: build <ifc>.cache.blend for a subset of the host's links with Bonsai's own load_link.

    Several of these run in parallel (estate-blend --jobs); the final make_estate then links every cache with
    use_cache=True in well under a second each. The temp .blend must sit in the site folder (Bonsai stores the
    IFC path relative to it) and is deleted afterwards.
    """
    import bpy
    _boot.boot()
    from estate.blender.make_blend import load_ifc
    site = Path(a["site"]).resolve()
    blend = site.parent / f".estate_warm_{os.getpid()}.blend"
    tmp = Path(tempfile.gettempdir()) / f"estate_links_{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)
    _boot.clear_scene()
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        load_ifc(site, a.get("deflection_tolerance", 0.05))
        with measured_children():
            links = load_links(a.get("query", QUERY), False, set(a["only"]))
    finally:
        blend.unlink(missing_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
        tempfile.tempdir = None
    errors = [f"cache not built: {l['filepath']}" for l in links if not l["is_loaded"]]
    res = {"site": str(site), "links": links, "children": CHILD_PEAKS, "errors": errors, "ok": not errors}
    if errors:
        res["error"] = "; ".join(errors)
    return res


def run(a: dict) -> dict:
    import bpy
    if a.get("warm"):
        return warm(a)
    _boot.boot()
    import bonsai.tool as tool
    from estate.blender import views
    from estate.blender.make_blend import load_ifc

    site = Path(a["site"]).resolve() if a.get("site") else None
    blend = Path(a.get("blend") or (site.parent / "ESTATE.blend")).resolve()
    query = a.get("query", QUERY)
    use_cache = bool(a.get("use_cache", False))
    tmp = Path(tempfile.gettempdir()) / f"estate_links_{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)        # Bonsai writes its child-process script with NamedTemporaryFile(delete=False)
    _boot.clear_scene()
    blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    t0 = time.time()
    mode, links, glbs, notes = None, [], [], []
    with measured_children():
        if site is not None and site.exists():
            try:
                t_host = load_ifc(site, a.get("deflection_tolerance", 0.05))
                notes.append(f"host loaded in {t_host:.1f} s with {len(tool.Project.get_project_props().links)} links")
                links = load_links(query, use_cache)
                mode = "host"
            except Exception as e:  # noqa: BLE001
                notes.append(f"host failed: {type(e).__name__}: {e}")
        elif site is not None:
            notes.append(f"host {site} does not exist")
        if mode is None or not links or not all(l["is_loaded"] for l in links):
            if a.get("links") and tool.Ifc.get() is None:
                links = link_files(a["links"], query, use_cache)
                mode = "link_ifc"
        if not links or not any(l["is_loaded"] for l in links):
            glbs = import_glbs(a.get("glb") or [])
            if glbs:
                mode = "glb"
    t_links = time.time() - t0

    bpy.ops.file.make_paths_relative()
    libs = [lib.filepath for lib in bpy.data.libraries]
    for l in links:
        empty = bpy.data.objects.get(l["empty"] or "")
        if empty is not None and empty.instance_collection is not None and empty.instance_collection.library:
            l["library"] = empty.instance_collection.library.filepath
    views.setup_workbench()
    bbox = views.world_bbox()
    if bbox is not None:
        views.qa_cameras(bbox)
    stored = tool.Blender.get_bim_props().ifc_file
    bpy.context.scene["estate"] = json.dumps({"ifc": stored, "mode": mode, "links": [l["filepath"] for l in links],
                                              "ifc_sha256": _boot.sha256(site) if site and site.exists() else None,
                                              "bbox": [list(bbox[0]), list(bbox[1])] if bbox else None})
    bpy.ops.wm.save_mainfile(compress=True)
    leftovers = sorted(p.name for p in tmp.glob("*.py"))
    shutil.rmtree(tmp, ignore_errors=True)
    tempfile.tempdir = None
    errors = []
    if mode is None:
        errors.append("nothing was linked or imported")
    errors += [f"link not loaded: {l['filepath']}" for l in links if not l["is_loaded"]]
    errors += [f"library path not relative: {p}" for p in libs if p.endswith(".cache.blend") and not p.startswith("//")]
    errors += [f"link misplaced by {l['placement_offset']} (host and linked IFC need the same IfcMapConversion): "
               f"{l['filepath']}" for l in links if l.get("placement_ok") is False]
    res = {"blend": str(blend), "site": str(site) if site else None, "mode": mode, "ifc_file_stored": stored,
           "links": links, "glb": glbs, "libraries": libs, "links_seconds": round(t_links, 2),
           "child_peak_mb": max((c["peak_mb"] for c in CHILD_PEAKS), default=None), "children": CHILD_PEAKS,
           "temp_py_removed": leftovers, "notes": notes, "bbox": [list(bbox[0]), list(bbox[1])] if bbox else None,
           "blend_bytes": blend.stat().st_size, "errors": errors, "ok": not errors}
    if errors:
        res["error"] = "; ".join(errors)
    return res


if __name__ == "__main__":
    _boot.main(run)
