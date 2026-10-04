"""Tessellate an IFC once with IfcOpenShell and cache per-element triangle meshes (world coordinates, metres, Z up)."""
import multiprocessing
import os
import pickle

import numpy as np
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as uel

SKIP = {"IfcOpeningElement"}


def _colour(m):
    d = m.diffuse
    rgb = (d.r(), d.g(), d.b()) if hasattr(d, "r") else tuple(d)
    return (*rgb, 1.0 - (m.transparency or 0.0))


def load(path, cache=True):
    """Return (ifc_file, {guid: dict(cls, name, storey, verts, faces, colours)})."""
    f = ifcopenshell.open(path)
    cpath = path + ".mesh.pkl"
    if cache and os.path.exists(cpath) and os.path.getmtime(cpath) > os.path.getmtime(path):
        with open(cpath, "rb") as fh:
            return f, pickle.load(fh)
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    it = ifcopenshell.geom.iterator(s, f, multiprocessing.cpu_count(),
                                    exclude=[e for c in SKIP for e in f.by_type(c)])
    out = {}
    if it.initialize():
        while True:
            sh = it.get()
            g = sh.geometry
            el = f.by_guid(sh.guid)
            cont = uel.get_container(el) or uel.get_aggregate(el)
            while cont is not None and not cont.is_a("IfcSpatialStructureElement"):
                cont = uel.get_container(cont) or uel.get_aggregate(cont)
            ids = np.array(g.material_ids)
            pal = np.array([_colour(mm) for mm in g.materials]) if g.materials else np.array([[0.7, 0.7, 0.7, 1.0]])
            faces = np.array(g.faces, dtype=np.int64).reshape(-1, 3)
            ids = np.where(ids < 0, 0, ids) if len(ids) else np.zeros(len(faces), int)
            out[sh.guid] = dict(cls=sh.type, name=el.Name or "", storey=cont.Name if cont is not None else "",
                                verts=np.array(g.verts).reshape(-1, 3), faces=faces, colours=pal[ids])
            if not it.next():
                break
    if cache:
        with open(cpath, "wb") as fh:
            pickle.dump(out, fh)
    return f, out
