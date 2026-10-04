"""Tessellate an IFC once with IfcOpenShell and cache per-element triangle meshes (lifted from legacy ifc_mesh.py).

World coordinates (metres, Z up). The cache lives in build/cache keyed by the file's sha256, so validation,
navigation, drawings and engine export share one tessellation per IFC version.
"""
from __future__ import annotations

import hashlib
import multiprocessing
import os
import pickle
from pathlib import Path

import numpy as np
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as uel

from estate import env

SKIP = ("IfcOpeningElement",)


def _colour(m):
    d = m.diffuse
    rgb = (d.r(), d.g(), d.b()) if hasattr(d, "r") else tuple(d)
    tr = m.transparency
    tr = 0.0 if tr is None or tr != tr else tr      # default (unstyled) materials report NaN
    return (*rgb, 1.0 - tr)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


CACHE_VERSION = "mesh-v2"     # bump when the tessellation settings or record layout change


def cache_path(path) -> Path:
    p = Path(path)
    key = sha256(p)[:16] + "-" + hashlib.sha256(f"{CACHE_VERSION}/{ifcopenshell.version}".encode()).hexdigest()[:6]
    return env.BUILD / "cache" / f"{p.stem}-{key}.mesh.pkl"


def prune(keep: int = 2):
    """Keep only the newest cache files per IFC stem."""
    by_stem = {}
    folder = env.BUILD / "cache"
    files = list(folder.glob("*.mesh.pkl")) + list(folder.glob("*.ground.pkl")) + list(folder.glob("*.overhead*.pkl"))
    for c in sorted(files, key=lambda q: q.stat().st_mtime, reverse=True):
        kind = c.name.rsplit(".", 2)[-2] if c.name.count(".") >= 2 else ""
        by_stem.setdefault((c.name.split("-")[0], kind), []).append(c)
    for files in by_stem.values():
        for c in files[keep:]:
            c.unlink(missing_ok=True)


def container_name(el):
    cont = uel.get_container(el) or uel.get_aggregate(el)
    while cont is not None and not cont.is_a("IfcSpatialStructureElement"):
        cont = uel.get_container(cont) or uel.get_aggregate(cont)
    return cont.Name if cont is not None else ""


def tessellate(f, num_threads=None, include=None):
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    threads = num_threads or max(1, min(8, multiprocessing.cpu_count() // 3))   # several workers run at once
    kwargs = {"include": include} if include is not None else {"exclude": [e for c in SKIP for e in f.by_type(c)]}
    it = ifcopenshell.geom.iterator(s, f, threads, **kwargs)
    out = {}
    if it.initialize():
        while True:
            sh = it.get()
            g = sh.geometry
            el = f.by_guid(sh.guid)
            ids = np.array(g.material_ids)
            pal = np.array([_colour(mm) for mm in g.materials]) if g.materials else np.array([[0.7, 0.7, 0.7, 1.0]])
            faces = np.array(g.faces, dtype=np.int64).reshape(-1, 3)
            ids = np.where(ids < 0, 0, ids) if len(ids) else np.zeros(len(faces), int)
            out[sh.guid] = dict(cls=sh.type, name=el.Name or "", storey=container_name(el),
                                verts=np.array(g.verts).reshape(-1, 3), faces=faces, colours=pal[ids])
            if not it.next():
                break
    return out


def load(path, cache=True, num_threads=None):
    """Return (ifc_file, {guid: dict(cls, name, storey, verts, faces, colours)})."""
    f = ifcopenshell.open(str(path))
    cp = cache_path(path)
    if cache and cp.exists():
        with open(cp, "rb") as fh:
            return f, pickle.load(fh)
    out = tessellate(f, num_threads)
    if cache:
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cp)
    return f, out
