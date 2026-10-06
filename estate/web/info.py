"""model/export_info.json: where an export came from (the commit, whether its code was edited, the tools) and the
hashes a downstream packer checks it against. Written by cmd_build right after the manifest it hashes; no clock.

    {"schema": "sample-town-n5/export-info/1",
     "commit": "<git rev-parse HEAD>", "dirty": false,
     "dirty_scope": "git status --porcelain -- estate config estate.py estate.sh estate.cmd",
     "seed": 20261004, "tools": {"blender": "5.2.x", "ifcopenshell": "0.9.x", "bonsai": "...", "python": "3.13.x"},
     "walk": {"radius": 0.2, "voxel": 0.1, "band_pad": 0.25, "sites": 14, "stamped": 0, "leaves": 0,
              "leaf_blocked": 0, "leaf_narrowed": 0, "leaf_passthrough": 0, "leaf_overhead": 0,
              "stair_points_moved": 0, "stair_off_grid": 0},
     "manifest_sha256": "<sha256 of model/estate_manifest.json>",
     "files": {"BLK_509/BLK_509_walk.bin": {"sha256": "...", "bytes": 0}, "BLK_509/BLK_509_web.json": {...}, ...}}

``dirty`` covers only the generator (the scope above): reports the build rewrites do not make an export dirty.
The walk figures add up the web JSONs present (``stair_off_grid``: samples of the stair paths with no floor on
the walk grid, 0 when every path can be walked); ``files`` lists every walk grid and web JSON under model/, by path
relative to model/.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from functools import lru_cache
from pathlib import Path

from estate import env

SCHEMA = "sample-town-n5/export-info/1"
SCOPE = ("estate", "config", "estate.py", "estate.sh", "estate.cmd")
WALK_SUMS = ("stamped", "leaves", "leaf_blocked", "leaf_narrowed", "leaf_passthrough", "leaf_overhead",
             "stair_points_moved", "stair_off_grid")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args, root=None) -> str | None:
    """Output of a git command in the project (stripped), or None without git or outside a checkout."""
    try:
        r = subprocess.run(["git", *args], cwd=root or env.ROOT, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def provenance(root=None) -> tuple:
    """(HEAD commit, dirty) of the generator: dirty when git status lists any change under SCOPE."""
    status = git("status", "--porcelain", "--", *SCOPE, root=root)
    return git("rev-parse", "HEAD", root=root), (None if status is None else bool(status))


@lru_cache(maxsize=1)
def tool_versions() -> dict:
    """Blender (blender --version, else the pinned series), IfcOpenShell, Bonsai (its extension manifest) and
    Python."""
    blender = env.PINS["blender"]
    try:
        out = subprocess.run([str(env.BLENDER_EXE), "--version"], capture_output=True, text=True, timeout=120).stdout
        m = re.search(r"Blender (\d+\.\d+(?:\.\d+)?)", out or "")
        blender = m.group(1) if m else blender
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import ifcopenshell
        ifc = ifcopenshell.version
    except Exception:  # noqa: BLE001
        ifc = None
    bonsai = None
    manifest = env.BONSAI_SITE.parents[3] / "blender_org" / "bonsai" / "blender_manifest.toml"
    try:
        m = re.search(r'^version\s*=\s*"([^"]+)"', manifest.read_text(encoding="utf-8"), re.M)
        bonsai = m.group(1) if m else None
    except OSError:
        pass
    return {"blender": blender, "ifcopenshell": ifc, "bonsai": bonsai, "python": platform.python_version()}


def walk_summary(model_dir: Path, cfg: dict) -> dict:
    w = (cfg or {}).get("web", {})
    out = {"radius": float(w.get("radius", 0.20)), "voxel": float(w.get("cell", 0.10)),
           "band_pad": float(w.get("band_pad", 0.25)), "sites": 0, **{k: 0 for k in WALK_SUMS}}
    for p in sorted(model_dir.glob("*/*_web.json")):
        if p.name != f"{p.parent.name}_web.json":
            continue
        doc = json.loads(p.read_text(encoding="utf-8")).get("walk", {})
        out["sites"] += 1
        for k in WALK_SUMS:
            out[k] += int(doc.get(k, 0))
    return out


def export_files(model_dir: Path) -> dict:
    """{path relative to model/: {sha256, bytes}} of every building's walk grid and web JSON."""
    out = {}
    for pattern in ("*/*_walk.bin", "*/*_web.json"):
        for p in model_dir.glob(pattern):
            if p.name.startswith(f"{p.parent.name}_"):
                out[p.relative_to(model_dir).as_posix()] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    return dict(sorted(out.items()))


def export_info(cfg=None, model_dir=None, manifest=None, root=None) -> dict:
    from estate import config
    cfg = cfg if cfg is not None else config.load()
    model_dir = Path(model_dir) if model_dir else env.MODEL
    manifest = Path(manifest) if manifest else model_dir / "estate_manifest.json"
    commit, dirty = provenance(root)
    return {"schema": SCHEMA, "commit": commit, "dirty": dirty,
            "dirty_scope": "git status --porcelain -- " + " ".join(SCOPE),
            "seed": cfg.get("estate", {}).get("seed"), "tools": tool_versions(),
            "walk": walk_summary(model_dir, cfg),
            "manifest_sha256": sha256(manifest) if manifest.exists() else None,
            "files": export_files(model_dir)}


def write_export_info(cfg=None, model_dir=None, manifest=None, out=None, root=None) -> Path:
    """Write export_info.json (default model/export_info.json) for the manifest just written; returns its path."""
    model_dir = Path(model_dir) if model_dir else env.MODEL
    out = Path(out) if out else model_dir / "export_info.json"
    doc = export_info(cfg, model_dir, manifest, root)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out)
    return out
