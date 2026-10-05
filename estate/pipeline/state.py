"""Incremental build state: a stage of a target is skipped when its input key matches and its outputs exist.

key = sha256(target spec + hashes of the code and config it depends on + tool versions). model/.state.json maps
target -> stage -> {key, outputs, seconds}. Frozen targets (hand-edited in Bonsai) are never rebuilt.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from estate import env

STATE = env.MODEL / ".state.json"


@lru_cache(maxsize=None)
def tree_hash(*globs: str) -> str:
    h = hashlib.sha256()
    for g in globs:
        for p in sorted(env.ROOT.glob(g)):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(p.relative_to(env.ROOT).as_posix().encode())
                h.update(p.read_bytes())
    return h.hexdigest()


def versions() -> str:
    import ifcopenshell
    return f"ifcopenshell {ifcopenshell.version}; blender {env.PINS['blender']}"


STAGE_DEPS = {   # code/config each stage depends on (globs relative to the project root)
    "ifc": ("estate/blocks/*.py", "estate/geom/*.py", "estate/ifc/*.py", "estate/flats/*.py", "estate/site/*.py",
            "estate/rules.py", "estate/guids.py", "estate/config.py", "estate/masterplan.py", "estate/env.py",
            "estate/pipeline/runner.py", "config/flats/*.toml", "config/rules.toml"),
    "ifc4": ("estate/pipeline/runner.py",),
    "check": ("estate/validate/*.py", "estate/validate/ids/*", "estate/flats/check.py", "estate/flats/template.py",
              "estate/export/meshcache.py", "config/rules.toml"),
    "nav": ("estate/validate/nav*.py", "estate/export/meshcache.py"),
    "blend": ("estate/blender/*.py",),
    "glb": ("estate/export/*.py", "estate/validate/nav_doorpose.py"),
    "drawings": ("estate/draw/*.py",),
    "render": ("estate/blender/*.py",),
    # the web stage's own modules only: estate/web/info.py and release.py (export_info.json, the release zips) shape
    # no walk grid or web JSON, so editing them reruns nothing
    "web": ("estate/web/__init__.py", "estate/web/export.py", "estate/web/walk.py", "estate/web/sn5w.py",
            "estate/web/stairs.py", "estate/validate/nav3d.py", "estate/validate/nav_doorpose.py",
            "estate/export/meshcache.py"),
}


def stage_key(stage: str, spec: dict, inputs: list[Path] | None = None) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(spec, sort_keys=True, default=str).encode())
    h.update(tree_hash(*STAGE_DEPS.get(stage, ())).encode())
    h.update(versions().encode())
    for p in inputs or []:
        p = Path(p)
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode() if p.exists() else b"missing")
    return h.hexdigest()[:32]


def load() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def save(state: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, STATE)


def fresh(state: dict, target: str, stage: str, key: str) -> bool:
    rec = state.get(target, {}).get(stage)
    if not rec or rec.get("key") != key:
        return False
    return all((env.ROOT / o).exists() for o in rec.get("outputs", []))


def code_hash(stage: str) -> str:
    """The hash of a stage's code and config alone (one part of its key). Recorded with the stage's outputs, so a
    reader can tell outputs of the current code from outputs an older version wrote (tests/test_leaks.py)."""
    return tree_hash(*STAGE_DEPS.get(stage, ()))[:16]


def record(state: dict, target: str, stage: str, key: str, outputs: list, seconds: float, extra: dict | None = None):
    state.setdefault(target, {})[stage] = dict(key=key, code=code_hash(stage), outputs=[env.rel(o) for o in outputs],
                                               seconds=round(seconds, 1), **(extra or {}))
