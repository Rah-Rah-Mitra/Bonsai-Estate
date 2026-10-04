"""Load config/estate.toml and resolve shared values (seeds, placements, georeference, project identity)."""
from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

import numpy as np

from estate import env, guids
from estate.geom.walls import rotate_z, translate


@lru_cache(maxsize=4)
def load(path: str | None = None) -> dict:
    p = Path(path) if path else env.CONFIG / "estate.toml"
    cfg = tomllib.loads(p.read_text(encoding="utf-8"))
    cfg["_path"] = str(p)
    return cfg


def blocks(cfg: dict) -> list[dict]:
    return list(cfg.get("block", []))


def block(cfg: dict, blk: str) -> dict:
    for b in blocks(cfg):
        if str(b["blk"]) == str(blk):
            return b
    raise KeyError(f"no block {blk} in {cfg['_path']}")


def placement_matrix(at, rot=0) -> np.ndarray:
    """Block-local -> estate: rotate by rot quarter turns about the local origin, then translate to at."""
    return translate(float(at[0]), float(at[1]), 0.0) @ rotate_z(int(rot))


def identity(cfg: dict) -> dict:
    """Shared IfcProject / IfcSite identity so every federated file describes the same project and site."""
    code = cfg["estate"]["code"]
    return {"project_name": cfg["estate"]["name"], "project_guid": guids.from_text(f"{code}/IfcProject"),
            "site_name": f"{cfg['estate']['name']} site", "site_guid": guids.from_text(f"{code}/IfcSite"),
            "georef": dict(cfg.get("georef", {}))}


def target_id(kind: str, blk: str) -> str:
    """Folder / file stem of a building: BLK_501, MSCP_513, NC_514."""
    return {"block": f"BLK_{blk}", "mscp": f"MSCP_{blk}", "nc": f"NC_{blk}"}[kind]
