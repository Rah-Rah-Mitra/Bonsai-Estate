"""Resolve the masterplan: every building's footprint, lift lobbies and entrances in estate coordinates.

This is the single source of truth the site works (roads, paths, linkways, landscape), the siting checks and
the engine manifest read. Buildings are planned (plates only, no IFC) through the typology registry; a
typology that is not implemented yet falls back to an approximate rectangle so site work can proceed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields

import numpy as np
import shapely
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from estate import config, env
from estate.blocks import registry
from estate.flats.template import catalogue

NOMINAL_WIDTH = {"2RF": 6.6, "3R": 8.4, "4R": 10.6, "5R": 12.6, "3G": 12.6, "EA": 14.0}
MODULE_WIDTH = {"stair": 2.8, "core": 9.8}


@dataclass
class Site:
    kind: str                 # block | mscp | nc
    blk: str
    id: str                   # BLK_501, MSCP_513, NC_514
    name: str
    typology: str
    storeys: int
    at: tuple
    rot: int
    footprint: object         # shapely Polygon, estate coords
    lift_lobbies: list = field(default_factory=list)
    entrances: list = field(default_factory=list)
    height: float = 0.0
    approximate: bool = False
    plan: object = None       # BuildingPlan (blocks only)


def _xf(M, pts):
    return [tuple(np.round((M @ np.array([p[0], p[1], 0.0, 1.0]))[:2], 3)) for p in pts]


def _xf_poly(M, poly):
    a, b, d, e, xo, yo = M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[0, 3], M[1, 3]
    return shapely.affinity.affine_transform(poly, [a, b, d, e, xo, yo])


def _approx_slab(spec, cat):
    seq = spec.get("sequence") or (spec.get("wing_a", []) + [spec.get("corner", "")] + spec.get("wing_b", []))
    w = 0.0
    for item in seq:
        if item in MODULE_WIDTH:
            w += MODULE_WIDTH[item]
        elif item in cat:
            w += cat[item].width
        elif item:
            w += NOMINAL_WIDTH.get(item.split("-")[0], 10.0)
    return box(-w / 2, -3.1, w / 2, 9.6)


def resolve(cfg: dict | None = None, plans: bool = True) -> dict:
    cfg = cfg or config.load()
    seed = int(cfg["estate"]["seed"])
    cat = catalogue()
    sites = []
    for spec in config.blocks(cfg):
        spec = dict(spec, street=cfg["estate"].get("street", ""))
        M = config.placement_matrix(spec["at"], spec.get("rot", 0))
        names_storeys = int(spec["storeys"])
        height = 3.6 + 2.8 * (names_storeys - 1) + 3.4
        plan, approx = None, False
        if plans and registry.available(spec["typology"]):
            try:
                plan = registry.planner(spec["typology"])(spec, seed, cat)
            except Exception as e:  # noqa: BLE001  (missing templates during development)
                print(f"  masterplan: Blk {spec['blk']} not planned yet ({e}); using an approximate footprint")
        if plan is not None:
            fp = _xf_poly(M, plan.footprint)
            lobbies = _xf(M, plan.meta.get("lift_lobbies", []))
            ents = _xf(M, plan.meta.get("entrances", []))
        else:
            approx = True
            fp = _xf_poly(M, _approx_slab(spec, cat) if spec["typology"] != "PT4" else box(-14, -10, 14, 10))
            c = fp.centroid
            lobbies, ents = [(c.x, c.y)], [(c.x, fp.bounds[1])]
        sites.append(Site("block", str(spec["blk"]), config.target_id("block", spec["blk"]), f"Blk {spec['blk']}",
                          spec["typology"], names_storeys, tuple(spec["at"]), int(spec.get("rot", 0)), fp, lobbies, ents,
                          round(height, 2), approx, plan))
    for kind in ("mscp", "nc"):
        if kind in cfg:
            c = cfg[kind]
            r = box(*c["rect"])
            tid = config.target_id(kind, c["blk"])
            lobbies, ents = [(r.centroid.x, r.centroid.y)], []
            bj = env.MODEL / tid / f"{tid}.build.json"     # written by estate/site/mscp.py and centre.py
            if bj.exists():
                info = json.loads(bj.read_text(encoding="utf-8"))
                lobbies = [tuple(p) for p in info.get("lift_lobbies", lobbies)]
                for k, pts in info.get("entrances", {}).items():
                    if isinstance(pts, list) and k not in ("vehicle", "bin_centre_service"):
                        ents += [tuple(p) for p in pts]
            sites.append(Site(kind, str(c["blk"]), tid, c["name"], kind.upper(),
                              int(c.get("decks", c.get("shops", {}).get("storeys", 1))), (r.centroid.x, r.centroid.y), 0,
                              r, lobbies, ents, 0.0, False, None))
    roads = []
    for rd in cfg.get("road", []):
        line = shapely.LineString(rd["centreline"])
        roads.append(dict(rd, reserve_poly=line.buffer(rd["reserve"] / 2, cap_style="flat"),
                          carriage_poly=line.buffer(rd["carriageway"] / 2, cap_style="flat")))
    return {"cfg": cfg, "sites": sites, "roads": roads, "greens": cfg.get("green", []),
            "bus_stops": cfg.get("bus_stop", []), "extent": box(*cfg["estate"]["extent"])}


def siting_checks(mp: dict) -> list[str]:
    """Setbacks from road reserves, facade and gable spacing, overlaps (thresholds in config/rules.toml)."""
    from estate.flats.check import rules
    R = rules()["siting"]
    issues = []
    sites = mp["sites"]
    reserves = unary_union([r["reserve_poly"] for r in mp["roads"]])
    for s in sites:
        if s.kind != "block":
            continue
        d = s.footprint.distance(reserves)
        if d < R["setback"] - 1e-6:
            issues.append(f"{s.id}: {d:.1f} m from a road reserve (< {R['setback']} m setback)")
        if not mp["extent"].contains(s.footprint):
            issues.append(f"{s.id}: outside the estate extent")
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            if a.footprint.intersects(b.footprint) and a.footprint.intersection(b.footprint).area > 0.01:
                issues.append(f"{a.id} overlaps {b.id}")
                continue
            if a.kind == "block" and b.kind == "block":
                d = a.footprint.distance(b.footprint)
                if d < R["gable_spacing"] - 1e-6:
                    issues.append(f"{a.id} - {b.id}: {d:.1f} m apart (< {R['gable_spacing']} m)")
    return issues


def write_json(mp: dict, path=None):
    path = path or (env.MODEL / "masterplan.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"estate": mp["cfg"]["estate"], "georef": mp["cfg"].get("georef", {}),
           "sites": [dict({f.name: getattr(s, f.name) for f in fields(s) if f.name not in ("footprint", "plan")},
                          footprint=mapping(s.footprint), bounds=list(s.footprint.bounds)) for s in mp["sites"]],
           "roads": [{k: v for k, v in r.items() if not k.endswith("_poly")} for r in mp["roads"]],
           "greens": mp["greens"], "bus_stops": mp["bus_stops"]}
    path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return path
