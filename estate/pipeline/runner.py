"""Build targets: residential blocks, the MSCP, the neighbourhood centre and the site, stage by stage.

IFC authoring runs one building per process (ProcessPoolExecutor, spawn) on Blender's Python. Bonsai stages run
blender.exe headless through estate/blender/run.py. Each worker returns a small info dict; the parent records
the incremental state.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from estate import config, env


@dataclass
class Target:
    id: str               # BLK_501, MSCP_513, NC_514, SITE
    kind: str             # block | mscp | nc | site
    blk: str
    spec: dict

    @property
    def folder(self) -> Path:
        return env.MODEL if self.kind == "site" else env.MODEL / self.id

    @property
    def ifc(self) -> Path:
        return self.folder / f"{self.id}.ifc"

    @property
    def ifc4(self) -> Path:
        return self.folder / f"{self.id}_ifc4.ifc"

    @property
    def blend(self) -> Path:
        return self.folder / f"{self.id}.blend"


def targets(cfg=None, only=None, include_site=True) -> list[Target]:
    cfg = cfg or config.load()
    out = [Target(config.target_id("block", b["blk"]), "block", str(b["blk"]), dict(b)) for b in config.blocks(cfg)]
    for kind in ("mscp", "nc"):
        if kind in cfg:
            out.append(Target(config.target_id(kind, cfg[kind]["blk"]), kind, str(cfg[kind]["blk"]), dict(cfg[kind])))
    if include_site:
        out.append(Target("SITE", "site", "", {"roads": cfg.get("road", []), "greens": cfg.get("green", []),
                                               "bus_stops": cfg.get("bus_stop", []), "linkways": cfg.get("linkways", {})}))
    if only:
        want = {o.upper() for o in only}
        out = [t for t in out if t.id.upper() in want or t.blk in only]
    return out


# ----------------------------------------------------------------------------- block IFC worker
def plan_block(spec: dict, cfg: dict):
    from estate.blocks import registry
    from estate.blocks.builder import derive_plan
    from estate.flats.template import catalogue
    spec = dict(spec, street=cfg["estate"].get("street", ""))
    plan = registry.planner(spec["typology"])(spec, int(cfg["estate"]["seed"]), catalogue())
    return plan, derive_plan(plan)


def build_block_ifc(target_id: str, schema: str = "IFC4X3", out: str | None = None) -> dict:
    """Worker: plan, derive and author one residential block; returns an info dict (never raises)."""
    t0 = time.time()
    try:
        from estate.blocks.builder import build_building, plan_errors
        from estate.ifc.writer import IfcWriter
        cfg = config.load()
        blk = target_id.split("_", 1)[1]
        spec = config.block(cfg, blk)
        plan, dp = plan_block(spec, cfg)
        errs = plan_errors(dp)
        if errs:
            return dict(target=target_id, ok=False, error="plan errors", details=errs[:40], seconds=time.time() - t0)
        ident = config.identity(cfg)
        W = IfcWriter(schema=schema, guid_key=f"{cfg['estate']['code']}/{target_id}/{schema}",
                      guid_base=f"{cfg['estate']['code']}/{target_id}", typed_openings=True, **ident)
        _, info = build_building(W, plan, dp, placement=config.placement_matrix(spec["at"], spec.get("rot", 0)))
        path = Path(out) if out else (env.MODEL / target_id / (f"{target_id}.ifc" if schema == "IFC4X3" else f"{target_id}_ifc4.ifc"))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.ifc")
        W.write(tmp)
        W.close()
        tmp.replace(path)
        counts = dict(sorted(W.counts.items()))
        if schema == "IFC4X3":
            (path.parent / f"{target_id}_flats.json").write_text(json.dumps(info["flats"], indent=0), encoding="utf-8")
            (path.parent / f"{target_id}.spec.json").write_text(json.dumps(dict(
                spec=spec, levels=info["levels"], flats_per_floor=len(plan.typical.flats),
                stacks={k: dict(template=v.template, type=v.flat_type, variant=v.variant, signature=v.signature)
                        for k, v in sorted(plan.typical.flats.items())},
                meta={k: v for k, v in plan.meta.items()}), indent=1, default=str), encoding="utf-8")
        n_el = sum(v for k, v in counts.items())
        return dict(target=target_id, ok=True, path=env.rel(path), schema=schema, seconds=round(time.time() - t0, 1),
                    flats=len(info["flats"]), zones=info["zones"], counts=counts, elements=n_el)
    except Exception as e:  # noqa: BLE001
        return dict(target=target_id, ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:],
                    seconds=round(time.time() - t0, 1))


def build_nonres_ifc(target_id: str, schema: str = "IFC4X3") -> dict:
    t0 = time.time()
    try:
        cfg = config.load()
        out = env.MODEL / target_id / (f"{target_id}.ifc" if schema == "IFC4X3" else f"{target_id}_ifc4.ifc")
        out.parent.mkdir(parents=True, exist_ok=True)
        if target_id.startswith("MSCP"):
            from estate.site.mscp import build_mscp_ifc
            info = build_mscp_ifc(cfg, out, schema=schema)
        else:
            from estate.site.centre import build_nc_ifc
            info = build_nc_ifc(cfg, out, schema=schema)
        outs = [out] + ([out.parent / f"{target_id}.build.json"] if schema == "IFC4X3" else [])
        return dict(target=target_id, ok=True, path=env.rel(out), schema=schema, seconds=round(time.time() - t0, 1),
                    outputs=[env.rel(p) for p in outs if p.exists()], info=json.loads(json.dumps(info, default=str)))
    except Exception as e:  # noqa: BLE001
        return dict(target=target_id, ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:],
                    seconds=round(time.time() - t0, 1))


def drawings_job(target_id: str, ifc: str, out_dir: str) -> dict:
    """Stair sections and L1 / typical / RF plans cut from the IFC (runs in a worker process)."""
    t0 = time.time()
    try:
        from estate.draw.iplans import draw_all
        return dict(target=target_id, ok=True, outputs=[env.rel(p) for p in draw_all(target_id, ifc, out_dir)],
                    seconds=round(time.time() - t0, 1))
    except Exception as e:  # noqa: BLE001
        return dict(target=target_id, ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:],
                    seconds=round(time.time() - t0, 1))


def build_site_ifc(schema: str = "IFC4X3") -> dict:
    """SITE.ifc; the IFC4X3 run also writes the pedestrian graph, the site plan and masterplan.json from the same
    layout, so nav2d and the engine manifest never read a graph that no longer matches the IFC."""
    t0 = time.time()
    try:
        from estate import masterplan
        from estate.site.builder import build_site
        mp = masterplan.resolve()
        out = env.MODEL / ("SITE.ifc" if schema == "IFC4X3" else "SITE_ifc4.ifc")
        outputs = [out]
        if schema == "IFC4X3":
            graph, png = env.MODEL / "SITE_graph.json", env.REPORTS / "site_plan.png"
            info = build_site(mp, out, schema=schema, graph_path=graph, plan_png=png)
            sb = env.REPORTS / "site_build.json"
            sb.write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
            outputs += [graph, png, masterplan.write_json(mp), sb]
        else:
            info = build_site(mp, out, schema=schema)
        return dict(target="SITE", ok=True, path=env.rel(out), schema=schema, seconds=round(time.time() - t0, 1),
                    outputs=[env.rel(p) for p in outputs], info=json.loads(json.dumps(info, default=str)))
    except Exception as e:  # noqa: BLE001
        return dict(target="SITE", ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:],
                    seconds=round(time.time() - t0, 1))


def migrate_ifc4(target_id: str) -> dict:
    """IFC4 copy of a building made from its finished IFC4X3 master (ifcpatch Migrate): identical GlobalIds and
    names, and it follows hand edits of frozen blocks."""
    t0 = time.time()
    try:
        import ifcopenshell
        import ifcpatch
        src = env.MODEL / target_id / f"{target_id}.ifc"
        out = env.MODEL / target_id / f"{target_id}_ifc4.ifc"
        f = ifcopenshell.open(str(src))
        import ifcopenshell.api.root
        for k in f.by_type("IfcKerb"):          # IFC4X3-only classes with no IFC4 equivalent
            el = ifcopenshell.api.root.reassign_class(f, product=k, ifc_class="IfcBuildingElementProxy",
                                                      predefined_type="USERDEFINED")
            el.ObjectType = "Kerb"
        for ty in f.by_type("IfcGeographicElementType"):
            if ty.PredefinedType == "VEGETATION":
                ty.PredefinedType, ty.ElementType = "USERDEFINED", ty.ElementType or "Vegetation"
        g = ifcpatch.execute({"input": str(src), "file": f, "recipe": "Migrate", "arguments": ["IFC4"]})
        tmp = out.with_suffix(".tmp.ifc")
        ifcpatch.write(g, str(tmp))
        tmp.replace(out)
        return dict(target=target_id, ok=True, path=env.rel(out), schema="IFC4", seconds=round(time.time() - t0, 1),
                    outputs=[env.rel(out)])
    except Exception as e:  # noqa: BLE001
        return dict(target=target_id, ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:],
                    seconds=round(time.time() - t0, 1))


def ifc_job(target: Target, schema: str) -> dict:
    if schema == "IFC4" and target.kind != "site":
        return migrate_ifc4(target.id)
    if target.kind == "block":
        return build_block_ifc(target.id, schema)
    if target.kind in ("mscp", "nc"):
        return build_nonres_ifc(target.id, schema)
    return build_site_ifc(schema)
