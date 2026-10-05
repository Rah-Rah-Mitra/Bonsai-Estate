"""Build stages that wrap the area modules (validation, navigation, Bonsai .blend, engine export, renders, estate,
web export).

Each stage function takes (args, cfg, targets, state) and returns a list of failure dicts. Stages are keyed on
the target's IFC sha256 plus the stage's code, so they rerun exactly when the model or the tool changes.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from estate import env


FAMILIES = ("ifcqa", "programme", "geometry")    # estate.validate.cli.FAMILIES


def _todo(a, targets, state, stage, extra=None, need_blend=False):
    from estate.pipeline import state as st
    out = []
    for t in targets:
        src = t.blend if need_blend else t.ifc
        if not src.exists():
            continue
        key = st.stage_key(stage, dict({"target": t.id}, **(extra or {})), [t.ifc] + ([t.blend] if need_blend else []))
        if not getattr(a, "force", False) and st.fresh(state, t.id, stage, key):
            print(f"  {t.id:<9} {stage}: up to date")
            continue
        out.append((t, key))
    return out


def _spawn_pool(n):
    return ProcessPoolExecutor(max(1, n), mp_context=mp.get_context("spawn"))


# ----------------------------------------------------------------------------- check (IFC QA + IDS + geometry)
def stage_check(a, cfg, targets, state):
    from estate.pipeline import state as st
    from estate.validate.cli import validate_one
    todo = _todo(a, targets, state, "check")
    failed = []
    if not todo:
        write_validation_summary(targets)
        return failed
    with _spawn_pool(min(getattr(a, "jobs", 4), len(todo), 8)) as ex:
        futs = [(t, key, ex.submit(validate_one, t.ifc)) for t, key in todo]
        for t, key, fut in futs:
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                failed.append(dict(target=t.id, stage="check", error=f"{type(e).__name__}: {e}"))
                continue
            out = t.folder / f"{t.ifc.stem}_validation.json"
            out.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
            s = rec["summary"]
            ok = s["failed_errors"] == 0
            print(f"  {t.id:<9} check: {'ok' if ok else 'FAIL'}  errors {s['failed_errors']} ({s['error_items']} items), "
                  f"warnings {s['failed_warnings']} ({s['warning_items']})  {rec['runtime_s']} s")
            if ok:
                st.record(state, t.id, "check", key, [out], rec["runtime_s"], {"warnings": s["failed_warnings"]})
                st.save(state)
            else:
                bad = [r for fam in FAMILIES for r in rec.get(fam, {}).get("results", [])
                       if r["level"] == "error"]
                for r in bad[:6]:
                    print(f"      {r['check']}: {r['count']}  e.g. {str(r['examples'][:1])[:140]}")
                failed.append(dict(target=t.id, stage="check", errors=[(r["check"], r["count"]) for r in bad]))
    fed = write_validation_summary(targets)
    bad_fed = [r for r in (fed or {}).get("results", []) if r.get("level") == "error" and r.get("count")]
    print(f"  federation: {'ok' if not bad_fed else 'FAIL'} ({len((fed or {}).get('files', []))} files)")
    if bad_fed:
        failed.append(dict(target="FEDERATION", stage="check", errors=[(r["check"], r["count"]) for r in bad_fed]))
    return failed


def write_validation_summary(targets):
    """reports/validation_summary.json: every target's latest validation summary plus the federation check."""
    from estate.validate import ifcqa
    files = {}
    for t in targets:
        rec = env.MODEL / (t.id if t.kind != "site" else "") / f"{t.id}_validation.json"
        if rec.exists():
            r = json.loads(rec.read_text(encoding="utf-8"))
            files[t.id] = dict(summary=r["summary"], runtime_s=r.get("runtime_s"),
                               failing=[dict(family=fam, check=x["check"], level=x["level"], count=x["count"])
                                        for fam in FAMILIES for x in r.get(fam, {}).get("results", [])
                                        if x["level"] in ("error", "warn")])
    masters = [t.ifc for t in targets if t.ifc.exists()]
    fed = ifcqa.check_federation(masters) if masters else None
    out = env.REPORTS / "validation_summary.json"
    out.write_text(json.dumps(dict(files=files, federation=fed), indent=1, default=str), encoding="utf-8")
    return fed


# ----------------------------------------------------------------------------- nav (walk tests per building)
def stage_nav(a, cfg, targets, state):
    from estate.pipeline import state as st
    from estate.validate.nav_cli import _line, _run_one
    agent = {"radius": 0.30, "height": 1.80, "step": 0.40, "voxel": 0.10}
    agent.update({k: float(v) for k, v in cfg.get("agent", {}).items() if k in agent})
    todo = [(t, k) for t, k in _todo(a, targets, state, "nav", agent) if t.kind != "site"]
    failed = []
    if todo:
        tasks = [(str(t.ifc), str(t.folder), [agent["radius"]], agent, None, False) for t, _ in todo]
        with _spawn_pool(min(6, len(tasks))) as ex:
            futs = [ex.submit(_run_one, t) for t in tasks]
            results = []
            for (t, _), fut in zip(todo, futs):
                try:
                    results.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    results.append(dict(stem=t.id, error=f"worker crashed: {type(e).__name__}: {e}", seconds=0))
        for (t, key), s in zip(todo, results):
            print(_line(s).replace(f"  {s['stem']:<12}", f"  {t.id:<9} nav:", 1))
            if "error" in s or not s["ok"]:
                failed.append(dict(target=t.id, stage="nav", error=s.get("error") or s.get("results")))
            else:
                st.record(state, t.id, "nav", key, [t.folder / f"{t.id}_nav.json"], s["seconds"])
                st.save(state)
    return failed


# ----------------------------------------------------------------------------- blend (Bonsai .blend per building)
def stage_blend(a, cfg, targets, state):
    from estate.blender.cli import blend_one
    from estate.pipeline import state as st
    todo = _todo(a, targets, state, "blend")
    failed = []
    if not todo:
        return failed
    with ThreadPoolExecutor(max(1, getattr(a, "blender_jobs", 4))) as ex:
        futs = [(t, key, ex.submit(blend_one, t.ifc)) for t, key in todo]
        for t, key, fut in futs:
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                r = dict(ok=False, error=f"{type(e).__name__}: {e}")
            if r.get("ok"):
                m = r["make"]
                objs = m.get("counts", {}).get("objects", "")
                print(f"  {t.id:<9} blend: {env.rel(r['blend'])}  {objs} objects, {m['total_seconds']} s, "
                      f"{m['blend_bytes'] / 1e6:.0f} MB, verified {'verify' in r}")
                st.record(state, t.id, "blend", key, [Path(r["blend"])], m.get("total_seconds", 0) or 0)
                st.save(state)
            else:
                print(f"  {t.id:<9} blend: FAILED {str(r.get('error'))[:300]}")
                failed.append(dict(target=t.id, stage="blend", error=r.get("error")))
    return failed


# ----------------------------------------------------------------------------- render (Workbench QA renders)
def stage_render(a, cfg, targets, state):
    from estate.blender.run import BlenderError, run_blender
    from estate.pipeline import state as st
    todo = [(t, k) for t, k in _todo(a, targets, state, "render", need_blend=True)]
    failed = []
    out_dir = env.REPORTS / "renders"

    def one(t):
        return run_blender("render", {"blend": str(t.blend), "out": str(out_dir / t.id), "width": 1600,
                                      "views": ["iso", "aerial_NE", "cutaway"]}, timeout=1800)
    with ThreadPoolExecutor(max(1, getattr(a, "blender_jobs", 4))) as ex:
        futs = [(t, key, ex.submit(one, t)) for t, key in todo]
        for t, key, fut in futs:
            try:
                r = fut.result()
                print(f"  {t.id:<9} render: {len(r['files'])} images")
                st.record(state, t.id, "render", key, [Path(f) for f in r["files"]], r.get("wall_seconds", 0))
                st.save(state)
            except (BlenderError, KeyError, OSError) as e:
                print(f"  {t.id:<9} render: FAILED {str(e)[:300]}")
                failed.append(dict(target=t.id, stage="render", error=str(e)))
    return failed


# ----------------------------------------------------------------------------- glb (engine export)
def stage_glb(a, cfg, targets, state):
    from estate.export.cli import stage_glb as glb
    # the manifest is written once by cmd_build after the last stage (it hashes the .blend files too)
    return glb(a, cfg, [t for t in targets if t.kind != "site" or t.ifc.exists()], state)


# ----------------------------------------------------------------------------- estate (ESTATE.blend + site walk test)
def stage_estate(a, cfg, targets, state):
    import argparse
    from estate.blender.cli import cmd_estate_blend
    from estate.pipeline import state as st
    failed = []
    site = env.MODEL / "SITE.ifc"
    if not site.exists():
        return [dict(target="ESTATE", stage="estate", error="model/SITE.ifc missing")]
    ifcs = sorted(p for p in env.MODEL.glob("*/*.ifc") if p.parent.name == p.stem)
    key = st.stage_key("blend", {"target": "ESTATE"}, [site] + ifcs)
    out = env.MODEL / "ESTATE.blend"
    if not getattr(a, "force", False) and st.fresh(state, "ESTATE", "blend", key):
        print("  ESTATE    blend: up to date")
    else:
        t0 = time.time()
        rc = cmd_estate_blend(argparse.Namespace(site=str(site), out=str(out), use_cache=False,
                                                 jobs=max(1, getattr(a, "blender_jobs", 4)),
                                                 query="IfcElement, ! IfcFeatureElement", verify=True, timeout=7200))
        if rc == 0 and out.exists():
            st.record(state, "ESTATE", "blend", key, [out], time.time() - t0)
            st.save(state)
        else:
            failed.append(dict(target="ESTATE", stage="blend", error=f"estate-blend exit {rc}"))
    # estate renders (the headline images of the report) plus the street view from bus stop BS1, redone whenever
    # ESTATE.blend, the view list or the street camera (it follows the masterplan and the site graph) changes.
    # street.py sits outside every stage's code hash, so tuning it re-renders ESTATE alone, through this key.
    if out.exists():
        from estate.blender.run import BlenderError, run_blender
        from estate.report import street
        views, cameras = ["iso", "top", "aerial_NE", "aerial_SW"], {}
        try:
            cameras["street_BS1"] = street.camera("BS1", env.MODEL / "masterplan.json", env.MODEL / "SITE_graph.json")
        except (KeyError, ValueError, FileNotFoundError) as e:      # BS1 dropped from config/estate.toml, ...
            print(f"  ESTATE    render: no street view ({e})")
        rkey = st.stage_key("render", {"target": "ESTATE", "views": views, "cameras": cameras}, [out])
        if getattr(a, "force", False) or not st.fresh(state, "ESTATE", "render", rkey):
            try:
                r = run_blender("render", {"blend": str(out), "out": str(env.REPORTS / "renders" / "ESTATE"),
                                           "width": 1600, "views": views, "cameras": cameras}, timeout=3600)
                st.record(state, "ESTATE", "render", rkey, [Path(f) for f in r["files"]], r.get("wall_seconds", 0))
                st.save(state)
                print(f"  ESTATE    render: {len(r['files'])} images")
            except (BlenderError, KeyError, OSError) as e:
                failed.append(dict(target="ESTATE", stage="render", error=str(e)[:500]))
    # site walk test + estate graph
    from estate import masterplan
    from estate.validate import nav2d, navgraph
    mp_ = masterplan.resolve(cfg)
    site_res = nav2d.check_site(mp_, out_dir=env.REPORTS, log=lambda *x: None)
    est = navgraph.estate_report(mp_, sorted(env.MODEL.glob("*/*_nav.json")), site_res)
    print("  ESTATE    nav: " + navgraph.summary_text(est).replace("\n", "\n      "))
    if not (site_res["summary"]["ok"] and est["summary"]["ok"]):
        failed.append(dict(target="ESTATE", stage="nav", error="site or estate navigation failed"))
    return failed


# ----------------------------------------------------------------------------- web (walk grids, stairs, door poses)
def stage_web(a, cfg, targets, state):
    """<ID>_walk.bin and <ID>_web.json for every building (estate/web/export.py), from its IFC and the engine JSON
    the glb stage wrote, keyed on both, the web code (with nav3d and the door poses) and [web] / the agent. Runs in
    the process pool, one building per worker; the site has no walk grid (a viewer samples the ground)."""
    from estate.pipeline import state as st
    from estate.web.export import line, web_one
    wcfg = {"web": dict(cfg.get("web", {})),
            "agent": {k: v for k, v in cfg.get("agent", {}).items() if k in ("height", "step")}}
    todo, failed = [], []
    for t in targets:
        if t.kind == "site" or not t.ifc.exists():
            continue
        eng = t.folder / f"{t.id}_engine.json"
        if not eng.exists():
            failed.append(dict(target=t.id, stage="web", error=f"{env.rel(eng)} missing (run the glb stage)"))
            print(f"  {t.id:<9} web: no engine JSON")
            continue
        key = st.stage_key("web", dict({"target": t.id}, **wcfg), [t.ifc, eng])
        if not getattr(a, "force", False) and st.fresh(state, t.id, "web", key):
            print(f"  {t.id:<9} web: up to date")
            continue
        todo.append((t, key))
    if not todo:
        return failed
    tasks = [(str(t.ifc), str(t.folder / f"{t.id}_engine.json"), str(t.folder), t.id, wcfg) for t, _ in todo]
    with _spawn_pool(min(6, len(tasks))) as ex:
        futs = [ex.submit(web_one, task) for task in tasks]
        results = []
        for (t, _), fut in zip(todo, futs):
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001  (a crashed worker fails its target)
                results.append(dict(stem=t.id, ok=False, error=f"worker crashed: {type(e).__name__}: {e}"))
    for (t, key), r in zip(todo, results):
        print(f"  {t.id:<9} web: {line(r)}")
        if r.get("ok"):
            st.record(state, t.id, "web", key, [Path(p) for p in r["files"]], r["seconds"],
                      {"walk_gz_bytes": r["gz_bytes"], "leaf_passthrough": r["walk"]["leaf_passthrough"]})
            st.save(state)
        else:
            failed.append(dict(target=t.id, stage="web", error=r.get("error"), trace=r.get("trace")))
    return failed


STAGES = {"check": stage_check, "nav": stage_nav, "blend": stage_blend, "glb": stage_glb, "render": stage_render,
          "estate": stage_estate, "web": stage_web}
