"""CLI for the walk tests: ``estate.cmd nav [--only BLK_501 ...] [--radius 0.3 0.4] [--site]``.

For every building IFC in model/<target>/<target>.ifc it runs nav3d and writes <target>_nav.json and the
nav/L??_r30.png maps next to the IFC. Door linings stay as obstacles and every passable door is re-tested at
``--door-voxel`` (0.025 m), so a 0.70 m clear opening is judged by its width, not by the 0.1 m grid.
``--ifc`` runs ad-hoc files (such as the legacy block) into build/nav.
``--site`` adds the site test (nav2d) and the estate graph (navgraph), which write reports/nav_site.json,
reports/nav_site.png and reports/nav_estate.json. Buildings run in spawned processes (one building each).
Hook: ``from estate.validate import nav_cli; nav_cli.register(sub)``.
"""
from __future__ import annotations

import time
from pathlib import Path

from estate import env


def register(sub):
    p = sub.add_parser("nav", help="walk tests: buildings (nav3d), site (nav2d) and the estate graph (navgraph)")
    p.add_argument("--only", nargs="*", metavar="TARGET", help="building targets, e.g. BLK_501 MSCP_513")
    p.add_argument("--radius", type=float, nargs="+", help="agent radii in m; the first decides pass/fail "
                                                            "(default: [agent] radius in config/estate.toml)")
    p.add_argument("--site", action="store_true", help="also run the site test and the estate navigation graph")
    p.add_argument("--ifc", nargs="*", help="ad-hoc IFC files instead of the model targets")
    p.add_argument("--out", help="output folder for --ifc runs (default build/nav/<stem>)")
    p.add_argument("--levels", nargs="*", help="levels to draw (default: lowest, L5 or a middle floor, roof)")
    p.add_argument("--door-voxel", type=float, help="voxel of the door-band test (default 0.025; 0 = off)")
    p.add_argument("--jobs", type=int, default=0, help="buildings in parallel (default min(4, n))")
    p.add_argument("--trace-memory", action="store_true", help="also report the tracemalloc peak")
    p.set_defaults(fn=cmd_nav)


def _run_one(task):
    """One building in a worker process. Returns a compact summary (the full report is on disk)."""
    from estate.validate import nav3d
    ifc, out, radii, agent, levels, trace = task
    t = time.time()
    try:
        rep = nav3d.check_building(ifc, out, radii, agent["height"], agent["step"], agent["voxel"], levels=levels,
                                   trace_memory=trace, door_voxel=agent.get("door_voxel", nav3d.DOOR_VOXEL))
    except Exception as e:  # noqa: BLE001  (report the failure, keep the other buildings going)
        return dict(stem=Path(ifc).stem, error=f"{type(e).__name__}: {e}", seconds=round(time.time() - t, 1))
    return dict(stem=rep["stem"], ok=rep["ok"], step_free_ok=rep["step_free_ok"], memory=rep["memory"],
                seconds=round(time.time() - t, 1), warnings=rep.get("warnings", []),
                results={r: res["summary"] for r, res in rep["results"].items()})


def _line(s):
    from estate.validate.nav3d import ROOF_WORD
    if "error" in s:
        return f"  {s['stem']:<12} ERROR {s['error']}"
    parts = []
    for r, x in s["results"].items():
        if x["flats"]:
            sf = f"main doors step-free {x['flats_main_door_step_free']}/{x['flats']}"
        else:
            sf = f"spaces step-free {x['spaces_step_free']}/{x['spaces']}"
        blocked = f", doors impassable {x['doors_impassable']}" if x.get("doors_impassable") else ""
        parts.append(f"r={r}: {x['reachable']}/{x['spaces']} spaces, roof {ROOF_WORD[x['roof_reachable']]}, "
                     f"flats {x['flats_reachable']}/{x['flats']}, {sf}{blocked}")
    mem = s["memory"].get("peak_working_set_mb")
    warn = "".join(f"\n      warning: {w}" for w in s.get("warnings", []))
    return (f"  {s['stem']:<12} {'ok  ' if s['ok'] else 'FAIL'} " + " | ".join(parts)
            + f"  ({s['seconds']} s, {mem} MB)" + warn)


def cmd_nav(a):
    from estate import config, masterplan
    cfg = config.load()
    agent = {"radius": 0.30, "height": 1.80, "step": 0.40, "voxel": 0.10}
    agent.update({k: float(v) for k, v in cfg.get("agent", {}).items() if k in agent})
    radii = a.radius or [agent["radius"]]
    if a.door_voxel is not None:
        agent["door_voxel"] = a.door_voxel
    tasks = []
    if a.ifc:
        for f in a.ifc:
            f = Path(f)
            if a.out:
                out = Path(a.out) if len(a.ifc) == 1 else Path(a.out) / f.stem
            else:
                out = env.BUILD / "nav" / f.stem
            tasks.append((str(f), str(out), radii, agent, a.levels, a.trace_memory))
    else:
        mp = masterplan.resolve(cfg, plans=False)
        only = set(a.only or [])
        known = set()
        for s in mp["sites"]:
            known.add(s.id)
            ifc = env.MODEL / s.id / f"{s.id}.ifc"
            if only and s.id not in only:
                continue
            if ifc.exists():
                tasks.append((str(ifc), str(ifc.parent), radii, agent, a.levels, a.trace_memory))
            elif only:
                print(f"  {s.id}: no IFC at {env.rel(ifc)}")
        for t in sorted(only - known):
            print(f"  {t}: not a masterplan target")
    bad = 0
    if tasks:
        jobs = a.jobs or min(4, len(tasks))
        print(f"nav3d: {len(tasks)} building(s), radii {radii}, {jobs} job(s)")
        t0 = time.time()
        if jobs > 1 and len(tasks) > 1:
            import multiprocessing as mp_
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(jobs, mp_context=mp_.get_context("spawn")) as ex:
                results = list(ex.map(_run_one, tasks))
        else:
            results = [_run_one(t) for t in tasks]
        for s in results:
            print(_line(s))
            bad += ("error" in s) or not s["ok"]
        print(f"nav3d done in {time.time() - t0:.1f} s")
    elif not a.site:
        print("nav3d: nothing to do (no building IFCs found)")
    if a.site:
        from estate.validate import nav2d, navgraph
        mp = masterplan.resolve(cfg)
        site = nav2d.check_site(mp, out_dir=env.REPORTS, log=print)
        s = site["summary"]
        print(f"nav2d ({site['mode']}): {s['reachable']}/{s['routes']} bus stop -> lobby routes, step-free "
              f"{s['step_free']}/{s['routes']}, nearest-stop walk <= {s['nearest_stop_max_walk_m']} m, covered >= "
              f"{s['nearest_stop_min_covered_share']}  -> {'ok' if s['ok'] else 'NOT OK'} "
              f"({env.rel(env.REPORTS / 'nav_site.json')})")
        for x in site.get("assumptions", []):
            print(f"    assumption: {x}")
        gc_ = site.get("graph_checks")
        if gc_:
            print(f"    graph vs SITE.ifc: {'ok' if gc_['ok'] else 'MISMATCH'} (covered {gc_['covered_measured_m']} of "
                  f"{gc_['covered_claimed_m']} m claimed, {gc_['edges_over_carriageway']} road-crossing edges over "
                  f"{len(gc_.get('zebras_crossed', []))} zebras, kerb drops {len(gc_['kerb_drops'])}, zebras missing "
                  f"{len(gc_['zebras_missing'])}, off paving {gc_['off_paving_m']} m)")
            for x in gc_["kerb_drops"][:10]:
                print(f"    kerb drop: {x}")
        navs = sorted(env.MODEL.glob("*/*_nav.json"))
        est = navgraph.estate_report(mp, navs, site)
        print(navgraph.summary_text(est))
        bad += (not s["ok"]) + (not est["summary"]["ok"])
    return 1 if bad else 0
