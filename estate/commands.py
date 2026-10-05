"""Estate commands registered on the CLI: catalogue, plan, build, report (+ the area CLIs: site, nonres, validate,
mutate, nav, export, blend, estate-blend, render, release)."""
from __future__ import annotations

import csv
import importlib
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from estate import env

AREA_CLIS = ["estate.site.cli", "estate.site.cli_buildings", "estate.validate.cli", "estate.validate.nav_cli",
             "estate.export.cli", "estate.blender.cli", "estate.web.release"]
FLAT_TYPES_ORDER = ["2RF", "3R", "4R", "5R", "3G", "EA"]


# ----------------------------------------------------------------------------- catalogue
def cmd_catalogue(a):
    """Check every flat template and variant in its harness cases and render the flat gallery."""
    from estate.draw.raster import render_plate
    from estate.flats.check import check_template
    from estate.flats.template import catalogue
    cat = catalogue()
    only = set(a.only or [])
    out_dir = env.REPORTS / "flat_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not only and not a.no_draw:     # full run: drop tiles of variants that no longer exist
        for old in out_dir.glob("*.png"):
            old.unlink()
    summary, bad = [], 0
    for tid, t in cat.items():
        if only and tid not in only:
            continue
        res = check_template(t, stretch=not a.no_stretch)
        n_err = sum(1 for r in res for i in r["issues"] if i.level == "error")
        n_warn = sum(1 for r in res for i in r["issues"] if i.level == "warn")
        bad += n_err > 0
        print(f"{tid:<10} {t.flat_type:<4} {len(res):>3} cases  errors {n_err:>3}  warnings {n_warn:>3}   {t.title}")
        seen = set()
        for r in res:
            for i in r["issues"]:
                if i.level == "error" or a.verbose:
                    line = f"    [{r['case']}] {r['variant']}: {i}"
                    if line not in seen:
                        seen.add(line)
                        print(line)
            if r.get("dp") is not None and not a.no_draw:
                v = r["variant"]
                tag = "_".join([("M" if v.get("mirror") else "N"), v.get("kitchen", "closed")[0].upper(),
                                v.get("outdoor", "none").replace("_", ""), str(v.get("width", "") or "")]).strip("_")
                path = out_dir / f"{tid}_{tag}_{r['case']}.png"
                lf = r["lf"]
                title = (f"{tid}  {t.flat_type}  {lf.width:.2f} x {lf.depth:.2f} m  [{r['case']}]  {v}"
                         f"  sig {lf.signature()}  errors {sum(1 for i in r['issues'] if i.level == 'error')}")
                render_plate(r["dp"], path, title, scale=45)
        summary.append(dict(template=tid, type=t.flat_type, title=t.title, cases=len(res), errors=n_err, warnings=n_warn,
                            signatures=sorted({r["lf"].signature() for r in res if r["lf"] is not None}),
                            canonical=sorted({r["lf"].signature(True) for r in res if r["lf"] is not None}),
                            issues=[dict(case=r["case"], variant=r["variant"], level=i.level, code=i.code, room=i.room,
                                         msg=i.msg) for r in res for i in r["issues"]]))
    (env.REPORTS / "catalogue.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n{len(summary)} templates, {bad} with errors -> reports/catalogue.json, reports/flat_gallery/")
    return 1 if bad else 0


# ----------------------------------------------------------------------------- plan
def _neighbour_pairs(plan):
    stacks = sorted(plan.typical.flats)
    if plan.typology == "PT4":
        ring = ["101", "103", "105", "107"]
        return [(ring[i], ring[(i + 1) % 4]) for i in range(4)]
    return list(zip(stacks[:-1], stacks[1:]))


def cmd_plan(a):
    """Resolve the masterplan and run every plan-level check before any IFC is written."""
    from estate import masterplan
    from estate.blocks.builder import derive_plan, plan_errors
    from estate.draw.raster import render_plate
    from estate.flats.check import check_flat, rules
    from estate.flats.template import catalogue, flat_ifa
    t0 = time.time()
    mp_ = masterplan.resolve()
    cat = catalogue()
    errors, warnings = [], []
    sched, sigs, types_by_sig = [], [], {}
    mix = Counter()
    plans_dir = env.REPORTS / "plans"
    for old in plans_dir.glob("*.png") if plans_dir.exists() else []:   # never leave plates of an older plan
        old.unlink()
    for s in mp_["sites"]:
        if s.kind != "block":
            continue
        if s.plan is None:
            errors.append(f"{s.id}: not planned ({s.typology} typology or templates missing)")
            continue
        dp = derive_plan(s.plan)
        errors += [f"{s.id} {e}" for e in plan_errors(dp)]
        for stack, fi in sorted(s.plan.typical.flats.items()):
            known = set(cat[fi.template].known_issues) if fi.template in cat else set()
            for i in check_flat(dp["typical"], stack, known):
                (errors if i.level == "error" else warnings).append(f"{s.id} #{stack} {fi.template}: {i.code} {i.room} {i.msg}")
            if fi.template in cat and not cat[fi.template].estate:
                errors.append(f"{s.id} #{stack}: template {fi.template} is reference-only (estate = false)")
            n = s.plan.storeys - 1
            mix[fi.flat_type] += n
            sigs.append((s.id, stack, fi.flat_type, fi.signature, fi.variant.get("canonical", fi.signature),
                         fi.variant.get("topology", "")))
            for li in range(2, s.plan.storeys + 1):
                ifa = flat_ifa(fi, s.plan.typical)
                sched.append(dict(blk=s.blk, unit=f"#{li:02d}-{stack}", storey=f"L{li}", stack=stack, type=fi.flat_type,
                                  template=fi.template, signature=fi.signature, ifa=round(ifa, 1),
                                  variant=json.dumps({k: v for k, v in fi.variant.items() if k != "canonical"}, sort_keys=True)))
        from estate.blocks import plate_ops
        for name, fid, dist in plate_ops.window_outlook(dp["typical"]):
            errors.append(f"{s.id} {fid} '{name}': outlook {dist:.2f} m to a wall (< {plate_ops.OUTLOOK_MIN} m clear)")
        by_stack = {stack: fi.signature for stack, fi in s.plan.typical.flats.items()}
        for x, y in _neighbour_pairs(s.plan):
            if x in by_stack and y in by_stack and by_stack[x] == by_stack[y]:
                errors.append(f"{s.id}: adjacent stacks {x} and {y} have identical layouts")
        if not a.no_draw:
            for lvl in ("ground", "typical", "roof"):
                render_plate(dp[lvl], plans_dir / f"{s.id}_{lvl}.png", f"{s.name} {lvl} ({s.typology}, {s.storeys} storeys)",
                             scale=22 if s.typology != "PT4" else 30, flat_labels=lvl == "typical")
    errors += masterplan.siting_checks(mp_)
    # variety
    R = rules()["variety"]
    raw = {x[3] for x in sigs}
    canon = {x[4] for x in sigs}
    per_type = defaultdict(set)
    per_type_layout = defaultdict(set)
    for _, _, ft, sig, can, topo in sigs:
        per_type[ft].add(topo)          # per-type minimum: genuinely different plans (topology), not resizes
        per_type_layout[ft].add(can)
    expected = mp_["cfg"].get("mix", {}).get("expected", {})
    print(f"masterplan: {sum(1 for s in mp_['sites'] if s.kind == 'block')} blocks, {sum(mix.values())} flats "
          f"in {time.time() - t0:.1f} s")
    print("  mix: " + ", ".join(f"{t} {mix.get(t, 0)}" + (f"/{expected[t]}" if t in expected else "") for t in FLAT_TYPES_ORDER))
    for t in FLAT_TYPES_ORDER:
        if t in expected and mix.get(t, 0) != expected[t]:
            errors.append(f"mix: {t} has {mix.get(t, 0)} flats, expected {expected[t]}")
    topo = {x[5] for x in sigs}
    print(f"  variety: {len(raw)} raw / {len(canon)} mirror-canonical stack layouts, {len(topo)} distinct plans "
          "(topology); plans per type " + ", ".join(f"{t} {len(per_type[t])}" for t in FLAT_TYPES_ORDER if per_type[t]))
    if len(raw) < R["min_raw"] or len(canon) < R["min_canonical"]:
        errors.append(f"variety: {len(raw)} raw / {len(canon)} canonical < {R['min_raw']} / {R['min_canonical']}")
    for t, ss in per_type.items():
        need = R["min_per_type_small"] if t in ("2RF", "3G", "EA") else R["min_per_type"]
        if len(ss) < need:
            errors.append(f"variety: {t} has {len(ss)} distinct plans (topology) < {need}")
    masterplan.write_json(mp_)
    env.REPORTS.mkdir(parents=True, exist_ok=True)
    with open(env.REPORTS / "unit_schedule.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sched[0].keys()) if sched else ["blk"])
        w.writeheader()
        w.writerows(sched)
    with open(env.REPORTS / "mix.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["type", "flats", "expected", "share"])
        tot = sum(mix.values()) or 1
        for t in FLAT_TYPES_ORDER:
            w.writerow([t, mix.get(t, 0), expected.get(t, ""), f"{mix.get(t, 0) / tot:.3f}"])
    (env.REPORTS / "plan_check.json").write_text(json.dumps(dict(
        errors=errors, warnings=warnings, mix=mix, raw=len(raw), canonical=len(canon),
        topology=len(topo), per_type={t: len(per_type[t]) for t in FLAT_TYPES_ORDER},
        per_type_layouts={t: len(per_type_layout[t]) for t in FLAT_TYPES_ORDER}), indent=1), encoding="utf-8")
    for e in errors[: a.max_print]:
        print("  ERROR", e)
    if len(errors) > a.max_print:
        print(f"  ... {len(errors) - a.max_print} more errors (reports/plan_check.json)")
    print(f"  {len(errors)} errors, {len(warnings)} warnings -> model/masterplan.json, reports/unit_schedule.csv, "
          f"reports/mix.csv, reports/plans/")
    return 1 if errors else 0


# ----------------------------------------------------------------------------- build
def _site_spec(cfg):
    """Everything SITE.ifc depends on: the resolved footprints, entrances and lobbies of every building."""
    from estate import masterplan
    mp_ = masterplan.resolve(cfg)
    return dict(sites=[(s.id, s.footprint.wkt, s.lift_lobbies, s.entrances) for s in mp_["sites"]],
                roads=cfg.get("road", []), greens=cfg.get("green", []), bus_stops=cfg.get("bus_stop", []),
                linkways=cfg.get("linkways", {}), park=cfg.get("park_connector", {}), mscp=cfg.get("mscp", {}),
                nc=cfg.get("nc", {}))


def _pool(jobs):
    return ProcessPoolExecutor(max(1, jobs), mp_context=mp.get_context("spawn"))


def write_failures(failed: list, path: Path | None = None) -> Path:
    """reports/build_failures.json: the build's failures, their tracebacks and Blender log tails scrubbed of this
    machine's folders (env.scrub), since the file is tracked and report.json / report.html copy it."""
    path = path or env.REPORTS / "build_failures.json"
    path.write_text(json.dumps(env.scrub(json.loads(json.dumps(failed, default=str))), indent=1), encoding="utf-8")
    return path


def default_stages(cfg: dict) -> list:
    """Every build stage in order, minus what config/estate.toml [outputs] switches off."""
    out = cfg.get("outputs", {})
    return [s for s, flag in (("ifc", True), ("drawings", True), ("check", True), ("nav", out.get("nav", True)),
                              ("glb", out.get("glb", True)), ("blend", out.get("blend", True)),
                              ("render", out.get("renders", True)), ("estate", out.get("estate_blend", True)),
                              ("web", out.get("web", True)))
            if flag]


def write_provenance(stages, failed: list) -> None:
    """After the last stage: the estate manifest (it hashes the .blend files, so it comes last) and then
    export_info.json, which hashes the manifest (estate/web/info.py), whenever a stage that feeds them ran."""
    if not {"glb", "blend", "estate", "web"} & set(stages):
        return
    try:
        from estate.export.manifest import write_manifest
        write_manifest()
    except Exception as e:  # noqa: BLE001
        failed.append(dict(target="MANIFEST", stage="glb", error=f"{type(e).__name__}: {e}"))
        return
    try:
        from estate.web.info import write_export_info
        write_export_info()
    except Exception as e:  # noqa: BLE001
        failed.append(dict(target="EXPORT_INFO", stage="web", error=f"{type(e).__name__}: {e}"))


def cmd_build(a):
    from estate import config
    from estate.pipeline import runner, state as st
    cfg = config.load()
    stages = a.stages.split(",") if a.stages else default_stages(cfg)
    tg = runner.targets(cfg, a.only, include_site=not a.no_site)
    S = st.load()
    failed = []
    t0 = time.time()
    if "ifc" in stages:
        ifc4 = bool(cfg["estate"].get("ifc4_copy") and not a.no_ifc4)
        shared = dict(estate=cfg["estate"], georef=cfg.get("georef"), budget=cfg.get("budget"))
        buildings = [t for t in tg if t.kind != "site"]
        sites = [t for t in tg if t.kind == "site"]

        def wanted(t, schema):
            stage = "ifc" if schema == "IFC4X3" else "ifc4"
            if schema == "IFC4X3" and t.spec.get("frozen"):
                print(f"  {t.id:<9} ifc: frozen (hand-edited in Bonsai), not regenerated")
                return None
            if schema == "IFC4" and t.kind != "site":       # migrated from the master: keyed on the master's bytes
                key = st.stage_key("ifc4", {"target": t.id}, [t.ifc])
            elif t.kind == "site":                          # the site depends on every building's footprint
                key = st.stage_key("ifc", dict(_site_spec(cfg), schema=schema, **shared),
                                   sorted(env.MODEL.glob("*/*.build.json"))
                                   + sorted(p for p in env.MODEL.glob("*/*.ifc") if not p.name.endswith("_ifc4.ifc")))
            else:
                key = st.stage_key("ifc", dict(t.spec, schema=schema, target=t.id, **shared))
            if not a.force and st.fresh(S, t.id, stage, key):
                print(f"  {t.id:<9} {stage}: up to date")
                return None
            return (t, schema, stage, key)

        # masters first, then IFC4 migrations + the site (it reads the MSCP/NC build info), then the site's IFC4
        batches = [lambda: [wanted(t, "IFC4X3") for t in buildings],
                   lambda: ([wanted(t, "IFC4") for t in buildings] if ifc4 else []) + [wanted(t, "IFC4X3") for t in sites],
                   lambda: [wanted(t, "IFC4") for t in sites] if ifc4 else []]
        for make in batches:
            batch = [j for j in make() if j]
            if not batch:
                continue
            with _pool(min(a.jobs, len(batch))) as ex:
                futs = {ex.submit(runner.ifc_job, t, schema): (t, schema, stage, key) for t, schema, stage, key in batch}
                for fut in as_completed(futs):
                    t, schema, stage, key = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as e:  # noqa: BLE001  (a crashed worker is a failed target, not a crashed build)
                        r = dict(target=t.id, ok=False, error=f"worker crashed: {type(e).__name__}: {e}")
                    if r.get("ok"):
                        outs = [env.ROOT / p for p in r.get("outputs", [r["path"]])]
                        st.record(S, t.id, stage, key, outs, r["seconds"],
                                  {k: r[k] for k in ("flats", "elements") if k in r})
                        st.save(S)
                        extra = f"{r.get('flats', '')} flats, {r.get('elements', '')} elements" if t.kind == "block" else ""
                        print(f"  {t.id:<9} {stage}: {r['path']} in {r['seconds']} s {extra}")
                    else:
                        failed.append(r)
                        print(f"  {t.id:<9} {stage}: FAILED {r.get('error')}")
                        for d in r.get("details", [])[:8]:
                            print(f"      {d}")
                        if a.verbose and r.get("trace"):
                            print(r["trace"])
    if "drawings" in stages:
        from estate.draw.raster import render_plate
        todo = []
        for t in tg:
            if t.kind == "site" or not t.ifc.exists():
                continue
            key = st.stage_key("drawings", dict(t.spec, target=t.id, estate=cfg["estate"], drawn="ifc-1"), [t.ifc])
            if not a.force and st.fresh(S, t.id, "drawings", key):
                continue
            outs = []
            if t.kind == "block" and not t.spec.get("frozen"):   # config plates only where the config is the IFC
                plan, dp = runner.plan_block(t.spec, cfg)
                for lvl in ("ground", "typical", "roof"):
                    p = t.folder / "plans" / f"{t.id}_{lvl}.png"
                    render_plate(dp[lvl], p, f"{plan.name} {lvl} floor plan ({plan.typology}, {plan.storeys} storeys)",
                                 scale=30, flat_labels=lvl == "typical")
                    outs.append(p)
            todo.append((t, key, outs))
        if todo:    # stair sections + plans cut from the IFC (frozen blocks, MSCP and NC included), in parallel
            with _pool(min(a.jobs, len(todo))) as ex:
                futs = {ex.submit(runner.drawings_job, t.id, str(t.ifc), str(t.folder / "plans")): (t, key, outs)
                        for t, key, outs in todo}
                for fut in as_completed(futs):
                    t, key, outs = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as e:  # noqa: BLE001
                        r = dict(ok=False, error=f"worker crashed: {type(e).__name__}: {e}")
                    if not r.get("ok"):
                        failed.append(dict(target=t.id, stage="drawings", error=r.get("error")))
                        print(f"  {t.id:<9} drawings: FAILED {r.get('error')}")
                        continue
                    outs = outs + [env.ROOT / p for p in r["outputs"]]
                    keep = {p.resolve() for p in outs}
                    for old in (t.folder / "plans").glob("*.png"):    # sheets from older code or a frozen block
                        if old.resolve() not in keep:
                            old.unlink()
                    st.record(S, t.id, "drawings", key, outs, r["seconds"])
                    st.save(S)
                    print(f"  {t.id:<9} drawings: {len(outs)} sheets")
    from estate.pipeline import hooks
    try:
        for stage in stages:
            if stage in hooks.STAGES:
                try:
                    failed += hooks.STAGES[stage](a, cfg, tg, S)
                except Exception as e:  # noqa: BLE001  (a stage that crashes fails its stage, not the build)
                    import traceback
                    failed.append(dict(target="*", stage=stage, error=f"{type(e).__name__}: {e}",
                                       trace=traceback.format_exc()[-3000:]))
                    print(f"  stage {stage}: CRASHED {type(e).__name__}: {e}")
        write_provenance(stages, failed)
    finally:
        print(f"build finished in {time.time() - t0:.0f} s, {len(failed)} failures")
        write_failures(failed)
        from estate.export import meshcache
        meshcache.prune(keep=1)
        if not a.only:
            try:
                from estate.report.html import outputs_line, write_report
                print(f"report: {outputs_line(write_report())}")
            except Exception as e:  # noqa: BLE001
                print(f"report: not written ({type(e).__name__}: {e})")
    return 1 if failed else 0


# ----------------------------------------------------------------------------- report
def cmd_report(a):
    """reports/report.html, its JSON twin reports/report.json and reports/issues.bcf (BCF 2.1, one topic per failing
    validation item) from the latest build artefacts; --out (when registered) writes the three into another folder."""
    from estate.report.html import outputs_line, write_report
    out = getattr(a, "out", None)
    p = write_report(Path(out) / "report.html" if out else None)
    print(f"wrote {outputs_line(p)}")
    return 0


def ensure_test_fixtures() -> None:
    """Build the shared fixtures the suites read from build/ when missing (a fresh clone or worktree has none):
    build/t_pt4.ifc (tests/fixtures/make_t_pt4.py) and build/legacy/hdb_block_legacy.ifc (estate legacy)."""
    import subprocess
    jobs = []
    if not (env.BUILD / "t_pt4.ifc").exists():
        jobs.append([str(env.ROOT / "tests" / "fixtures" / "make_t_pt4.py")])
    if not (env.BUILD / "legacy" / "hdb_block_legacy.ifc").exists():
        jobs.append([str(env.ROOT / "estate.py"), "legacy"])
    for args in jobs:
        print("building test fixture:", " ".join(Path(x).name for x in args))
        subprocess.run([sys.executable, "-I", "-B", *args], cwd=env.ROOT, check=True)


def cmd_test(a):
    import unittest
    if not a.no_fixtures:
        ensure_test_fixtures()
    suite = unittest.defaultTestLoader.discover(str(env.ROOT / "tests"), pattern=a.pattern)
    r = unittest.TextTestRunner(verbosity=2 if a.verbose else 1).run(suite)
    return 0 if r.wasSuccessful() else 1


def cmd_clean(a):
    """Remove caches and leftovers: tessellation / ground caches (all or older versions), temp files, Bonsai
    link caches next to the building IFCs (they are rebuilt by the estate stage)."""
    import shutil
    from estate.export import meshcache
    cache = env.BUILD / "cache"
    if a.all_caches and cache.exists():
        shutil.rmtree(cache)
        print("removed build/cache")
    else:
        meshcache.prune(keep=1)
        print("pruned build/cache to the newest tessellation per file")
    n = 0
    for p in list(env.MODEL.rglob("*.tmp*")) + list(env.MODEL.rglob("*.blend1")):
        p.unlink()
        n += 1
    if a.link_caches:
        for p in env.MODEL.glob("*/*.ifc.cache.*"):
            p.unlink()
            n += 1
    print(f"removed {n} temporary / cache files under model/")
    return 0


def register(sub):
    p = sub.add_parser("clean", help="remove caches and temporary files")
    p.add_argument("--all-caches", action="store_true", help="delete build/cache entirely (else keep the newest)")
    p.add_argument("--link-caches", action="store_true", help="also delete Bonsai's *.ifc.cache.* link caches")
    p.set_defaults(fn=cmd_clean)

    p = sub.add_parser("test", help="run the unit tests in tests/")
    p.add_argument("--pattern", default="test_*.py")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--no-fixtures", action="store_true", help="do not build missing build/ fixtures first")
    p.set_defaults(fn=cmd_test)

    p = sub.add_parser("catalogue", help="check flat templates in test harnesses and render the flat gallery")
    p.add_argument("--only", nargs="*")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--no-draw", action="store_true")
    p.add_argument("--no-stretch", action="store_true")
    p.set_defaults(fn=cmd_catalogue)

    p = sub.add_parser("plan", help="resolve the masterplan and run plan-level checks (no IFC)")
    p.add_argument("--no-draw", action="store_true")
    p.add_argument("--max-print", type=int, default=40)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("build", help="build targets stage by stage (incremental)")
    p.add_argument("--only", nargs="*", help="target ids (BLK_507, MSCP_513, NC_514, SITE) or block numbers")
    p.add_argument("--stages", help="comma list: ifc,drawings,check,nav,glb,blend,render,estate,web (default: all, in that "
                                    "order)")
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--blender-jobs", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-ifc4", action="store_true")
    p.add_argument("--no-site", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("report", help="write reports/report.html, report.json and issues.bcf from the latest build "
                                      "artefacts")
    p.add_argument("--out", help="folder for the three files (default reports/)")
    p.set_defaults(fn=cmd_report)

    import sys
    for mod in AREA_CLIS:
        try:
            importlib.import_module(mod).register(sub)
        except ModuleNotFoundError:
            pass
        except Exception as e:  # noqa: BLE001  (an area module mid-edit must not break the other commands)
            print(f"[estate] {mod} not loaded: {type(e).__name__}: {e}", file=sys.stderr)
