"""``estate export``: game-engine export of each building (LOD0-2 glb + engine JSON) and the estate manifest.

  estate export                        every model/<ID>/<ID>.ifc that exists (and model/SITE.ifc), then
                                       model/estate_manifest.json
  estate export --only BLK_501 SITE    selected targets
  estate export path/to/x.ifc ...      explicit IFC files, written next to each file (or into --out)

Every glb is re-read with the in-house reader (spec checks); --blender also round-trips them through Blender's
glTF importer and requires identical object / triangle / DOOR_* counts and bounds within 1 mm. An export also
fails when a passable door's portal has a side that is neither a room nor outside (engine.portals).
Estate targets take their spawn points from the masterplan entrances set out on the pedestrian graph
(manifest.site_spawns); explicit IFC paths derive them from the ground storey, unless --model-dir names a folder
with masterplan.json, SITE_graph.json and SITE.ifc (a scratch build), whose entrances then apply to the IFC of
the same stem.
``stage_glb`` is the build-pipeline hook for ``estate build --stages glb`` (signature of pipeline/hooks.py). It does
not write the manifest: the build writes it once after its last stage, when the .blend files exist.
"""
from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from estate import env


def export_one(ifc, out_dir=None, stem=None, frame="building", model_dir=None) -> dict:
    """Worker: export one IFC and re-read its glbs. Never raises; returns a report dict with ``ok``. Spawn points
    come from the masterplan site ``stem`` (or, with ``model_dir``, the site named like the IFC) in model_dir
    (default model/)."""
    t0 = time.time()
    try:
        from estate.export.engine import check_glb, export_building
        from estate.export.manifest import site_spawns
        sid = stem or (Path(ifc).stem if model_dir else None)
        kind, entrances = site_spawns(sid, model_dir) if sid and frame == "building" else (None, None)
        r = export_building(ifc, out_dir, stem, local_matrix="building" if frame == "building" else None,
                            entrances=entrances, site_kind=kind)
        problems = {}
        for k, p in r["files"].items():
            if p.endswith(".glb"):
                c = check_glb(p)
                if c["problems"]:
                    problems[k] = c["problems"]
        if r["problems"]:
            problems["portals"] = r["problems"]
        return dict(r, ok=not problems, problems=problems)
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, ifc=str(ifc), stem=stem, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc()[-3000:], seconds=round(time.time() - t0, 1))


def _run(jobs, n):
    """jobs: [export_one arguments] -> results in job order."""
    if n <= 1 or len(jobs) <= 1:
        return [export_one(*j) for j in jobs]
    out = [None] * len(jobs)
    with ProcessPoolExecutor(min(n, len(jobs)), mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(export_one, *j): i for i, j in enumerate(jobs)}
        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
    return out


def _line(r):
    label = r.get("stem") or Path(r.get("ifc", "?")).stem
    if not r.get("ok") and "error" in r:
        return f"  {label:<18} export: FAILED {r['error']}"
    tri = " / ".join(f"{r[k]['triangles']:,}" for k in ("lod0", "lod1", "lod2") if k in r)
    mb = " / ".join(f"{r[k]['bytes'] / 1e6:.1f}" for k in ("lod0", "lod1") if k in r)
    b = r["budget"]
    flag = "" if not b["over_budget"] else f"  OVER BUDGET: {len(b['over_budget'])} flats > {b['limit']}"
    probs = r.get("problems") or {}
    bad = "" if r.get("ok") else "".join(
        f"\n      {k}: {len(v)} problem(s), e.g. {v[0]}" for k, v in probs.items() if v)
    tiles = f", {r['tiles']} tiles" if r.get("tiles") else ""
    return (f"  {label:<18} LOD0/1/2 {tri} tris (LOD0/1 {mb} MB), {r['door_nodes']} doors / "
            f"{r['door_leaves']} leaves, {r['lifts']} lifts, {r['rooms']} rooms, {r['flats']} flats "
            f"(max {b['max']:,.0f} tris/flat), {r['spawns']} spawns{tiles}, {r['seconds']} s{flag}{bad}")


def _glbs(results):
    return [p for r in results if r.get("files") for p in r["files"].values() if p.endswith(".glb")]


def blender_check(paths) -> bool:
    from estate.export.engine import blender_roundtrip
    ok = True
    for p, r in blender_roundtrip(paths).items():
        ok &= r["ok"]
        b = r["blender"]
        print(f"  [{'ok' if r['ok'] else 'FAIL'}] {Path(p).name}: {b['objects']} objects, {b['triangles']:,} tris, "
              f"{b['door_nodes']} DOOR_*, bounds dev {r['bounds_dev_mm']} mm (per object {r['object_bounds_dev_mm']} mm) "
              f"{'; '.join(r['diffs'])}")
    return ok


def model_targets(only=None):
    """(target id, IFC path) for every estate target (pipeline order)."""
    from estate.pipeline.runner import targets
    return [(t.id, t.ifc) for t in targets(only=only)]


def cmd_export(a):
    t0 = time.time()
    jobs = []
    if a.paths:
        jobs = [(Path(p), Path(a.out) if a.out else None, None) for p in a.paths]
    else:
        for tid, ifc in model_targets(a.only):
            if not ifc.exists():
                print(f"  {tid:<18} no IFC yet ({env.rel(ifc)}), skipped")
                continue
            jobs.append((ifc, (Path(a.out) / tid) if a.out else None, tid))
    print(f"estate export: {len(jobs)} IFC files")
    results = _run([(i, o, s, a.frame, a.model_dir) for i, o, s in jobs], a.jobs)
    for r in results:
        print(_line(r))
        if not r.get("ok") and a.verbose and r.get("trace"):
            print(r["trace"])
    ok = all(r.get("ok") for r in results)
    if a.blender and results:
        print("Blender glTF importer round trip:")
        ok &= blender_check(_glbs(results))
    if not a.paths and not a.no_manifest:
        from estate.export.manifest import write_manifest
        p = write_manifest()
        print(f"  manifest -> {env.rel(p)}")
    print(f"export finished in {time.time() - t0:.0f} s, {sum(not r.get('ok') for r in results)} failures")
    return 0 if ok else 1


def stage_glb(a, cfg, targets, state) -> list:
    """Build stage: export every target whose IFC or spawn points changed (keyed on the IFC sha, the export code
    and the entrance spawns from the masterplan and pedestrian graph). The caller writes the manifest."""
    from estate.export.manifest import site_spawns
    from estate.pipeline import state as st
    todo = []
    for t in targets:
        if not t.ifc.exists():
            continue
        key = st.stage_key("glb", {"target": t.id, "spawns": site_spawns(t.id)}, [t.ifc])
        if not getattr(a, "force", False) and st.fresh(state, t.id, "glb", key):
            print(f"  {t.id:<9} glb: up to date")
            continue
        todo.append((t, key))
    results = _run([(t.ifc, None, t.id, "building") for t, _ in todo], getattr(a, "jobs", 1))
    failed = []
    for (t, key), r in zip(todo, results):
        print(_line(r).replace(f"  {t.id:<18}", f"  {t.id:<9} glb:", 1))
        if r.get("ok"):
            st.record(state, t.id, "glb", key, [Path(p) for p in r["files"].values()], r["seconds"],
                      {"lod0_triangles": r["lod0"]["triangles"], "max_tris_per_flat": r["budget"]["max"]})
            st.save(state)
        else:
            failed.append(dict(target=t.id, stage="glb", error=r.get("error") or r.get("problems")))
    return failed


def register(sub):
    p = sub.add_parser("export", help="game-engine export: LOD0-2 glb + engine JSON per building, then the estate manifest")
    p.add_argument("paths", nargs="*", help="IFC files to export (default: every model/<ID>/<ID>.ifc that exists)")
    p.add_argument("--only", nargs="*", help="target ids (BLK_501, MSCP_513, NC_514, SITE) or block numbers")
    p.add_argument("--out", help="output folder (default: next to each IFC)")
    p.add_argument("--model-dir", help="folder with masterplan.json, SITE_graph.json and SITE.ifc for the spawn points "
                                       "(default model/); explicit IFC paths then take the entrances of their stem")
    p.add_argument("--frame", choices=("building", "world"), default="building",
                   help="building: centre the glbs/JSON on the IfcBuilding placement (default); world: IFC coordinates")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--blender", action="store_true", help="round-trip every glb through Blender's glTF importer")
    p.add_argument("--no-manifest", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(fn=cmd_export)
