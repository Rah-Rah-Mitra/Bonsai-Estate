"""CLI commands for the Blender stages (hooked into the estate CLI with ``register(sub)``).

    estate.cmd blend <ifc...> [--jobs 4] [--no-verify]      one .blend per IFC (same folder, same stem), then reopen
                                                           each in a fresh Blender to verify it
    estate.cmd estate-blend [--site model/SITE.ifc]        ESTATE.blend: host SITE.ifc + every building linked
    estate.cmd render <blend> [--out DIR] [--views ...]    Workbench QA renders (1600 px)
    estate.cmd dev <blend|ifc>                             open in the Blender GUI for hand edits with Bonsai

Each Blender job runs in its own background blender.exe (estate/blender/run.py); --jobs runs several at once
(about 0.9 GB per building job). Results are printed as a table and written to build/logs/blender/*.json.
"""
from __future__ import annotations

import glob
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from estate import env


def _paths(items, suffix=".ifc") -> list[Path]:
    """Resolve CLI paths (cwd first, then the project root) and expand globs (cmd.exe does not)."""
    out = []
    for it in items:
        cands = []
        for base in (Path.cwd(), env.ROOT):
            pat = str(Path(it) if Path(it).is_absolute() else base / it)
            cands = sorted(Path(p) for p in glob.glob(pat)) if any(c in it for c in "*?[") else \
                ([Path(pat)] if Path(pat).exists() else [])
            if cands:
                break
        if not cands:
            raise SystemExit(f"no such file: {it}")
        out += [p.resolve() for p in cands if p.suffix.lower() == suffix]
    return list(dict.fromkeys(out))


def _mb(n) -> str:
    return f"{(n or 0) / 2**20:6.1f}"


def _summary(name, payload):
    path = env.BUILD / "logs" / "blender" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return path


def blend_one(ifc: Path, verify=True, timeout=1800, deflection=0.05) -> dict:
    """make_blend then (optionally) verify_blend for one IFC; never raises."""
    from estate.blender.run import BlenderError, run_blender
    rec = {"ifc": str(ifc), "ok": False}
    try:
        made = run_blender("make_blend", {"ifc": str(ifc), "deflection_tolerance": deflection}, timeout=timeout)
        rec.update(make=made, blend=made["blend"])
        if verify:
            rec["verify"] = run_blender("verify_blend", {"blend": made["blend"], "expected": made["counts"]["by_class"]},
                                        timeout=timeout)
        rec["ok"] = True
    except BlenderError as e:
        rec["error"] = str(e)
    return rec


def cmd_blend(a) -> int:
    ifcs = [p for p in _paths(a.ifc) if not p.stem.endswith("_ifc4")]
    if not ifcs:
        raise SystemExit("no IFC files given")
    print(f"blend: {len(ifcs)} IFC, {a.jobs} parallel Blender jobs{'' if a.verify else ', no verify'}")
    t = time.time()
    recs = []
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {ex.submit(blend_one, p, a.verify, a.timeout, a.deflection): p for p in ifcs}
        for fut in as_completed(futs):
            r = fut.result()
            recs.append(r)
            name = Path(r["ifc"]).stem
            if not r["ok"]:
                print(f"  FAIL {name}: {r['error'].splitlines()[0]}")
                continue
            m, v = r["make"], r.get("verify", {})
            print(f"  ok   {name:<14} objects {m['counts']['ifc_objects']:>6}  elements {m['counts']['element_objects']:>6}"
                  f"  load {m['load_seconds']:5.1f} s  job {m['wall_seconds']:5.1f} s  .blend {_mb(m['blend_bytes'])} MB"
                  f"  peak {m.get('peak_working_set_mb', 0):6.0f} MB"
                  + (f"  verify {v.get('wall_seconds', 0):4.1f} s" if v else ""))
    recs.sort(key=lambda r: r["ifc"])
    bad = sum(not r["ok"] for r in recs)
    path = _summary("blend_summary", recs)
    print(f"{len(recs) - bad}/{len(recs)} ok in {time.time() - t:.1f} s -> {env.rel(path)}")
    return 1 if bad else 0


def _discover(folder: Path):
    links = sorted(p for p in folder.glob("*/*.ifc") if not p.stem.endswith("_ifc4") and p.parent.name == p.stem)
    glbs = sorted(folder.glob("*/*_lod1.glb"))
    return links, glbs


def cmd_estate_blend(a) -> int:
    from estate.blender.run import BlenderError, run_blender
    site = _paths([a.site])[0] if (Path.cwd() / a.site).exists() or (env.ROOT / a.site).exists() else None
    folder = site.parent if site else (env.ROOT / Path(a.site).parent)
    links, glbs = _discover(folder)
    args = {"site": str(site) if site else None, "blend": a.out or str(folder / "ESTATE.blend"),
            "use_cache": a.use_cache, "query": a.query, "links": [str(p) for p in links], "glb": [str(p) for p in glbs]}
    print(f"estate-blend: host {env.rel(site) if site else '(none)'}, {len(links)} building IFCs found")
    t = time.time()
    warm = {}
    if site and not a.use_cache and a.jobs > 1:
        from estate.blender.make_estate import host_links
        from estate.blender.run import run_many
        locs = host_links(site)
        groups = [locs[i::a.jobs] for i in range(min(a.jobs, len(locs)))]
        if len(groups) > 1:
            res = run_many([("make_estate", {"site": str(site), "warm": True, "only": g, "query": a.query})
                            for g in groups], max_workers=a.jobs, timeout=a.timeout)
            failed = [r for r in res if isinstance(r, Exception)]
            warm = {"seconds": round(time.time() - t, 1), "groups": groups,
                    "results": [r if isinstance(r, dict) else {"error": str(r)} for r in res]}
            print(f"  caches built by {len(groups)} parallel jobs in {warm['seconds']} s"
                  + (f", {len(failed)} failed (the final job rebuilds them)" if failed else ""))
            args["use_cache"] = not failed
    try:
        r = run_blender("make_estate", args, timeout=a.timeout)
    except BlenderError as e:
        print(f"  FAIL {e}")
        return 1
    for l in r["links"]:
        print(f"  {'ok  ' if l['is_loaded'] else 'FAIL'} {l['filepath']:<28} {l.get('seconds') or 0:5.1f} s  "
              f"{l.get('instanced_polygons', 0):>8} polygons  library {l.get('library')}")
    child = [c["peak_mb"] for w in warm.get("results", []) for c in w.get("children", [])]
    child_peak = max(child + [r.get("child_peak_mb") or 0.0])
    print(f"  mode {r['mode']}, links {r['links_seconds']} s, host peak {r.get('peak_working_set_mb')} MB, "
          f"child peak {child_peak} MB, .blend {_mb(r['blend_bytes'])} MB")
    out = {"warm": warm, "make": r}
    rc = 0
    if a.verify and r["mode"] in ("host", "link_ifc"):
        try:
            v = run_blender("verify_blend", {"blend": r["blend"], "links": True}, timeout=a.timeout)
            print(f"  verify ok: {len(v['links'])} links loaded after reopen, libraries {v['libraries']}")
            out["verify"] = v
        except BlenderError as e:
            print(f"  verify FAIL {e}")
            rc = 1
    path = _summary("estate_blend", out)
    print(f"{r['blend']} in {time.time() - t:.1f} s -> {env.rel(path)}")
    return rc


def cmd_render(a) -> int:
    from estate.blender.run import BlenderError, run_blender
    blend = _paths([a.blend], ".blend")[0]
    args = {"blend": str(blend), "out": a.out, "width": a.width}
    if a.views:
        args["views"] = a.views
    if a.storey is not None:
        args["storey"] = a.storey
    try:
        r = run_blender("render", args, timeout=a.timeout)
    except BlenderError as e:
        print(f"FAIL {e}")
        return 1
    for f in r["files"]:
        print(f"  {env.rel(f)}")
    if r["skipped"]:
        print(f"  skipped: {', '.join(r['skipped'])}")
    print(f"{len(r['files'])} renders in {r['wall_seconds']} s (storey {r['storey']})")
    return 0


def cmd_dev(a) -> int:
    from estate.blender.run import launch_gui
    target = _paths([a.path], Path(a.path).suffix.lower() or ".blend")[0]
    p = launch_gui("dev_session", {"path": str(target), "ifc_only": a.ifc_only})
    print(f"Blender started (pid {p.pid}) with {env.rel(target)}; see estate/blender/dev_session.py for the "
          f"hand-edit workflow (Save Project, then Ctrl+S, then frozen = true in config/estate.toml)")
    return 0


def register(sub) -> None:
    p = sub.add_parser("blend", help="build a Bonsai .blend next to each IFC and verify it in a fresh Blender")
    p.add_argument("ifc", nargs="+", help="IFC files (globs allowed), e.g. model/*/*.ifc")
    p.add_argument("--jobs", "-j", type=int, default=4, help="parallel Blender processes (about 0.9 GB each)")
    p.add_argument("--no-verify", dest="verify", action="store_false")
    p.add_argument("--deflection", type=float, default=0.05, help="Bonsai deflection tolerance (m)")
    p.add_argument("--timeout", type=float, default=1800)
    p.set_defaults(fn=cmd_blend)

    p = sub.add_parser("estate-blend", help="ESTATE.blend: the host SITE.ifc with every building linked via Bonsai")
    p.add_argument("--site", default="model/SITE.ifc", help="host IFC with LINKED_MODEL references")
    p.add_argument("--out", help="output .blend (default ESTATE.blend next to the site)")
    p.add_argument("--use-cache", action="store_true", help="reuse <ifc>.cache.blend files (stale after a rebuild)")
    p.add_argument("--jobs", "-j", type=int, default=4, help="parallel jobs building the link caches")
    p.add_argument("--query", default="IfcElement, ! IfcFeatureElement", help="Bonsai selector for linked elements")
    p.add_argument("--no-verify", dest="verify", action="store_false")
    p.add_argument("--timeout", type=float, default=3600)
    p.set_defaults(fn=cmd_estate_blend)

    p = sub.add_parser("render", help="Workbench QA renders of a .blend (iso, top, aerials, storey cutaway)")
    p.add_argument("blend")
    p.add_argument("--out", help="output folder (default <blend folder>/renders)")
    p.add_argument("--views", nargs="*", help="iso top aerial_NE aerial_SW cutaway cutplan")
    p.add_argument("--storey", help="cutaway storey name or index (default the sixth storey)")
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--timeout", type=float, default=1800)
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("dev", help="open a .blend or IFC in the Blender GUI for hand edits with Bonsai")
    p.add_argument("path")
    p.add_argument("--ifc-only", action="store_true", help="load the IFC even if a sibling .blend exists")
    p.set_defaults(fn=cmd_dev)
