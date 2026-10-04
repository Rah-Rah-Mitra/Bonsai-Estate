"""CLI for the IFC / geometry validators: ``estate validate`` and ``estate mutate``.

``register(sub)`` adds the two sub-commands to the estate argparse tree (hooked in by estate/commands.py).
validate runs three check families per building file (in a spawn process pool with --jobs): ifcqa (+ IDS), the
flat programme read back from the IFC (programme.py) and geometry; optionally the federation checks. It prints a
summary, writes reports/validation.json and, for files under model/, a <stem>_validation.json next to the IFC.
mutate runs the mutation suite and writes reports/mutations.json.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from estate import env

FIXED_DATE = "2026-10-04"
FAMILIES = ("ifcqa", "programme", "geometry")      # per-file result families of a validation record


def discover() -> list[Path]:
    """Building and site files of the estate (IFC4X3 masters, not the _ifc4 engine copies)."""
    out = sorted(p for p in env.MODEL.glob("*/*.ifc") if not p.stem.endswith("_ifc4") and p.parent.name == p.stem)
    site = env.MODEL / "SITE.ifc"
    return out + ([site] if site.exists() else [])


def _crashed(path, check, e):
    msg = f"{type(e).__name__}: {e}"
    return {"file": env.rel(path), "results": [{"check": check, "level": "error", "severity": "error", "count": 1,
                                                "examples": [msg], "items": [{"guids": [], "xyz": None, "text": msg}]}]}


def validate_one(path, schema=True, express_rules=False, ids=True, geometry=True, programme=True) -> dict:
    """ifcqa (+ IDS), programme and geometry of one file; importable by spawn workers. The record also names the
    file's IfcProject GlobalId (the BCF header of its issues, estate/report/bcf_out.py)."""
    import ifcopenshell

    from estate.validate import geometry as geom
    from estate.validate import ifcqa
    from estate.validate import programme as prog
    t = time.time()
    rec = {"file": env.rel(path)}
    f = ifcopenshell.open(str(path))
    rec["ifc_project"] = next((p.GlobalId for p in f.by_type("IfcProject")), None)
    rec["ifcqa"] = ifcqa.check_file(path, schema=schema, express_rules=express_rules, ids=ids, f=f)
    if programme:
        try:
            rec["programme"] = prog.check_file(path, f=f)
        except Exception as e:  # noqa: BLE001 - a crash is a failed check, not a crashed run
            rec["programme"] = _crashed(path, "programme_run", e)
    if geometry:
        try:
            rec["geometry"] = geom.check_file(path)
        except Exception as e:  # noqa: BLE001 - a crash is a failed check, not a crashed run
            rec["geometry"] = _crashed(path, "geometry_run", e)
    results = [r for fam in FAMILIES for r in rec.get(fam, {}).get("results", [])]
    rec["summary"] = ifcqa.summarize(results)
    rec["runtime_s"] = round(time.time() - t, 2)
    return rec


def _failing(results):
    return [r for r in results if r["level"] in ("error", "warn")]


def _print_record(rec):
    s = rec["summary"]
    print(f"{rec['file']:<44} errors {s['failed_errors']:>2} ({s['error_items']:>4} items)  "
          f"warnings {s['failed_warnings']:>2} ({s['warning_items']:>4})  {rec['runtime_s']:>6.1f} s")
    for fam in FAMILIES:
        for r in _failing(rec.get(fam, {}).get("results", [])):
            ex = f"  e.g. {str(r['examples'][0])[:110]}" if r["examples"] else ""
            print(f"    {r['level']:<5} {fam:<9} {r['check']:<26} {r['count']:>5}{ex}")


def cmd_validate(a):
    from estate.validate import ifcqa
    paths = [Path(p) for p in a.paths] if a.paths else discover()
    if not paths:
        print("no IFC files given and none found under model/ (pass paths, e.g. build/t_pt4.ifc)")
        return 2
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("missing: " + ", ".join(map(str, missing)))
        return 2
    opts = dict(schema=not a.no_schema, express_rules=a.rules, ids=not a.no_ids, geometry=not a.no_geometry,
                programme=not a.no_programme)
    t = time.time()
    records = []
    if a.jobs > 1 and len(paths) > 1:
        import multiprocessing as mp
        with ProcessPoolExecutor(min(a.jobs, len(paths)), mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(validate_one, str(p), **opts) for p in paths]
            for fu in futs:
                records.append(fu.result())
                _print_record(records[-1])
    else:
        for p in paths:
            records.append(validate_one(p, **opts))
            _print_record(records[-1])
    fed = None
    if a.federation:
        fed = ifcqa.check_federation(paths)
        s = fed["summary"]
        print(f"federation of {len(paths)} files: errors {s['failed_errors']} ({s['error_items']} items)")
        for r in _failing(fed["results"]):
            print(f"    {r['level']:<5} {r['check']:<26} {r['count']:>5}  {str(r['examples'][:1])[:120]}")
    for rec, p in zip(records, paths):
        try:
            Path(p).resolve().relative_to(env.MODEL.resolve())
        except ValueError:
            continue
        out = Path(p).with_name(f"{Path(p).stem}_validation.json")
        out.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    errors = sum(r["summary"]["failed_errors"] for r in records) + (fed["summary"]["failed_errors"] if fed else 0)
    report = {"date": FIXED_DATE, "files": records, "federation": fed, "options": opts,
              "summary": {"files": len(records), "files_with_errors": sum(1 for r in records if r["summary"]["failed_errors"]),
                          "failed_checks": errors, "runtime_s": round(time.time() - t, 2),
                          "by_family": by_family(records)}}
    out = Path(a.out) if a.out else env.REPORTS / "validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"{len(records)} files, {report['summary']['files_with_errors']} with errors, {errors} failed error checks, "
          f"{report['summary']['runtime_s']:.1f} s -> {env.rel(out)}")
    return 1 if errors else 0


def by_family(records) -> dict:
    """Failed checks and items per family over a set of validation records."""
    from estate.validate import ifcqa
    out = {}
    for fam in FAMILIES:
        res = [r for rec in records for r in rec.get(fam, {}).get("results", [])]
        if res:
            out[fam] = ifcqa.summarize(res)
    return out


def cmd_mutate(a):
    from estate.validate import mutations
    path = Path(a.path) if a.path else next(iter(p for p in discover() if p.stem.startswith("BLK_")),
                                            env.BUILD / "t_pt4.ifc")
    if not path.exists():
        print(f"missing: {path}")
        return 2
    print(f"mutation suite on {env.rel(path)}")
    r = mutations.run(path, only=a.only, schema=a.schema, log=print)
    out = Path(a.out) if a.out else env.REPORTS / "mutations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    print(f"{r['detected']}/{r['total']} mutations detected ({100 * r['rate']:.0f}%) in {r['runtime_s']:.1f} s -> {env.rel(out)}")
    return 0 if r["total"] and r["detected"] == r["total"] else 1


def register(sub):
    p = sub.add_parser("validate", help="IFC QA, IDS, flat programme and geometric validation of building files")
    p.add_argument("paths", nargs="*", help="IFC files (default: model/<id>/<id>.ifc and model/SITE.ifc)")
    p.add_argument("--federation", action="store_true", help="also check GUIDs / project / georeference across the files")
    p.add_argument("--no-geometry", action="store_true", help="skip the tessellation-based checks")
    p.add_argument("--rules", action="store_true",
                   help="also run the full EXPRESS rule executor (10-30 s per block; WR11 is always checked directly)")
    p.add_argument("--no-schema", action="store_true", help="skip ifcopenshell.validate")
    p.add_argument("--no-ids", action="store_true", help="skip the IDS (ids/estate.ids)")
    p.add_argument("--no-programme", action="store_true", help="skip the flat programme checks (programme.py)")
    p.add_argument("--jobs", "-j", type=int, default=1, help="files validated in parallel (spawn processes)")
    p.add_argument("--out", help="report path (default reports/validation.json)")
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("mutate", help="inject known defects into copies of a building file and require detection")
    p.add_argument("path", nargs="?", help="IFC file (default: first model/BLK_*/BLK_*.ifc, else build/t_pt4.ifc)")
    p.add_argument("--only", nargs="*", help="mutation names")
    p.add_argument("--schema", action="store_true", help="include ifcopenshell.validate in the ifcqa reruns")
    p.add_argument("--out", help="report path (default reports/mutations.json)")
    p.set_defaults(fn=cmd_mutate)
