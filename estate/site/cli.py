"""CLI for the site works: ``estate.cmd site [--schema IFC4X3] [--out model/SITE.ifc] [--ifc4] [--validate]``.

Resolves the masterplan, prints the siting checks, plans the external works once and writes SITE.ifc (and the
IFC4 copy for Unreal Datasmith with --ifc4), the pedestrian graph model/SITE_graph.json and the site plan
reports/site_plan.png plus its editable vector twin reports/site_plan.svg (--no-png / --no-svg to skip either).
The building IFCs under --model-dir (default model/) are read for their ground floors, so build them first; the
pedestrian graph is checked against them and against the SITE.ifc just written, and a mismatch fails the command.
Hooked into estate/commands.py by the lead through ``register(sub)``.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from estate import env

_VALIDATE = r"""
import sys, json
sys.path.insert(0, {root!r})
from estate.env import bootstrap; bootstrap()
import ifcopenshell, ifcopenshell.validate
out = {{}}
for p in {paths!r}:
    f = ifcopenshell.open(p)
    lg = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(f, lg, express_rules=True)
    out[p] = [dict(type=str(s.get("type")), attribute=str(s.get("attribute")), message=str(s.get("message"))[:300])
              for s in lg.statements]
print(json.dumps(out))
"""


def validate_files(paths) -> dict:
    """ifcopenshell.validate (schema + express rules) in a fresh interpreter: authoring swaps the entity hash for
    deterministic output, so validating in the authoring process can report spurious set-based rule failures."""
    code = _VALIDATE.format(root=str(env.ROOT), paths=[str(p) for p in paths])
    r = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def cmd_site(a):
    from estate import masterplan
    from estate.site import builder
    t0 = time.time()
    mp = masterplan.resolve()
    issues = masterplan.siting_checks(mp)
    print(f"masterplan: {len(mp['sites'])} sites, {len(mp['roads'])} roads, {len(mp['greens'])} greens, "
          f"{len(mp['bus_stops'])} bus stops")
    approx = [s.id for s in mp["sites"] if s.approximate]
    if approx:
        print(f"  approximate footprints (typology not planned yet): {', '.join(approx)}")
    print(f"siting checks: {len(issues)} issue(s)")
    for i in issues:
        print(f"  - {i}")
    print("planning site works")
    model_dir = Path(a.model_dir) if a.model_dir else None
    layout = builder.plan_site(mp, log=print, model_dir=model_dir)
    for w in layout.warnings:
        print(f"  warning: {w}")
    for n in layout.notes:
        print(f"  note: {n}")
    fallback = [sid for sid, g in sorted(layout.grounds.items()) if g.source == "footprint"]
    if fallback:
        print(f"  no IFC yet (ground floor taken as open footprint): {', '.join(fallback)}")
    out = Path(a.out) if a.out else env.MODEL / "SITE.ifc"
    graph = Path(a.graph) if a.graph else env.MODEL / "SITE_graph.json"
    png = None if a.no_png else (Path(a.png) if a.png else env.REPORTS / "site_plan.png")
    svg = None if a.no_svg else (Path(a.svg) if a.svg else env.REPORTS / "site_plan.svg")
    bad = 0
    try:
        info = builder.build_site(mp, out, a.schema, graph_path=graph, plan_png=png, layout=layout, log=print,
                                  model_dir=model_dir, plan_svg=svg)
    except builder.GraphCheckError as e:
        info, bad = e.info, 1
    issues = info["graph_check"]["issues"]
    print(f"pedestrian graph vs {env.rel(out)} and the building ground floors: "
          f"{'ok' if not issues else f'{len(issues)} issue(s)'}")
    for i in issues[:20]:
        print(f"  ERROR: {i}")
    written = [out]
    if a.ifc4 and a.schema.upper() != "IFC4":
        out4 = out.with_name(out.stem + "_ifc4.ifc")
        i4 = builder.write_site(layout, out4, "IFC4", log=print)
        info["ifc4"] = dict(path=env.rel(out4), elements=i4["elements"], by_class=i4["by_class"])
        written.append(out4)
    st = info["graph_stats"]
    print(f"IfcElement by class ({a.schema}): " + ", ".join(f"{k} {v}" for k, v in info["by_class"].items()))
    print(f"pedestrian graph: {st['nodes']} nodes, {st['edges']} edges, {st['components']} component(s); "
          f"{st['length_m']:.0f} m ({st['covered_m']:.0f} m covered); unreachable stop-lobby pairs "
          f"{st['n_unreachable']}")
    near = st.get("covered_share_from_nearest_stop", {})
    print(f"  covered share from the nearest bus stop: min {near.get('min')} mean {near.get('mean')}; "
          f"longest walk {near.get('max_walk_m')} m")
    print(f"  {info['crossings']} crossings, {info['crossovers']} driveway crossovers, {info['assemblies']} assemblies, "
          f"{info['trees']} trees")
    for e in info.get("escape_paths", []):
        how = {"paved": f"{e['width']} m path, {e['length']} m to the footway", "on_footway": "opens onto a footway",
               "on_floor": "lets out onto the building's own floor", "failed": "NO PATH"}[e["status"]]
        print(f"  stair discharge {e['site']} at ({e['at'][0]}, {e['at'][1]}): {how}")
    if a.validate:
        res = validate_files(written)
        for p, errs in res.items():
            print(f"validate {env.rel(p)}: {len(errs)} issue(s)")
            for e in errs[:10]:
                print(f"  {e['type']} {e['attribute']}: {e['message'][:160]}")
            bad += len(errs)
        info["validation"] = {env.rel(p): len(e) for p, e in res.items()}
    over = [p for p in [info] + ([info["ifc4"]] if "ifc4" in info else []) if p["elements"] > 25000]
    if over:
        print("  ERROR: more than 25,000 IfcElement in a file")
        bad += 1
    rep = (out.parent if a.out else env.REPORTS) / "site_build.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    builder.write_site_report(info, rep)
    print(f"done in {time.time() - t0:.1f} s -> {env.rel(out)}, {env.rel(graph)}"
          f"{', ' + env.rel(png) if png else ''}{', ' + env.rel(svg) if svg else ''}, {env.rel(rep)}")
    return 1 if bad else 0


def register(sub):
    p = sub.add_parser("site", help="build SITE.ifc (external works), the pedestrian graph and the site plan")
    p.add_argument("--schema", default="IFC4X3", choices=["IFC4X3", "IFC4"])
    p.add_argument("--out", help="output IFC (default model/SITE.ifc)")
    p.add_argument("--ifc4", action="store_true", help="also write <out>_ifc4.ifc for Unreal Datasmith")
    p.add_argument("--graph", help="pedestrian graph JSON (default model/SITE_graph.json)")
    p.add_argument("--png", help="site plan PNG (default reports/site_plan.png)")
    p.add_argument("--no-png", action="store_true")
    p.add_argument("--svg", help="site plan SVG, editable vector twin of the PNG (default reports/site_plan.svg)")
    p.add_argument("--no-svg", action="store_true")
    p.add_argument("--model-dir", help="folder holding <id>/<id>.ifc of the buildings (default model/)")
    p.add_argument("--validate", action="store_true", help="run ifcopenshell.validate on the written files")
    p.set_defaults(fn=cmd_site)
