"""reports/report.html: one page that shows what was built and whether every check passed.

Reads whatever artefacts exist (masterplan, plan checks, build state, validation, navigation, catalogue,
drawings, renders) so it can be regenerated at any point of a build.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

from estate import env


def _load(p):
    p = Path(p)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def _img(path, caption, width=None):
    rel = os.path.relpath(path, env.REPORTS).replace("\\", "/")
    w = f' style="max-width:{width}px"' if width else ""
    return f'<figure><a href="{rel}"><img src="{rel}" loading="lazy"{w}></a><figcaption>{html.escape(caption)}</figcaption></figure>'


def _badge(ok, text=None):
    cls = "ok" if ok else "bad"
    return f'<span class="badge {cls}">{html.escape(text or ("pass" if ok else "fail"))}</span>'


def _table(rows, cols):
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in rows)
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


CSS = """
:root{--fg:#1d2733;--muted:#5b6876;--bg:#ffffff;--card:#f5f6f8;--line:#d9dee4;--ok:#2e7d4f;--bad:#b3261e}
@media (prefers-color-scheme: dark){:root{--fg:#e6e9ee;--muted:#9aa6b2;--bg:#14181d;--card:#1d232a;--line:#323a44}}
body{font:14px/1.45 system-ui,Segoe UI,Arial,sans-serif;color:var(--fg);background:var(--bg);margin:0 auto;max-width:1280px;padding:16px}
h1{font-size:24px;margin:8px 0}h2{font-size:18px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.muted{color:var(--muted)}table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}th{background:var(--card)}
.badge{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px;color:#fff}.ok{background:var(--ok)}.bad{background:var(--bad)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
figure{margin:0;background:var(--card);padding:6px;border-radius:6px}figure img{width:100%;height:auto;display:block}
figcaption{font-size:12px;color:var(--muted);margin-top:4px}.kpi{display:flex;gap:12px;flex-wrap:wrap}
.kpi div{background:var(--card);padding:8px 12px;border-radius:6px}.kpi b{display:block;font-size:20px}
pre{background:var(--card);padding:8px;overflow:auto;font-size:12px}.tw{overflow-x:auto;max-width:100%}
"""


def write_report(out=None) -> Path:
    out = Path(out or env.REPORTS / "report.html")
    mp = _load(env.MODEL / "masterplan.json") or {}
    pc = _load(env.REPORTS / "plan_check.json") or {}
    stt = _load(env.MODEL / ".state.json") or {}
    val = _load(env.REPORTS / "validation_summary.json")
    if not isinstance(val, dict):
        val = None
    nav = _load(env.REPORTS / "nav_estate.json")
    cat = _load(env.REPORTS / "catalogue.json") or []
    parts = [f"<h1>{html.escape(mp.get('estate', {}).get('name', 'Sample Town N5'))} - build report</h1>",
             '<p class="muted">Walkable HDB neighbourhood generated as federated IFC4X3 (IfcOpenShell 0.9 / Bonsai). '
             'All figures below are read from the build artefacts in model/ and reports/.</p>']
    fails = _load(env.REPORTS / "build_failures.json")
    if isinstance(fails, list):
        if fails:
            rows = [[html.escape(str(f.get("target", ""))), html.escape(str(f.get("stage", ""))),
                     html.escape(str(f.get("error") or f.get("errors") or "")[:300])] for f in fails]
            parts.append("<h2>Build status " + _badge(False, f"{len(fails)} failure(s)") + "</h2>"
                         + _table(rows, ["target", "stage", "error"]))
        else:
            parts.append("<h2>Build status " + _badge(True, "all stages passed") + "</h2>")
    mix = pc.get("mix", {})
    total = sum(mix.values()) if mix else 0
    n_blocks = sum(1 for s in mp.get("sites", []) if s.get("kind") == "block")
    parts.append('<div class="kpi">' + "".join(f"<div><b>{v}</b>{k}</div>" for k, v in [
        ("residential blocks", n_blocks), ("flats", total), ("stack layouts (raw)", pc.get("raw", "-")),
        ("mirror-canonical layouts", pc.get("canonical", "-")), ("plan errors", len(pc.get("errors", [])))]) + "</div>")
    if mix:
        parts.append("<h2>Unit mix</h2>" + _table([[t, mix.get(t, 0), f"{mix.get(t, 0) / max(total, 1) * 100:.1f}%"]
                                                   for t in ["2RF", "3R", "4R", "5R", "3G", "EA"]], ["type", "flats", "share"]))
    # buildings: one row per target with every stage's result
    rows = []
    for site in mp.get("sites", []) + [{"id": "SITE", "typology": "site works", "storeys": ""}]:
        tid = site["id"]
        rec = stt.get(tid, {})
        folder = env.MODEL if tid == "SITE" else env.MODEL / tid
        v = _load(folder / f"{tid}_validation.json") or {}
        vs = v.get("summary", {})
        n = _load(folder / f"{tid}_nav.json") or {}
        nr = next(iter(n.get("results", {}).values()), {}).get("summary", {}) if n else {}
        glb = rec.get("glb", {})
        check = _badge(vs.get("failed_errors", 1) == 0, f"{vs.get('failed_errors', '-')} err / {vs.get('warning_items', '-')} warn")             if vs else "-"
        nav_cell = (_badge(n.get("ok", False), f"{nr.get('reachable', 0)}/{nr.get('spaces', 0)} spaces")
                    + (f" step-free {nr.get('flats_main_door_step_free', 0)}/{nr.get('flats', 0)}" if nr.get("flats") else ""))             if n else "-"
        rows.append([tid, site.get("typology", ""), site.get("storeys", ""), rec.get("ifc", {}).get("flats", ""),
                     rec.get("ifc", {}).get("elements", ""), check, nav_cell,
                     f"{glb.get('lod0_triangles', ''):,}" if glb.get("lod0_triangles") else "-",
                     _badge("blend" in rec, "verified") if "blend" in rec else "-"])
    if rows:
        parts.append("<h2>Buildings</h2>" + _table(rows, ["id", "typology", "storeys", "flats", "IFC elements",
                                                           "IFC / IDS / programme / geometry checks", "walk test (r 0.30 m)",
                                                           "LOD0 triangles", "Bonsai .blend"]))
    site_nav = _load(env.REPORTS / "nav_site.json")
    if nav or site_nav:
        es = (nav or {}).get("summary", {})
        ss = (site_nav or {}).get("summary", {})
        parts.append("<h2>Estate walkability</h2>" + _table([
            ["flats reachable from every bus stop", f"{es.get('reachable_from_every_bus_stop', '-')} / {es.get('flats', '-')}",
             _badge(es.get("reachable_from_every_bus_stop") == es.get("flats"))],
            ["step-free from every bus stop", f"{es.get('step_free_from_every_bus_stop', '-')} / {es.get('flats', '-')}",
             _badge(es.get("step_free_from_every_bus_stop") == es.get("flats"))],
            ["longest walk, nearest bus stop to a lift lobby", f"{ss.get('nearest_stop_max_walk_m', '-')} m (target <= 400 m)",
             _badge(bool(ss.get("walk_ok")))],
            ["lowest covered share of a nearest-stop route", f"{ss.get('nearest_stop_min_covered_share', '-')} (target >= 0.90)",
             _badge(bool(ss.get("covered_ok")))],
            ["bus stop to lobby routes checked", f"{ss.get('reachable', '-')} / {ss.get('routes', '-')}",
             _badge(bool(ss.get("reachability_ok")))],
        ], ["measure", "result", "check"]))
    if pc.get("errors"):
        parts.append("<h2>Plan-level errors</h2><pre>" + html.escape("\n".join(pc["errors"][:200])) + "</pre>")
    # images
    # validation totals and federation
    if val:
        fam = {}
        for tid, rec in (val.get("files") or {}).items():
            for x in rec.get("failing", []):
                k = (x["family"], x["check"], x["level"])
                fam[k] = fam.get(k, 0) + x["count"]
        fed = val.get("federation") or {}
        fed_bad = [r for r in fed.get("results", []) if r.get("level") == "error" and r.get("count")]
        rows = [[f, c, _badge(lv != "error", lv), n] for (f, c, lv), n in sorted(fam.items())]
        parts.append("<h2>Validation</h2><p>"
                     + f"{len(val.get('files') or {})} files; federation across "
                     + f"{len(fed.get('files', []))} masters: {_badge(not fed_bad)} (shared IfcProject / IfcSite, "
                     + "unique GlobalIds, one EPSG:3414 map conversion). Checks that are not clean:</p>"
                     + (_table(rows, ["family", "check", "level", "items (all files)"]) if rows else "<p>none</p>"))
    mut = _load(env.REPORTS / "mutations.json")
    if mut:
        parts.append("<h2>Validator mutation suite</h2><p>"
                     + f"{mut.get('detected')}/{mut.get('total')} injected defects detected "
                     + _badge(mut.get("detected") == mut.get("total")) + f" on {html.escape(str(mut.get('file')))}</p>"
                     + _table([[html.escape(r["mutation"]), html.escape(r.get("describe", "")), _badge(r["detected"])]
                               for r in mut.get("rows", [])], ["mutation", "defect", "caught"]))
    leg = _load(env.BUILD / "legacy" / "legacy_counts.json")
    if leg:
        same = leg.get("baseline") == leg.get("rebuilt")
        parts.append("<h2>Legacy regression</h2><p>The original Blk 123 rebuilt by the estate code: "
                     + _badge(same, "identical counts" if same else "counts differ") + "</p>"
                     + _table([[c, leg["baseline"].get(c), leg["rebuilt"].get(c)] for c in leg.get("baseline", {})],
                              ["class", "hdb_block.ifc", "rebuilt"]))
    glbs = [(tid, r["glb"]) for tid, r in stt.items() if isinstance(r, dict) and "glb" in r]
    if glbs:
        worst = max((g.get("max_tris_per_flat") or 0) for _, g in glbs)
        total = sum((g.get("lod0_triangles") or 0) for _, g in glbs)
        parts.append("<h2>Engine export</h2>" + _table([
            ["LOD0 triangles, whole estate", f"{total:,} (budget 8,000,000)", _badge(total <= 8_000_000)],
            ["worst flat, LOD0 triangles", f"{worst:,.0f} (budget 5,000)", _badge(worst <= 5000)],
            ["manifest", "model/estate_manifest.json", _badge((env.MODEL / "estate_manifest.json").exists(), "written")],
        ], ["measure", "value", "check"]))
    eb = _load(env.BUILD / "logs" / "blender" / "estate_blend.json")
    if eb:
        mk = eb.get("make", {})
        links = mk.get("links", [])
        ok = all(l.get("is_loaded") for l in links) and bool(links)
        parts.append("<h2>ESTATE.blend</h2>" + _table([
            ["linked buildings loaded", f"{sum(1 for l in links if l.get('is_loaded'))}/{len(links)}", _badge(ok)],
            ["link time", f"{mk.get('links_seconds')} s (budget 60 s)", _badge((mk.get("links_seconds") or 0) <= 60)],
            ["peak memory", f"{mk.get('peak_working_set_mb')} MB (budget 8 GB)",
             _badge((mk.get("peak_working_set_mb") or 0) <= 8192)],
            ["verified after reopen", "yes" if eb.get("verify") else "no", _badge(bool(eb.get("verify")))],
        ], ["measure", "value", "check"]))
    if pc.get("per_type"):
        parts.append("<h2>Layout variety</h2><p>Distinct stack layouts (mirror images counted once) per flat type; the "
                     "plan asks for at least 3 per type (2 for 2-room Flexi, 3Gen and Executive), 40 raw and 20 "
                     "canonical overall.</p>" + _table([[t, n] for t, n in pc["per_type"].items()], ["type", "layouts"]))
    site_plan = env.REPORTS / "site_plan.png"
    if site_plan.exists():
        parts.append("<h2>Site plan</h2>" + _img(site_plan, "Site plan", 1200))
    figs = []
    for d in sorted(env.MODEL.glob("*/plans/*_typical.png")):
        figs.append(_img(d, d.stem.replace("_", " ")))
    if not figs:
        figs = [_img(p, p.stem.replace("_", " ")) for p in sorted((env.REPORTS / "plans").glob("*_typical.png"))]
    if figs:
        parts.append("<h2>Typical floor plans</h2><div class='grid'>" + "".join(figs) + "</div>")
    renders = sorted((env.REPORTS / "renders").rglob("*.png")) if (env.REPORTS / "renders").exists() else []
    renders.sort(key=lambda p: (p.parent.name != "ESTATE", p.parent.name, p.name))
    if renders:
        parts.append("<h2>Renders (Blender Workbench, from the Bonsai .blend files)</h2><div class='grid'>"
                     + "".join(_img(p, p.stem.replace("_", " ")) for p in renders) + "</div>")
    navs = sorted(env.MODEL.glob("*/nav/*.png"))
    if navs:
        parts.append("<h2>Walk-test maps</h2><div class='grid'>" + "".join(_img(p, f"{p.parent.parent.name} {p.stem}")
                                                                           for p in navs[:60]) + "</div>")
    if cat:
        rows = [[c["template"], c["type"], html.escape(c["title"]), c["cases"], len(c.get("signatures", [])),
                 _badge(c["errors"] == 0, f"{c['errors']} errors")] for c in cat]
        parts.append("<h2>Flat catalogue</h2>" + _table(rows, ["template", "type", "title", "cases", "layouts", "checks"]))
        gal = []
        for c in cat:
            pngs = sorted((env.REPORTS / "flat_gallery").glob(f"{c['template']}_*.png"))
            if pngs:
                gal.append(_img(pngs[0], f"{c['template']} ({c['type']})"))
        parts.append("<div class='grid'>" + "".join(gal) + "</div>")
    page = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>" \
           f"<title>Sample Town N5 report</title><style>{CSS}</style></head><body>{''.join(parts)}</body></html>"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
