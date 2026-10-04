"""reports/report.html and reports/report.json: one page that shows what was built and whether every check passed,
its JSON twin, and reports/issues.bcf with every failing validation item as a BCF topic.

``collect()`` reads whatever artefacts exist (masterplan, plan checks, build state, validation, navigation,
catalogue, drawings, renders) into one dict, section by section in page order, with a fixed key order and no clock,
so it can run at any point of a build and the same artefacts always give the same dict. Its "files" list names every
artefact it read or links (project-relative path, every data key it feeds - "report" for the title, then the
sections in page order - size and sha256), so a reader of
report.json knows exactly which outputs the figures come from. ``render(data)`` turns the dict into the page and
does no reading of its own besides; ``write_report()`` writes report.html, report.json and issues.bcf
(estate/report/bcf_out.py) side by side, image links relative to that folder.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath

from estate import env
from estate.report import bcf_out

BCF_NAME = "issues.bcf"
JSON_NAME = "report.json"
FLAT_TYPES = ["2RF", "3R", "4R", "5R", "3G", "EA"]
WALK_MAPS_SHOWN = 60
SECTIONS = {      # data key -> the page's <h2> heading (kpi is the unheaded strip under the title)
    "build": "Build status", "kpi": None, "mix": "Unit mix", "buildings": "Buildings",
    "walkability": "Estate walkability", "plan_errors": "Plan-level errors", "validation": "Validation",
    "mutations": "Validator mutation suite", "legacy": "Legacy regression", "engine": "Engine export",
    "estate_blend": "ESTATE.blend", "layout_variety": "Layout variety", "site_plan": "Site plan",
    "typical_plans": "Typical floor plans", "renders": "Renders (Blender Workbench, from the Bonsai .blend files)",
    "walk_maps": "Walk-test maps", "catalogue": "Flat catalogue"}


def _load(p):
    p = Path(p)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


class _Inputs:
    """The artefacts collect() read or links, in first-use order: {path, sections, bytes, sha256}; "sections" are
    the data keys the file feeds, in the order collect() reaches them."""

    def __init__(self):
        self.rows = {}

    def add(self, p, section) -> bool:
        p = Path(p)
        if not p.is_file():
            return False
        k = env.rel(p)
        if k not in self.rows:
            b = p.read_bytes()
            self.rows[k] = {"path": k, "sections": [section], "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
        elif section not in self.rows[k]["sections"]:
            self.rows[k]["sections"].append(section)
        return True

    def load(self, p, section):
        return _load(p) if self.add(p, section) else None

    def also(self, p, section, used=True):
        """A later section reads an artefact an earlier one loaded: list it there too (when it gave that section
        anything), so "sections" names every section a file feeds, not just the first one that opened it."""
        if used:
            self.add(p, section)

    def image(self, p, section, caption) -> dict:
        self.add(p, section)
        return {"path": env.rel(p), "caption": caption}


# ----------------------------------------------------------------------------- collect
def collect() -> dict:
    """Every section's data in page order (None or empty when its artefacts are missing), plus "files"."""
    inp = _Inputs()
    # masterplan, plan checks and build state feed several sections each: loaded once here, and every section that
    # reads them adds itself with inp.also() (the title, kpi strip, mix, buildings, plan errors, engine, variety)
    mp_p, pc_p, st_p = env.MODEL / "masterplan.json", env.REPORTS / "plan_check.json", env.MODEL / ".state.json"
    mp = inp.load(mp_p, "report") or {}
    pc = inp.load(pc_p, "kpi") or {}
    stt = inp.load(st_p, "buildings") or {}
    name = mp.get("estate", {}).get("name", "Sample Town N5")
    data = {"report": {"title": f"{name} - build report", "estate": name, "html": "report.html", "json": JSON_NAME,
                       "bcf": BCF_NAME},
            "files": None}                      # filled last, kept here in the key order
    fails = inp.load(env.REPORTS / "build_failures.json", "build")
    data["build"] = {"ok": not fails, "failures": fails} if isinstance(fails, list) else None

    mix = pc.get("mix") or {}
    total = sum(mix.values()) if mix else 0
    inp.also(mp_p, "kpi")
    inp.also(pc_p, "mix", bool(mix))
    data["kpi"] = {"residential_blocks": sum(1 for s in mp.get("sites", []) if s.get("kind") == "block"),
                   "flats": total, "stack_layouts_raw": pc.get("raw"), "mirror_canonical_layouts": pc.get("canonical"),
                   "plan_errors": len(pc.get("errors", []))}
    data["mix"] = {"total": total, "flats": dict(mix)} if mix else None

    rows = []         # buildings: one row per target with every stage's result
    inp.also(mp_p, "buildings", bool(mp.get("sites")))
    for site in mp.get("sites", []) + [{"id": "SITE", "typology": "site works", "storeys": None}]:
        tid = site["id"]
        rec = stt.get(tid, {})
        folder = env.MODEL if tid == "SITE" else env.MODEL / tid
        v = inp.load(folder / f"{tid}_validation.json", "buildings") or {}
        n = inp.load(folder / f"{tid}_nav.json", "buildings") or {}
        nr = next(iter(n.get("results", {}).values()), {}).get("summary", {}) if n else {}
        rows.append({"id": tid, "typology": site.get("typology", ""), "storeys": site.get("storeys"),
                     "flats": rec.get("ifc", {}).get("flats"), "elements": rec.get("ifc", {}).get("elements"),
                     "validation": v.get("summary") or None,
                     "nav": {"ok": n.get("ok", False), "summary": nr} if n else None,
                     "lod0_triangles": rec.get("glb", {}).get("lod0_triangles"), "blend_verified": "blend" in rec})
    data["buildings"] = rows

    nav = inp.load(env.REPORTS / "nav_estate.json", "walkability")
    site_nav = inp.load(env.REPORTS / "nav_site.json", "walkability")
    data["walkability"] = ({"estate": (nav or {}).get("summary", {}), "site": (site_nav or {}).get("summary", {})}
                           if nav or site_nav else None)
    data["plan_errors"] = list(pc.get("errors") or [])
    inp.also(pc_p, "plan_errors", bool(data["plan_errors"]))
    data["validation"] = _validation(inp)

    mut = inp.load(env.REPORTS / "mutations.json", "mutations")
    data["mutations"] = {"file": mut.get("file"), "detected": mut.get("detected"), "total": mut.get("total"),
                         "rows": [{"mutation": r["mutation"], "describe": r.get("describe", ""),
                                   "detected": r["detected"]} for r in mut.get("rows", [])]} if mut else None
    leg = inp.load(env.BUILD / "legacy" / "legacy_counts.json", "legacy")
    data["legacy"] = {"identical": leg.get("baseline") == leg.get("rebuilt"), "baseline": leg.get("baseline", {}),
                      "rebuilt": leg.get("rebuilt", {})} if leg else None
    glbs = [r["glb"] for r in stt.values() if isinstance(r, dict) and "glb" in r]
    inp.also(st_p, "engine", bool(glbs))
    manifest = env.MODEL / "estate_manifest.json"
    data["engine"] = {"buildings": len(glbs),
                      "lod0_triangles": sum((g.get("lod0_triangles") or 0) for g in glbs), "lod0_budget": 8_000_000,
                      "worst_flat_lod0_triangles": max((g.get("max_tris_per_flat") or 0) for g in glbs),
                      "flat_budget": 5000,
                      "interior_chunks": sum((g.get("interior_chunks") or 0) for g in glbs),
                      "chunks_ok": all(r["glb"].get("interior_chunks") for tid, r in stt.items()
                                       if isinstance(r, dict) and "glb" in r and tid != "SITE"),
                      "manifest": env.rel(manifest) if inp.add(manifest, "engine") else None} if glbs else None
    eb = inp.load(env.BUILD / "logs" / "blender" / "estate_blend.json", "estate_blend")
    if eb:
        mk = eb.get("make", {})
        links = mk.get("links", [])
        eb = {"links": len(links), "links_loaded": sum(1 for x in links if x.get("is_loaded")),
              "links_seconds": mk.get("links_seconds"), "links_budget_s": 60,
              "peak_working_set_mb": mk.get("peak_working_set_mb"), "memory_budget_mb": 8192,
              "verified": bool(eb.get("verify"))}
    data["estate_blend"] = eb or None
    data["layout_variety"] = pc.get("per_type") or None
    inp.also(pc_p, "layout_variety", bool(data["layout_variety"]))

    sp = env.REPORTS / "site_plan.png"
    svg = env.REPORTS / "site_plan.svg"
    data["site_plan"] = dict(inp.image(sp, "site_plan", "Site plan"),
                             svg=env.rel(svg) if inp.add(svg, "site_plan") else None) if sp.exists() else None
    plans = sorted(env.MODEL.glob("*/plans/*_typical.png")) or sorted((env.REPORTS / "plans").glob("*_typical.png"))
    data["typical_plans"] = [inp.image(p, "typical_plans", p.stem.replace("_", " ")) for p in plans]
    renders = sorted((env.REPORTS / "renders").rglob("*.png")) if (env.REPORTS / "renders").exists() else []
    renders.sort(key=lambda p: (p.parent.name != "ESTATE", p.parent.name, p.name))
    data["renders"] = [inp.image(p, "renders", p.stem.replace("_", " ")) for p in renders]
    data["walk_maps"] = [inp.image(p, "walk_maps", f"{p.parent.parent.name} {p.stem}")
                         for p in sorted(env.MODEL.glob("*/nav/*.png"))[:WALK_MAPS_SHOWN]]
    cat = inp.load(env.REPORTS / "catalogue.json", "catalogue") or []
    rows = []
    for c in cat:
        pngs = sorted((env.REPORTS / "flat_gallery").glob(f"{c['template']}_*.png"))
        rows.append({"template": c["template"], "type": c["type"], "title": c["title"], "cases": c["cases"],
                     "layouts": len(c.get("signatures", [])), "errors": c["errors"],
                     "gallery": inp.image(pngs[0], "catalogue", f"{c['template']} ({c['type']})") if pngs else None})
    data["catalogue"] = rows
    data["files"] = list(inp.rows.values())
    return data


def _validation(inp) -> dict | None:
    """Validation totals per (family, check, level) over all files, the federation, and the BCF topics the
    per-file records give (bcf_out.issues: one per failing item)."""
    val = inp.load(env.REPORTS / "validation_summary.json", "validation")
    if not isinstance(val, dict):
        val = None
    for p in bcf_out.validation_paths():
        inp.add(p, "validation")
    records = bcf_out.validation_records()
    fed_rec = bcf_out.federation_record() if val else None
    found = bcf_out.issues(records + ([fed_rec] if fed_rec else []))
    if val is None and not records:
        return None
    fam = {}
    for tid, rec in ((val or {}).get("files") or {}).items():
        for x in rec.get("failing", []):
            k = (x["family"], x["check"], x["level"])
            fam[k] = fam.get(k, 0) + x["count"]
    fed = (val or {}).get("federation") or {}
    fed_bad = [r for r in fed.get("results", []) if r.get("level") == "error" and r.get("count")]
    return {"summary": val is not None,
            "files": len((val or {}).get("files") or {}),
            "per_file": {tid: rec.get("summary") for tid, rec in ((val or {}).get("files") or {}).items()},
            "federation": {"files": len(fed.get("files", [])), "ok": not fed_bad,
                           "failing": [{"check": r["check"], "count": r["count"]} for r in fed_bad]},
            "failing_checks": [{"family": f, "check": c, "level": lv, "items": n}
                               for (f, c, lv), n in sorted(fam.items())],
            "bcf": dict(file=BCF_NAME, **bcf_out.summary(found))}


# ----------------------------------------------------------------------------- render
def _rel(path, base):
    """A link from the page's folder to a project-relative artefact path."""
    return os.path.relpath(env.ROOT / path, base).replace("\\", "/")


def _img(img, base, width=None):
    rel = _rel(img["path"], base)
    w = f' style="max-width:{width}px"' if width else ""
    return (f'<figure><a href="{rel}"><img src="{rel}" loading="lazy"{w}></a>'
            f'<figcaption>{html.escape(img["caption"])}</figcaption></figure>')


def _badge(ok, text=None):
    cls = "ok" if ok else "bad"
    return f'<span class="badge {cls}">{html.escape(text or ("pass" if ok else "fail"))}</span>'


def _table(rows, cols):
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in rows)
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _h2(key, extra=""):
    return f"<h2>{SECTIONS[key]}{extra}</h2>"


def _v(x, missing=""):
    return missing if x is None else x


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


def _bcf_line(b) -> str:
    if b.get("error"):
        return f"<p>BCF issues: {_badge(False, 'not written')} {html.escape(b['error'])}</p>"
    return (f"<p>Issues for BIM tools: <a href=\"{b['file']}\">{b['file']}</a> (BCF 2.1) has {b['topics']} topic(s), "
            f"{b['errors']} error(s) and {b['warnings']} warning(s): one per failing item, with its elements selected "
            f"and, for {b['with_viewpoint']} of them, a viewpoint on the spot. Open it in Bonsai's BCF panel or any BCF "
            f"viewer next to the building IFCs.</p>")


def render(data, base=None) -> str:
    """The page for collect()'s dict; image links are relative to ``base`` (the page's folder, default reports/)."""
    base = Path(base or env.REPORTS)
    rep = data["report"]
    parts = [f"<h1>{html.escape(rep['estate'])} - build report</h1>",
             '<p class="muted">Walkable HDB neighbourhood generated as federated IFC4X3 (IfcOpenShell 0.9 / Bonsai). '
             'All figures below are read from the build artefacts in model/ and reports/; the same data, with the list '
             f'of artefacts it was read from, is in <a href="{rep["json"]}">{rep["json"]}</a>.</p>']
    b = data.get("build")
    if b is not None:
        if b["failures"]:
            rows = [[html.escape(str(f.get("target", ""))), html.escape(str(f.get("stage", ""))),
                     html.escape(str(f.get("error") or f.get("errors") or "")[:300])] for f in b["failures"]]
            parts.append(_h2("build", " " + _badge(False, f"{len(b['failures'])} failure(s)"))
                         + _table(rows, ["target", "stage", "error"]))
        else:
            parts.append(_h2("build", " " + _badge(True, "all stages passed")))
    k = data["kpi"]
    parts.append('<div class="kpi">' + "".join(f"<div><b>{v}</b>{t}</div>" for t, v in [
        ("residential blocks", k["residential_blocks"]), ("flats", k["flats"]),
        ("stack layouts (raw)", _v(k["stack_layouts_raw"], "-")),
        ("mirror-canonical layouts", _v(k["mirror_canonical_layouts"], "-")),
        ("plan errors", k["plan_errors"])]) + "</div>")
    mix = data.get("mix")
    if mix:
        fl, total = mix["flats"], mix["total"]
        parts.append(_h2("mix") + _table([[t, fl.get(t, 0), f"{fl.get(t, 0) / max(total, 1) * 100:.1f}%"]
                                          for t in FLAT_TYPES], ["type", "flats", "share"]))
    rows = []
    for r in data["buildings"]:
        vs, nv = r["validation"], r["nav"]
        check = (_badge(vs.get("failed_errors", 1) == 0,
                        f"{vs.get('failed_errors', '-')} err / {vs.get('warning_items', '-')} warn") if vs else "-")
        nr = (nv or {}).get("summary", {})
        nav_cell = (_badge(nv["ok"], f"{nr.get('reachable', 0)}/{nr.get('spaces', 0)} spaces")
                    + (f" step-free {nr.get('flats_main_door_step_free', 0)}/{nr.get('flats', 0)}" if nr.get("flats")
                       else "")) if nv else "-"
        rows.append([r["id"], r["typology"], _v(r["storeys"]), _v(r["flats"]), _v(r["elements"]), check, nav_cell,
                     f"{r['lod0_triangles']:,}" if r["lod0_triangles"] else "-",
                     _badge(True, "verified") if r["blend_verified"] else "-"])
    if rows:
        parts.append(_h2("buildings") + _table(rows, ["id", "typology", "storeys", "flats", "IFC elements",
                                                      "IFC / IDS / programme / geometry checks", "walk test (r 0.30 m)",
                                                      "LOD0 triangles", "Bonsai .blend"]))
    w = data.get("walkability")
    if w:
        es, ss = w["estate"], w["site"]
        parts.append(_h2("walkability") + _table([
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
    if data["plan_errors"]:
        parts.append(_h2("plan_errors") + "<pre>" + html.escape("\n".join(data["plan_errors"][:200])) + "</pre>")
    v = data.get("validation")
    if v:
        bcf = _bcf_line(v["bcf"])
        if v["summary"]:
            rows = [[x["family"], x["check"], _badge(x["level"] != "error", x["level"]), x["items"]]
                    for x in v["failing_checks"]]
            parts.append(_h2("validation") + "<p>"
                         + f"{v['files']} files; federation across "
                         + f"{v['federation']['files']} masters: {_badge(v['federation']['ok'])} (shared IfcProject / "
                         + "IfcSite, unique GlobalIds, one EPSG:3414 map conversion). Checks that are not clean:</p>"
                         + (_table(rows, ["family", "check", "level", "items (all files)"]) if rows else "<p>none</p>")
                         + bcf)
        else:
            parts.append(_h2("validation") + bcf)
    mut = data.get("mutations")
    if mut:
        parts.append(_h2("mutations") + "<p>"
                     + f"{mut['detected']}/{mut['total']} injected defects detected "
                     + _badge(mut["detected"] == mut["total"]) + f" on {html.escape(str(mut['file']))}</p>"
                     + _table([[html.escape(r["mutation"]), html.escape(r["describe"]), _badge(r["detected"])]
                               for r in mut["rows"]], ["mutation", "defect", "caught"]))
    leg = data.get("legacy")
    if leg:
        parts.append(_h2("legacy") + "<p>The original Blk 123 rebuilt by the estate code: "
                     + _badge(leg["identical"], "identical counts" if leg["identical"] else "counts differ") + "</p>"
                     + _table([[c, leg["baseline"].get(c), leg["rebuilt"].get(c)] for c in leg["baseline"]],
                              ["class", "hdb_block.ifc", "rebuilt"]))
    e = data.get("engine")
    if e:
        parts.append(_h2("engine") + _table([
            ["LOD0 triangles, whole estate", f"{e['lod0_triangles']:,} (budget 8,000,000)",
             _badge(e["lod0_triangles"] <= e["lod0_budget"])],
            ["worst flat, LOD0 triangles", f"{e['worst_flat_lod0_triangles']:,.0f} (budget 5,000)",
             _badge(e["worst_flat_lod0_triangles"] <= e["flat_budget"])],
            ["interior chunks (per-storey glb)", f"{e.get('interior_chunks', 0):,}", _badge(e.get("chunks_ok", False))],
            ["manifest", "model/estate_manifest.json", _badge(e["manifest"] is not None, "written")],
        ], ["measure", "value", "check"]))
    eb = data.get("estate_blend")
    if eb:
        parts.append(_h2("estate_blend") + _table([
            ["linked buildings loaded", f"{eb['links_loaded']}/{eb['links']}",
             _badge(eb["links_loaded"] == eb["links"] and eb["links"] > 0)],
            ["link time", f"{eb['links_seconds']} s (budget 60 s)",
             _badge((eb["links_seconds"] or 0) <= eb["links_budget_s"])],
            ["peak memory", f"{eb['peak_working_set_mb']} MB (budget 8 GB)",
             _badge((eb["peak_working_set_mb"] or 0) <= eb["memory_budget_mb"])],
            ["verified after reopen", "yes" if eb["verified"] else "no", _badge(eb["verified"])],
        ], ["measure", "value", "check"]))
    if data.get("layout_variety"):
        parts.append(_h2("layout_variety") + "<p>Distinct stack layouts (mirror images counted once) per flat type; the "
                     "plan asks for at least 3 per type (2 for 2-room Flexi, 3Gen and Executive), 40 raw and 20 "
                     "canonical overall.</p>"
                     + _table([[t, n] for t, n in data["layout_variety"].items()], ["type", "layouts"]))
    sp = data.get("site_plan")
    if sp:
        svg = sp.get("svg")
        link = (f'<p class="muted">Vector drawing: <a href="{_rel(svg, base)}">{PurePosixPath(svg).name}</a></p>'
                if svg else "")
        parts.append(_h2("site_plan") + _img(sp, base, 1200) + link)
    if data["typical_plans"]:
        parts.append(_h2("typical_plans") + "<div class='grid'>" + "".join(_img(p, base) for p in data["typical_plans"])
                     + "</div>")
    if data["renders"]:
        parts.append(_h2("renders") + "<div class='grid'>" + "".join(_img(p, base) for p in data["renders"]) + "</div>")
    if data["walk_maps"]:
        parts.append(_h2("walk_maps") + "<div class='grid'>" + "".join(_img(p, base) for p in data["walk_maps"])
                     + "</div>")
    if data["catalogue"]:
        rows = [[c["template"], c["type"], html.escape(c["title"]), c["cases"], c["layouts"],
                 _badge(c["errors"] == 0, f"{c['errors']} errors")] for c in data["catalogue"]]
        parts.append(_h2("catalogue") + _table(rows, ["template", "type", "title", "cases", "layouts", "checks"]))
        parts.append("<div class='grid'>" + "".join(_img(c["gallery"], base) for c in data["catalogue"] if c["gallery"])
                     + "</div>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>Sample Town N5 report</title><style>{CSS}</style>"
            f"</head><body>{''.join(parts)}</body></html>")


# ----------------------------------------------------------------------------- write
def write_report(out=None) -> Path:
    """report.html (``out``, default reports/report.html) with report.json and issues.bcf next to it; returns the
    page's path. A BCF that cannot be written is reported on the page instead of failing the report."""
    out = Path(out or env.REPORTS / "report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = collect()
    bcf = (data.get("validation") or {}).get("bcf")
    try:
        n = bcf_out.write_bcf(out.with_name(BCF_NAME), project_name=data["report"]["estate"])
    except Exception as e:  # noqa: BLE001 - the page and its JSON are still worth writing
        if bcf is not None:
            bcf["error"] = f"{type(e).__name__}: {e}"
    else:
        if bcf is not None:
            bcf["topics"] = n
    out.write_text(render(data, out.parent), encoding="utf-8", newline="\n")
    out.with_name(JSON_NAME).write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str) + "\n",
                                        encoding="utf-8", newline="\n")
    return out


def outputs_line(page) -> str:
    """One line naming what write_report wrote next to ``page`` (for the CLI): the page, its JSON and the BCF."""
    page = Path(page)
    data = _load(page.with_name(JSON_NAME)) or {}
    b = (data.get("validation") or {}).get("bcf") or {}
    bcf = (f"{env.rel(page.with_name(BCF_NAME))} ({b.get('topics', 0)} topics)" if not b.get("error")
           else f"no BCF ({b['error']})")
    return f"{env.rel(page)}, {env.rel(page.with_name(JSON_NAME))}, {bcf}"
