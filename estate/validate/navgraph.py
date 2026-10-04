"""Estate navigation graph: can every flat be reached from every bus stop, with and without stairs?

It joins the two walk tests on one networkx graph:
  bus stop -> lift lobby   site routes from nav2d (walk length, covered share, step-free)
  lobby    -> street ring  the void deck of the building's nav3d run; step-free when the lift landings on L1
                           are step-free from the street
  street   -> flat         nav3d per flat: every room reachable (stairs allowed), and the main door
                           crossed without steps by lift (step-free: the street component continues into
                           the entry room, so a threshold step or a blocked door breaks it)
A flat counts as reachable from a bus stop when a path exists in the full graph. It counts as step-free when
a path exists using step-free edges only. Buildings without a nav3d result are listed as missing and not
counted as passes.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from estate import env


def _load(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    return json.loads(Path(x).read_text(encoding="utf-8"))


def _building_result(nav: dict, radius: str | None = None):
    res = nav.get("results", {})
    if not res:
        return None
    key = radius if radius in res else next(iter(res))
    return key, res[key]


def estate_report(mp: dict, building_nav_jsons, site_result, out_path=None, radius: str | None = None) -> dict:
    """Per-flat reachability from every bus stop. building_nav_jsons: paths or dicts of nav3d reports (any
    order; matched to masterplan sites by file stem, e.g. BLK_501). site_result: nav2d result or its JSON path.
    Writes reports/nav_estate.json (or out_path) and returns the report."""
    site = _load(site_result) or {}
    navs = {}
    for x in building_nav_jsons or []:
        d = _load(x)
        if d:
            navs[d.get("stem") or Path(d.get("file", "")).stem] = d
    sites = {s.id: s for s in mp["sites"]}
    bus_ids = [b["id"] for b in mp["bus_stops"]]

    edges = []                     # (a, b, attrs): stairs = usable at all, step_free = usable without steps
    for r in site.get("routes", []):
        if r.get("reachable"):
            edges.append((("bus", r["bus_stop"]), ("lobby", r["target"], r.get("lobby", 0)),
                          dict(length=r["length_m"], stairs=True, step_free=bool(r.get("step_free")))))

    buildings, flats, missing = [], [], []
    for sid, s in sorted(sites.items()):
        nav = navs.get(sid)
        n_lobbies = max(1, len(s.lift_lobbies))
        if nav is None:
            if s.kind == "block":
                missing.append(sid)
            buildings.append(dict(id=sid, kind=s.kind, nav=False))
            continue
        key, res = _building_result(nav, radius)
        lifts = res.get("lifts", [])
        sf_street = any(lf.get("step_free_from_street") for lf in lifts) if lifts else bool(
            res["summary"].get("start_cells"))
        for li in range(n_lobbies):
            edges.append((("lobby", sid, li), ("street", sid), dict(length=0.0, stairs=True, step_free=sf_street)))
        buildings.append(dict(id=sid, kind=s.kind, nav=True, radius=key, ok=res["summary"].get("ok"),
                              step_free_ok=res["summary"].get("step_free_ok"), flats=len(res.get("flats", [])),
                              lifts_step_free_from_street=sf_street,
                              doors_impassable=res["summary"].get("doors_impassable"),
                              main_doors_checked_in=res["summary"].get("main_doors_checked_in")))
        for fl in res.get("flats", []):
            node = ("flat", sid, fl["unit"])
            edges.append((("street", sid), node, dict(length=0.0, stairs=bool(fl["reachable"]),
                                                       step_free=bool(fl.get("main_door_step_free")))))
            flats.append(dict(building=sid, unit=fl["unit"], type=fl.get("type", ""), storey=fl.get("storey", ""),
                              main_door_checked_in=fl.get("main_door_checked_in"), node=node))

    G, Gsf = nx.Graph(), nx.Graph()
    for gr in (G, Gsf):
        gr.add_nodes_from([("bus", b) for b in bus_ids] + [fl["node"] for fl in flats])
    G.add_edges_from((a, b, d) for a, b, d in edges if d["stairs"])
    Gsf.add_edges_from((a, b, d) for a, b, d in edges if d["step_free"])
    per_bus = {}
    for b in bus_ids:
        src = ("bus", b)
        per_bus[b] = (nx.single_source_dijkstra_path_length(G, src, weight="length"),
                      nx.single_source_dijkstra_path_length(Gsf, src, weight="length"))
    out_flats = []
    for fl in flats:
        node = fl.pop("node")
        frm = {}
        for b in bus_ids:
            d_all, d_sf = per_bus[b]
            frm[b] = dict(reachable=node in d_all, step_free=node in d_sf,
                          walk_m=round(d_all[node], 1) if node in d_all else None)
        fl["from"] = frm
        fl["reachable_from_all"] = all(v["reachable"] for v in frm.values())
        fl["step_free_from_all"] = all(v["step_free"] for v in frm.values())
        out_flats.append(fl)

    n = len(out_flats)
    walks = [v["walk_m"] for f in out_flats for v in f["from"].values() if v["walk_m"] is not None]
    expected = sum(len(s.plan.typical.flats) * (s.storeys - 1) for s in sites.values()
                   if s.kind == "block" and getattr(s, "plan", None) is not None)
    summary = dict(
        bus_stops=len(bus_ids), buildings_with_nav=sum(b["nav"] for b in buildings), missing_buildings=missing,
        flats=n, flats_expected_from_plans=expected,
        reachable_from_every_bus_stop=sum(f["reachable_from_all"] for f in out_flats),
        step_free_from_every_bus_stop=sum(f["step_free_from_all"] for f in out_flats),
        max_walk_to_lobby_m=max(walks) if walks else None, site_mode=site.get("mode"),
        site_ok=site.get("summary", {}).get("ok"),
        site_graph_matches_ifc=(site.get("graph_checks") or {}).get("ok"),
        main_door_threshold_not_checked=sum(f.get("main_door_checked_in") == "near door" for f in out_flats))
    summary["reachable_share"] = round(summary["reachable_from_every_bus_stop"] / n, 4) if n else 0.0
    summary["step_free_share"] = round(summary["step_free_from_every_bus_stop"] / n, 4) if n else 0.0
    summary["ok"] = bool(n) and not missing and summary["reachable_share"] == 1.0 and summary["step_free_share"] == 1.0
    report = dict(summary=summary, buildings=buildings, assumptions=site.get("assumptions", []), flats=out_flats)
    out_path = Path(out_path) if out_path else env.REPORTS / "nav_estate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return report


def summary_text(report: dict) -> str:
    s = report["summary"]
    lines = [f"estate navigation: {s['flats']} flats with a nav result in {s['buildings_with_nav']} buildings "
             f"({s['flats_expected_from_plans']} expected from plans), {s['bus_stops']} bus stops",
             f"  reachable from every bus stop: {s['reachable_from_every_bus_stop']}/{s['flats']} "
             f"({s['reachable_share'] * 100:.1f}%)",
             f"  step-free from every bus stop: {s['step_free_from_every_bus_stop']}/{s['flats']} "
             f"({s['step_free_share'] * 100:.1f}%)",
             f"  longest walk bus stop -> lift lobby: {s['max_walk_to_lobby_m']} m (site mode {s['site_mode']})"]
    if s["missing_buildings"]:
        lines.append(f"  no nav3d result yet: {', '.join(s['missing_buildings'])}")
    if s.get("main_door_threshold_not_checked"):
        lines.append(f"  main door crossing not checked (no entry room found): {s['main_door_threshold_not_checked']}")
    if s.get("site_graph_matches_ifc") is False:
        lines.append("  the pedestrian graph does not match SITE.ifc (see nav_site.json graph_checks)")
    lines.append(f"  {'OK' if s['ok'] else 'NOT OK'}")
    return "\n".join(lines)
