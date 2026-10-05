"""Estate manifest for game engines: one JSON that places every building glb in the estate (model/estate_manifest.json).

The glbs and engine JSONs are block-local (see engine.py); this file carries what an importer needs to assemble the
neighbourhood: the CRS / georeference, every site with its block -> estate transform, bounds, real lift lobbies and
entrances (the masterplan resolved with block planning) and the relative paths (+ sha256) of its IFC, IFC4 copy,
.blend, LOD glbs, per-storey interior chunks (``glb_int``, bottom up) and engine JSON, the roads, the pedestrian
graph written by the site work (model/SITE_graph.json, embedded when present), bus stops and spawn points. All
coordinates are estate metres (origin at the south-west corner, +x east, +y north, Z up).

Spawn points stand on the pedestrian network: ``entrance_spawns`` sets each one out from a masterplan entrance
along the outdoor path that leaves it for the network (footpaths and linkways, not the indoor or void-deck routes
through the building) and keeps it AGENT_RADIUS clear of every vehicle surface in SITE.ifc (IfcPavement:
carriageways, junctions, driveways, crossovers, zebra crossings), pulling it back towards the entrance, or onto
the void deck behind it, where a driveway runs up to the door. The engine export (cli.export_one) writes them into
each building's engine JSON, and the manifest copies them from there. ``bus_stop_spawns`` puts one on every bus
stop's platform node (under the shelter), AGENT_RADIUS clear of the carriageway.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, shape
from shapely.ops import nearest_points

from estate import config, env
from estate.rules import AGENT_RADIUS

SCHEMA = "sample-town-n5/estate-manifest/1"
PEDESTRIAN = ("footpath", "linkway", "sidewalk", "park")    # graph edge kinds a spawn may stand on (not 'crossing')
THROUGH_BUILDING = ("indoor", "void_deck")                  # graph edge 'via': routes inside a building, not away
ENTRANCE_NODES = ("entrance", "nc", "mscp")
SPAWN_OFFSET = 1.5           # m from the entrance, along the path
SNAP = 6.0                   # an entrance the graph does not know snaps onto a path at most this far away
MATCH = 0.75                 # an entrance matches an entrance node of its site at most this far away
MATCH_PLANNED = 1.0          # ... or the graph meta entrance planned at most this far away


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file(path: Path, base: Path) -> dict:
    rel = Path(os.path.relpath(path, base)).as_posix()
    if path.exists():
        return {"path": rel, "sha256": _sha(path), "bytes": path.stat().st_size}
    return {"path": rel, "exists": False}


def _mat(M, nd=9):
    return [[round(float(x), nd) + 0.0 for x in row] for row in np.asarray(M)]


def _xf2(M, pts, z=0.0):
    return [[round(float(v), 3) for v in (M @ np.array([p[0], p[1], p[2] if len(p) > 2 else z, 1.0]))[:3]] for p in pts]


def _rings(poly):
    parts = [[[round(x, 3), round(y, 3)] for x, y in g.exterior.coords] for g in getattr(poly, "geoms", [poly])
             if g.geom_type == "Polygon" and not g.is_empty]
    return parts[0] if len(parts) == 1 else parts


def site_files(folder: Path, stem: str, base: Path, lods=(0, 1, 2), interiors=True,
               engine: dict | None = None) -> dict:
    """Files of one site; the external works (SITE) have no LOD2 (engine.py: no massing for a site-only file) and
    no interior chunks (``interiors`` False). ``glb_int`` lists a building's per-storey interior chunks
    [{storey, elevation, path, sha256, bytes}] bottom up, as its engine JSON names them (``engine``, the parsed
    <stem>_engine.json, read here when not given): the JSON is written with the chunks, so a chunk file left from
    an older export is never listed. Empty without an engine JSON. A building also lists the web stage's walk grid
    and web JSON (``walk``, ``web``: estate/web); the site has neither."""
    out = {"ifc": _file(folder / f"{stem}.ifc", base), "ifc4": _file(folder / f"{stem}_ifc4.ifc", base),
           "blend": _file(folder / f"{stem}.blend", base)}
    out.update({f"glb_lod{k}": _file(folder / f"{stem}_lod{k}.glb", base) for k in lods})
    out["engine"] = _file(folder / f"{stem}_engine.json", base)
    if interiors:
        eng = engine if engine is not None else _load(folder / f"{stem}_engine.json")
        chunks = sorted((eng or {}).get("interior_chunks", {}).items(), key=lambda kv: kv[1]["elevation"])
        out["glb_int"] = [dict(storey=s, elevation=c["elevation"], **_file(folder / c["file"], base))
                          for s, c in chunks]
        out["walk"] = _file(folder / f"{stem}_walk.bin", base)
        out["web"] = _file(folder / f"{stem}_web.json", base)
    return out


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _compass(v) -> str:
    return ("E" if v[0] > 0 else "W") if abs(v[0]) >= abs(v[1]) else ("N" if v[1] > 0 else "S")


_PAVING = {}


def vehicle_paving(model_dir=None):
    """Plan polygons of every vehicle surface in SITE.ifc (all its IfcPavement), cached per file version; None
    without a SITE.ifc."""
    path = (Path(model_dir) if model_dir else env.MODEL) / "SITE.ifc"
    if not path.exists():
        return None
    key = (str(path.resolve()), path.stat().st_mtime_ns)
    if key not in _PAVING:
        import ifcopenshell
        import shapely
        from shapely.geometry import Polygon
        from estate.export import glb, meshcache
        f = ifcopenshell.open(str(path))
        pav = f.by_type("IfcPavement")
        polys = []
        for d in (meshcache.tessellate(f, include=pav) if pav else {}).values():
            tris = d["verts"][d["faces"]]
            n, ln = glb.face_normals(tris)
            polys += [q for q in (Polygon(t[:, :2]) for t in tris[(n[:, 2] / (ln + 1e-12)) > 0.9]) if q.area > 1e-6]
        _PAVING.clear()
        _PAVING[key] = shapely.union_all(polys, grid_size=1e-3) if polys else None
    return _PAVING[key]


def _on_path(pts, t):
    """The point at arc length t along the polyline pts; t < 0 extends its first segment backwards."""
    pts = [np.asarray(p, float) for p in pts]
    if t <= 0 or len(pts) < 2:
        d = pts[1] - pts[0] if len(pts) > 1 else np.zeros(2)
        n = float(np.hypot(*d))
        return pts[0] + (d / n * t if n > 1e-9 else 0.0)
    for a, b in zip(pts[:-1], pts[1:]):
        n = float(np.hypot(*(b - a)))
        if t <= n:
            return a + (b - a) * (t / n if n > 1e-9 else 0.0)
        t -= n
    return pts[-1]


def _clear_spot(pts, offset, footprint, keep_out):
    """Where to stand on the path pts (a polyline from the entrance; a ray is two points): arc length ``offset`` if
    that is clear of keep_out, else the largest clear one nearer the entrance, else a negative one on the deck
    behind it (inside the footprint). Returns (t, point), or None if no spot is clear."""
    for t in [offset, *np.arange(offset - 0.25, 0.24, -0.25), -0.5, -1.0, -1.5]:
        p = _on_path(pts, t)
        if (keep_out is None or not keep_out.contains(Point(*p))) and (t >= 0 or footprint.contains(Point(*p))):
            return float(t), p
    return None


def _walkable(edge, node, site_id, outdoor):
    """A pedestrian edge towards ``node`` that a spawn may follow away from site_id: not into the site's own
    nodes (lift lobbies, its other entrances) and, when ``outdoor``, not along a route through a building."""
    return edge["kind"] in PEDESTRIAN and node.get("ref") != site_id and \
        not (outdoor and edge.get("via") in THROUGH_BUILDING)


def _walk(start, first, nodes, adj, site_id, length, outdoor):
    """The path from node ``start`` over the edge to ``first`` and on along the network until it is ``length`` m
    long: at each node it takes the walkable edge that turns least, never back to a node it has passed (graph
    nodes split paths into short pieces, some only centimetres long). Returns the polyline (estate xy)."""
    pts, seen, nxt = [np.array(nodes[start]["xy"], float)], {start}, first
    total = 0.0
    while nxt is not None:
        q = np.array(nodes[nxt]["xy"], float)
        seg = float(np.hypot(*(q - pts[-1])))
        if seg > 1e-9:
            d = (q - pts[-1]) / seg
            pts.append(q)
            total += seg
        seen.add(nxt)
        if total >= length:
            break
        best = None
        for b, edge in adj.get(nxt, []):
            if b in seen or not _walkable(edge, nodes[b], site_id, outdoor):
                continue
            r = np.array(nodes[b]["xy"], float) - q
            n = float(np.hypot(*r))
            turn = 1.0 if n < 1e-9 or len(pts) < 2 else float(r @ d) / n
            if best is None or turn > best[0]:
                best = (turn, b)
        nxt = best[1] if best else None
    return pts


def _entrance_node(site_id, e, nodes, graph):
    """The graph node of a planned entrance: the one the site work recorded for it (graph meta entrances: planned
    point -> node, which may sit a little off the planned point), else the nearest entrance node of the site
    within MATCH m. None when the graph does not know the entrance."""
    meta = [m for m in (graph or {}).get("meta", {}).get("entrances", []) if m.get("site") == site_id]
    cands = [n for n in nodes.values() if n["kind"] in ENTRANCE_NODES and n.get("ref") in (None, site_id)]

    def dist(p, q):
        return float(np.hypot(*(np.asarray(p, float) - np.asarray(q, float))))
    if meta:
        m = min(meta, key=lambda m: dist(m["planned"], e))
        if dist(m["planned"], e) <= MATCH_PLANNED:
            n = min(cands, key=lambda n: dist(n["xy"], m["node"]), default=None)
            if n is not None and dist(n["xy"], m["node"]) <= 1e-3:
                return n
    n = min(cands, key=lambda n: dist(n["xy"], e), default=None)
    return n if n is not None and dist(n["xy"], e) <= MATCH else None


def entrance_spawns(site_id: str, kind: str, entrances, footprint, graph: dict | None,
                    offset: float = SPAWN_OFFSET, paving=None) -> list[dict]:
    """Spawn points (estate frame) at a site's pedestrian entrances. Each entrance is matched to its node in the
    pedestrian graph (_entrance_node); the spawn sits ``offset`` m out along the path that leaves it for the
    network (_walk: outdoor edges first, so not into a hawker hall or along a void deck; of several, the one that
    gets farthest, then farthest from the footprint), facing the entrance. An entrance the graph does not know
    steps ``offset`` m out of the footprint along the nearest axis and snaps onto a pedestrian edge within SNAP m.
    With ``paving`` (vehicle surfaces), a spot within AGENT_RADIUS of it moves towards the entrance (_clear_spot).
    Named 'Void deck entrance <compass>' for residential blocks, 'Entrance <compass>' otherwise."""
    keep_out = paving.buffer(AGENT_RADIUS) if paving is not None else None
    nodes = {n["id"]: n for n in (graph or {}).get("nodes", [])}
    adj, paths = {}, []
    for e in (graph or {}).get("edges", []):
        if e["a"] not in nodes or e["b"] not in nodes:
            continue
        adj.setdefault(e["a"], []).append((e["b"], e))
        adj.setdefault(e["b"], []).append((e["a"], e))
        if e["kind"] in PEDESTRIAN:
            paths.append(LineString([nodes[e["a"]]["xy"], nodes[e["b"]]["xy"]]))
    label = "Void deck entrance" if kind == "block" else "Entrance"
    out, seen = [], {}
    for ent in entrances:
        e = np.array(ent[:2], float)
        pos, source = None, "graph"
        node = _entrance_node(site_id, e, nodes, graph)
        if node is not None:
            a = np.array(node["xy"], float)
            best = None
            for outdoor in (True, False):        # an entrance reached only through the building: any pedestrian edge
                for b, edge in adj.get(node["id"], []):
                    if not _walkable(edge, nodes[b], site_id, outdoor):
                        continue
                    pts = _walk(node["id"], b, nodes, adj, site_id, offset, outdoor)
                    if len(pts) < 2:
                        continue
                    length = sum(float(np.hypot(*(q - p))) for p, q in zip(pts[:-1], pts[1:]))
                    spot = _clear_spot(pts, min(offset, length), footprint, keep_out)
                    if spot is None:
                        continue
                    score = (spot[0], footprint.distance(Point(*spot[1])))
                    if best is None or score > best[0]:
                        best = (score, spot[1])
                if best is not None:
                    break
            if best is not None:
                pos, e = best[1], a
        if pos is None:
            source = "footprint"
            axes = [np.array(v, float) for v in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            out_axes = [d for d in axes
                        if footprint.contains(Point(*(e - d))) and not footprint.contains(Point(*(e + d)))]
            c = np.array(footprint.centroid.coords[0])
            d = out_axes[0] if out_axes else max(axes, key=lambda v: float(v @ (e - c)))
            pos = e + d * offset
            if paths:
                pt = Point(*pos)
                line = min(paths, key=lambda ln: ln.distance(pt))
                if 1e-6 < line.distance(pt) <= SNAP:
                    q = line.interpolate(line.project(pt))
                    pos, source = np.array([q.x, q.y]), "snapped"
            if keep_out is not None and keep_out.contains(Point(*pos)):
                n = float(np.hypot(*(pos - e)))
                first = (pos - e) / n if n > 1e-9 else d
                spots = [s for u in [first, *axes]
                         for s in [_clear_spot([e, e + u * max(offset, 1e-3)], offset, footprint, keep_out)]
                         if s is not None]
                if spots:
                    pos = max(spots, key=lambda s: s[0])[1]
                else:                               # boxed in by vehicle paving: stand at the entrance itself
                    pos, source = e.copy(), "entrance"
        f = e - pos if np.hypot(*(e - pos)) > 1e-9 else np.array(footprint.centroid.coords[0]) - e
        f = f / np.linalg.norm(f) if np.linalg.norm(f) > 1e-9 else np.zeros(2)
        name = f"{label} {_compass(-f)}"
        seen[name] = seen.get(name, 0) + 1
        out.append(dict(name=name if seen[name] == 1 else f"{name} {seen[name]}",
                        position=[round(float(v), 3) for v in pos], facing=[round(float(v), 3) + 0.0 for v in f],
                        entrance=[round(float(v), 3) for v in e], source=source))
    return out


def site_spawns(site_id: str, model_dir=None) -> tuple[str | None, list[dict] | None]:
    """(site kind, entrance spawns) of one estate site from model/masterplan.json and model/SITE_graph.json;
    (None, None) for an IFC outside the masterplan, (kind, None) when the site lists no entrances."""
    model_dir = Path(model_dir) if model_dir else env.MODEL
    mp = _load(model_dir / "masterplan.json")
    site = next((s for s in (mp or {}).get("sites", []) if s.get("id") == site_id), None)
    if site is None:
        return None, None
    if not site.get("entrances"):
        return site.get("kind"), None
    return site.get("kind"), entrance_spawns(site_id, site.get("kind"), site["entrances"], shape(site["footprint"]),
                                             _load(model_dir / "SITE_graph.json"), paving=vehicle_paving(model_dir))


def bus_stop_spawns(bus_stops, graph: dict | None = None, paving=None, roads: dict | None = None) -> list[dict]:
    """Spawn points (estate frame) at the bus stops ({id, road, position} as in the manifest's bus_stops). A
    masterplan point may mark the stop on the kerb line, where a spawn would stand half over the carriageway; the
    spawn takes the stop's node in the pedestrian graph instead (graph meta bus_stops, else the bus_stop node of
    that ref: the platform under the shelter). With ``paving`` (vehicle surfaces), a spot still within AGENT_RADIUS
    of it steps to the nearest clear point (0.25 m steps along the axes, at most 4 m). It faces its road
    (``roads``: {road id: centreline}), else the nearest vehicle paving."""
    keep_out = paving.buffer(AGENT_RADIUS) if paving is not None else None
    meta = {b["id"]: b for b in (graph or {}).get("meta", {}).get("bus_stops", [])}
    by_ref = {n.get("ref"): n for n in (graph or {}).get("nodes", []) if n.get("kind") == "bus_stop"}
    out = []
    for b in bus_stops:
        at = np.array(b["position"][:2], float)
        node = by_ref.get(b["id"])
        xy = meta.get(b["id"], {}).get("node") or (node or {}).get("xy")
        pos, source = (np.array(xy, float), "graph") if xy is not None else (at, "masterplan")
        if keep_out is not None and keep_out.contains(Point(*pos)):
            steps = [pos + np.array(u) * t for t in np.arange(0.25, 4.01, 0.25)
                     for u in ((0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0))]
            clear = next((p for p in steps if not keep_out.contains(Point(*p))), None)
            if clear is not None:
                pos, source = clear, source + "+clear"
        road = (roads or {}).get(b.get("road"))
        target = LineString(road) if road is not None and len(road) > 1 else paving
        if target is not None and not target.is_empty:
            q = nearest_points(target, Point(*pos))[0]
            f = np.array([q.x, q.y]) - pos
        else:
            f = at - pos
        n = float(np.hypot(*f))
        f = f / n if n > 1e-9 else np.zeros(2)
        z = float((node or {}).get("z", 0.0)) if source.startswith("graph") else 0.0
        out.append(dict(name=f"Bus stop {b['id']}", position=[round(float(pos[0]), 3), round(float(pos[1]), 3), z],
                        facing=[round(float(f[0]), 3) + 0.0, round(float(f[1]), 3) + 0.0, 0.0], stop=b["id"],
                        source=source))
    return out


def crs(cfg: dict) -> dict:
    g = cfg.get("georef", {})
    north = float(g.get("true_north_deg", 0.0))
    return {"epsg": int(g.get("epsg", 3414)), "name": f"EPSG:{int(g.get('epsg', 3414))}",
            "map_conversion": {"eastings": float(g.get("eastings", 0.0)), "northings": float(g.get("northings", 0.0)),
                               "orthogonal_height": float(g.get("height", 0.0)),
                               "x_axis_abscissa": round(float(np.cos(np.radians(north))), 12),
                               "x_axis_ordinate": round(float(np.sin(np.radians(north))), 12), "scale": 1.0},
            "note": "IfcMapConversion shared by every federated IFC: E = eastings + x*a - y*o, N = northings + x*o + "
                    "y*a, H = orthogonal_height + z (a, o = x_axis_abscissa / ordinate). Eastings / northings are "
                    "SVY21 placeholders."}


def write_manifest(mp: dict | None = None, out=None, model_dir=None) -> Path:
    """Write the estate manifest (default model/estate_manifest.json) from a resolved masterplan
    (estate.masterplan.resolve, with block planning when omitted, so lift lobbies and entrances are the planned
    ones; a site still approximate is flagged and warned about). Returns the path."""
    if mp is None:
        from estate import masterplan
        mp = masterplan.resolve()     # plans=True: real lift lobbies and entrances, not placeholders
    cfg = mp["cfg"]
    model_dir = Path(model_dir) if model_dir else env.MODEL
    out = Path(out) if out else model_dir / "estate_manifest.json"
    base = out.parent
    graph_path = model_dir / "SITE_graph.json"
    graph = _load(graph_path)
    warnings, sites, spawns = [], [], []
    for s in mp["sites"]:
        if s.approximate:
            warnings.append(f"{s.id}: not planned; approximate footprint, lift lobbies and entrances")
        folder = model_dir / s.id
        eng = _load(folder / f"{s.id}_engine.json")
        files = site_files(folder, s.id, base, engine=eng)
        M = config.placement_matrix(s.at, s.rot)
        src = "config"
        fp = _rings(s.footprint)
        b = s.footprint.bounds
        bounds = [[round(b[0], 3), round(b[1], 3), 0.0], [round(b[2], 3), round(b[3], 3), round(float(s.height), 3)]]
        here, extra = [], {}
        if eng is not None:
            T = np.array(eng["transform"], float)
            if not np.allclose(T, M, atol=1e-6):          # the glbs are local to the IfcBuilding placement: trust it
                warnings.append(f"{s.id}: engine JSON transform differs from config placement (using the engine JSON)")
                extra["config_transform"] = _mat(M)
            M, src = T, "engine_json"
            bounds = eng.get("bounds_estate", bounds)
            if eng.get("massing", {}).get("footprint"):
                rings = [[p[:2] for p in _xf2(M, ring)] for ring in eng["massing"]["footprint"]]
                fp = rings[0] if len(rings) == 1 else rings
            for sp in eng.get("spawns", []):
                facing = M[:3, :3] @ np.array(sp.get("facing", [0.0, 0.0, 0.0]))
                here.append(dict(site=s.id, name=sp["name"], position=_xf2(M, [sp["position"]])[0],
                                 facing=[round(float(v), 3) + 0.0 for v in facing],
                                 **({"room": sp["room"]} if "room" in sp else {})))
        elif s.entrances:
            here = [dict(site=s.id, name=sp["name"], position=[*sp["position"], 0.0], facing=[*sp["facing"], 0.0],
                                approximate=bool(s.approximate))
                           for sp in entrance_spawns(s.id, s.kind, s.entrances, s.footprint, graph,
                                                     paving=vehicle_paving(model_dir))]
        spawns += here
        sites.append(dict(id=s.id, kind=s.kind, blk=s.blk, name=s.name, typology=s.typology, storeys=s.storeys,
                          height=s.height, approximate=bool(s.approximate), at=list(s.at), rot=s.rot,
                          transform=_mat(M), transform_source=src, **extra, bounds=bounds, footprint=fp,
                          lift_lobbies=[list(p) for p in s.lift_lobbies], entrances=[list(p) for p in s.entrances],
                          files=files))
    estate_files = dict(site_files(model_dir, "SITE", base, lods=(0, 1), interiors=False),
                        estate_blend=_file(model_dir / "ESTATE.blend", base),
                        masterplan=_file(model_dir / "masterplan.json", base))
    site_eng = _load(model_dir / "SITE_engine.json")
    roads = [dict(id=r["id"], name=r.get("name"), road_class=r.get("class"),
                  centreline=[[float(x), float(y)] for x, y in r["centreline"]],
                  reserve_width=float(r["reserve"]), carriageway_width=float(r["carriageway"])) for r in mp["roads"]]
    bus = [dict(id=b["id"], road=b.get("road"), position=[float(b["at"][0]), float(b["at"][1]), 0.0])
           for b in mp["bus_stops"]]
    spawns += [dict(site="SITE", **sp) for sp in bus_stop_spawns(bus, graph, vehicle_paving(model_dir),
                                                                  {r["id"]: r["centreline"] for r in roads})]
    e = cfg["estate"]
    man = {
        "schema": SCHEMA,
        "estate": {k: e[k] for k in ("name", "code", "seed", "extent") if k in e},
        "frame": "Estate frame: metres, origin at the south-west corner, +x east, +y north, Z up. Each site's glbs and "
                 "engine JSON are block-local; estate = transform @ local (row-major 4x4). glTF is Y up: convert a glb "
                 "point (gx, gy, gz) to block-local (gx, -gz, gy) before applying the transform.",
        "crs": crs(cfg),
        "agent": cfg.get("agent", {}),
        "sites": sites,
        "site": {"id": "SITE", "files": estate_files, "tiles": (site_eng or {}).get("tiles", [])},
        "roads": roads,
        "park_connector": cfg.get("park_connector"),
        "greens": [dict(id=g["id"], name=g.get("name"), rect=g.get("rect"), items=g.get("items", [])) for g in mp["greens"]],
        "bus_stops": bus,
        "pedestrian_graph": graph,
        "pedestrian_graph_file": _file(graph_path, base),
        "spawn_points": spawns,
        "warnings": warnings,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    tmp.replace(out)
    return out
