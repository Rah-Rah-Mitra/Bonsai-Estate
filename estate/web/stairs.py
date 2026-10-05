"""Stairs of a building for the web export: flights, landings and a walking path, block-local, Z up, metres.

Every IfcStair climbs one storey (estate/ifc/stairs.py: a dog-leg of two IfcStairFlights with a mid landing and a
floor landing on the next storey, both IfcSlab LANDING parts of the stair). A flight's riser count, riser height
and going are its IFC attributes (NumberOfRisers, RiserHeight, TreadLength); where it runs comes from its solid:
the treads are its up-facing faces, the run direction goes from the lowest tread to the highest, and the flight's
centre line is the middle of the narrowest tread. ``start`` is the foot of the first riser, ``end`` the top of the
last one, both on the centre line.

``path`` is a walking line from the centre of the floor landing the first flight leaves (the floor landing of the
stair one storey down, else this stair's own floor landing set down a storey: stairs are stacked), straight onto
each flight (APPROACH before its first riser), up it tread by tread, straight off it, through the centre of the
landing it arrives on (the mid landing, then the next floor's landing): consecutive points never rise more than
one riser, and the last one stands on the next floor. Once the walk grid is encoded, ``fit_paths`` moves a landing
point that the decoded grid has no floor under, or whose line to its neighbours crosses a cell without one (even at
a corner, for a few mm: estate/web/walkcheck.py crossed), onto the nearest cell of its landing from which both lines
are clear, so a viewer can walk every path on the grid it is exported with.
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient

import ifcopenshell.util.element as uel

from estate.web import walkcheck

R = 3                       # decimals of every coordinate written
APPROACH = 0.3              # m of landing walked straight before a flight's first riser and after its last
CLEARANCE = 0.01            # m a moved landing point's lines keep from every cell without a floor where its landing
                            # allows (else walkcheck.EDGE, the check's own margin): not 3 mm past a blocked corner


def _r(v):
    return [round(float(x), R) + 0.0 for x in v]


def _ring(poly) -> list:
    return [_r(c) for c in orient(poly, 1.0).exterior.coords]


def _tris(meshes, guid, inv):
    d = meshes[guid]
    T = d["verts"][d["faces"]]
    return T @ inv[:3, :3].T + inv[:3, 3]


def _up(T, limit=0.99):
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(n, axis=1)
    return (n[:, 2] / np.maximum(ln, 1e-12)) > limit


def _clusters(z, gap=0.005):
    order = np.argsort(z, kind="stable")
    cut = np.flatnonzero(np.diff(z[order]) > gap) + 1
    return np.split(order, cut)


def flight(el, T) -> dict | None:
    """Run of one IfcStairFlight from its triangles T (block-local): start, end, width, the tread points and the
    IFC riser figures; None when it has no treads."""
    up = T[_up(T)]
    if not len(up):
        return None
    zc = up[:, :, 2].mean(1)
    treads = sorted(((float(zc[g].mean()), up[g][:, :, :2].reshape(-1, 2)) for g in _clusters(zc)),
                    key=lambda t: t[0])
    c = [(xy.min(0) + xy.max(0)) / 2 for _, xy in treads]
    d = c[-1] - c[0] if len(treads) > 1 else np.zeros(2)
    if np.linalg.norm(d) < 1e-9:              # one tread: towards the highest vertices
        V = T.reshape(-1, 3)
        d = V[V[:, 2] > V[:, 2].max() - 0.01, :2].mean(0) - V[V[:, 2] < V[:, 2].min() + 0.01, :2].mean(0)
    u = d / max(float(np.linalg.norm(d)), 1e-12)
    v = np.array([-u[1], u[0]])
    s0 = np.array([(xy @ u).min() for _, xy in treads])
    s1 = np.array([(xy @ u).max() for _, xy in treads])
    l0 = np.array([(xy @ v).min() for _, xy in treads])
    l1 = np.array([(xy @ v).max() for _, xy in treads])
    lmid = (l0.max() + l1.min()) / 2
    zt = [t[0] for t in treads]
    risers = int(el.NumberOfRisers or (len(treads) + 1))
    riser = float(el.RiserHeight or (np.median(np.diff(zt)) if len(zt) > 1 else 0.175))
    going = float(el.TreadLength or float(np.median(s1 - s0)))
    rise = (zt[-1] - zt[0]) / (len(zt) - 1) if len(zt) > 1 else riser    # the solid's riser: RiserHeight is rounded
    z0 = zt[0] - rise                         # the floor at the foot of the first riser, and at the head:
    z1 = zt[-1] + rise

    def at(s, z):
        p = s * u + lmid * v
        return [float(p[0]), float(p[1]), float(z)]
    points = [at(s0[0], z0)] + [at((a + b) / 2, z) for a, b, (z, _) in zip(s0, s1, treads)] \
        + [at(s1[-1], z1)]
    return dict(name=el.Name or "", start=points[0], end=points[-1], width=float((l1 - l0).min()), risers=risers,
                riser=riser, going=going, points=points, dir=u)


def _landing(T):
    """(top z, plan polygon of the top faces) of a landing slab's triangles."""
    up = T[_up(T, 0.9)]
    if not len(up):
        return None
    z = float(up[:, :, 2].max())
    top = up[np.abs(up[:, :, 2] - z).max(1) < 0.01]
    polys = [p for p in (Polygon(t[:, :2]) for t in top) if p.area > 1e-8]
    if not polys:
        return None
    u = shapely.union_all(polys, grid_size=1e-4).simplify(1e-4)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda q: q.area)
    return z, u


def _centre(poly) -> np.ndarray:
    c = poly.centroid
    if not poly.contains(c):
        c = poly.representative_point()
    return np.array([c.x, c.y])


def stairs(f, meshes, transform, storeys: dict, rooms: list) -> list[dict]:
    """Every IfcStair of IFC ``f`` (``meshes``: meshcache triangles, estate frame) in block-local coordinates
    (``transform``: block -> estate, as in the engine JSON; ``storeys``: block-local FFLs; ``rooms``: the engine
    JSON's rooms). Sorted by storey, then name."""
    inv = np.linalg.inv(np.asarray(transform, float))
    ffl = dict(storeys)
    landings = []                             # every LANDING slab of the file: (z, polygon)
    for s in f.by_type("IfcSlab"):
        if (s.PredefinedType or "") == "LANDING" and s.GlobalId in meshes:
            lg = _landing(_tris(meshes, s.GlobalId, inv))
            if lg is not None:
                landings.append(lg)
    room_polys = [(r["storey"], r["name"], Polygon([p[:2] for p in r["polygon"]])) for r in rooms
                  if len(r.get("polygon") or ()) >= 4]
    out = []
    for st in f.by_type("IfcStair"):
        parts = list(uel.get_decomposition(st))
        fl = [flight(p, _tris(meshes, p.GlobalId, inv)) for p in parts
              if p.is_a("IfcStairFlight") and p.GlobalId in meshes]
        fl = sorted((x for x in fl if x is not None), key=lambda x: (x["start"][2], x["name"]))
        if not fl:
            continue
        own = [lg for lg in (_landing(_tris(meshes, p.GlobalId, inv)) for p in parts
                             if p.is_a("IfcSlab") and (p.PredefinedType or "") == "LANDING" and p.GlobalId in meshes)
               if lg is not None]
        own.sort(key=lambda lg: lg[0])
        cont = uel.get_container(st)
        storey = cont.Name if cont is not None and cont.Name in ffl else min(
            ffl, key=lambda k: abs(ffl[k] - fl[0]["start"][2]))
        z_lo, z_hi = fl[0]["start"][2], fl[-1]["end"][2]
        to = min(ffl, key=lambda k: abs(ffl[k] - z_hi))
        to = to if abs(ffl[to] - z_hi) < 0.05 else None
        # the floor landing the walk starts on: the landing slab at this floor that the first flight leaves from,
        # else this stair's own floor landing (at its top) set down onto this floor
        p0 = Point(fl[0]["start"][:2])
        below = [poly for z, poly in landings if abs(z - z_lo) < 0.05 and poly.buffer(0.05).contains(p0)]
        top = [lg for lg in own if abs(lg[0] - z_hi) < 0.05]
        start_poly = below[0] if below else (top[0][1] if top else None)
        path = [] if start_poly is None else [[*_centre(start_poly), z_lo]]
        lands = [] if start_poly is None else [start_poly]        # each path point's landing (None: on a flight)
        for x in fl:                          # each flight met head on: APPROACH before its foot and after its head
            a = np.array(x["start"][:2]) - APPROACH * x["dir"]
            e = np.array(x["end"][:2]) + APPROACH * x["dir"]
            path += [[float(a[0]), float(a[1]), x["start"][2]], *x["points"], [float(e[0]), float(e[1]), x["end"][2]]]
            lands += [None] * (len(x["points"]) + 2)
            land = [poly for z, poly in own if abs(z - x["end"][2]) < 0.05]
            if land:
                path.append([*_centre(land[0]), x["end"][2]])
                lands.append(land[0])
        half = np.array(fl[0]["points"][len(fl[0]["points"]) // 2][:2])
        room = next((name for sto, name, poly in room_polys if sto == storey and poly.contains(Point(half))), None)
        out.append(dict(name=st.Name or "", room=room, storey=storey, to=to, from_ffl=round(ffl[storey], R),
                        to_ffl=round(ffl[to], R) if to else round(z_hi, R),
                        flights=[dict(start=_r(x["start"]), end=_r(x["end"]), width=round(x["width"], R),
                                      risers=x["risers"], riser=round(x["riser"], 4), going=round(x["going"], 4))
                                 for x in fl],
                        landings=[dict(z=round(z, R), polygon=_ring(poly)) for z, poly in own],
                        path=[_r(p) for p in path], _lands=lands))
    return sorted(out, key=lambda s: (s["from_ffl"], s["name"]))


def _cells_in(grid, poly, z, dz) -> list:
    """Centres [(x, y)] of the cells of ``grid`` (walkcheck.Grid) inside ``poly`` (block-local) with a floor within dz
    of z."""
    x0, y0, x1, y1 = poly.bounds
    (i0, j0), (i1, j1) = grid.cell_of(x0, y0), grid.cell_of(x1, y1)
    pts = []
    for iy in range(max(0, j0), min(grid.ny - 1, j1) + 1):
        for ix in range(max(0, i0), min(grid.nx - 1, i1) + 1):
            f = grid.floor_in_cell(ix, iy, z)
            if f is not None and abs(f[0] - z) <= dz:
                pts.append(grid.centre(ix, iy))
    if not pts:
        return []
    xy = np.array(pts, float)
    return [tuple(p) for p in xy[shapely.contains_xy(poly, xy[:, 0], xy[:, 1])].tolist()]


def fit_paths(stairs: list, grid) -> dict:
    """Put every stair's path on the walk grid it is exported with (``grid``: the encoded file decoded, as a viewer
    reads it: estate/web/walkcheck.py Grid), so that a viewer walking it never leaves the grid: every cell each of its
    segments crosses has a floor within a step of it all the way across (Grid.gaps, the stage's own check). Flights
    stand on their treads; it is a landing point (a landing's centre, or the start landing set down a storey) that
    can fall where the grid has no floor (in an opened leaf's sweep, within the radius of a wall or column) or whose
    straight line to the flight crosses a blocked cell, if only at its corner. Only such a point moves: keeping its
    height, to the cell of its landing nearest the landing's centre from which both of its lines clear every blocked
    cell by CLEARANCE, else (no such cell) by walkcheck.EDGE, the check's own margin; an inner point with neither is
    dropped when the line past it is clear. Removes each stair's ``_lands``; returns dict(moved, dropped, off): the
    landing points moved and dropped, and the cells the fitted paths still cross without a floor (the stage refuses
    any; none on the estate's buildings)."""
    moved = dropped = off = 0
    for s in stairs:
        lands = s.pop("_lands")
        P = [list(p) for p in s["path"]]
        i = 0
        while i < len(P):
            prev = P[i - 1] if i else None
            nxt = P[i + 1] if i + 1 < len(P) else None

            def clear(q, eps=walkcheck.EDGE, prev=prev, nxt=nxt):
                return not grid.gaps(q, q, eps) and (prev is None or not grid.gaps(prev, q, eps)) and \
                    (nxt is None or not grid.gaps(q, nxt, eps))
            if lands[i] is None or clear(P[i]):
                i += 1
                continue
            x0, y0, z = P[i]
            cands = sorted({(round(x, R) + 0.0, round(y, R) + 0.0) for x, y in _cells_in(grid, lands[i], z, 0.1)},
                           key=lambda q: (round(float(np.hypot(q[0] - x0, q[1] - y0)), 6), q))
            pick = next(([x, y, z] for eps in (CLEARANCE, walkcheck.EDGE) for x, y in cands
                         if clear([x, y, z], eps)), None)
            if pick is not None:
                P[i] = pick
                moved += 1
            elif prev is not None and nxt is not None and not grid.gaps(prev, nxt):
                del P[i], lands[i]
                dropped += 1
                continue
            i += 1
        s["path"] = [_r(p) for p in P]
        P = s["path"]
        off += sum(len(grid.gaps(p, q)) for p, q in zip(P[:-1], P[1:])) if len(P) > 1 else \
            sum(len(grid.gaps(p, p)) for p in P)
    return dict(moved=moved, dropped=dropped, off=off)
