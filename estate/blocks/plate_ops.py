"""Whole-plate operations: move, rotate, mirror, rename, merge and sanity-check floor plates and building plans.

Typologies are composed from strips authored in their own frame (a corridor slab wing, a corner module) and
then placed by 90-degree rotations, mirrors and translations. Everything a plate carries must move with it:
face polygons, door centre and hinge points, window points, AC ledges with their railing bands, flat
envelopes, and the entry side of every stair enclosure. Mirrors are allowed: walls and door swings are
re-derived from the geometry (blocks/builder.py), so a mirrored plate stays valid.
Coordinates are snapped to 0.1 mm after every transform so that shared edges stay bit-identical and unions
and differences of faces do not produce slivers.
The checks at the end run before any wall is derived: plate structure (rectilinear, non-overlapping faces,
references), flat access (one main door per flat, onto a common corridor or lobby) and the longest corridor
walk to a stair door. window_outlook runs on the derived plate: counting glazing area says nothing about what a
window faces, so it sweeps every habitable / kitchen window outwards and reports the ones that look onto a wall
close by (the L-block corner unit once faced the next wing's flank 1.9 m away).
"""
from __future__ import annotations

import copy
import itertools

import numpy as np
import shapely
from shapely.ops import unary_union

from estate.blocks.plate import OUT, DoorSpec, Face, Ledge, Plate, WindowSpec
from estate.flats.template import Affine2
from estate.geom.arrangement import wall_footprint

GRID = 4                 # decimals kept by snap() (0.1 mm, as geom/arrangement.py)
DIRS = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}


# ----------------------------------------------------------------------------- affine helpers
def snap_geom(g):
    if g is None or g.is_empty:
        return g
    return shapely.transform(g, lambda c: np.round(c, GRID) + 0.0)


def snap_pt(p):
    return (round(float(p[0]) + 0.0, GRID), round(float(p[1]) + 0.0, GRID))


def translation(dx: float, dy: float) -> Affine2:
    return Affine2(1, 0, 0, 1, dx, dy)


def rotation(k: int, about=(0.0, 0.0)) -> Affine2:
    """k quarter turns counter-clockwise about a point."""
    k %= 4
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]
    A = np.array([[c, -s], [s, c]], float)
    t = np.asarray(about, float) - A @ np.asarray(about, float)
    return Affine2(c, -s, s, c, float(t[0]), float(t[1]))


def mirror_x(about: float = 0.0) -> Affine2:
    """Mirror in the vertical line x = about (x -> 2 about - x)."""
    return Affine2(-1, 0, 0, 1, 2.0 * about, 0.0)


def mirror_y(about: float = 0.0) -> Affine2:
    return Affine2(1, 0, 0, -1, 0.0, 2.0 * about)


def compose(*xfs: Affine2) -> Affine2:
    """compose(a, b, c)(p) == a(b(c(p)))."""
    A, t = np.eye(2), np.zeros(2)
    for xf in reversed(xfs):
        A, t = xf.A @ A, xf.A @ t + xf.t
    out = Affine2()
    out.A, out.t = A, t
    return out


def map_entry(entry: str, xf: Affine2) -> str:
    """Side name (N/S/E/W) after the linear part of xf (stair entries, corridor sides)."""
    v = xf.A @ np.asarray(DIRS[entry], float)
    for k, d in DIRS.items():
        if np.allclose(v, d):
            return k
    raise ValueError(f"{xf.A} is not a multiple of 90 degrees")


def _poly(g, xf: Affine2):
    return snap_geom(xf.poly(g))


# ----------------------------------------------------------------------------- plates
def transform_plate(plate: Plate, xf: Affine2) -> Plate:
    """A new plate with every geometric item mapped by xf (face ids, kinds and metadata unchanged)."""
    out = Plate()
    for f in plate.faces.values():
        out.add(Face(f.id, _poly(f.poly, xf), f.kind, f.flat, f.room, f.name, dict(f.meta)))
    out.doors = [DoorSpec(d.a, d.b, snap_pt(xf(d.at)), d.width, d.kind, d.into, snap_pt(xf(d.hinge)), d.name, d.height)
                 for d in plate.doors]
    out.windows = [WindowSpec(w.face, snap_pt(xf(w.at)), w.width, w.sill, w.height, w.name, w.kind, w.required)
                   for w in plate.windows]
    out.open_pairs = set(plate.open_pairs)
    out.ledges = [Ledge(_poly(l.poly, xf), l.flat, l.name, [_poly(b, xf) for b in l.rail], l.kind) for l in plate.ledges]
    for stack, fi in plate.flats.items():
        nf = copy.copy(fi)
        nf.envelope = _poly(fi.envelope, xf)
        nf.variant = dict(fi.variant)
        out.flats[stack] = nf
    out.notes = list(plate.notes)
    return out


def translate(plate: Plate, dx: float, dy: float) -> Plate:
    return transform_plate(plate, translation(dx, dy))


def rotate(plate: Plate, k: int, about=(0.0, 0.0)) -> Plate:
    return transform_plate(plate, rotation(k, about))


def mirror(plate: Plate, axis: str = "x", about: float = 0.0) -> Plate:
    """axis 'x': mirror x (in the line x = about); 'y': mirror y (in the line y = about)."""
    return transform_plate(plate, mirror_x(about) if axis == "x" else mirror_y(about))


def snap(plate: Plate) -> Plate:
    """Snap a plate in place to the 0.1 mm grid (after placing flats with float cursors)."""
    for f in plate.faces.values():
        f.poly = snap_geom(f.poly)
    for d in plate.doors:
        d.at, d.hinge = snap_pt(d.at), snap_pt(d.hinge)
    for w in plate.windows:
        w.at = snap_pt(w.at)
    for l in plate.ledges:
        l.poly = snap_geom(l.poly)
        l.rail = [snap_geom(b) for b in l.rail]
    for fi in plate.flats.values():
        fi.envelope = snap_geom(fi.envelope)
    return plate


def rename(plate: Plate, fn) -> Plate:
    """A copy with face ids mapped by fn(old_id) -> new_id (doors, windows and open pairs follow)."""
    def m(fid):
        return fid if fid == OUT else fn(fid)
    out = Plate()
    for f in plate.faces.values():
        out.add(Face(m(f.id), f.poly, f.kind, f.flat, f.room, f.name, dict(f.meta)))
    out.doors = [DoorSpec(m(d.a), m(d.b), d.at, d.width, d.kind, m(d.into), d.hinge, d.name, d.height) for d in plate.doors]
    out.windows = [WindowSpec(m(w.face), w.at, w.width, w.sill, w.height, w.name, w.kind, w.required) for w in plate.windows]
    out.open_pairs = {frozenset(m(x) for x in p) for p in plate.open_pairs}
    out.ledges = list(plate.ledges)
    out.flats = dict(plate.flats)
    out.notes = list(plate.notes)
    return out


def merge(*plates: Plate) -> Plate:
    out = Plate()
    for p in plates:
        out.extend(p)
    return out


def union(plate: Plate, kinds=None):
    polys = [f.poly for f in plate.faces.values() if kinds is None or f.kind in kinds]
    return snap_geom(unary_union(polys)) if polys else shapely.Polygon()


def bounds(plate: Plate, ledges=False):
    g = [f.poly for f in plate.faces.values()] + ([l.poly for l in plate.ledges] if ledges else [])
    return unary_union(g).bounds


def single_polygon(g, what="face", min_part=0.01):
    """The polygon of a difference / union result; raises if it falls apart into significant pieces."""
    g = snap_geom(g)
    if g.geom_type == "Polygon":
        return g
    parts = sorted((p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"), key=lambda p: -p.area)
    if not parts:
        raise ValueError(f"{what}: empty")
    if len(parts) > 1 and parts[1].area > min_part:
        raise ValueError(f"{what} falls apart into {len(parts)} pieces {[round(p.area, 2) for p in parts]}")
    return parts[0]


# ----------------------------------------------------------------------------- plans
def transform_plan(plan, xf: Affine2):
    """A copy of a BuildingPlan with its plates, stairs, roof items, columns, meta points and void-deck furniture
    mapped by xf."""
    out = copy.copy(plan)
    out.ground = transform_plate(plan.ground, xf)
    out.typical = transform_plate(plan.typical, xf)
    out.roof = transform_plate(plan.roof, xf)
    out.stairs = [type(s)(s.face, map_entry(s.entry, xf), s.name) for s in plan.stairs]
    out.roof_items = [(k, _poly(p, xf), h) for k, p, h in plan.roof_items]
    out.columns = None if plan.columns is None else [snap_pt(xf(p)) for p in plan.columns]
    meta = dict(plan.meta)
    for key in ("lift_lobbies", "entrances"):
        if key in meta:
            meta[key] = [snap_pt(xf(p)) for p in meta[key]]
    if "corridor_side" in meta:
        meta["corridor_side"] = map_entry(meta["corridor_side"], xf)
    if "furniture" in meta:
        meta["furniture"] = [dict(it, poly=_poly(it["poly"], xf)) for it in meta["furniture"]]
    out.meta = meta
    return out


# ----------------------------------------------------------------------------- checks
def check_plate(plate: Plate, tol: float = 1e-6) -> list[str]:
    """Structural sanity of a plate (before wall derivation): valid rectilinear polygons, no overlaps, every
    door / window / open pair refers to existing faces, door points lie on the shared boundary."""
    errs = []
    faces = list(plate.faces.values())
    for f in faces:
        g = f.poly
        if g.geom_type != "Polygon" or not g.is_valid or g.is_empty:
            errs.append(f"face {f.id}: not a valid polygon ({g.geom_type})")
            continue
        for ring in [g.exterior, *g.interiors]:
            c = list(ring.coords)
            for p, q in zip(c[:-1], c[1:]):
                if abs(p[0] - q[0]) > tol and abs(p[1] - q[1]) > tol:
                    errs.append(f"face {f.id}: non-rectilinear edge {p} -> {q}")
                    break
    tree = shapely.STRtree([f.poly for f in faces])
    for i, j in zip(*tree.query([f.poly for f in faces], predicate="intersects")):
        if i < j:
            a = faces[i].poly.intersection(faces[j].poly).area
            if a > 1e-4:
                errs.append(f"faces {faces[i].id} and {faces[j].id} overlap ({a:.3f} m2)")
    ids = set(plate.faces) | {OUT}
    for d in plate.doors:
        for fid in (d.a, d.b, d.into):
            if fid not in ids:
                errs.append(f"door '{d.name}': unknown face {fid}")
        if d.a in plate.faces and d.b in plate.faces:
            shared = plate.faces[d.a].poly.boundary.intersection(plate.faces[d.b].poly.boundary)
            if shared.distance(shapely.Point(d.at)) > 1e-3:
                errs.append(f"door '{d.name}' ({d.a}|{d.b}) at {d.at} is not on their shared boundary")
    for w in plate.windows:
        if w.face not in plate.faces:
            errs.append(f"window '{w.name}': unknown face {w.face}")
    for p in plate.open_pairs:
        for fid in p:
            if fid not in ids:
                errs.append(f"open pair {sorted(p)}: unknown face {fid}")
    return errs


def access_issues(plate: Plate) -> list[str]:
    """Every flat has exactly one door to a non-flat face, it is its main door and it opens onto a common
    corridor / lobby; flat rooms open onto nothing else outside the flat (no doors to cores or amenities)."""
    from estate.blocks.plate import COMMON_KINDS
    out = []
    for stack in sorted(plate.flats):
        mains = []
        for d in plate.doors:
            fa, fb = plate.faces.get(d.a), plate.faces.get(d.b)
            ins = [f for f in (fa, fb) if f is not None and f.flat == stack]
            if len(ins) != 1:
                continue
            other = fb if ins[0] is fa else fa
            if other is not None and other.flat == stack:
                continue
            if other is not None and other.flat is not None:
                out.append(f"{stack}: door '{d.name}' opens into flat {other.flat}")
            elif other is None or other.kind not in COMMON_KINDS:
                out.append(f"{stack}: door '{d.name}' opens onto {other.kind if other else 'outside'}")
            elif d.kind != "main":
                out.append(f"{stack}: '{d.name}' ({d.kind}) opens onto the corridor but is not the main door")
            else:
                mains.append(d)
        if len(mains) != 1:
            out.append(f"{stack}: {len(mains)} main doors onto a corridor / lobby")
    return out


def stair_walk(plate: Plate, cell: float = 0.25) -> tuple[float, tuple | None]:
    """Longest walk (m) from any point of the common corridors / lobbies to the nearest stair door, over a
    cell grid of the corridor faces (8-neighbour moves). Returns (max distance, worst point)."""
    import networkx as nx
    from estate.blocks.plate import COMMON_KINDS, STAIR
    area = union(plate, COMMON_KINDS)
    if area.is_empty:
        return 0.0, None
    x0, y0, x1, y1 = area.bounds
    nx_, ny_ = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    xs = x0 + (np.arange(nx_) + 0.5) * cell
    ys = y0 + (np.arange(ny_) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    inside = shapely.contains_xy(area, X, Y)
    G = nx.Graph()
    diag = cell * 2 ** 0.5
    for i, j in zip(*np.nonzero(inside)):
        for di, dj, w in ((1, 0, cell), (0, 1, cell), (1, 1, diag), (1, -1, diag)):
            a, b = i + di, j + dj
            if 0 <= a < nx_ and 0 <= b < ny_ and inside[a, b]:
                G.add_edge((i, j), (a, b), weight=w)
    stairs = {f.id for f in plate.faces.values() if f.kind == STAIR}
    src = set()
    for d in plate.doors:
        if {d.a, d.b} & stairs and plate.kind(d.b if d.a in stairs else d.a) in COMMON_KINDS:
            i, j = int((d.at[0] - x0) / cell), int((d.at[1] - y0) / cell)
            best = min(((abs(xs[a] - d.at[0]) + abs(ys[b] - d.at[1]), (a, b)) for a in range(max(0, i - 3), min(nx_, i + 4))
                        for b in range(max(0, j - 3), min(ny_, j + 4)) if inside[a, b]), default=None)
            if best is not None:
                src.add(best[1])
    if not src:
        return float("inf"), None
    dist = nx.multi_source_dijkstra_path_length(G, src)
    cells = list(zip(*np.nonzero(inside)))
    worst = max(cells, key=lambda c: dist.get(c, float("inf")))
    return float(dist.get(worst, float("inf"))), (round(float(xs[worst[0]]), 2), round(float(ys[worst[1]]), 2))


OUTLOOK_MIN = 2.8        # m clear in front of a window: 3.0 m between wall centrelines, as across a point-block lobby
OUTLOOK_RANGE = 30.0     # how far the outlook is measured


def window_outlook(dp, min_clear: float = OUTLOOK_MIN) -> list[tuple]:
    """Glazed windows of habitable rooms and kitchens (and any required window) that look onto a full-height wall
    less than min_clear away: the window's width is swept outwards from the outer face of its wall, and storey-high
    walls count as obstructions (parapets, railings and balcony guards do not). Works on a derived plate (after the
    walls exist, so a merged L-block or a rotated wing is tested as built). Returns
    [(window name, face id, clear distance m)], sorted."""
    plate = dp.plate
    tall = [wall_footprint(r) for r in dp.runs
            if r.wall and r.t > 0 and r.wall not in ("RAILING", "PARAPET") and r.height in ("storey", "tower")]
    obstacles = unary_union(tall) if tall else shapely.Polygon()
    required = {(w.face, w.name) for w in plate.windows if w.required}
    out = []
    for o in dp.openings:
        fid = o.meta.get("face")
        f = plate.faces.get(fid)
        if o.kind != "window" or f is None:
            continue
        glazed = o.meta.get("kind", "window") == "window" and (f.meta.get("category") == "habitable" or f.room == "K")
        if not (glazed or (fid, o.name) in required):
            continue
        r = dp.runs[o.run]
        mid = (o.s0 + o.s1) / 2
        parts = r.meta.get("parts") or [(0.0, r.length, r.left, r.right)]
        left = next((lf for a0, a1, lf, _ in parts if a0 - 1e-6 <= mid <= a1 + 1e-6), r.left)
        u = r.u
        n = np.array([-u[1], u[0]]) * (-1.0 if left == fid else 1.0)     # outwards, away from the room
        a = np.asarray(r.p0, float) + u * (o.s0 + 0.02)
        b = np.asarray(r.p0, float) + u * (o.s1 - 0.02)
        near = r.t / 2 + 0.01
        sweep = shapely.Polygon([a + n * near, b + n * near, b + n * OUTLOOK_RANGE, a + n * OUTLOOK_RANGE])
        hit = obstacles.intersection(sweep)
        if hit.is_empty or hit.area < 1e-6:
            continue
        pts = shapely.get_coordinates(hit)
        clear = float(np.min((pts - a) @ n)) - r.t / 2
        if clear < min_clear - 1e-3:
            out.append((o.name, fid, round(clear, 2)))
    return sorted(out)


COLUMN = 0.6             # void deck column, square (blocks/builder.py _columns)
COLUMN_STEP = 7.2        # at most this far apart along a facade / party wall line
COLUMN_KEEP_OFF = 0.35   # no column centre within this of a core or amenity room
COLUMN_SPACING = 1.5     # two columns are never closer than this
COLUMN_LINES = ("EXT", "PARTY", "PARAPET", "HS250", "HS")


def column_candidates(dp_typical, ground: Plate, keep_off: float = COLUMN_KEEP_OFF) -> list[tuple]:
    """Every point a void deck column may stand on, sorted: the ends of each facade (wall to outside) and party
    wall run of the derived typical floor and points every <= COLUMN_STEP along it, except within keep_off of a
    ground-floor core or amenity room. column_points thins them out; anything that must stay clear of the
    columns in every orientation of the plan (letterbox banks, blocks/point.py) avoids all of them, as which
    point of a close pair survives the thinning depends on the sort order."""
    from estate.blocks.plate import VOID
    blocked = unary_union([f.poly.buffer(keep_off) for f in ground.faces.values() if f.kind != VOID])
    pts = []
    for r in dp_typical.runs:
        if r.wall not in COLUMN_LINES or not (OUT in (r.left, r.right) or r.wall == "PARTY"):
            continue
        n_seg = max(1, int(np.ceil(r.length / COLUMN_STEP)))
        for i in range(n_seg + 1):
            pts.append(tuple(np.asarray(r.p0) + (np.asarray(r.p1) - np.asarray(r.p0)) * i / n_seg))
    pts = sorted(set((float(round(x, 2)), float(round(y, 2))) for x, y in pts))   # numpy rounding, as the builder
    return [p for p in pts if not blocked.contains(shapely.Point(p))]


def column_points(dp_typical, ground: Plate) -> list[tuple]:
    """The void deck columns: column_candidates, dropping any closer than COLUMN_SPACING to one already kept."""
    kept = []
    for p in column_candidates(dp_typical, ground):
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= COLUMN_SPACING for q in kept):
            kept.append(p)
    return kept


def overlaps(polys: list, tol: float = 1e-4) -> list[tuple]:
    """Index pairs of polygons that overlap by more than tol m2 (ledges, amenities, roof items)."""
    out = []
    for (i, a), (j, b) in itertools.combinations(enumerate(polys), 2):
        if a.intersects(b) and a.intersection(b).area > tol:
            out.append((i, j))
    return out
