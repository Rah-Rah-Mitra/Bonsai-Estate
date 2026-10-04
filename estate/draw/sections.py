"""Stair sections cut from the IFC (port of legacy/toolkit/draw_block.stair_section without trimesh or matplotlib).

The drawing is taken from the authored model, not from the plan it was generated from, so it shows what is in the
file, including a block hand-edited in Bonsai. Triangles come from the shared tessellation cache
(estate.export.meshcache). A vertical plane through the centreline of flight 1 cuts them (numpy), the cut segments
are closed into polygons with shapely and drawn with PIL. The stair parts beyond the plane (flight 2, balustrades,
well guards, doors in the far wall) are drawn in elevation from their silhouettes, lighter.

Each distinct stair enclosure (IfcStairs grouped by footprint) gets one sheet covering its lowest storeys, with the
storey FFLs and the riser / going figures read back from the IfcStairFlight attributes.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import shapely
import ifcopenshell.util.element as uel
import ifcopenshell.util.placement as upl
from PIL import Image, ImageDraw
from shapely.geometry import MultiPoint, box

from estate.draw.raster import Canvas, font

NEAR = {"IfcSlab": (140, 140, 140), "IfcWall": (59, 59, 59), "IfcColumn": (59, 59, 59),
        "IfcStairFlight": (176, 141, 87), "IfcRailing": (68, 68, 68), "IfcDoor": (192, 57, 43),
        "IfcWindow": (127, 179, 213)}
BEYOND = {"IfcStairFlight": ((226, 214, 188), (176, 150, 110)), "IfcRailing": (None, (160, 160, 160)),
          "IfcDoor": (None, (205, 120, 110))}     # class: (fill, outline) of the elevation beyond the cut
FLIGHT_TEXT = ((122, 90, 42), (160, 128, 80))
COMPASS = ("north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west")
_MEASURE = ImageDraw.Draw(Image.new("RGB", (1, 1)))


# ----------------------------------------------------------------------------- mesh cuts
def plane_segments(verts, faces, origin, normal):
    """Segments where a triangle mesh crosses a plane, as an (n, 2, 3) array (numpy only).

    Each crossing edge is interpolated from its vertex above the plane, so the two triangles sharing an edge give
    bit-identical points and the cut loops close exactly. A vertex on the plane counts as just above it.
    """
    tri = verts[faces]
    s = (tri - np.asarray(origin, float)) @ np.asarray(normal, float)
    s[s == 0.0] = 1e-12
    above = s > 0
    k = above.sum(1)
    sel = (k == 1) | (k == 2)
    tri, s, above = tri[sel], s[sel], above[sel]
    out = np.zeros((len(tri), 2, 3))
    slot = np.zeros(len(tri), int)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        hit = np.nonzero(above[:, i] != above[:, j])[0]
        up = np.where(above[hit, i], i, j)
        dn = np.where(above[hit, i], j, i)
        su, sd = s[hit, up], s[hit, dn]
        pu, pd = tri[hit, up], tri[hit, dn]
        out[hit, slot[hit]] = pu + (pd - pu) * (su / (su - sd))[:, None]
        slot[hit] += 1
    return out


def polygons(seg2d):
    """Close 2D cut segments into the section area (even-odd, so a solid with a through-hole stays hollow)."""
    seg2d = np.round(np.asarray(seg2d, float), 6)
    if not len(seg2d):
        return None
    seg2d = seg2d[np.abs(seg2d[:, 0] - seg2d[:, 1]).max(1) > 1e-7]
    if not len(seg2d):
        return None
    noded = shapely.union_all(shapely.linestrings(seg2d), grid_size=1e-5)
    faces = shapely.get_parts(shapely.polygonize(shapely.get_parts(noded)))
    faces = faces[shapely.area(faces) > 1e-8]
    if len(faces) > 1:   # a face inside an odd number of loop shells is solid; holes come out even
        shells = shapely.polygons(shapely.get_exterior_ring(faces))
        inside = shapely.contains(shells[None, :], shapely.point_on_surface(faces)[:, None])
        faces = faces[inside.sum(1) % 2 == 1]
    return shapely.union_all(faces) if len(faces) else None


def cut(d, origin, normal, axes, offset=(0.0, 0.0)):
    """Section of one tessellated element by a plane, projected to 2D by the 3 x 2 matrix `axes` (+ offset)."""
    seg = plane_segments(d["verts"], d["faces"], origin, normal)
    return polygons(seg @ axes + offset) if len(seg) else None


def silhouette(d, axes, offset=(0.0, 0.0)):
    """Outline of an element seen along the out-of-page axis of `axes` (for a plan: the footprint). The union of
    the triangles facing the viewer covers a closed mesh's projection exactly."""
    tri = d["verts"][d["faces"]] @ axes + offset
    a = (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - \
        (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0])
    front = tri[a > 1e-9]
    if not len(front):
        front = tri[a < -1e-9][:, ::-1]
    if not len(front):
        return None
    g = shapely.union_all(shapely.polygons(np.round(front, 6)), grid_size=1e-5)
    return None if g.is_empty else g


# ----------------------------------------------------------------------------- model queries
def natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s or "")]


def storey_levels(f):
    """World FFL of every storey from its placement (not the Elevation attribute: a building may sit on terrain)."""
    out = {}
    for s in f.by_type("IfcBuildingStorey"):
        out[s.Name] = float(upl.get_local_placement(s.ObjectPlacement)[2, 3]) if s.ObjectPlacement else \
            float(s.Elevation or 0.0)
    return out


def _stem(name, storey):
    return name[len(storey) + 1:] if storey and name.startswith(storey + " ") else name


def stair_groups(f, meshes, zs=None):
    """IfcStairs grouped by enclosure (footprint), bottom to top: [dict(name, footprint, stairs, storeys)]."""
    zs = zs or storey_levels(f)
    items = []
    for st in f.by_type("IfcStair"):
        parts = [p for p in uel.get_decomposition(st) if p.GlobalId in meshes]
        if not parts:
            continue
        cont = uel.get_container(st)
        storey = cont.Name if cont is not None else ""
        xy = np.vstack([meshes[p.GlobalId]["verts"][:, :2] for p in parts])
        items.append((zs.get(storey, 0.0), natural(_stem(st.Name or "", storey)), storey, st,
                      box(*xy.min(0), *xy.max(0))))
    items.sort(key=lambda t: (t[0], t[1]))
    groups = []
    for _, _, storey, st, fp in items:
        for g in groups:
            if g["footprint"].contains(fp.centroid) and storey not in g["storeys"]:
                g["stairs"].append(st)
                g["storeys"].append(storey)
                break
        else:
            groups.append(dict(name=_stem(st.Name or "", storey), footprint=fp, stairs=[st], storeys=[storey]))
    groups.sort(key=lambda g: (natural(g["name"]), g["footprint"].centroid.x, g["footprint"].centroid.y))
    return groups


def _flights(st, meshes):
    return sorted((p for p in uel.get_decomposition(st) if p.is_a("IfcStairFlight") and p.GlobalId in meshes),
                  key=lambda p: natural(p.Name))


def run_direction(verts):
    """Horizontal direction in which a flight rises: the gradient of a plane fitted through its vertices."""
    xy = verts[:, :2] - verts[:, :2].mean(0)
    (a, b, _), *_ = np.linalg.lstsq(np.c_[xy, np.ones(len(xy))], verts[:, 2], rcond=None)
    g = np.array([a, b])
    if np.hypot(a, b) > 1e-6:
        return g / np.linalg.norm(g)
    rect = np.asarray(MultiPoint(verts[:, :2]).minimum_rotated_rectangle.exterior.coords)
    e = max((rect[1] - rect[0], rect[2] - rect[1]), key=np.linalg.norm)
    return e / np.linalg.norm(e)


# ----------------------------------------------------------------------------- drawing helpers
def text_size(s, size):
    x0, y0, x1, y1 = _MEASURE.multiline_textbbox((0, 0), s, font=font(size))
    return x1 - x0, y1 - y0


def draw_text(cv, p, s, size=11, fill=(40, 40, 40), anchor="mm", bg=None):
    """Canvas text with any anchor (PIL's multiline_text refuses top/bottom anchors), optionally on a backdrop."""
    multi = "\n" in s
    kw = dict(font=font(size), anchor=anchor[0] + "m" if multi else anchor, **({"align": "center"} if multi else {}))
    if bg is not None:
        x0, y0, x1, y1 = (cv.d.multiline_textbbox if multi else cv.d.textbbox)(cv.P(p), s, **kw)
        cv.d.rectangle((x0 - 2, y0 - 1, x1 + 2, y1 + 1), fill=bg)
    (cv.d.multiline_text if multi else cv.d.text)(cv.P(p), s, fill=fill, **kw)


def dashed(cv, p0, p1, fill, width=1, dash=0.15, gap=0.1):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    if L < 1e-9:
        return
    u = (p1 - p0) / L
    t = 0.0
    while t < L:
        cv.line([p0 + u * t, p0 + u * min(t + dash, L)], fill, width)
        t += dash + gap


def paint(cv, g, fill=None, edge=None, width=1):
    """Fill and/or outline a (multi)polygon. Holes stay transparent (Canvas.poly paints them white, which would
    erase whatever was drawn underneath), so polygons with holes are filled through a mask cropped to their box."""
    if g is None or g.is_empty:
        return
    for part in getattr(g, "geoms", [g]):
        if part.geom_type != "Polygon" or part.is_empty:
            continue
        if fill is not None and not part.interiors:
            cv.d.polygon([cv.P(c) for c in part.exterior.coords], fill=fill)
        elif fill is not None:
            px = np.array([cv.P(c) for c in part.exterior.coords])
            x0, y0 = np.floor(px.min(0)).astype(int)
            x1, y1 = np.ceil(px.max(0)).astype(int)
            mask = Image.new("L", (int(x1 - x0) + 1, int(y1 - y0) + 1), 0)
            md = ImageDraw.Draw(mask)
            md.polygon([(x - x0, y - y0) for x, y in px], fill=255)
            for hole in part.interiors:
                md.polygon([(x - x0, y - y0) for x, y in (cv.P(c) for c in hole.coords)], fill=0)
            cv.img.paste(fill, (int(x0), int(y0), int(x1) + 1, int(y1) + 1), mask)
        if edge is not None:
            for ring in [part.exterior, *part.interiors]:
                cv.line(list(ring.coords), edge, width)


# ----------------------------------------------------------------------------- the section
def _section(f, meshes, zs, g, out_png, storeys=4, title=None, scale=70.0):
    stairs, levels = g["stairs"][:storeys], g["storeys"][:storeys]
    order = sorted(zs, key=zs.get)
    i_top = order.index(levels[-1]) + 1 if levels[-1] in order else len(order)
    top = order[i_top] if i_top < len(order) else None
    flights = _flights(stairs[0], meshes)
    if not flights:
        return None
    # section plane through the centreline of flight 1, looking across the well towards flight 2
    run = run_direction(meshes[flights[0].GlobalId]["verts"])
    n = np.array([-run[1], run[0]])
    mids = []
    for fl in flights:
        v = meshes[fl.GlobalId]["verts"][:, :2] @ n
        mids.append(float(v.min() + v.max()) / 2)
    if len(mids) > 1 and mids[-1] < mids[0]:
        n, mids = -n, [-m for m in mids]
    right = np.array([n[1], -n[0]])
    axes = np.array([[right[0], 0.0], [right[1], 0.0], [0.0, 1.0]])
    normal, origin = np.array([n[0], n[1], 0.0]), np.array([n[0] * mids[0], n[1] * mids[0], 0.0])

    fp = g["footprint"]
    u_fp = np.asarray(fp.exterior.coords)[:, :2] @ right
    u0, u1 = float(u_fp.min()) - 1.0, float(u_fp.max()) + 1.0
    z_lo = zs.get(levels[0], 0.0)
    z_hi = zs[top] if top else max(float(meshes[p.GlobalId]["verts"][:, 2].max())
                                   for s in stairs for p in uel.get_decomposition(s) if p.GlobalId in meshes)
    z0, z1 = z_lo - 0.7, z_hi + 0.35
    win = box(u0, z0, u1, z1)

    near_zone, far_zone = fp.buffer(1.2), fp.buffer(0.45)
    keep = set(levels)
    near, beyond = [], []
    for d in meshes.values():
        cls = d["cls"]
        if cls not in NEAR or not (d["storey"] in keep or (d["storey"] == top and cls == "IfcSlab")):
            continue
        lo, hi = d["verts"].min(0), d["verts"].max(0)
        foot = box(lo[0], lo[1], hi[0], hi[1])
        if not near_zone.intersects(foot):
            continue
        c = cut(d, origin, normal, axes)
        if c is not None:
            near.append((cls, c))
        if cls in BEYOND and far_zone.intersects(foot):
            depth = float(d["verts"][:, :2].mean(0) @ n)
            if depth > mids[0] + 0.05:
                sil = silhouette(d, axes)
                if sil is not None:
                    beyond.append((depth, cls, d["name"], sil))

    lab_l = max(text_size(f"{lv}  FFL {zs[lv] - z_lo:+.2f}", 12)[0] for lv in levels + ([top] if top else []))
    rows = []
    for st, lv in zip(stairs, levels):
        for k, fl in enumerate(_flights(st, meshes)):
            v = meshes[fl.GlobalId]["verts"][:, 2]
            n_r, r, t = (getattr(fl, a, None) for a in ("NumberOfRisers", "RiserHeight", "TreadLength"))
            txt = f"{lv} flight {k + 1}" + (f": {n_r} risers x {r * 1000:.0f} mm, going {t * 1000:.0f} mm"
                                            if n_r and r and t else "")
            rows.append(((v.min() + v.max()) / 2, txt, FLIGHT_TEXT[min(k, 1)]))
    lab_r = max((text_size(t, 12)[0] for _, t, _ in rows), default=0)
    cv = Canvas((u0 - (lab_l + 20) / scale, z0 - 0.8, u1 + (lab_r + 30) / scale, z1), scale, margin=30, top=50)

    for _, cls, _, sil in sorted(beyond, key=lambda t: (-t[0], t[1], t[2])):     # farthest first
        paint(cv, sil.intersection(win), *BEYOND[cls])
    for cls in ("IfcSlab", "IfcWall", "IfcColumn", "IfcWindow", "IfcStairFlight", "IfcRailing", "IfcDoor"):
        for c_cls, c in near:
            if c_cls == cls:
                paint(cv, c.intersection(win), NEAR[cls])

    for lv in levels + ([top] if top else []):
        z = zs[lv]
        dashed(cv, (u0, z), (u1, z), (150, 150, 150))
        draw_text(cv, (u0 - 0.1, z + 0.08), f"{lv}  FFL {z - z_lo:+.2f}", 12, (80, 80, 80), "rs")
    for z, txt, col in rows:
        cv.line([(u1 + 0.05, z), (u1 + 0.25, z)], col, 1)
        draw_text(cv, (u1 + 0.3, z), txt, 12, col, "lm")
    sb = (u0, z0 - 0.45)
    cv.line([sb, (sb[0] + 2.0, sb[1])], (0, 0, 0), 3)
    for t in (0.0, 1.0, 2.0):
        cv.line([(sb[0] + t, sb[1] - 0.06), (sb[0] + t, sb[1] + 0.06)], (0, 0, 0), 1)
    draw_text(cv, (sb[0] + 1.0, sb[1] - 0.12), "2 m", 11, (0, 0, 0), "mt")
    head = title or Path(out_png).stem
    looking = COMPASS[int(round(np.degrees(np.arctan2(n[0], n[1])) / 45.0)) % 8]
    cv.d.text((cv.m, 8), f"{head}: {g['name']} section through the flight 1 centreline, {levels[0]} to {top or 'top'}, "
                         f"looking {looking}", fill=(30, 30, 30), font=font(15))
    cv.d.text((cv.m, 28), "dark: cut (flights brown, slabs and landings grey, walls black, doors red); "
                          "light: flight, balustrades and doors beyond", fill=(90, 90, 90), font=font(11))
    cv.save(out_png)
    return Path(out_png)


def stair_section(ifc_path, out_png, stair_name_prefix=None, storeys=4, data=None, title=None, scale=70.0):
    """Draw the section of one stair enclosure (the first one, or the first whose name after the storey, e.g.
    'stair 2', or full IfcStair name starts with stair_name_prefix). data=(ifc_file, meshes) skips loading."""
    if data is None:
        from estate.export import meshcache
        data = meshcache.load(ifc_path)
    f, meshes = data
    zs = storey_levels(f)
    groups = stair_groups(f, meshes, zs)
    if stair_name_prefix:
        groups = [g for g in groups if g["name"].startswith(stair_name_prefix) or
                  any((s.Name or "").startswith(stair_name_prefix) for s in g["stairs"])]
    return _section(f, meshes, zs, groups[0], out_png, storeys, title, scale) if groups else None


def stair_sections(ifc_path, out_dir, stem=None, storeys=4, data=None, scale=70.0):
    """One section per distinct stair enclosure: <stem>_stair_<n>.png in out_dir. Returns the paths written."""
    if data is None:
        from estate.export import meshcache
        data = meshcache.load(ifc_path)
    f, meshes = data
    stem = stem or Path(ifc_path).stem
    zs = storey_levels(f)
    out = []
    for k, g in enumerate(stair_groups(f, meshes, zs)):
        p = _section(f, meshes, zs, g, Path(out_dir) / f"{stem}_stair_{k + 1}.png", storeys, stem, scale)
        if p is not None:
            out.append(p)
    return out
