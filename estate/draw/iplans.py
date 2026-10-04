"""Floor plans cut from the IFC at FFL + 1.2 m (port of legacy/toolkit/draw_block.plan, generalised to any building,
storey and bounds; PIL instead of matplotlib).

The plates in raster.py are drawn from the derived plan that the IFC is authored from. These sheets are drawn from
the IFC itself, so they show what the file contains, a block hand-edited in Bonsai included. Everything is read
back from the model: walls, columns and glazing cut by the plane (elements wholly below it in outline, as a plan
shows them), spaces with their names and NetFloorArea, door swings from each IfcDoor's placement and OperationType
(SINGLE_SWING_LEFT hinges at local x = 0, RIGHT at x = OverallWidth, the leaf opens towards local +y), lift shafts
from the cars, flats from the IfcZones. The sheet is drawn in the building's own frame (its IfcBuilding placement)
with a north arrow.

draw_all() is the drawings-stage entry point: the stair sections plus L1, a typical floor and the roof.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import shapely
import ifcopenshell.util.element as uel
import ifcopenshell.util.placement as upl
from shapely.geometry import MultiPoint, box
from shapely.ops import polylabel

from estate.blocks.plate import CORRIDOR, DECK, LIFT, REFUSE, STAIR, VOID
from estate.draw.raster import FILL, FLAT_EDGE, WALL, Canvas, font
from estate.draw.sections import (cut, dashed, draw_text, natural, paint, silhouette, stair_sections, storey_levels,
                                  text_size)
from estate.rules import WALL_TYPES

CUT = {"IfcWall": (59, 59, 59), "IfcColumn": (45, 45, 45), "IfcRailing": (85, 85, 85),
       "IfcStairFlight": (176, 141, 87), "IfcRampFlight": (170, 170, 166), "IfcSlab": (200, 196, 186),
       "IfcTransportElement": (154, 165, 173), "IfcTank": (127, 167, 201), "IfcFurniture": (150, 140, 125)}
CUT_DEFAULT = (120, 120, 120)
BELOW = {"IfcStairFlight": ((233, 223, 199), (176, 150, 110)), "IfcRampFlight": ((228, 228, 224), (160, 160, 160)),
         "IfcSlab": ((226, 226, 224), (150, 150, 150)), "IfcRailing": (None, (110, 110, 110)),
         "IfcWall": ((175, 175, 175), (120, 120, 120)), "IfcFurniture": ((238, 234, 226), (150, 140, 125)),
         "IfcGeographicElement": ((214, 236, 204), (120, 160, 110)), "IfcKerb": ((220, 220, 215), (160, 160, 160))}
BELOW_DEFAULT = ((236, 236, 232), (160, 160, 160))
CUT_ORDER = ["IfcSlab", "IfcRampFlight", "IfcStairFlight", "IfcFurniture", "IfcRailing", "IfcTransportElement",
             "IfcTank", "IfcColumn", "IfcWall"]
BASE_SLABS = ("FLOOR", "BASESLAB", "ROOF")        # drawn under the spaces when below the cut
NOT_CUT = ("IfcSpace", "IfcDoor", "IfcWindow", "IfcOpeningElement", "IfcAnnotation", "IfcVirtualElement")
GLASS, GLASS_EDGE, GLASS_HIGH = (127, 179, 213), (60, 120, 170), (205, 226, 240)
DOOR, LIFT_DOOR, TREAD = (192, 57, 43), (150, 150, 150), (176, 150, 110)
PARKING = (226, 228, 230)
WALL_KEY = {v[0]: k for k, v in WALL_TYPES.items()}


# ----------------------------------------------------------------------------- frame, styles, labels
def building_frame(f, storey=None):
    """3 x 2 axes and offset taking world points to the frame of the building that holds `storey`."""
    b = uel.get_aggregate(storey) if storey is not None else None
    b = b if b is not None and b.is_a("IfcBuilding") else (f.by_type("IfcBuilding") or [None])[0]
    M = upl.get_local_placement(b.ObjectPlacement) if b is not None and b.ObjectPlacement else np.eye(4)
    R, t = M[:2, :2], M[:2, 3]
    return np.array([[R[0, 0], R[0, 1]], [R[1, 0], R[1, 1]], [0.0, 0.0]]), -t @ R


def space_fill(sp, psets):
    cat = psets.get("SampleCity_Room", {}).get("Category")
    if cat in FILL:
        return FILL[cat]
    s = f" {sp.LongName or ''} {sp.Name or ''} ".lower()
    if sp.PredefinedType == "PARKING" and " lot " in s:
        return PARKING
    for words, kind in ((("stair",), STAIR), (("refuse", "bin "), REFUSE), (("void deck",), VOID),
                        (("roof", "garden"), DECK), (("lobby", "corridor", "walkway", "passage", "aisle"), CORRIDOR)):
        if any(w in s for w in words):
            return FILL[kind]
    if "toilet" in s:
        return FILL["wet"]
    if sp.PredefinedType == "PARKING":
        return PARKING
    return FILL["outdoor"] if sp.PredefinedType == "EXTERNAL" else FILL["circulation"]


def _short(name, storey):
    if name.startswith("#") and " " in name:
        return name.split(" ", 1)[1]
    head, _, rest = name.partition("-")
    return rest if rest and head == storey else name


def _wall_colour(el):
    t = uel.get_type(el)
    return WALL.get(WALL_KEY.get(t.Name if t is not None else "", ""), CUT["IfcWall"])


def _fits(poly, s, size, scale):
    """Centre for label s (font size) inside poly, or None when it does not fit."""
    w, h = text_size(s, size)
    w, h = (w + 4) / scale, (h + 2) / scale
    part = max(getattr(poly, "geoms", [poly]), key=lambda g: g.area)
    if part.geom_type != "Polygon" or part.bounds[2] - part.bounds[0] < w or part.bounds[3] - part.bounds[1] < h:
        return None
    c = polylabel(part, tolerance=0.05)
    return (c.x, c.y) if part.buffer(0.1).contains(box(c.x - w / 2, c.y - h / 2, c.x + w / 2, c.y + h / 2)) else None


def _space_labels(sp, psets, poly, storey):
    """Label candidates, most informative first: long name + area, code + area, code."""
    long = (sp.LongName or "").replace(" / ", "/").replace("Household Shelter", "Shelter").replace(" shaft", "")
    code = psets.get("SampleCity_Room", {}).get("RoomCode") or _short(sp.Name or "", storey)
    if " lot " in f" {long.lower()} ":
        return [code]
    area = psets.get("Qto_SpaceBaseQuantities", {}).get("NetFloorArea") or poly.area
    return [s for s in (f"{long}\n{area:.1f} m²" if long else None, f"{code}\n{area:.1f} m²", f"{code}\n{area:.1f}",
                        code) if s]


def _zone_text(zn):
    flat = uel.get_psets(zn).get("SampleCity_Flat", {})
    if flat:
        ifa = flat.get("InternalFloorArea")
        return f"{zn.Name}  {flat.get('Template', '')}\n{zn.LongName or flat.get('FlatType', '')}" + \
            (f", {ifa:.0f} m² IFA" if ifa else "")
    return f"{zn.Name}\n{zn.LongName}" if zn.LongName else zn.Name


def place_outside(env, w, h, busy, centre, gap=0.65):
    """Box (w x h) beside env on the side where it overlaps `busy` least; ties go to the outward side."""
    x0, y0, x1, y1 = env.bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cand = {"N": box(cx - w / 2, y1 + gap, cx + w / 2, y1 + gap + h),
            "S": box(cx - w / 2, y0 - gap - h, cx + w / 2, y0 - gap),
            "E": box(x1 + gap, cy - h / 2, x1 + gap + w, cy + h / 2),
            "W": box(x0 - gap - w, cy - h / 2, x0 - gap, cy + h / 2)}
    pref = (["N", "S"] if cy >= centre[1] else ["S", "N"]) + (["E", "W"] if cx >= centre[0] else ["W", "E"])
    side = min(pref, key=lambda s: (round(cand[s].intersection(busy).area, 2) if busy is not None else 0.0,
                                    pref.index(s)))
    return cand[side]


def treads(d, axes, off, z_max=None):
    """Tread outlines of a flight: its horizontal upward faces grouped by level (only those below z_max)."""
    tri = d["verts"][d["faces"]]
    p = tri @ axes + off
    a = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
    flat = (a > 1e-9) & (np.ptp(tri[:, :, 2], axis=1) < 1e-4)
    zc = np.round(tri[flat][:, 0, 2], 3)
    out = []
    for zv in np.unique(zc):
        if z_max is None or zv < z_max:
            out.append(shapely.union_all(shapely.polygons(np.round(p[flat][zc == zv], 6)), grid_size=1e-5))
    return out


def arc_points(c, r, a, b, n=16):
    """Quarter circle about c from direction a to direction b."""
    return [c + r * (math.cos(t) * a + math.sin(t) * b) for t in np.linspace(0.0, math.pi / 2, n)]


# ----------------------------------------------------------------------------- the plan
def _collect(f, meshes, storey, z, axes, off):
    """Sort the storey's elements into spaces, base slabs, below-the-cut outlines, cut polygons and windows."""
    order = {c: i for i, c in enumerate(CUT_ORDER)}
    els = sorted(((g, d) for g, d in meshes.items() if d["storey"] == storey),
                 key=lambda t: (order.get(t[1]["cls"], -1), t[1]["name"], t[0]))
    g = dict(spaces=[], base=[], below=[], cuts=[], windows=[], treads=[])
    for gid, d in els:
        cls = d["cls"]
        zmin, zmax = float(d["verts"][:, 2].min()), float(d["verts"][:, 2].max())
        if cls == "IfcSpace":
            poly = silhouette(d, axes, off)
            if poly is not None:
                sp = f.by_guid(gid)
                g["spaces"].append((sp, poly, uel.get_psets(sp)))
        elif cls == "IfcWindow":
            g["windows"].append((MultiPoint(d["verts"] @ axes + off).convex_hull, zmin <= z <= zmax))
        elif cls in NOT_CUT or zmin > z:
            continue
        elif zmax < z:
            sil = silhouette(d, axes, off)
            if sil is None:
                continue
            if cls == "IfcStairFlight":
                g["treads"] += treads(d, axes, off)
            elif cls == "IfcSlab" and getattr(f.by_guid(gid), "PredefinedType", None) in BASE_SLABS:
                g["base"].append(sil)
            else:
                g["below"].append((zmax, cls, sil))
        else:
            c = cut(d, (0.0, 0.0, z), (0.0, 0.0, 1.0), axes, off)
            if c is None:
                continue
            colour = _wall_colour(f.by_guid(gid)) if cls == "IfcWall" else CUT.get(cls, CUT_DEFAULT)
            g["cuts"].append((cls, colour, c))
            if cls == "IfcStairFlight":                        # the treads below the cut, as the toolkit drew it
                g["treads"] += treads(d, axes, off, z)
            elif cls == "IfcRampFlight":
                sil = silhouette(d, axes, off)
                if sil is not None:
                    g["below"].append((zmin, cls, sil.difference(c)))
    return g


def _lifts(f, meshes, storey, axes, off):
    """(label, car footprint) of every lift serving this storey; the car marks its shaft on every floor."""
    out = []
    for gid, d in sorted(meshes.items(), key=lambda t: (t[1]["name"], t[0])):
        if d["cls"] != "IfcTransportElement":
            continue
        serves = uel.get_psets(f.by_guid(gid)).get("Navigation", {}).get("ServesLevels")
        if serves and storey not in str(serves).split(","):
            continue
        m = re.match(r"(lift \d+)", d["name"].lower())
        out.append(((m.group(1) if m else "lift").upper(), MultiPoint(d["verts"] @ axes + off).convex_hull))
    return out


def ifc_plan(ifc_path, storey, out_png, bounds=None, scale=None, title=None, data=None, cut_height=1.2):
    """Draw `storey` of the IFC cut at FFL + cut_height. bounds (building frame, metres) default to the storey's
    extent; scale (px per m) defaults to fit about 2400 x 1800 px. data=(ifc_file, meshes) skips loading."""
    if data is None:
        from estate.export import meshcache
        data = meshcache.load(ifc_path)
    f, meshes = data
    st = next((s for s in f.by_type("IfcBuildingStorey") if s.Name == storey), None)
    if st is None:
        raise KeyError(f"no storey {storey!r} in {ifc_path}")
    z = storey_levels(f)[storey] + cut_height
    axes, off = building_frame(f, st)
    R = axes[:2]                                    # world vector -> frame vector (row vector @ R)
    g = _collect(f, meshes, storey, z, axes, off)
    lifts = _lifts(f, meshes, storey, axes, off)
    content = [p for _, p, _ in g["spaces"]] + [c for _, _, c in g["cuts"]] + [s for _, _, s in g["below"]] + \
        [h for h, _ in g["windows"]] + g["base"]
    if not content:
        return None
    ext = shapely.union_all([box(*c.bounds) for c in content if not c.is_empty]).bounds
    if scale is None:
        w, h = (bounds[2] - bounds[0], bounds[3] - bounds[1]) if bounds else (ext[2] - ext[0] + 5, ext[3] - ext[1] + 7)
        scale = float(max(8.0, min(40.0, math.floor(min(2400.0 / w, 1800.0 / h) * 2) / 2)))
    fs = 11 if scale >= 34 else 10 if scale >= 27 else 9 if scale >= 21 else 8

    # flats (or any zone) on this storey, labelled outside their envelope where it collides least
    poly_of = {sp.GlobalId: p for sp, p, _ in g["spaces"]}
    zones = []
    for zn in sorted(f.by_type("IfcZone"), key=lambda z_: (natural(z_.Name), z_.GlobalId)):
        members = uel.get_grouped_by(zn)
        ps = [poly_of[s.GlobalId] for s in members if s.GlobalId in poly_of]
        if ps and not (len(members) == 1 and members[0].Name == zn.Name):    # a one-space zone is its space
            zones.append((zn, shapely.union_all(ps).buffer(0.2, join_style="mitre").buffer(-0.2, join_style="mitre")))
    solid = [p for _, p, _ in g["spaces"]] + [c for _, _, c in g["cuts"]] + [s for _, _, s in g["below"]] + \
        [h for h, _ in g["windows"]]
    busy = shapely.union_all(solid).buffer(0.25) if solid else None
    centre = (busy.centroid.x, busy.centroid.y) if busy is not None else (0.0, 0.0)
    zone_labels = []
    for k, (zn, env) in enumerate(zones):
        s = _zone_text(zn)
        w, h = text_size(s, fs + 1)
        spot = place_outside(env, (w + 6) / scale, (h + 4) / scale, busy, centre)
        busy = spot if busy is None else busy.union(spot)
        zone_labels.append((s, spot, FLAT_EDGE[k % len(FLAT_EDGE)]))

    pad, strip = 1.5, 80 / scale                    # strip under the plan for the scale bar and north arrow
    if bounds is None:
        allb = shapely.union_all([box(*ext)] + [b for _, b, _ in zone_labels]).bounds
        bounds = (allb[0] - pad, allb[1] - pad - strip, allb[2] + pad, allb[3] + pad)
    cv = Canvas(bounds, scale, margin=30, top=50)

    for b in g["base"]:
        paint(cv, b, (244, 244, 242), (175, 175, 175))
    for sp, poly, ps in g["spaces"]:
        paint(cv, poly, space_fill(sp, ps))
    for _, car in lifts:                             # shaft symbol: the car outline crossed
        paint(cv, car, FILL[LIFT], (120, 126, 130))
        x0, y0, x1, y1 = car.bounds
        cv.line([(x0, y0), (x1, y1)], (120, 126, 130), 1)
        cv.line([(x0, y1), (x1, y0)], (120, 126, 130), 1)
    for _, cls, s in sorted(g["below"], key=lambda t: t[0]):
        fill, edge = BELOW.get(cls, BELOW_DEFAULT)
        paint(cv, s, fill, edge)
    for t in g["treads"]:
        paint(cv, t, BELOW["IfcStairFlight"][0], TREAD)
    for _, colour, c in g["cuts"]:
        paint(cv, c, colour)
    for hull, is_cut in g["windows"]:
        paint(cv, hull, GLASS if is_cut else GLASS_HIGH, GLASS_EDGE if is_cut else GLASS)
    _doors(cv, f, meshes, st, R, off, scale)
    for k, (zn, env) in enumerate(zones):
        paint(cv, env, None, FLAT_EDGE[k % len(FLAT_EDGE)], 2)

    # labels: spaces, stairs without a space of their own, lifts, flats
    stair_spaces = []
    for sp, poly, ps in g["spaces"]:
        if "stair" in (sp.LongName or sp.Name or "").lower():
            stair_spaces.append(poly)
        for s in _space_labels(sp, ps, poly, storey):
            p = _fits(poly, s, fs, scale)
            if p is not None:
                draw_text(cv, p, s, fs, (60, 60, 60))
                break
    for stair in sorted(f.by_type("IfcStair"), key=lambda s: (s.Name or "", s.GlobalId)):
        if uel.get_container(stair) != st:
            continue
        parts = [meshes[p.GlobalId]["verts"] for p in uel.get_decomposition(stair) if p.GlobalId in meshes]
        if not parts:
            continue
        c = MultiPoint(np.vstack(parts) @ axes + off).convex_hull.centroid
        if not any(p.contains(c) for p in stair_spaces):
            name = stair.Name or "stair"
            draw_text(cv, (c.x, c.y), (name[len(storey) + 1:] if name.startswith(storey + " ") else name).upper(),
                      fs - 1, (90, 90, 90))
    for label, car in lifts:
        c = car.centroid
        draw_text(cv, (c.x, c.y), label, fs - 1, (60, 60, 60), bg=FILL[LIFT])
    for s, spot, colour in zone_labels:
        c = spot.centroid
        draw_text(cv, (c.x, c.y), s, fs + 1, colour)

    # sheet furniture: title, legend, scale bar, north arrow
    x0, y0, x1, y1 = bounds
    head = title or Path(out_png).stem
    cv.d.text((cv.m, 8), f"{head}: {storey} plan, cut at FFL + {cut_height:.1f} m (drawn from the IFC)",
              fill=(30, 30, 30), font=font(15))
    cv.d.text((cv.m, 28), "cut: walls and columns dark, glazing blue; below the cut in outline; red arcs: door swings; "
                          "spaces with NetFloorArea (m²); flats outlined from IfcZone", fill=(90, 90, 90), font=font(11))
    L = 5.0 if scale >= 20 else 10.0
    sb = np.array([x1 - pad - L, y0 + 25 / scale])
    cv.line([sb, sb + (L, 0)], (0, 0, 0), 3)
    for t in (0.0, L / 2, L):
        cv.line([sb + (t, -4 / scale), sb + (t, 4 / scale)], (0, 0, 0), 1)
    draw_text(cv, tuple(sb + (L / 2, 8 / scale)), f"{L:.0f} m", 11, (0, 0, 0), "mb")
    north, u = np.array([0.0, 1.0]) @ R, 1.0 / scale
    na = np.array([x0 + pad + 20 * u, y0 + 35 * u])
    side = np.array([-north[1], north[0]]) * 7 * u
    cv.line([na - north * 22 * u, na + north * 10 * u], (0, 0, 0), 2)
    cv.d.polygon([cv.P(na + north * 24 * u), cv.P(na + north * 6 * u + side), cv.P(na + north * 6 * u - side)],
                 fill=(0, 0, 0))
    draw_text(cv, tuple(na + side * 2.2 + north * 14 * u), "N", 12, (0, 0, 0))
    cv.save(out_png)
    return Path(out_png)


def swing_leaves(o, ux, uy, width, op, face=0.0):
    """(hinge, closed direction, radius) of each swinging leaf, the IFC way: SINGLE_SWING_LEFT hinges at local
    x = 0, RIGHT at x = width, double doors at both jambs; every leaf opens towards local +y from the wall face."""
    o, ux, uy = (np.asarray(v, float) for v in (o, ux, uy))
    if op.startswith("DOUBLE_DOOR") or op == "DOUBLE_SWING_DOUBLE_DOOR":
        leaves = [(o, ux, width / 2), (o + ux * width, -ux, width / 2)]
    elif op.endswith("RIGHT"):
        leaves = [(o + ux * width, -ux, width)]
    else:
        leaves = [(o, ux, width)]
    return [(h + uy * face, c, r) for h, c, r in leaves]


def door_symbol(door, meshes, R, off):
    """How to draw one IfcDoor in the plan frame: dict(kind: lift|rolling|sliding|swing, a, b, uy, leaves)."""
    M = upl.get_local_placement(door.ObjectPlacement)
    o_w, ux_w, uy_w = M[:2, 3], M[:2, 0], M[:2, 1]
    w = float(door.OverallWidth or 0.9)
    face, mid = 0.15, 0.05                       # host wall's opening-side face and centre, from the door origin
    try:
        wall = door.FillsVoids[0].RelatingOpeningElement.VoidsElements[0].RelatingBuildingElement
        v = meshes[wall.GlobalId]["verts"][:, :2]
        along = (v - o_w) @ ux_w
        across = (v[(along > -0.5) & (along < w + 0.5)] - o_w) @ uy_w
        if len(across):
            face, mid = float(np.clip(across.max(), 0.0, 0.6)), float(across.max() + across.min()) / 2
    except (IndexError, KeyError, AttributeError):
        pass
    o, ux, uy = o_w @ R + off, ux_w @ R, uy_w @ R
    op = door.OperationType or ""
    if op in ("", "NOTDEFINED", "USERDEFINED"):
        op = getattr(uel.get_type(door), "OperationType", None) or op
    kind = ("lift" if "lift" in (door.Name or "").lower() else "rolling" if "ROLLING" in op else
            "sliding" if ("SLIDING" in op or "FOLDING" in op) else "swing")
    return dict(kind=kind, a=o + uy * mid, b=o + ux * w + uy * mid, uy=uy,
                leaves=swing_leaves(o, ux, uy, w, op, face) if kind == "swing" else [])


def _doors(cv, f, meshes, st, R, off, scale):
    """Door leaves and swing arcs from each IfcDoor's placement and OperationType."""
    lw = max(1, int(scale * 0.04))
    doors = sorted((d for d in f.by_type("IfcDoor") if uel.get_container(d) == st),
                   key=lambda d: (d.Name or "", d.GlobalId))
    for door in doors:
        sym = door_symbol(door, meshes, R, off)
        a, b = sym["a"], sym["b"]
        if sym["kind"] == "lift":
            cv.line([a, b], LIFT_DOOR, max(2, int(scale * 0.05)))
        elif sym["kind"] == "rolling":
            dashed(cv, a, b, DOOR, lw, dash=0.25, gap=0.15)
        elif sym["kind"] == "sliding":
            d = sym["uy"] * 0.04
            cv.line([a + d, a + (b - a) * 0.55 + d], DOOR, lw)
            cv.line([b - d, a + (b - a) * 0.45 - d], DOOR, lw)
        for hinge, closed, r in sym["leaves"]:
            cv.line([hinge, hinge + sym["uy"] * r], DOOR, lw)
            cv.line(arc_points(hinge, r, closed, sym["uy"]), DOOR, 1)


# ----------------------------------------------------------------------------- the drawings stage
def plan_storeys(f):
    """L1, a typical floor (L5, else the middle storey) and the roof (RF, else the top storey), bottom to top."""
    zs = storey_levels(f)
    names = sorted(zs, key=zs.get)
    if not names:
        return []
    typical = "L5" if "L5" in names[1:-1] else names[len(names) // 2]
    first = "L1" if "L1" in names else names[0]
    roof = "RF" if "RF" in names else names[-1]
    return sorted(dict.fromkeys([first, typical, roof]), key=zs.get)


def draw_all(target_id, ifc_path, out_dir, storeys=None, data=None):
    """Stair sections (<id>_stair_<n>.png) and IFC-cut plans (<id>_<storey>_ifc.png) for L1, a typical floor and
    the roof. Returns the list of files written."""
    if data is None:
        from estate.export import meshcache
        data = meshcache.load(ifc_path)
    out_dir = Path(out_dir)
    out = list(stair_sections(ifc_path, out_dir, stem=target_id, data=data))
    for s in storeys or plan_storeys(data[0]):
        p = ifc_plan(ifc_path, s, out_dir / f"{target_id}_{s}_ifc.png", title=target_id, data=data)
        if p is not None:
            out.append(p)
    return out
