"""Turn floor plates into an IFC building (walls, openings, slabs, stairs, lifts, spaces, zones, ledges).

A typology module (point.py, slab.py, lblock.py) describes a building as a ``BuildingPlan``: one plate for the
void deck, one for the typical residential floor, one for the roof, the stair enclosures with their entry side
and the lift shafts. This module derives the walls of each plate once (geom/arrangement.py), checks every
opening against them, and instantiates the plates storey by storey with IfcWriter.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import Point, box
from shapely.ops import unary_union

import ifcopenshell.api.geometry as geometry

from estate.blocks.plate import (AMENITY, BALCONY, CORE_KINDS, CORRIDOR, DECK, LIFT, LOBBY, OUT, PLANT, REFUSE, ROOM,
                                 STAIR, VOID, Plate)
from estate.geom.arrangement import Run, finish_runs, merge_collinear, runs_from_faces, wall_footprint
from estate.flats.template import flat_ifa
from estate.geom.walls import Opening, WallSpec, frame, rotate_z, translate
from estate.ifc.stairs import flight_run, stair
from estate.rules import (DOOR_H, FTF, PARAPET_H, RAILING_H, SLAB, VOID_DECK_FTF, WALL_TYPES)

WET = {"wet"}
WINDOW_TO = {"out", CORRIDOR, LOBBY, BALCONY, VOID, DECK}   # a window may open to these (HDB: kitchen/bedroom windows on corridors)
TOWER_H = {STAIR: 3.0, LIFT: 3.2, REFUSE: 3.0, PLANT: 3.0}


class PlanError(Exception):
    pass


# ----------------------------------------------------------------------------- wall classification
def classify(plate: Plate, r: Run, level: str):
    """Set wall type, thickness and height mode of a run from the faces on either side."""
    if frozenset((r.left, r.right)) in plate.open_pairs:
        return
    fa = plate.faces.get(r.left)
    fb = plate.faces.get(r.right)
    ka = "out" if fa is None else fa.kind
    kb = "out" if fb is None else fb.kind
    kinds = {ka, kb}

    def set_(wall, height="storey", external=False, climbable=False):
        r.wall = wall
        r.t = WALL_TYPES[wall][1] if wall in WALL_TYPES else 0.05
        r.height, r.external, r.climbable = height, external, climbable

    # vertical cores win over everything
    if kinds & CORE_KINDS:
        core = fa if ka in CORE_KINDS else fb
        other_k = kb if core is fa else ka
        set_("CORE", "tower" if level == "roof" else "storey", external=other_k in ("out", DECK))
        r.meta["core_kind"] = core.kind
        return
    if level == "roof":
        if kinds == {DECK, "out"}:
            set_("PARAPET", "parapet", True, True)
        return
    if VOID in kinds:
        other = fb if ka == VOID else fa
        if other is not None and other.kind == AMENITY:
            set_("EXT", external=True)
        return                                   # void deck to outside / void: open
    if BALCONY in kinds:
        bal = fa if ka == BALCONY else fb
        other = fb if bal is fa else fa
        if other is None:
            r.wall, r.t, r.height, r.climbable = "RAILING", 0.05, "railing", True
        elif other.kind == ROOM and other.flat == bal.flat:
            set_("EXT", external=True)
        elif other.kind == BALCONY and other.flat == bal.flat:
            return
        else:
            set_("PARTY", external=True)
        return
    if kinds <= {CORRIDOR, LOBBY, "out"}:
        if "out" in kinds:
            set_("PARAPET", "parapet", True, True)
        return                                   # corridor to lobby: open
    rooms = [f for f in (fa, fb) if f is not None and f.kind in (ROOM, AMENITY)]
    if len(rooms) == 2:
        a, b = rooms
        if a.kind == AMENITY or b.kind == AMENITY:
            set_("EXT" if a.kind != b.kind else "INT")
        elif a.flat != b.flat:
            set_("PARTY")
        elif "shelter" in (a.meta.get("category"), b.meta.get("category")):
            set_(a.meta.get("shelter_wall", "HS250") if a.meta.get("category") == "shelter" else b.meta.get("shelter_wall", "HS250"))
        elif WET & {a.meta.get("category"), b.meta.get("category")}:
            set_("WET")
        else:
            set_("INT")
        return
    room = rooms[0]
    if room.meta.get("category") == "shelter":
        set_(room.meta.get("shelter_wall", "HS250"), external=True)
    else:
        set_("EXT", external=True)


# ----------------------------------------------------------------------------- derived plate
@dataclass
class PlacedOpening:
    run: int
    s0: float
    s1: float
    kind: str                 # door | window
    name: str
    door_kind: str = ""
    swing: int = 1
    hinge: str = "start"
    sill: float = 0.0
    height: float = DOOR_H
    meta: dict = field(default_factory=dict)


@dataclass
class DerivedPlate:
    plate: Plate
    level: str
    runs: list
    openings: list
    spaces: dict              # face id -> net polygon
    wall_union: object
    errors: list
    warnings: list
    dropped_windows: list


def derive(plate: Plate, level: str) -> DerivedPlate:
    errors, warnings, dropped = [], [], []
    runs = runs_from_faces({k: f.poly for k, f in plate.faces.items()})
    for r in runs:
        classify(plate, r, level)
    finish_runs(runs)
    walls = [r for r in runs if r.wall and r.wall != "RAILING"]
    wall_union = unary_union([wall_footprint(r) for r in walls]) if walls else shapely.Polygon()

    openings = []

    def fit(r, s, width, name):
        a, b = s - width / 2, s + width / 2
        if a < r.clear0 - 1e-6 or b > r.length - r.clear1 + 1e-6:
            errors.append(f"{name}: {width:.2f} m opening at {s:.2f} does not fit run {r.left}|{r.right} "
                          f"(length {r.length:.2f}, clear {r.clear0:.2f}/{r.clear1:.2f})")
            return None
        return a, b

    for d in plate.doors:
        pair = frozenset((d.a, d.b))
        cand = [(r, r.contains(d.at)) for r in runs if r.faces() == pair]
        cand = [(r, s) for r, s in cand if s is not None]
        if not cand:
            errors.append(f"door '{d.name}' ({d.a}|{d.b}) at {d.at}: no shared boundary there")
            continue
        r, s = cand[0]
        if not r.wall or r.wall == "RAILING":
            errors.append(f"door '{d.name}' ({d.a}|{d.b}): the boundary has no wall ({r.wall})")
            continue
        iv = fit(r, s, d.width, d.name)
        if iv is None:
            continue
        swing = 1 if d.into == r.left else -1
        sh = r.contains(d.hinge, tol=0.05)
        hinge = "start" if (sh is not None and sh < s) else "end"
        openings.append(PlacedOpening(r.idx, iv[0], iv[1], "door", d.name, d.kind, swing, hinge, 0.0, d.height,
                                      {"from": d.a, "to": d.b}))
    for w in plate.windows:
        cand = [(r, r.contains(w.at)) for r in runs if w.face in (r.left, r.right)]
        cand = [(r, s) for r, s in cand if s is not None]
        reason = None
        if not cand:
            reason = "not on the room boundary"
        else:
            r, s = cand[0]
            other = r.right if r.left == w.face else r.left
            ok_kind = plate.kind(other)
            if r.wall not in ("EXT", "HS250", "HS", "CORE") or ok_kind not in WINDOW_TO:
                reason = f"faces {ok_kind} ({r.wall})"
        if reason is None:
            iv = fit(r, s, w.width, w.name)
            if iv is None:
                continue
            openings.append(PlacedOpening(r.idx, iv[0], iv[1], "window", w.name, sill=w.sill, height=w.height,
                                          meta={"face": w.face, "kind": w.kind}))
        else:
            (errors if w.required else dropped).append(f"window '{w.name}' of {w.face}: {reason}")
    # overlapping openings on one run
    by_run = {}
    for o in openings:
        by_run.setdefault(o.run, []).append(o)
    for rid, ops in by_run.items():
        ops.sort(key=lambda o: o.s0)
        for a, b in zip(ops[:-1], ops[1:]):
            if b.s0 < a.s1 + 0.1 - 1e-6:
                errors.append(f"openings '{a.name}' and '{b.name}' overlap or sit closer than 0.1 m")
    runs, openings = merge_collinear(runs, openings)

    spaces = {}
    for fid, f in plate.faces.items():
        if f.kind in (LIFT, PLANT):
            continue
        net = f.poly.difference(wall_union) if not wall_union.is_empty else f.poly
        if net.geom_type == "MultiPolygon":
            parts = sorted(net.geoms, key=lambda g: -g.area)
            if parts[1].area > 0.05:
                warnings.append(f"space {fid} is split by walls ({[round(p.area, 2) for p in parts]})")
            net = parts[0]
        spaces[fid] = net
    return DerivedPlate(plate, level, runs, openings, spaces, wall_union, errors, warnings, dropped)


# ----------------------------------------------------------------------------- building plan
@dataclass
class StairSpec:
    face: str
    entry: str                 # N | S | E | W: side of the enclosure the door (and floor landing) is on
    name: str


@dataclass
class BuildingPlan:
    blk: str
    name: str
    long_name: str
    typology: str
    storeys: int               # storeys incl. the void deck (roof added on top)
    ground: Plate
    typical: Plate
    roof: Plate
    stairs: list               # StairSpec
    lifts: list                # lift face ids
    address: dict | None = None
    void_deck_uses: list = field(default_factory=list)
    roof_items: list = field(default_factory=list)
    columns: list | None = None
    meta: dict = field(default_factory=dict)

    @property
    def footprint(self):
        return unary_union([f.poly for f in self.typical.faces.values()]).buffer(0.1, join_style="mitre")

    def levels(self):
        names = [f"L{i}" for i in range(1, self.storeys + 1)] + ["RF"]
        ffl = [0.0, VOID_DECK_FTF] + [VOID_DECK_FTF + FTF * i for i in range(1, self.storeys)]
        return names, ffl[:len(names)]


def derive_plan(plan: BuildingPlan) -> dict:
    return {"ground": derive(plan.ground, "ground"), "typical": derive(plan.typical, "typical"),
            "roof": derive(plan.roof, "roof")}


def plan_errors(dp: dict) -> list:
    out = []
    for lvl, d in dp.items():
        out += [f"[{lvl}] {e}" for e in d.errors]
    return out


# ----------------------------------------------------------------------------- IFC instantiation
def _wallspec(r: Run, z, h, storey, name, openings):
    u = r.u
    p0 = np.asarray(r.p0, float) - u * r.ext0
    L = r.length + r.ext0 + r.ext1
    w = WallSpec(p0, u, L, r.t, -r.t / 2, z, h, name, r.wall, r.external, storey, r.climbable,
                 faces=(r.left, r.right))
    for o in openings:
        w.openings.append(Opening(o.s0 + r.ext0, o.s1 + r.ext0, o.sill, o.height, o.kind, o.name, o.door_kind,
                                  o.swing, dict(o.meta, hinge=o.hinge)))
    return w


def slab_polygon(plate: Plate, keep_cores=False):
    allf = unary_union([f.poly for f in plate.faces.values()]).buffer(0.1, join_style="mitre")
    if keep_cores:
        return allf
    holes = [f.poly.buffer(-0.1, join_style="mitre") for f in plate.faces.values() if f.kind in (LIFT, STAIR)]
    return allf.difference(unary_union(holes)) if holes else allf


def build_building(W, plan: BuildingPlan, dp: dict | None = None, placement=None, guid_prefix=None):
    """Author one building into writer W. Returns (building, info dict)."""
    dp = dp or derive_plan(plan)
    errs = plan_errors(dp)
    if errs:
        raise PlanError(f"Blk {plan.blk}: " + "; ".join(errs[:12]) + (f" (+{len(errs) - 12} more)" if len(errs) > 12 else ""))
    m = W.m
    names, ffl = plan.levels()
    bldg = W.building(plan.name, plan.long_name, plan.address)
    W.props(bldg, "SampleCity_Building", {"Block": plan.blk, "Typology": plan.typology, "Storeys": plan.storeys,
                                         "Flats": len(plan.typical.flats) * (plan.storeys - 1)})
    storeys = W.storeys(bldg, names, ffl)
    runs_next = [flight_run(ffl[i + 1] - ffl[i]) for i in range(len(ffl) - 1)]
    zones, flat_records = [], []

    for li, (st, z) in enumerate(zip(storeys, ffl)):
        nm = names[li]
        is_roof = nm == "RF"
        top = ffl[li + 1] if li + 1 < len(ffl) else None
        sh = (top - z) if top is not None else None
        d = dp["roof"] if is_roof else dp["ground"] if li == 0 else dp["typical"]
        plate = d.plate

        # ---- slabs
        if li == 0:
            W.slab(slab_polygon(plate, keep_cores=True), 0.0, "L1 void deck slab", st, "BASESLAB")
        elif is_roof:
            W.slab(slab_polygon(dp["typical"].plate), z, "Roof slab", st, "ROOF", "roof")
        else:
            W.slab(slab_polygon(plate), z, f"{nm} floor slab", st, "FLOOR")

        # ---- walls with openings
        ops_by_run, wall_names = {}, {}
        for o in d.openings:
            ops_by_run.setdefault(o.run, []).append(o)
        for r in d.runs:
            if not r.wall:
                continue
            if r.wall == "RAILING":
                _railing(W, r, z, RAILING_H, st, f"{nm} balcony railing {r.idx}")
                continue
            if r.height == "storey":
                h = sh - SLAB
            elif r.height == "parapet":
                h = PARAPET_H
            else:
                h = TOWER_H.get(r.meta.get("core_kind"), 3.0)
            ops = [o for o in ops_by_run.get(r.idx, [])]
            for o in ops:
                o.name_full = f"{_unit(nm, plate, o)}{o.name}"
            wname = f"{nm} {r.wall} {r.left}|{r.right}"
            wall_names[wname] = wall_names.get(wname, 0) + 1
            if wall_names[wname] > 1:            # the same pair of faces meets on several runs
                wname = f"{wname} ({wall_names[wname]})"
            w = _wallspec(r, z, h, st, wname, [])
            w.meta["key"] = f"{nm}/{r.wall}/{r.p0[0]:.3f},{r.p0[1]:.3f}/{r.p1[0]:.3f},{r.p1[1]:.3f}"
            for o in ops:   # the derived openings are shared by every storey: build per-storey metadata
                meta = dict(o.meta, hinge=o.hinge)
                meta.update({k: space_name(nm, li, plate, o.meta[k]) for k in ("from", "to") if k in o.meta})
                w.openings.append(Opening(o.s0 + r.ext0, o.s1 + r.ext0, o.sill, o.height, o.kind, o.name_full,
                                          o.door_kind, o.swing, meta))
            W.build_wall(w)

        # ---- stairs to the next level
        if top is not None:
            for s in plan.stairs:
                b = plate.faces[s.face].poly.bounds if s.face in plate.faces else dp["typical"].plate.faces[s.face].poly.bounds
                x0, y0, x1, y1 = b
                inner = (x0 + 0.1, x1 - 0.1, y1 - 0.1, y0 + 0.1)
                rn = runs_next[li + 1] if li + 1 < len(runs_next) else None
                stair(W, f"{nm} {s.name}", inner, z, top, st, rn, entry=s.entry)

        # ---- ground: columns, lift cars, void deck furniture
        if li == 0:
            _columns(W, plan, dp, st, sh - SLAB)
            _furniture(W, plan, st, "ground")
            for k, lf in enumerate(plan.lifts):
                lb = plate.faces[lf].poly.buffer(-0.1, join_style="mitre").bounds
                car = box(lb[0] + 0.25, lb[1] + 0.25, lb[2] - 0.25, lb[3] - 0.25)
                el = W.product("IfcTransportElement", f"Lift {k + 1} car", W.extrusion(car, 2.4, car.centroid.coords[0]),
                               translate(*car.centroid.coords[0], 0.05), st, "ELEVATOR", "lift")
                W.props(el, "Navigation", {"Elevator": True, "ServesLevels": ",".join(names[:-1]), "Shaft": lf})

        # ---- spaces and zones
        level_h = (sh - SLAB) if sh else 2.4
        by_flat = {}
        for fid, net in d.spaces.items():
            f = plate.faces[fid]
            if f.kind in (ROOM, BALCONY):
                uid = f"#{li + 1:02d}-{f.flat}"
                sp = W.space(net, z, level_h, f"{uid} {f.room}", f.name, st, external=f.kind == BALCONY, flat=uid,
                             room=f.room, extra={"Category": f.meta.get("category", "")})
                by_flat.setdefault(f.flat, []).append(sp)
            elif f.kind in (CORRIDOR, LOBBY):
                W.space(net, z, level_h, f"{nm}-{fid}", f.name or "Common corridor", st)
            elif f.kind == VOID:
                W.space(net, z, level_h, f"{nm}-{fid}", f.name or "Void deck", st, external=True)
            elif f.kind == AMENITY:
                W.space(net, z, level_h, f"{nm}-{fid}", f.name, st)
            elif f.kind == DECK:
                W.space(net, z, 2.4, f"RF-{fid}", "Roof deck", st, external=True)
            elif f.kind in (STAIR, REFUSE):
                W.space(net, z, 3.0 if is_roof else level_h, f"{nm}-{fid}", f.name or fid, st)
        for stack, sps in by_flat.items():
            fi = plate.flats[stack]
            uid = f"#{li + 1:02d}-{stack}"
            ifa = round(flat_ifa(fi, plate), 2)
            zn = W.zone(uid, fi.long_name, sps, {
                "FlatType": fi.flat_type, "UnitNumber": uid, "Block": plan.blk, "Stack": stack, "Storey": nm,
                "Template": fi.template, "Variant": json.dumps({k: v for k, v in fi.variant.items() if k != "canonical"},
                                                               sort_keys=True),
                "Signature": fi.signature, "CanonicalSignature": fi.variant.get("canonical", ""),
                "GrossArea": fi.gross_area, "InternalFloorArea": ifa})
            zones.append(zn)
            flat_records.append(dict(blk=plan.blk, unit=uid, storey=nm, stack=stack, type=fi.flat_type,
                                     template=fi.template, signature=fi.signature,
                                     canonical=fi.variant.get("canonical", ""), ifa=ifa, gross=fi.gross_area,
                                     variant={k: v for k, v in fi.variant.items() if k != "canonical"}))

        # ---- AC ledges and bay-window ledges
        for led in plate.ledges:
            uid = f"#{li + 1:02d}-{led.flat}" if led.flat else nm
            bay = led.kind == "bay"
            el = W.slab(led.poly, z, f"{uid} {led.name}", st, "USERDEFINED", "slab", 0.15,
                        "Bay window ledge" if bay else "AC ledge")
            W.props(el, "Navigation", {"ClimbableTopEdge": not bay, "TopEdgeHeightAboveFloor": 0.0})
            for band in led.rail:
                c = band.centroid
                h = 1.0
                rl = W.product("IfcRailing", f"{uid} {led.name} {'glazing' if bay else 'railing'}",
                               W.extrusion(band, h, (c.x, c.y)), translate(c.x, c.y, z), st, "GUARDRAIL",
                               "glass" if bay else "alu")
                W.props(rl, "Navigation", {"ClimbableTopEdge": not bay, "TopEdgeHeightAboveFloor": h})

        # ---- roof extras
        if is_roof:
            for k, (kind, poly, h) in enumerate(plan.roof_items):
                if kind == "tank":
                    el = W.box_element("IfcTank", "Roof water tank", poly, z, h, st, "STORAGE", "tank")
                    W.props(el, "Navigation", {"ClimbableTopEdge": True, "TopEdgeHeightAboveFloor": h})
            for s in plan.stairs:
                W.slab(plate.faces[s.face].poly.buffer(0.1, join_style="mitre"), z + 3.2, f"{s.name} tower roof", st,
                       "ROOF", "roof")
            for k, lf in enumerate(plan.lifts):
                W.slab(plate.faces[lf].poly.buffer(0.1, join_style="mitre"), z + 3.4, f"Lift {k + 1} motor room roof",
                       st, "ROOF", "roof")

    W.flush()
    if placement is not None:
        geometry.edit_object_placement(m, product=bldg, matrix=placement, should_transform_children=True)
    return bldg, {"levels": dict(zip(names, ffl)), "flats": flat_records, "zones": len(zones)}


def space_name(nm, li, plate, fid):
    """Name of the IfcSpace a plate face becomes on a storey (doors record FromSpace / ToSpace with it)."""
    f = plate.faces.get(fid)
    if f is None:
        return "outside"
    if f.kind in (ROOM, BALCONY):
        return f"#{li + 1:02d}-{f.flat} {f.room}"
    if f.kind == DECK:
        return f"RF-{fid}"
    if f.kind in (LIFT, PLANT):
        return f"{nm}-{fid} (shaft)"
    return f"{nm}-{fid}"


def _furniture(W, plan, st, level):
    """Void-deck furniture from plan.meta['furniture']: dicts(name, poly, height, object_type, level, mat)."""
    for item in plan.meta.get("furniture", []):
        if item.get("level", "ground") != level:
            continue
        el = W.box_element("IfcFurniture", item["name"], item["poly"], item.get("z", 0.0), item.get("height", 1.8), st,
                           "USERDEFINED", item.get("mat", "steel"), item.get("object_type", item["name"]))
        W.props(el, "Navigation", {"Obstacle": True, "ClimbableTopEdge": False})
        if item.get("boxes"):
            W.props(el, "SampleCity_LetterboxBank", {"Letterboxes": int(item["boxes"])})


def _unit(nm, plate, o):
    f = plate.faces.get(o.meta.get("from")) or plate.faces.get(o.meta.get("face"))
    if f is not None and f.flat:
        return f"#{int(nm[1:]):02d}-{f.flat} " if nm.startswith("L") else ""
    return f"{nm} "


def _rail_line(W, p, q, z, h, st, name, pre="GUARDRAIL", t=0.05):
    p, q = np.asarray(p, float), np.asarray(q, float)
    u = (q - p) / np.linalg.norm(q - p)
    n = np.array([-u[1], u[0]])
    poly = shapely.Polygon([p - n * t / 2, q - n * t / 2, q + n * t / 2, p + n * t / 2])
    c = poly.centroid
    rl = W.product("IfcRailing", name, W.extrusion(poly, h, (c.x, c.y)), translate(c.x, c.y, z), st, pre, "alu")
    W.props(rl, "Navigation", {"ClimbableTopEdge": True, "TopEdgeHeightAboveFloor": h})
    return rl


def _railing(W, r: Run, z, h, st, name):
    return _rail_line(W, r.p0, r.p1, z, h, st, name, "GUARDRAIL")


def _columns(W, plan: BuildingPlan, dp, st, h):
    """Void deck columns under the residential envelope: at facade corners, party-wall ends and every <= 7.2 m."""
    from estate.blocks import plate_ops
    m = W.m
    ground = dp["ground"].plate
    if plan.columns is None:      # the same rule the letterbox placement keeps clear of (blocks/plate_ops.py)
        kept = plate_ops.column_points(dp["typical"], ground)
    else:
        blocked = unary_union([f.poly.buffer(plate_ops.COLUMN_KEEP_OFF) for f in ground.faces.values()
                               if f.kind != VOID])
        kept = []
        for p in sorted(set((round(x, 2), round(y, 2)) for x, y in plan.columns)):
            if blocked.contains(Point(p)) or any(math.hypot(p[0] - q[0], p[1] - q[1]) < plate_ops.COLUMN_SPACING
                                                 for q in kept):
                continue
            kept.append(p)
    prof = m.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=0.6, YDim=0.6)
    for k, p in enumerate(kept, 1):
        rep = geometry.add_profile_representation(m, context=W.body, profile=prof, depth=h)
        c = W.product("IfcColumn", f"L1 column C{k:02d}", rep, translate(p[0], p[1], 0.0), st, "COLUMN", "column")
        W.props(c, "Pset_ColumnCommon", {"IsExternal": True, "LoadBearing": True})
    return kept
