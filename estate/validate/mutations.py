"""Mutation testing of the validators: inject one known defect into an in-memory copy and require a check to fire.

A validator that never fails proves nothing. Each mutation below opens a fresh copy of a building file, breaks one
thing the way a generator bug or a careless hand edit in Bonsai would (an unfilled opening, a duplicated GlobalId,
a raised riser, a slab across a lift shaft, a shelter taken out of its flat, a bathroom vent window deleted, a
different map conversion in one federated file, ...), reruns the check family it targets (ifcqa / programme /
geometry / federation) and compares the per-check failure counts with the unmutated baseline. A mutation counts as detected when one of its expected checks reports more failures than in
the baseline. Geometry mutations re-tessellate only the elements they touch on top of the cached meshes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import ifcopenshell
import ifcopenshell.api.aggregate as aggregate
import ifcopenshell.api.feature as feature
import ifcopenshell.api.geometry as geometry
import ifcopenshell.api.georeference as georeference
import ifcopenshell.api.group as group
import ifcopenshell.api.pset as pset
import ifcopenshell.api.root as root
import ifcopenshell.api.spatial as spatial
import ifcopenshell.util.element as uel
import ifcopenshell.util.placement as upl

from estate import env, guids
from estate.export import meshcache
from estate.validate import geometry as geom
from estate.validate import ifcqa
from estate.validate import programme


@dataclass
class Mutation:
    name: str
    family: str                       # ifcqa | programme | geometry | federation
    expect: tuple                     # checks of which at least one must report more failures
    describe: str
    apply: object = field(repr=False)


# ----------------------------------------------------------------------------- helpers
def _named(f, cls, prefix=None, pred=None):
    els = sorted(f.by_type(cls), key=lambda e: (e.Name or "", e.id()))
    for e in els:
        if (prefix is None or (e.Name or "").startswith(prefix)) and (pred is None or pred(e)):
            return e
    for e in els:
        if pred is None or pred(e):
            return e
    raise LookupError(f"no {cls} {prefix or ''}")


def _mid_storey(f):
    sts = sorted(f.by_type("IfcBuildingStorey"), key=lambda s: upl.get_local_placement(s.ObjectPlacement)[2, 3])
    return sts[len(sts) // 2]


def _in_storey(e, storey):
    c = uel.get_container(e) or uel.get_aggregate(e)
    while c is not None and not c.is_a("IfcBuildingStorey"):
        c = uel.get_container(c) or uel.get_aggregate(c)
    return c == storey


def _body(f):
    for c in f.by_type("IfcGeometricRepresentationSubContext"):
        if c.ContextIdentifier == "Body":
            return c
    return f.by_type("IfcGeometricRepresentationContext")[0]


def _box(f, cls, name, x0, y0, x1, y1, z0, h, storey, predefined=None):
    """A prismatic element (rectangle extrusion) placed in world coordinates."""
    el = root.create_entity(f, cls, name=name, predefined_type=predefined)
    M = np.eye(4)
    M[:3, 3] = ((x0 + x1) / 2, (y0 + y1) / 2, z0)
    geometry.edit_object_placement(f, product=el, matrix=M)
    prof = f.createIfcRectangleProfileDef("AREA", None, None, float(x1 - x0), float(y1 - y0))
    solid = f.createIfcExtrudedAreaSolid(prof, None, f.createIfcDirection((0.0, 0.0, 1.0)), float(h))
    geometry.assign_representation(f, product=el, representation=f.createIfcShapeRepresentation(
        _body(f), "Body", "SweptSolid", [solid]))
    if cls == "IfcSpace":
        aggregate.assign_object(f, products=[el], relating_object=storey)
    else:
        spatial.assign_container(f, products=[el], relating_structure=storey)
    return el


_BASE = {}          # meshes of the unmutated file (set by run), reused while picking mutation targets


def _bounds(f, el):
    m = _BASE.get(el.GlobalId)
    if m is None:
        m = meshcache.tessellate(f, num_threads=1, include=[el])[el.GlobalId]
    return m["verts"]


def _move(f, el, dx=0.0, dy=0.0, dz=0.0):
    M = upl.get_local_placement(el.ObjectPlacement).copy()
    M[:3, 3] += (dx, dy, dz)
    geometry.edit_object_placement(f, product=el, matrix=M)


def _flight(f):
    st = _mid_storey(f)
    return _named(f, "IfcStairFlight", f"{st.Name} ", pred=lambda e: _in_storey(e, st))


# ----------------------------------------------------------------------------- ifcqa mutations
def m_remove_filling(f):
    d = _named(f, "IfcDoor", "L5")
    f.remove(d.FillsVoids[0])
    return {}


def m_duplicate_guid(f):
    a, b = sorted(f.by_type("IfcWall"), key=lambda e: e.id())[:2]
    b.GlobalId = a.GlobalId
    return {}


def m_uncontain_wall(f):
    spatial.unassign_container(f, products=[_named(f, "IfcWall", "L5")])
    return {}


def m_space_out_of_zone(f):
    sp = _named(f, "IfcSpace", "#05", pred=lambda s: ifcqa._zone_of(s) is not None)
    group.unassign_group(f, products=[sp], group=ifcqa._zone_of(sp))
    return {}


def m_strip_navigation(f):
    d = _named(f, "IfcDoor", "L5")
    for rel in list(d.IsDefinedBy):
        p = rel.RelatingPropertyDefinition
        if p.is_a("IfcPropertySet") and p.Name == "Navigation":
            pset.remove_pset(f, product=d, pset=p)
    return {}


def m_mirror_mapped(f):
    items = sorted(f.by_type("IfcMappedItem"), key=lambda e: e.id())
    windows = [i for i in items if any(p.is_a("IfcWindow") for r in f.get_inverse(i) for pd in f.get_inverse(r)
                                       for p in f.get_inverse(pd))]
    it = (windows or items)[len(windows or items) // 2]
    it.MappingTarget = f.createIfcCartesianTransformationOperator3D(
        f.createIfcDirection((1.0, 0.0, 0.0)), f.createIfcDirection((0.0, -1.0, 0.0)),
        f.createIfcCartesianPoint((0.0, 0.0, 0.0)), 1.0, f.createIfcDirection((0.0, 0.0, 1.0)))
    return {}


def m_degenerate_placement(f):
    w = _named(f, "IfcWall", "L5")
    rp = w.ObjectPlacement.RelativePlacement
    rp.RefDirection = f.createIfcDirection((0.0, 0.0, 1.0))
    rp.Axis = f.createIfcDirection((0.0, 0.0, 1.0))
    return {}


def m_ifa_out_of_band(f):
    z = _named(f, "IfcZone", "#05", pred=lambda z: bool(uel.get_pset(z, "SampleCity_Flat")))
    p = uel.get_pset(z, "SampleCity_Flat")
    band = ifcqa.rules_config().get("flat", {}).get("ifa_band", {}).get(p.get("FlatType"), [80.0, 90.0])
    pset.edit_pset(f, pset=f.by_id(p["id"]), properties={"InternalFloorArea": float(band[0]) - 10.0})
    return {}


def m_bad_flat_type(f):
    z = _named(f, "IfcZone", "#06", pred=lambda z: bool(uel.get_pset(z, "SampleCity_Flat")))
    pset.edit_pset(f, pset=f.by_id(uel.get_pset(z, "SampleCity_Flat")["id"]), properties={"FlatType": "4-Room"})
    return {}


def m_bad_room_code(f):
    _, s = _flat_room(programme.Index(f), "B2")
    pset.edit_pset(f, pset=f.by_id(uel.get_pset(s.el, "SampleCity_Room")["id"]), properties={"RoomCode": "Bedroom"})
    return {}


def m_bad_storey_name(f):
    _mid_storey(f).Name = "Level 5"
    return {}


def m_orphan_opening(f):
    o = sorted(f.by_type("IfcOpeningElement"), key=lambda e: e.id())[len(f.by_type("IfcOpeningElement")) // 2]
    f.remove(o.VoidsElements[0])
    return {}


def m_too_many_openings(f):
    w = _named(f, "IfcWall", "L5")
    for k in range(ifcqa.budget_config()["max_openings_per_wall"] + 1):
        o = root.create_entity(f, "IfcOpeningElement", name=f"extra opening {k}")
        geometry.edit_object_placement(f, product=o, matrix=upl.get_local_placement(w.ObjectPlacement))
        feature.add_feature(f, feature=o, element=w)
    return {}


def m_share_representation(f):
    a, b = [w for w in sorted(f.by_type("IfcWall"), key=lambda e: e.id()) if w.Representation][:2]
    body = [r for r in a.Representation.Representations if r.RepresentationIdentifier == "Body"][0]
    b.Representation.Representations = [body if r.RepresentationIdentifier == "Body" else r
                                        for r in b.Representation.Representations]
    return {}


# ----------------------------------------------------------------------------- programme mutations
def _flat_room(X, code, pred=None, prefix="#05"):
    """A room of the given code in the first flat (by name, storey prefix first) that satisfies pred(flat, space)."""
    for want in (prefix, None):
        for fl in X.flats:
            if want and not fl.name.startswith(want):
                continue
            s = fl.room(code)
            if s is not None and (pred is None or pred(fl, s)):
                return fl, s
    raise LookupError(f"no flat room {code}")


def _windows(X, s):
    return [o for o in X.served.get(s, ()) if o.kind == "window"]


def _remove_filled(f, el):
    """Delete a window / door together with the opening it fills."""
    op = el.FillsVoids[0].RelatingOpeningElement if el.FillsVoids else None
    root.remove_product(f, product=el)
    if op is not None:
        root.remove_product(f, product=op)


def m_shelter_out_of_zone(f):
    _, hs = _flat_room(programme.Index(f), "HS")
    group.unassign_group(f, products=[hs.el], group=ifcqa._zone_of(hs.el))
    return {}


def m_shrink_room_area(f):
    _, s = _flat_room(programme.Index(f), "B2")
    q = uel.get_pset(s.el, "Qto_SpaceBaseQuantities")
    amin = ifcqa.rules_config().get("room", {}).get("min_area", {}).get("B2", 8.5)
    pset.edit_qto(f, qto=f.by_id(q["id"]), properties={"NetFloorArea": float(amin) - 1.0})
    return {}


def m_shrink_shelter(f):
    _, hs = _flat_room(programme.Index(f), "HS")
    q = uel.get_pset(hs.el, "Qto_SpaceBaseQuantities")
    pset.edit_qto(f, qto=f.by_id(q["id"]), properties={"NetFloorArea": 1.2})
    return {}


def m_remove_bath_vent(f):
    X = programme.Index(f)
    vent = float(ifcqa.rules_config().get("glazing", {}).get("vent_min_area", 0.2))
    for code in sorted(programme.BATHS):
        try:
            _, s = _flat_room(X, code, lambda fl, s: s.code not in fl.mech_vent
                              and sum(o.width * o.height for o in _windows(X, s)) >= vent)
        except LookupError:
            continue
        for w in _windows(X, s):
            _remove_filled(f, w.el)
        return {}
    raise LookupError("no naturally ventilated bathroom")


def m_shrink_bedroom_windows(f):
    X = programme.Index(f)
    _, s = _flat_room(X, "B2", lambda fl, s: bool(_windows(X, s)))
    for w in _windows(X, s):
        w.el.OverallHeight = 0.2
    return {}


def m_stale_door_spaces(f):
    X = programme.Index(f)
    fl, b2 = _flat_room(X, "B2")
    d = next(d for d in X.flat_doors[fl] if b2 in d.sides and all(isinstance(x, programme.Space) for x in d.probed))
    other = next(s for _, v in sorted(fl.rooms.items()) for s in v if s not in d.probed)
    nav = uel.get_pset(d.el, "Navigation")
    pset.edit_pset(f, pset=f.by_id(nav["id"]), properties={"FromSpace": b2.name, "ToSpace": other.name})
    return {}


def m_remove_main_door(f):
    X = programme.Index(f)
    fl, _ = _flat_room(X, "HS", lambda fl, s: any(d.door_kind == "main" for d in X.flat_doors.get(fl, ())))
    for d in X.flat_doors[fl]:
        if d.door_kind == "main":
            _remove_filled(f, d.el)
    return {}


# ----------------------------------------------------------------------------- geometry mutations
def m_raise_riser(f):
    fl = _flight(f)
    r = float(fl.RiserHeight or uel.get_pset(fl, "Pset_StairFlightCommon").get("RiserHeight"))
    solid = fl.Representation.Representations[0].Items[0]
    k = 3
    for p in solid.SweptArea.OuterCurve.Points:
        x, y = p.Coordinates
        if abs(y - k * r) < 1e-6:
            p.Coordinates = (x, y + 0.02)
    return {"changed": [fl]}


def m_delete_flight(f):
    fl = _flight(f)
    g = fl.GlobalId
    root.remove_product(f, product=fl)
    return {"removed": [g]}


def m_delete_balustrade(f):
    fl = _flight(f)
    st = uel.get_aggregate(fl)
    rails = [p for p in uel.get_parts(st) if p.is_a("IfcRailing") and p.Name.startswith(fl.Name)]
    g = rails[0].GlobalId
    root.remove_product(f, product=rails[0])
    return {"removed": [g]}


def m_overlapping_wall(f):
    st = _mid_storey(f)
    w = max((w for w in f.by_type("IfcWall") if _in_storey(w, st)), key=lambda w: (np.ptp(_bounds(f, w)[:, :2], 0).max(), -w.id()))
    new = root.create_entity(f, "IfcWall", name=f"{w.Name} duplicate")
    M = upl.get_local_placement(w.ObjectPlacement).copy()
    M[:3, 3] += M[:3, 0] * 0.3
    geometry.edit_object_placement(f, product=new, matrix=M)
    body = [r for r in w.Representation.Representations if r.RepresentationIdentifier == "Body"][0]
    geometry.assign_representation(f, product=new, representation=ifcopenshell.util.element.copy_deep(f, body))
    spatial.assign_container(f, products=[new], relating_structure=st)
    return {"changed": [new]}


def m_slab_in_shaft(f):
    car = _named(f, "IfcTransportElement", pred=lambda e: e.PredefinedType in ("ELEVATOR", "LIFT"))
    v = _bounds(f, car)
    st = _mid_storey(f)
    z = upl.get_local_placement(st.ObjectPlacement)[2, 3]
    (x0, y0), (x1, y1) = v[:, :2].min(0), v[:, :2].max(0)
    s = _box(f, "IfcSlab", "Slab across the lift shaft", x0, y0, x1, y1, z - 0.2, 0.2, st, "FLOOR")
    return {"changed": [s]}


def m_remove_parapet(f):
    walls = [w for w in f.by_type("IfcWall") if uel.get_predefined_type(w) == "PARAPET"]
    roof = max({uel.get_container(w) for w in walls}, key=lambda s: upl.get_local_placement(s.ObjectPlacement)[2, 3])
    w = max((w for w in walls if uel.get_container(w) == roof), key=lambda w: (np.ptp(_bounds(f, w)[:, :2], 0).max(), -w.id()))
    g = w.GlobalId
    root.remove_product(f, product=w)
    return {"removed": [g]}


def m_overlapping_space(f):
    sp = _named(f, "IfcSpace", "#05", pred=lambda s: s.LongName and "Living" in s.LongName)
    v = _bounds(f, sp)
    st = uel.get_aggregate(sp)
    (x0, y0, z0), (x1, y1, z1) = v.min(0), v.max(0)
    s = _box(f, "IfcSpace", f"{sp.Name} copy", x0 + 0.5, y0, x1 + 0.5, y1, z0, z1 - z0, st)
    return {"changed": [s]}


def m_shift_wet_room(f):
    sp = _named(f, "IfcSpace", "#07", pred=lambda s: (uel.get_pset(s, "SampleCity_Room") or {}).get("RoomCode") == "CB")
    _move(f, sp, dx=0.6)
    return {"changed": [sp]}


def m_door_clash(f):
    d = _named(f, "IfcDoor", "#05")
    host = d.FillsVoids[0].RelatingOpeningElement.VoidsElements[0].RelatingBuildingElement
    st = uel.get_container(d)
    dv = _bounds(f, d)
    c = (dv.min(0) + dv.max(0)) / 2
    best = None
    for w in f.by_type("IfcWall"):
        if w == host or uel.get_container(w) != st:
            continue
        wv = _bounds(f, w)
        wc = (wv.min(0) + wv.max(0)) / 2
        dist = float(np.hypot(*(wc - c)[:2]))
        if best is None or dist < best[0]:
            best = (dist, wc)
    _move(f, d, dx=best[1][0] - c[0], dy=best[1][1] - c[1])
    return {"changed": [d]}


# ----------------------------------------------------------------------------- federation
def _with_georef(f, cfg_georef=None):
    if not f.by_type("IfcMapConversion"):
        g = cfg_georef or {"epsg": 3414, "eastings": 30000.0, "northings": 36000.0, "height": 15.0}
        georeference.add_georeferencing(f, ifc_class="IfcMapConversion", name=f"EPSG:{g['epsg']}")
        georeference.edit_georeferencing(f, projected_crs={"Name": f"EPSG:{g['epsg']}"}, coordinate_operation={
            "Eastings": float(g["eastings"]), "Northings": float(g["northings"]),
            "OrthogonalHeight": float(g.get("height", 0.0))})
    return f


def _rekey(f, key):
    keep = {e.id() for c in ("IfcProject", "IfcSite") for e in f.by_type(c)}
    for e in f.by_type("IfcRoot"):
        if e.id() not in keep:
            e.GlobalId = guids.from_text(f"{key}/{e.GlobalId}")
    return f


def federation_pair(path):
    a = _with_georef(ifcopenshell.open(str(path)))
    b = _rekey(_with_georef(ifcopenshell.open(str(path))), "mutation-B")
    return a, b


def m_fed_eastings(a, b):
    b.by_type("IfcMapConversion")[0].Eastings += 100.0


def m_fed_duplicate_guid(a, b):
    wa = sorted(a.by_type("IfcWall"), key=lambda e: e.id())[0]
    wb = sorted(b.by_type("IfcWall"), key=lambda e: e.id())[0]
    wb.GlobalId = wa.GlobalId


MUTATIONS = [
    Mutation("remove_door_filling", "ifcqa", ("filling",), "delete the IfcRelFillsElement of a door", m_remove_filling),
    Mutation("duplicate_guid", "ifcqa", ("unique_guids",), "give a wall the GlobalId of another wall", m_duplicate_guid),
    Mutation("uncontain_wall", "ifcqa", ("containment",), "remove a wall from its storey", m_uncontain_wall),
    Mutation("space_out_of_zone", "ifcqa", ("space_in_zone",), "unassign a flat room from its IfcZone", m_space_out_of_zone),
    Mutation("strip_navigation", "ifcqa", ("required_psets", "ids:SN5-DOOR-NAV"), "delete a door's Navigation pset",
             m_strip_navigation),
    Mutation("mirror_mapped_item", "ifcqa", ("placements_right_handed",), "mirror a mapped item, a window's if any (Axis2 = -Y)",
             m_mirror_mapped),
    Mutation("degenerate_placement", "ifcqa", ("placements_right_handed",), "make a wall's RefDirection parallel to its Axis",
             m_degenerate_placement),
    Mutation("zone_ifa_out_of_band", "ifcqa", ("zone_ifa",), "set a flat's InternalFloorArea 10 m2 under its band",
             m_ifa_out_of_band),
    Mutation("bad_flat_type", "ifcqa", ("ids:SN5-ZONE-TYPE", "zone_ifa"), "label a flat '4-Room'", m_bad_flat_type),
    Mutation("bad_room_code", "ifcqa", ("ids:SN5-ROOM-CODE",), "give a bedroom the RoomCode 'Bedroom'", m_bad_room_code),
    Mutation("bad_storey_name", "ifcqa", ("ids:SN5-STOREY-NAME",), "rename a storey 'Level 5'", m_bad_storey_name),
    Mutation("orphan_opening", "ifcqa", ("opening_voids",), "delete the IfcRelVoidsElement of an opening", m_orphan_opening),
    Mutation("too_many_openings", "ifcqa", ("openings_per_wall",), "cut 21 extra openings into one wall",
             m_too_many_openings),
    Mutation("share_representation", "ifcqa", ("representation_sharing",),
             "make two walls share one Body IfcShapeRepresentation", m_share_representation),
    Mutation("shelter_out_of_zone", "programme", ("required_rooms",), "take a household shelter out of its flat zone",
             m_shelter_out_of_zone),
    Mutation("shrink_room_area", "programme", ("room_area",), "set a bedroom's NetFloorArea 1 m2 under the minimum",
             m_shrink_room_area),
    Mutation("shrink_shelter", "programme", ("shelter_area",), "set a household shelter's NetFloorArea to 1.2 m2",
             m_shrink_shelter),
    Mutation("remove_bath_vent", "programme", ("bath_vent",), "delete the vent windows of a naturally ventilated bathroom",
             m_remove_bath_vent),
    Mutation("shrink_bedroom_windows", "programme", ("glazing",), "make a bedroom's windows 0.2 m high",
             m_shrink_bedroom_windows),
    Mutation("stale_door_spaces", "programme", ("door_spaces",),
             "point a bedroom door's Navigation.FromSpace / ToSpace at another room of the flat", m_stale_door_spaces),
    Mutation("remove_main_door", "programme", ("main_door", "reachable"), "delete a flat's main door",
             m_remove_main_door),
    Mutation("raise_riser", "geometry", ("stair_riser", "stair_riser_variation"), "raise the third tread of a flight by 20 mm",
             m_raise_riser),
    Mutation("delete_stair_flight", "geometry", ("stair_rise",), "delete one flight of a stair", m_delete_flight),
    Mutation("delete_balustrade", "geometry", ("stair_open_sides",), "delete the well-side balustrade of a flight",
             m_delete_balustrade),
    Mutation("overlapping_wall", "geometry", ("wall_overlap",), "duplicate the longest wall of a storey 0.3 m along itself",
             m_overlapping_wall),
    Mutation("slab_in_lift_shaft", "geometry", ("lift_shaft_slab",), "add a floor slab across a lift shaft", m_slab_in_shaft),
    Mutation("remove_roof_parapet", "geometry", ("falling_edges",), "delete the longest roof parapet", m_remove_parapet),
    Mutation("overlapping_space", "geometry", ("space_overlap",), "add a copy of a living room shifted 0.5 m",
             m_overlapping_space),
    Mutation("shift_wet_room", "geometry", ("wet_stack", "wet_over_dry"), "move one common bathroom 0.6 m sideways",
             m_shift_wet_room),
    Mutation("door_clash", "geometry", ("opening_clash",), "move a door into the nearest wall that does not host it",
             m_door_clash),
    Mutation("georef_eastings", "federation", ("federation_georef",), "change the IfcMapConversion eastings of one file",
             m_fed_eastings),
    Mutation("federation_duplicate_guid", "federation", ("federation_guids",), "reuse a wall GlobalId of another file",
             m_fed_duplicate_guid),
]


# ----------------------------------------------------------------------------- runner
def _counts(results):
    return {r["check"]: r["count"] for r in results}


def run(path, only=None, schema=False, log=None) -> dict:
    """Apply every mutation to a fresh copy of path; returns {baseline, rows, detected, total, runtime_s}."""
    t0 = time.time()
    path = str(path)
    _, meshes0 = meshcache.load(path)
    _BASE.clear()
    _BASE.update(meshes0)
    base_f = ifcopenshell.open(path)
    base = {"ifcqa": _counts(ifcqa.check(base_f, None, schema=schema, express_rules=False)),
            "programme": _counts(programme.check(base_f)),
            "geometry": _counts(geom.check(base_f, meshes0))}
    fa, fb = federation_pair(path)
    base["federation"] = _counts(ifcqa.check_federation([fa, fb])["results"])
    rows = []
    for m in MUTATIONS:
        if only and m.name not in only:
            continue
        t = time.time()
        err = None
        try:
            if m.family == "federation":
                fa, fb = federation_pair(path)
                m.apply(fa, fb)
                res = ifcqa.check_federation([fa, fb])["results"]
            else:
                f = ifcopenshell.open(path)
                info = m.apply(f) or {}
                if m.family == "ifcqa":
                    res = ifcqa.check(f, None, schema=schema, express_rules=False)
                elif m.family == "programme":
                    res = programme.check(f)
                else:
                    meshes = dict(meshes0)
                    for g in info.get("removed", []):
                        meshes.pop(g, None)
                    if info.get("changed"):
                        meshes.update(meshcache.tessellate(f, include=info["changed"]))
                    res = geom.check(f, meshes)
        except Exception as e:  # noqa: BLE001 - a crashing mutation is reported as undetected
            err, res = f"{type(e).__name__}: {e}", []
        counts = _counts(res)
        fired = sorted(c for c, n in counts.items() if n > base[m.family].get(c, 0))
        detected = err is None and any(c in fired for c in m.expect)
        ex = [r["examples"][0] for r in res if r["check"] in m.expect and r["count"] > base[m.family].get(r["check"], 0)
              and r["examples"]]
        rows.append(dict(mutation=m.name, family=m.family, describe=m.describe, expect=list(m.expect), fired=fired,
                         detected=detected, example=ex[0] if ex else err, runtime_s=round(time.time() - t, 2)))
        if log:
            log(f"  {'caught' if detected else 'MISSED':<7} {m.name:<26} {m.family:<10} {', '.join(fired) or err or '-'}")
    n = sum(r["detected"] for r in rows)
    return {"file": env.rel(path), "rows": rows, "detected": n, "total": len(rows),
            "rate": round(n / len(rows), 4) if rows else 0.0, "baseline": base, "runtime_s": round(time.time() - t0, 2)}
