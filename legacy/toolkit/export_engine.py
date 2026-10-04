#!/usr/bin/env python3
"""
export_engine.py - turn the IFC into game-engine inputs.

  <name>.glb          static architecture merged per storey + material (few actors), doors and lift doors as
                      separate nodes (swap for door / elevator blueprints), box-projected UVs (1 UV = 1 m) so
                      tiling materials work without unwrapping. glTF is Y-up metres; importers convert.
  <name>_engine.json  everything gameplay needs that geometry alone doesn't carry:
                      doors (hinge, width, swing side, passable), lifts (shafts, landing doors, levels),
                      rooms (flat number, room name, floor polygon -> trigger volumes), climbable edges
                      (parapet / AC-ledge / tank top edges as polylines), spawn point, verified agent settings.
Coordinates in the JSON are IFC world coordinates: metres, right-handed, Z up.
"""
import json
import re
import sys

import numpy as np
import shapely
import trimesh
from shapely.ops import unary_union
import ifcopenshell.util.element as uel
import ifcopenshell.util.placement as upl

from ifc_mesh import load

MERGE_SKIP = {"IfcDoor", "IfcSpace"}


def to_yup(v):
    return np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])


def box_uv_mesh(tris, colour):
    """Unshared vertices with box-projected UVs (dominant normal axis), 1 UV unit per metre."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ax = np.argmax(np.abs(n), axis=1)
    v = tris.reshape(-1, 3)
    axr = np.repeat(ax, 3)
    uv = np.where(axr[:, None] == 0, v[:, [1, 2]], np.where(axr[:, None] == 1, v[:, [0, 2]], v[:, [0, 1]]))
    faces = np.arange(len(v)).reshape(-1, 3)
    mesh = trimesh.Trimesh(to_yup(v), faces, process=False)
    rgba = (np.array(colour) * 255).astype(np.uint8)
    mat = trimesh.visual.material.PBRMaterial(baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.8,
                                              alphaMode="BLEND" if colour[3] < 0.99 else "OPAQUE", doubleSided=True)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    return mesh


def safe(s):
    return re.sub(r"[^A-Za-z0-9_\-]", "_", s)[:60]


def top_edges(d, z_tol=0.02):
    """Outline(s) of an element's topmost upward-facing faces, as closed polylines."""
    tris = d["verts"][d["faces"]]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    up = (n[:, 2] / (np.linalg.norm(n, axis=1) + 1e-12)) > 0.95
    if not up.any():
        return [], None
    zmax = tris[up][:, :, 2].max()
    sel = up & (np.abs(tris[:, :, 2].mean(axis=1) - zmax) < z_tol)
    poly = unary_union([shapely.Polygon(t[:, :2]) for t in tris[sel] if shapely.Polygon(t[:, :2]).area > 1e-8])
    out = []
    for g in getattr(poly, "geoms", [poly]):
        if g.geom_type == "Polygon":
            out.append([[round(x, 3), round(y, 3), round(float(zmax), 3)] for x, y in g.exterior.coords])
    return out, float(zmax)


def main(path):
    f, meshes = load(path)
    stem = path.rsplit(".", 1)[0]
    scene = trimesh.Scene()
    groups = {}
    doors = {d.GlobalId: d for d in f.by_type("IfcDoor")}
    for g, d in meshes.items():
        if d["cls"] in MERGE_SKIP:
            continue
        tris = d["verts"][d["faces"]]
        for colour in np.unique(d["colours"].round(3), axis=0):
            sel = np.all(np.abs(d["colours"] - colour) < 2e-3, axis=1)
            key = (d["storey"] or "SITE", tuple(colour))
            groups.setdefault(key, []).append(tris[sel])
    names = {}
    for (storey, colour), parts in groups.items():
        tris = np.concatenate(parts)
        base = f"{safe(storey)}_static_{int(colour[0]*255):02x}{int(colour[1]*255):02x}{int(colour[2]*255):02x}"
        names[base] = names.get(base, 0) + 1
        scene.add_geometry(box_uv_mesh(tris, colour), node_name=f"{base}_{names[base]}", geom_name=f"{base}_{names[base]}")
    door_info = []
    for g, door in doors.items():
        d = meshes.get(g)
        if d is None:
            continue
        nav = uel.get_psets(door).get("Navigation", {})
        node = f"DOOR_{safe(door.Name)}_{g}"
        for colour in np.unique(d["colours"].round(3), axis=0):
            sel = np.all(np.abs(d["colours"] - colour) < 2e-3, axis=1)
            scene.add_geometry(box_uv_mesh(d["verts"][d["faces"]][sel], colour), node_name=f"{node}_{len(door_info)}_{int(colour[0]*255)}",
                               geom_name=f"{node}_{int(colour[0]*255)}")
        M = upl.get_local_placement(door.ObjectPlacement)
        door_info.append(dict(guid=g, name=door.Name, node_prefix=node, storey=d["storey"], kind=nav.get("DoorKind"),
                              passable=nav.get("Passable", True), width=door.OverallWidth, height=door.OverallHeight,
                              hinge=[round(v, 3) for v in M[:3, 3]], along_wall=[round(v, 3) for v in M[:3, 0]],
                              swing_side=[round(v, 3) for v in M[:3, 1]], operation=door.OperationType))
    scene.export(stem + ".glb")

    storeys = {s.Name: s.Elevation for s in f.by_type("IfcBuildingStorey")}
    rooms = []
    for sp in f.by_type("IfcSpace"):
        d = meshes.get(sp.GlobalId)
        st = uel.get_aggregate(sp)
        poly = unary_union([shapely.Polygon(t[:, :2]) for t in d["verts"][d["faces"]] if shapely.Polygon(t[:, :2]).area > 1e-8])
        zone = next((z for z in f.by_type("IfcZone") if sp in uel.get_grouped_by(z)), None)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda q: q.area)
        rooms.append(dict(guid=sp.GlobalId, name=sp.Name, room=sp.LongName, flat=zone.Name if zone else None,
                          storey=st.Name, floor_level=storeys[st.Name],
                          height=uel.get_psets(sp).get("Qto_SpaceBaseQuantities", {}).get("Height"),
                          polygon=[[round(x, 3), round(y, 3)] for x, y in poly.exterior.coords]))
    climb = []
    for el in f.by_type("IfcElement"):
        nav = uel.get_psets(el).get("Navigation", {})
        if not nav.get("ClimbableTopEdge") or el.GlobalId not in meshes:
            continue
        d = meshes[el.GlobalId]
        loops, ztop = top_edges(d)
        if loops:
            climb.append(dict(guid=el.GlobalId, name=el.Name, ifc_class=el.is_a(), storey=d["storey"],
                              top_z=round(ztop, 3), floor_level=storeys.get(d["storey"]), polylines=loops))
    lifts = []
    for k in (1, 2):
        car = [t for t in f.by_type("IfcTransportElement") if t.Name == f"Lift {k} car"][0]
        cb = meshes[car.GlobalId]["verts"]
        lifts.append(dict(name=f"Lift {k}", car_guid=car.GlobalId,
                          shaft_xy_bounds=[round(float(v), 3) for v in (*cb[:, :2].min(0), *cb[:, :2].max(0))],
                          landing_doors=[dd["node_prefix"] for dd in door_info if dd["name"].endswith(f"lift {k} landing door")],
                          levels={k2: v for k2, v in storeys.items() if k2 != "RF"}))
    info = dict(
        frame="IFC world coordinates: metres, right-handed, Z up (the .glb is the same geometry, Y up)",
        spawn=dict(name="Street (footpath to void deck)", position=[5.5, -12.0, -0.12]),
        verified_agent=dict(radius=0.30, height=1.8, max_step=0.4,
                            note="nav_check.py: every room, lobby and the roof reachable at r=0.30; at r=0.40 rooms behind "
                                 "0.8-0.9 m doors are cut off. Keep capsule/navmesh agent radius <= 0.30 m or use fine navmesh cells indoors."),
        storeys=storeys, doors=door_info, lifts=lifts, rooms=rooms, climbable_edges=climb)
    with open(stem + "_engine.json", "w") as fh:
        json.dump(info, fh, indent=1)
    print(f"wrote {stem}.glb ({len(scene.geometry)} meshes) and {stem}_engine.json "
          f"({len(door_info)} doors, {len(rooms)} rooms, {len(climb)} climbable elements)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "hdb_block.ifc")
