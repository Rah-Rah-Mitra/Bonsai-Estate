#!/usr/bin/env python3
"""Draw a floor plan, a stair section and a cutaway straight from the IFC (via IfcOpenShell tessellation)."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
import trimesh
from matplotlib.patches import Arc
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union
import ifcopenshell.util.element as uel
import ifcopenshell.util.placement as upl

from ifc_mesh import load

CUT_COLOURS = {"IfcWall": "#3b3b3b", "IfcColumn": "#3b3b3b", "IfcWindow": "#7fb3d5", "IfcRailing": "#555555",
               "IfcStairFlight": "#c9b99a", "IfcSlab": "#d8d2c4", "IfcTransportElement": "#9aa5ad", "IfcTank": "#7fa7c9"}


def section(d, origin, normal, axes):
    tm = trimesh.Trimesh(d["verts"], d["faces"], process=False)
    segs = trimesh.intersections.mesh_plane(tm, normal, origin)
    if len(segs) == 0:
        return None
    lines = [LineString(np.round(s[:, axes], 4)) for s in segs if np.linalg.norm(s[0] - s[1]) > 1e-6]
    lines = [shapely.set_precision(ln, 1e-4) for ln in lines]
    polys = list(polygonize(unary_union([ln for ln in lines if not ln.is_empty and ln.length > 0])))
    return unary_union(polys) if polys else None


def fill(ax, geom, colour, **kw):
    if geom is None or geom.is_empty:
        return
    for g in getattr(geom, "geoms", [geom]):
        if g.geom_type != "Polygon":
            continue
        x, y = g.exterior.xy
        ax.fill(x, y, color=colour, linewidth=0, **kw)
        for hole in g.interiors:
            hx, hy = hole.xy
            ax.fill(hx, hy, color="white", linewidth=0)


def footprint_of(d):
    tris = d["verts"][d["faces"]][:, :, :2]
    polys = [shapely.Polygon(t) for t in tris if abs(shapely.Polygon(t).area) > 1e-8]
    return unary_union(polys)


def plan(f, meshes, storey, out):
    st = [s for s in f.by_type("IfcBuildingStorey") if s.Name == storey][0]
    z = st.Elevation + 1.2
    fig, ax = plt.subplots(figsize=(16, 12), dpi=110)
    # spaces
    for sp in f.by_type("IfcSpace"):
        if uel.get_aggregate(sp) != st or sp.GlobalId not in meshes:
            continue
        poly = footprint_of(meshes[sp.GlobalId])
        fill(ax, poly, "#f6efd9" if "lobby" not in (sp.LongName or "").lower() else "#e3ecf2")
        area = uel.get_psets(sp).get("Qto_SpaceBaseQuantities", {}).get("NetFloorArea", 0)
        p = poly.representative_point()
        label = (sp.LongName or "").replace("Living / Dining", "Living/Dining").replace("Household Shelter", "Shelter")
        ax.text(p.x, p.y, f"{label}\n{area:.1f} m²", ha="center", va="center", fontsize=6.3, color="#333")
    # cut elements at 1.2 m
    for g, d in meshes.items():
        if d["storey"] != storey or d["cls"] in ("IfcSpace", "IfcDoor"):
            continue
        cut = section(d, (0, 0, z), (0, 0, 1), [0, 1])
        fill(ax, cut, CUT_COLOURS.get(d["cls"], "#888"))
        if d["cls"] == "IfcStairFlight":   # show the flight in plan below the cut too
            fill(ax, footprint_of(d), "#e9dfc7", alpha=0.6)
    # doors: swing arcs from the IFC placement
    for door in f.by_type("IfcDoor"):
        if uel.get_container(door) != st:
            continue
        M = upl.get_local_placement(door.ObjectPlacement)
        o, ux, uy = M[:2, 3], M[:2, 0], M[:2, 1]
        w = door.OverallWidth
        lift = "lift" in door.Name.lower()
        col = "#999" if lift else "#c0392b"
        if lift:
            ax.plot(*np.array([o, o + ux * w]).T, color=col, lw=2)
            continue
        hinge = o + uy * 0.1
        ax.plot(*np.array([hinge, hinge + uy * w]).T, color=col, lw=0.8)
        ang0 = np.degrees(np.arctan2(ux[1], ux[0]))
        ang1 = np.degrees(np.arctan2(uy[1], uy[0]))
        a0, a1 = (ang0, ang1) if (ang1 - ang0) % 360 < 180 else (ang1, ang0)
        ax.add_patch(Arc(hinge, 2 * w, 2 * w, theta1=a0, theta2=a1, color=col, lw=0.6))
    # flat numbers
    for zn in f.by_type("IfcZone"):
        sps = [s for s in uel.get_grouped_by(zn) if uel.get_aggregate(s) == st]
        if not sps:
            continue
        u = unary_union([footprint_of(meshes[s.GlobalId]) for s in sps if s.GlobalId in meshes])
        c = u.centroid
        b = u.bounds
        ax.text((b[0] + b[2]) / 2, b[3] + 0.6 if c.y > 0 else b[1] - 0.9, f"{zn.Name}  ({zn.LongName}, {u.area:.0f} m² net)",
                ha="center", fontsize=9, weight="bold", color="#1f3b57")
    ax.text(-1.45, 2.85, "LIFT 1", ha="center", fontsize=7, color="#555")
    ax.text(1.45, 2.85, "LIFT 2", ha="center", fontsize=7, color="#555")
    ax.text(-1.45, -4.5, "STAIR 1", ha="center", fontsize=7, color="#555", rotation=90)
    ax.text(1.45, -4.5, "STAIR 2", ha="center", fontsize=7, color="#555", rotation=90)
    ax.set_aspect("equal")
    ax.set_xlim(-15.5, 15.5)
    ax.set_ylim(-12, 12)
    ax.axis("off")
    ax.set_title(f"{storey} plan, cut at 1.2 m above floor (drawn from the IFC). Red arcs = door swings; blue = glazing",
                 fontsize=11)
    ax.plot([10, 15], [-11.5, -11.5], color="k", lw=2)
    ax.text(12.5, -11.3, "5 m", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out)


def stair_section(f, meshes, out, x_east=-0.775, x_west=-2.125, levels=("L1", "L2", "L3", "L4")):
    fig, ax = plt.subplots(figsize=(9, 12), dpi=110)
    for g, d in meshes.items():
        if d["storey"] not in levels:
            continue
        if d["cls"] not in ("IfcStairFlight", "IfcSlab", "IfcWall", "IfcRailing", "IfcDoor"):
            continue
        if d["cls"] == "IfcWall" and "STAIR1" not in d["name"]:
            continue
        for x, colour, alpha in [(x_west, "#d9cdb3", 0.55), (x_east, None, 1.0)]:
            cut = section(d, (x, 0, 0), (1, 0, 0), [1, 2])
            if cut is None:
                continue
            cut = cut.intersection(shapely.box(-8.5, -1.0, 0.5, 40))
            c = colour or {"IfcStairFlight": "#b08d57", "IfcSlab": "#8c8c8c", "IfcWall": "#3b3b3b",
                           "IfcRailing": "#444", "IfcDoor": "#c0392b"}[d["cls"]]
            fill(ax, cut, c, alpha=alpha)
    st = {s.Name: s.Elevation for s in f.by_type("IfcBuildingStorey")}
    for lv in levels + ("L5",):
        ax.axhline(st[lv], color="#999", lw=0.5, ls=":")
        ax.text(-8.4, st[lv] + 0.05, f"{lv}  FFL {st[lv]:.2f} m", fontsize=8, color="#555")
    for flight in f.by_type("IfcStairFlight"):
        if flight.Name.startswith("L1 stair 1 flight 1") or flight.Name.startswith("L2 stair 1 flight 1"):
            ax.text(-5.2, st[flight.Name.split()[0]] + 0.9,
                    f"{flight.NumberOfRisers} risers x {flight.RiserHeight * 1000:.0f} mm\ngoing {flight.TreadLength * 1000:.0f} mm",
                    fontsize=8, color="#7a5a2a")
    ax.set_aspect("equal")
    ax.set_xlim(-8.6, 0.6)
    ax.set_ylim(-0.6, st["L5"] + 0.4)
    ax.set_xlabel("y (m)  - lobby is to the right")
    ax.set_ylabel("z (m)")
    ax.set_title("Stair 1 section (dark: flight cut on the near side; light: flight on the far side)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out)


def cutaway(f, meshes, storey, out):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    st = [s for s in f.by_type("IfcBuildingStorey") if s.Name == storey][0]
    below = {s.Name for s in f.by_type("IfcBuildingStorey") if s.Elevation < st.Elevation}
    tris, cols = [], []
    for g, d in meshes.items():
        keep = d["storey"] == storey and d["cls"] not in ("IfcSpace",)
        keep |= d["storey"] in below and d["cls"] not in ("IfcSpace",)
        if not keep or d["cls"] == "IfcGeographicElement":
            continue
        t = d["verts"][d["faces"]]
        if d["storey"] == storey and d["cls"] == "IfcSlab" and "floor slab" not in d["name"]:
            pass
        tris.append(t)
        cols.append(d["colours"])
    tris, cols = np.concatenate(tris), np.concatenate(cols)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    k = 0.55 + 0.45 * np.abs(n @ np.array([0.35, -0.45, 0.82]))
    shaded = cols.copy()
    shaded[:, :3] = np.clip(cols[:, :3] * k[:, None], 0, 1)
    fig = plt.figure(figsize=(15, 11), dpi=110)
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Poly3DCollection(tris, facecolors=shaded, linewidths=0))
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(st.Elevation - 12, st.Elevation + 18)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=48, azim=-62)
    ax.set_axis_off()
    ax.set_title(f"Cutaway at {storey}: everything above removed", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "hdb_block.ifc"
    f, meshes = load(path)
    plan(f, meshes, "L5", "plan_L5.png")
    stair_section(f, meshes, "stair_section.png")
    cutaway(f, meshes, "L5", "cutaway_L5.png")
