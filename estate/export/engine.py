"""Game-engine export of one building IFC (lifted from legacy/toolkit/export_engine.py, on the in-house glb writer).

  <stem>_lod0.glb     static architecture merged per storey (one primitive per colour) with box-projected UVs
                      (1 UV = 1 m). Doors, windows and lift cars are separate nodes DOOR_<name>_<guid> /
                      WIN_<name>_<guid> / LIFT_<name>_<guid> whose node matrix is the IFC placement, so an engine
                      can swap them for door, window or elevator blueprints; doors and windows hang under their
                      storey node. A DOOR_ node carries the frame (lining, casings, threshold) and one child per
                      leaf, DOOR_..._leaf (_leaf_L / _leaf_R for double doors). A swinging leaf's origin sits on its
                      hinge axis at the swing-side face: turning it about its local up axis (glTF +Y) by
                      open_sign * angle opens it towards swing_side. Sliding and rolling leaves sit on their
                      jamb-side edge and move along leaves[].travel. Furniture whose geometry repeats (hawker tables
                      and stools, letterbox banks: one type, or identical copies) is a FURN_<name>_<guid> node under
                      its storey. Repeated geometry (doors and windows of one type, furniture, trees) is written once
                      and shared by every node that uses it (glTF mesh reuse), which cuts LOD0 to about a third of
                      its merged size.
  <stem>_lod1.glb     exterior shell merged per material: no spaces, interior walls (INT/WET types, or
                      Pset_WallCommon.IsExternal = False) and the doors/windows they host, furniture, lift cars or
                      stairs inside cores. Walls of an open void deck and of the roof storey stay (seen from outside).
                      The doors and windows it keeps are instanced like LOD0's (closed, no leaf children): one node
                      L1DOOR_ / L1WIN_<name>_<guid> each under <stem>_openings, one shared mesh per type, so the
                      file stays smaller than LOD0 although it draws fewer triangles.
  <stem>_lod2.glb     massing: the slab footprint extruded to the roof parapet, plus boxes for roof structures.
  <stem>_int_<tag>.glb  one interior chunk per storey (tag: L5 -> L05, RF -> RF; storey_tag) for engines that
                      stream interiors floor by floor: exactly that storey's LOD0 nodes (the <storey>_static batch
                      and its DOOR_ with frame and leaves, WIN_ and FURN_ children: same names, meshes and matrices)
                      under a root <stem>_int_<tag>, written by the code that writes them into LOD0, in the same
                      block-local frame, so a chunk overlays LOD0 / LOD1 with the same transform. Lift cars span
                      every storey and stay in LOD0 only; the site group gets no chunk, nor does a site-only file.
  <stem>_engine.json  what gameplay needs that geometry does not carry: doors (with their leaves), lifts, rooms,
                      flats, portals, spawn points, climbable edges, verified agent, the per-flat triangle budget
                      and the interior chunks (file, elevation, triangle and node counts per storey).

A site-only file (SITE.ifc: no storeys) is cut into TILE x TILE m tiles so engines can cull and stream it: each tile
node SITE_<i>_<j> holds a ground, a structures (linkways, shelters) and a furniture mesh, with every tree an
instanced TREE_<name>_<guid> node. Its LOD1 keeps the ground and the linkway roofs and shelters (no furniture,
columns or posts), with an eight-triangle crown per tree, and it has no LOD2 (there is no massing to speak of).

Everything is block-local: when ``local_matrix`` (estate -> local, the inverse of the building placement) is
given, geometry and JSON are centred on the building origin and ``transform`` in the JSON maps them back into
the estate. JSON coordinates are IFC-style (metres, Z up); the glbs are glTF Y up: (x, y, z) -> (x, z, -y).
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

import ifcopenshell.geom
import ifcopenshell.util.element as uel
import ifcopenshell.util.placement as upl

from estate import env
from estate.export import glb, meshcache
from estate.validate.nav_doorpose import leaf_frame, leaf_record, leaf_spec
from estate.rules import AGENT_HEIGHT, AGENT_RADIUS, AGENT_STEP, PALETTE

LOD0_BUDGET_PER_FLAT = 5000
LOD1_DROP = {"IfcSpace", "IfcOpeningElement", "IfcVirtualElement", "IfcFurnishingElement", "IfcFurniture",
             "IfcSystemFurnitureElement", "IfcTransportElement", "IfcSanitaryTerminal"}
ALWAYS_INTERIOR = ("INT", "WET")             # wall type name prefixes (rules.WALL_TYPES)
INTERIOR_UNLESS_EXTERNAL = ("HS", "PARTY", "CORE")
UNIT_RE = re.compile(r"^(#\d+-\d+)\s")
EXPORT_PROPS = ("Navigation", "Pset_WallCommon", "Pset_DoorCommon", "SampleCity_Flat", "SampleCity_Room",
                "Qto_SpaceBaseQuantities", "Pset_SpaceCommon")
TILE = 100.0                                 # site tile size (m)
SITE_FURNITURE = {"IfcFurniture", "IfcFurnishingElement", "IfcSystemFurnitureElement", "IfcRailing", "IfcSign"}
SITE_STRUCTURE = {"IfcColumn", "IfcBeam", "IfcMember", "IfcRoof", "IfcCovering", "IfcPlate"}
SITE_LOD1_MIN = 1.0                          # site elements narrower than this in plan (m) leave LOD1: columns, posts
OUTSIDE = ("outside", "@out")                # Navigation From/ToSpace values for the world outside the building
LEAF_TOL = 0.005                             # leaf items sit strictly inside the jambs and above the threshold
FURNITURE = ("IfcFurniture", "IfcFurnishingElement", "IfcSystemFurnitureElement")    # instanced in LOD0 when repeated
ENTRANCE_LABELS = ("Void deck entrance", "Entrance")    # spawn name prefixes; 'Interior' when it stands in a room
CHUNK_COUNTS = ("door_nodes", "door_leaves", "window_nodes", "furniture_nodes")    # per chunk in the engine JSON


def safe(s, n=60):
    return re.sub(r"[^A-Za-z0-9_\-]", "_", s or "")[:n]


def unit_of(name):
    """'#02-101' from an element named '#02-101 Main door' (None for common elements)."""
    m = UNIT_RE.match(name or "")
    return m.group(1) if m else None


def door_node_name(name, guid):
    """DOOR_<name>_<guid>, at most 55 bytes, so the leaf children (_leaf, _leaf_L, _leaf_R) stay within the 63-byte
    object-name limit of Blender 4 and most engines."""
    return f"DOOR_{safe(name, 27)}_{guid}"


def leaf_suffix(k, n):
    return "_leaf" if n == 1 else f"_leaf_{'LR'[k] if n == 2 else k + 1}"


def lift_node_name(name, guid):
    return f"LIFT_{safe(name, 34)}_{guid}"


def window_node_name(name, guid):
    return f"WIN_{safe(name, 34)}_{guid}"


def tree_node_name(name, guid):
    return f"TREE_{safe(name, 34)}_{guid}"


def furniture_node_name(name, guid):
    return f"FURN_{safe(name, 34)}_{guid}"


def storey_tag(name):
    """File-name tag of a storey's interior chunk: L<n> with the number zero-padded to two digits (L5 -> L05, so a
    folder listing runs bottom up: L01 .. L25, then RF), RF and any other storey name through safe()."""
    m = re.fullmatch(r"L(\d+)", name or "")
    return f"L{int(m.group(1)):02d}" if m else (safe(name) or "storey")


def xf(M, v):
    v = np.asarray(v, float)
    return v @ M[:3, :3].T + M[:3, 3]


def r3(v, nd=3):
    return [round(float(x), nd) for x in v]


def mat_list(M, nd=9):
    return [[round(float(x), nd) + 0.0 for x in row] for row in np.asarray(M)]


# ----------------------------------------------------------------------------- IFC indexing helpers
def property_index(f, names=EXPORT_PROPS) -> dict:
    """{guid: {pset/qto name: {prop: value}}} for occurrence psets, from one pass over IfcRelDefinesByProperties."""
    out = {}
    for rel in f.by_type("IfcRelDefinesByProperties"):
        pd = rel.RelatingPropertyDefinition
        if not pd.is_a("IfcPropertySetDefinition") or pd.Name not in names:
            continue
        vals = {}
        if pd.is_a("IfcPropertySet"):
            for p in pd.HasProperties or ():
                if p.is_a("IfcPropertySingleValue"):
                    vals[p.Name] = p.NominalValue.wrappedValue if p.NominalValue is not None else None
        elif pd.is_a("IfcElementQuantity"):
            for q in pd.Quantities or ():
                vals[q.Name] = q[3]
        for o in rel.RelatedObjects:
            out.setdefault(o.GlobalId, {})[pd.Name] = vals
    return out


def parts_of(el):
    """The element and everything aggregated under it (walks IsDecomposedBy; avoids set()s of entities, which
    break while estate.guids has swapped the entity hash)."""
    out, todo = [], [el]
    while todo:
        e = todo.pop()
        out.append(e)
        for rel in getattr(e, "IsDecomposedBy", ()) or ():
            todo.extend(rel.RelatedObjects)
    return out


def top_edges(d, z_tol=0.02):
    """Outline(s) of an element's topmost upward-facing faces as closed polylines (legacy logic)."""
    tris = d["verts"][d["faces"]]
    n, ln = glb.face_normals(tris)
    up = (n[:, 2] / (ln + 1e-12)) > 0.95
    if not up.any():
        return [], None
    zmax = tris[up][:, :, 2].max()
    sel = up & (np.abs(tris[:, :, 2].mean(axis=1) - zmax) < z_tol)
    polys = [Polygon(t[:, :2]) for t in tris[sel]]
    poly = shapely.union_all([p for p in polys if p.area > 1e-8], grid_size=1e-4)
    out = []
    for g in getattr(poly, "geoms", [poly]):
        if g.geom_type == "Polygon" and not g.is_empty:
            out.append([[round(x, 3), round(y, 3), round(float(zmax), 3)] for x, y in g.exterior.coords])
    return out, float(zmax)


def plan_polygon(d):
    """Footprint of a space (union of its downward faces), cleaned of collinear vertices."""
    tris = d["verts"][d["faces"]]
    n, ln = glb.face_normals(tris)
    down = (n[:, 2] / (ln + 1e-12)) < -0.9
    sel = tris[down] if down.any() else tris
    polys = [p for p in (Polygon(t[:, :2]) for t in sel) if p.area > 1e-8]
    if not polys:
        return None
    u = shapely.union_all(polys, grid_size=1e-4).simplify(1e-4)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda q: q.area)
    return u if u.geom_type == "Polygon" and not u.is_empty else None


def prism(poly, z0, z1):
    """Side and cap triangles (Z up, outward normals) of a polygon extruded from z0 to z1."""
    poly = orient(poly, 1.0)
    caps = [orient(t, 1.0) for t in shapely.constrained_delaunay_triangles(poly).geoms if t.area > 1e-9]
    top = np.array([[(x, y, z1) for x, y in list(t.exterior.coords)[:3]] for t in caps])
    bottom = np.array([[(x, y, z0) for x, y in list(t.exterior.coords)[:3][::-1]] for t in caps])
    sides = []
    for ring in [poly.exterior, *poly.interiors]:
        c = list(ring.coords)
        for (ax, ay), (bx, by) in zip(c[:-1], c[1:]):
            a0, b0, b1, a1 = (ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1)
            sides += [(a0, b0, b1), (a0, b1, a1)]
    return np.array(sides, float).reshape(-1, 3, 3), top.reshape(-1, 3, 3), bottom.reshape(-1, 3, 3)


def clean_colours(c):
    """IfcOpenShell's default (unstyled) materials report NaN transparency: NaN alpha -> opaque, NaN rgb -> grey."""
    c = np.array(c, dtype=float, copy=True)
    c[:, 3] = np.where(np.isnan(c[:, 3]), 1.0, c[:, 3])
    c[:, :3] = np.where(np.isnan(c[:, :3]), 0.7, c[:, :3])
    return np.clip(c, 0.0, 1.0)


def geometry_key(tris, colours) -> str:
    """Hash of a triangle soup and its face colours at 0.1 mm: equal keys mean the same mesh (instancing)."""
    h = hashlib.sha1()
    h.update((np.round(np.asarray(tris, float), 4) + 0.0).tobytes())
    h.update((np.round(clean_colours(colours), 3) + 0.0).tobytes())
    return h.hexdigest()


def _clip_triangle(t, axis, c):
    """A triangle cut by the plane coordinate[axis] = c into triangles on either side (winding kept)."""
    d = t[:, axis] - c
    if (d >= -1e-9).all() or (d <= 1e-9).all():
        return [t]
    out = []
    for sgn in (1.0, -1.0):
        poly = []
        for i in range(3):
            a, b, da, db = t[i], t[(i + 1) % 3], d[i] * sgn, d[(i + 1) % 3] * sgn
            if da >= 0:
                poly.append(a)
            if da * db < 0:
                poly.append(a + (b - a) * (da / (da - db)))
        out += [np.array([poly[0], poly[j], poly[j + 1]]) for j in range(1, len(poly) - 1)]
    return out


def split_grid(tris, size=TILE):
    """Cut triangles (n, 3, 3) along the planes x = k * size and y = k * size, so each piece lies in one tile.
    Returns (triangles, index of the source triangle of each piece)."""
    tris = np.asarray(tris, float).reshape(-1, 3, 3)
    src = np.arange(len(tris))
    for axis in (0, 1):
        k0 = np.floor(tris[:, :, axis].min(axis=1) / size + 1e-9)
        k1 = np.floor(tris[:, :, axis].max(axis=1) / size - 1e-9)
        cross = k1 > k0
        if not cross.any():
            continue
        out_t, out_s = [tris[~cross]], [src[~cross]]
        for i in np.nonzero(cross)[0]:
            pieces = [tris[i]]
            for k in range(int(k0[i]) + 1, int(k1[i]) + 1):
                pieces = [q for p in pieces for q in _clip_triangle(p, axis, k * size)]
            out_t.append(np.array(pieces).reshape(-1, 3, 3))
            out_s.append(np.full(len(pieces), src[i]))
        tris, src = np.concatenate(out_t), np.concatenate(out_s)
    return tris, src


def tile_index(xy, size=TILE):
    return int(np.floor(xy[0] / size)), int(np.floor(xy[1] / size))


def octahedron(lo, hi):
    """Eight outward triangles of a diamond inscribed in the box lo..hi (Z up): a tree crown for LOD1."""
    c = (np.asarray(lo, float) + hi) / 2
    r = (np.asarray(hi, float) - lo) / 2
    px, nx, py, ny = c + (r[0], 0, 0), c - (r[0], 0, 0), c + (0, r[1], 0), c - (0, r[1], 0)
    top, bot = c + (0, 0, r[2]), c - (0, 0, r[2])
    ring = [px, py, nx, ny]
    return np.array([[ring[i], ring[(i + 1) % 4], top] for i in range(4)] +
                    [[ring[(i + 1) % 4], ring[i], bot] for i in range(4)])


class _Batch:
    """Triangles grouped by colour, remembering the source element of every triangle."""

    def __init__(self):
        self.parts = {}

    def add(self, d, elem_i, tris=None, sel=None):
        tris = d["verts"][d["faces"]] if tris is None else tris
        colours = d["colours"]
        if sel is not None:
            tris, colours = tris[sel], colours[sel]
        self.add_tris(tris, colours, np.full(len(tris), elem_i))

    def add_tris(self, tris, colours, eids):
        if not len(tris):
            return
        keys, inv = np.unique(np.round(clean_colours(colours), 3), axis=0, return_inverse=True)
        inv = inv.ravel()
        for k, key in enumerate(keys):
            sel = inv == k
            self.parts.setdefault(tuple(float(x) for x in key), []).append((tris[sel], eids[sel]))

    def _merged(self, materials):
        for key in sorted(self.parts, key=lambda k: (materials.name(k), k)):
            yield key, np.concatenate([t for t, _ in self.parts[key]]), np.concatenate([e for _, e in self.parts[key]])

    def key(self, materials):
        h = hashlib.sha1()
        for key, tris, _ in self._merged(materials):
            h.update(repr(key).encode())
            h.update((np.round(tris, 4) + 0.0).tobytes())
        return h.hexdigest()

    def emit(self, W, name, materials, parent=None, matrix=None, extras=None, shared=None, mesh_name=None):
        """Write one mesh node (one primitive per colour, box UVs in the node frame). ``matrix`` (Z up) is the
        node transform; the soup is already in the node frame. With ``shared`` (a dict kept per glb), a batch
        whose geometry was written before reuses that glTF mesh. Returns (node, [(kept tris, element ids)])."""
        merged = list(self._merged(materials))
        key = self.key(materials) if shared is not None else None
        if key is not None and key in shared:
            mesh, kept_idx = shared[key]
        else:
            prims, kept_idx = [], []
            for key_c, tris, _ in merged:
                p = glb.soup(glb.to_yup(tris), glb.box_uvs(tris))
                prims.append((p, materials.index(W, key_c)))
                kept_idx.append(p["kept"])
            mesh = W.mesh(mesh_name or name, prims)
            if key is not None:
                shared[key] = (mesh, kept_idx)
        kept = [(tris[k], eids[k]) for (_, tris, eids), k in zip(merged, kept_idx)]
        node = W.node(name, mesh, matrix=None if matrix is None else glb.matrix_to_yup(matrix), parent=parent,
                      extras=extras)
        return node, kept


class _Materials:
    """Colour -> glTF material, named after the palette entry it came from (rules.PALETTE)."""

    def __init__(self):
        self.palette = [(np.array([*rgb, 1.0 - tr]), name) for _, (name, rgb, tr) in sorted(PALETTE.items())]
        self._names = {}

    def name(self, key):
        if key not in self._names:
            c = np.array(key)
            best = min(self.palette, key=lambda p: float(np.abs(p[0] - c).max()))
            self._names[key] = best[1] if np.abs(best[0] - c).max() < 3e-3 else \
                "rgba_" + "".join(f"{int(round(v * 255)):02x}" for v in c)
        return self._names[key]

    def index(self, W, key):
        return W.material(self.name(key), key)


# ----------------------------------------------------------------------------- the exporter
class BuildingExport:
    def __init__(self, ifc_path, local_matrix=None, num_threads=None, cache=True):
        self.path = Path(ifc_path)
        self.f, raw = meshcache.load(self.path, cache=cache, num_threads=num_threads)
        f = self.f
        if isinstance(local_matrix, str) and local_matrix == "building":
            b = f.by_type("IfcBuilding")
            B = upl.get_local_placement(b[0].ObjectPlacement) if b and b[0].ObjectPlacement else np.eye(4)
            local_matrix = np.linalg.inv(B)
        self.L = np.eye(4) if local_matrix is None else np.asarray(local_matrix, float)
        self.T = np.linalg.inv(self.L)                     # block -> estate
        ident = np.allclose(self.L, np.eye(4))
        self.meshes = {g: (d if ident else dict(d, verts=xf(self.L, d["verts"]))) for g, d in sorted(raw.items())}
        self.props = property_index(f)
        # storeys (elevation in the local frame)
        st = []
        for s in f.by_type("IfcBuildingStorey"):
            z = float(s.Elevation or 0.0)
            if s.ObjectPlacement is not None:
                z = float((self.L @ upl.get_local_placement(s.ObjectPlacement))[2, 3])
            st.append((z, s.Name, s))
        st.sort(key=lambda t: (t[0], t[1]))
        self.storeys = {n: round(z, 6) for z, n, _ in st}
        self.storey_order = [n for _, n, _ in st]
        self.has_storeys = bool(st)
        self.lowest = self.storey_order[0] if st else None
        self.top = self.storey_order[-1] if st else None
        self.ent = {g: f.by_guid(g) for g in self.meshes}
        # stairs inside the building (cores): every part of every IfcStair
        self.stair_parts = set()
        for s in f.by_type("IfcStair"):
            self.stair_parts.update(p.GlobalId for p in parts_of(s))
        self.core_stair_parts = self._interior_stairs()
        # walls, hosts
        self.wall_type = {}
        for w in f.by_type("IfcWall"):
            t = uel.get_type(w)
            self.wall_type[w.GlobalId] = (t.Name if t is not None else "") or ""
        self.host, self.opening_of = {}, {}
        for e in f.by_type("IfcDoor") + f.by_type("IfcWindow"):
            for rel in e.FillsVoids or ():
                op = rel.RelatingOpeningElement
                self.opening_of[e.GlobalId] = op
                for v in op.VoidsElements or ():
                    self.host[e.GlobalId] = v.RelatingBuildingElement.GlobalId
        open_ground = any(self.props.get(sp.GlobalId, {}).get("Pset_SpaceCommon", {}).get("IsExternal")
                          for sp in f.by_type("IfcSpace")
                          if self.meshes.get(sp.GlobalId, {}).get("storey") == self.lowest)
        self.keep_all_walls_on = {self.top} | ({self.lowest} if open_ground else set())
        self.materials = _Materials()
        self._items, self._door_specs, self._parts = {}, {}, {}
        self.door_leaves, self.tiles = {}, []

    # ---- classification
    def is_site(self, g):
        return self.has_storeys and self.meshes[g]["storey"] not in self.storeys

    def is_tree(self, g):
        el = self.ent[g]
        return el.is_a("IfcGeographicElement") and getattr(el, "PredefinedType", None) != "TERRAIN" \
            and uel.get_type(el) is not None

    def site_category(self, g):
        """ground | structures | furniture | tree, for the tiles of a site-only file."""
        if self.is_tree(g):
            return "tree"
        cls = self.meshes[g]["cls"]
        if cls in SITE_FURNITURE:
            return "furniture"
        agg = uel.get_aggregate(self.ent[g])
        if cls in SITE_STRUCTURE or (agg is not None and agg.is_a("IfcElementAssembly")):
            return "structures"
        return "ground"

    def _interior_stairs(self):
        bb = [d["verts"] for g, d in self.meshes.items() if d["storey"] in self.storeys] if self.storeys else []
        if not bb:
            return set()
        allv = np.concatenate(bb)
        lo, hi = allv[:, :2].min(0) + 0.5, allv[:, :2].max(0) - 0.5
        out = set()
        for s in self.f.by_type("IfcStair"):
            parts = [p.GlobalId for p in parts_of(s)]
            v = [self.meshes[p]["verts"] for p in parts if p in self.meshes]
            if not v:
                continue
            v = np.concatenate(v)[:, :2]
            if (v.min(0) >= lo).all() and (v.max(0) <= hi).all():
                out.update(parts)
        return out

    def wall_is_exterior(self, g):
        tname = self.wall_type.get(g, "").upper()
        if tname.startswith(ALWAYS_INTERIOR):
            return False
        if self.meshes.get(g, {}).get("storey") in self.keep_all_walls_on:
            return True
        ext = self.props.get(g, {}).get("Pset_WallCommon", {}).get("IsExternal")
        if ext is not None:
            return bool(ext)
        return not tname.startswith(INTERIOR_UNLESS_EXTERNAL)

    def lod1_keep(self, g):
        d = self.meshes[g]
        cls = d["cls"]
        if cls in LOD1_DROP or g in self.core_stair_parts or (self.has_storeys and self.is_site(g)):
            return False
        if cls in ("IfcWall", "IfcWallStandardCase"):
            return self.wall_is_exterior(g)
        if cls in ("IfcDoor", "IfcWindow") and g in self.host:
            h = self.host[g]
            return self.wall_is_exterior(h) if h in self.wall_type else True
        return True

    def placement(self, el):
        return self.L @ upl.get_local_placement(el.ObjectPlacement)

    # ---- rooms (needed first: flat regions for the triangle budget)
    def rooms(self):
        zone_of = {}
        for z in self.f.by_type("IfcZone"):
            for rel in z.IsGroupedBy or ():
                for o in rel.RelatedObjects:
                    zone_of[o.GlobalId] = z
        out = []
        for sp in sorted(self.f.by_type("IfcSpace"), key=lambda s: s.GlobalId):
            d = self.meshes.get(sp.GlobalId)
            if d is None:
                continue
            poly = plan_polygon(d)
            if poly is None:
                continue
            pp = self.props.get(sp.GlobalId, {})
            z = zone_of.get(sp.GlobalId)
            flat = None
            if z is not None and "SampleCity_Flat" in self.props.get(z.GlobalId, {}):
                flat = self.props[z.GlobalId]["SampleCity_Flat"].get("UnitNumber") or z.Name
            st = d["storey"]
            height = pp.get("Qto_SpaceBaseQuantities", {}).get("Height", round(float(np.ptp(d["verts"][:, 2])), 3))
            rec = dict(guid=sp.GlobalId, name=sp.Name, room=sp.LongName, code=pp.get("SampleCity_Room", {}).get("RoomCode"),
                       flat=flat, zone=z.GlobalId if z else None, zone_name=z.Name if z else None, storey=st,
                       floor_level=self.storeys.get(st),
                       floor_z=round(float(d["verts"][:, 2].min()), 3), height=height,
                       external=bool(pp.get("Pset_SpaceCommon", {}).get("IsExternal", False)),
                       area=round(poly.area, 2),
                       polygon=[[round(x, 3), round(y, 3)] for x, y in poly.exterior.coords])
            if poly.interiors:
                rec["holes"] = [[[round(x, 3), round(y, 3)] for x, y in r.coords] for r in poly.interiors]
            rec["_poly"] = poly
            out.append(rec)
        return out

    # ---- LOD0
    def lod0(self, path, stem, rooms, include_site=True):
        W = glb.GlbWriter(extras={"frame": "block-local, glTF Y up", "source": self.path.name, "lod": 0})
        root = W.node(stem)
        shared = {}
        if not self.has_storeys:
            n_trees = self._site_lod0(W, root, stem, shared)
            stats = W.write(path, stem)
            stats.update(door_nodes=0, door_leaves=0, window_nodes=0, lift_nodes=0, furniture_nodes=0,
                         tree_nodes=n_trees, tiles=len(self.tiles))
            return stats, {}, {}, self._flat_budget([], {}, [], {})
        parts = self._lod0_parts(include_site)
        storey_node, kept_static, door_nodes, kept_nodes = self._emit_storeys(W, root, parts, shared)
        lift_nodes = {}
        for g in parts["lifts"]:                  # one car spans every storey it serves: LOD0 only, under the root
            d, el = self.meshes[g], self.ent[g]
            nm = lift_node_name(el.Name, g)
            self._element_node(W, d, self.placement(el), nm, root, {"guid": g, "ifc_class": d["cls"]})
            lift_nodes[g] = nm
        stats = W.write(path, stem)
        stats.update(door_nodes=len(door_nodes), door_leaves=sum(len(v) for v in self.door_leaves.values()),
                     window_nodes=len(parts["windows"]), lift_nodes=len(lift_nodes),
                     furniture_nodes=len(parts["furniture"]))
        budget = self._flat_budget(rooms, kept_static, parts["elems"], kept_nodes)
        return stats, door_nodes, lift_nodes, budget

    def _lod0_parts(self, include_site=True) -> dict:
        """What LOD0 draws, sorted once for LOD0 and the interior chunks (cached per include_site): the door,
        window, lift and instanced furniture guids (in meshes order, which fixes the node order), one static _Batch
        per storey group (its element ids index ``elems``, for the flat budget), the groups in draw order (storeys
        bottom up, then SITE) and each group's static node name (made unique over the whole file, so a chunk names
        its storey node exactly as LOD0 does)."""
        if include_site in self._parts:
            return self._parts[include_site]
        doors = [g for g, d in self.meshes.items() if d["cls"] == "IfcDoor"]
        windows = [g for g, d in self.meshes.items() if d["cls"] == "IfcWindow"]
        lifts = [g for g, d in self.meshes.items() if d["cls"] == "IfcTransportElement"]
        furniture = self.repeated_furniture([g for g, d in self.meshes.items() if d["cls"] in FURNITURE
                                             and (include_site or d["storey"] in self.storeys)])
        skip = set(doors) | set(windows) | set(lifts) | set(furniture)
        groups, elems = {}, []
        for g, d in self.meshes.items():
            if g in skip or d["cls"] == "IfcSpace":
                continue
            grp = self._group(g)
            if grp == "SITE" and not include_site:
                continue
            groups.setdefault(grp, _Batch()).add(d, len(elems))
            elems.append(g)
        placed = {self.meshes[g]["storey"] for g in doors + windows + furniture}
        order = [s for s in self.storey_order if s in groups or s in placed]
        order += ["SITE"] if "SITE" in groups else []
        static_name, used = {}, set()
        for grp in order:
            nm = f"{safe(grp)}_static"
            while nm in used:
                nm += "_"
            used.add(nm)
            static_name[grp] = nm
        self._parts[include_site] = dict(doors=doors, windows=windows, lifts=lifts, furniture=furniture,
                                         groups=groups, elems=elems, order=order, static_name=static_name)
        return self._parts[include_site]

    def _emit_storeys(self, W, root, parts, shared, storey=None):
        """The storey body of LOD0, under ``root``: for each storey group its <storey>_static node (the static
        batch, extras storey) and under it the storey's DOOR_ (frame + leaves pivoting on the hinge), WIN_ and
        FURN_ nodes. Without ``storey`` it writes every group and every door, window and furniture node (LOD0: one
        whose storey has no node hangs from root); with ``storey`` only that group and its elements (an interior
        chunk). Nodes are made in LOD0's order (all statics, then doors, windows, furniture) whichever caller
        writes them, which keeps LOD0 byte-identical. ``shared`` is the per-file mesh reuse dict. Returns (storey
        nodes, kept static triangles, door node names, kept element triangles) for the flat budget."""
        grps = parts["order"] if storey is None else [storey]
        mine = (lambda g: True) if storey is None else (lambda g: self._group(g) == storey)  # noqa: E731
        storey_node, kept_static = {}, {}
        for grp in grps:
            nm = parts["static_name"][grp]
            if grp in parts["groups"]:
                storey_node[grp], kept_static[grp] = parts["groups"][grp].emit(W, nm, self.materials, parent=root,
                                                                              extras={"storey": grp})
            else:
                storey_node[grp], kept_static[grp] = W.node(nm, parent=root, extras={"storey": grp}), []
        door_nodes, kept_nodes = {}, {}
        for g in filter(mine, parts["doors"]):
            nm = door_node_name(self.ent[g].Name, g)
            kept_nodes[g] = self._door_node(W, g, nm, storey_node.get(self.meshes[g]["storey"], root), shared)
            door_nodes[g] = nm
        for g in filter(mine, parts["windows"]):
            d, el = self.meshes[g], self.ent[g]
            t = uel.get_type(el)
            kept_nodes[g] = self._element_node(W, d, self.placement(el), window_node_name(el.Name, g),
                                               storey_node.get(d["storey"], root),
                                               {"guid": g, "ifc_class": "IfcWindow", "storey": d["storey"]}, shared,
                                               safe(t.Name, 50) if t is not None and t.Name else None)[1]
        for g in filter(mine, parts["furniture"]):
            d, el = self.meshes[g], self.ent[g]
            t = uel.get_type(el)
            extras = {"guid": g, "ifc_class": d["cls"], "storey": d["storey"]}
            if t is not None and t.Name:
                extras["type"] = t.Name
            kept_nodes[g] = self._element_node(W, d, self.placement(el), furniture_node_name(el.Name, g),
                                               storey_node.get(self._group(g), root), extras, shared,
                                               safe(t.Name, 50) if t is not None and t.Name else None)[1]
        return storey_node, kept_static, door_nodes, kept_nodes

    # ---- interior chunks
    def interior_chunks(self, out_dir, stem, include_site=True) -> dict:
        """One glb per storey for engines that stream interiors floor by floor: <stem>_int_<tag>.glb (storey_tag)
        with exactly the storey's LOD0 nodes, written by _emit_storeys (as LOD0 writes them) under a root
        <stem>_int_<tag> without a transform, so the chunk sits in LOD0's block-local Y-up frame and overlays LOD0 /
        LOD1 under the same manifest transform. Lift cars and the site group stay in LOD0 only; a site-only file
        has no chunks. Door leaves are recorded again exactly as LOD0 recorded them (self.door_leaves).
        Returns {storey: glb stats + node counts, path, tag, elevation}, bottom up."""
        if not self.has_storeys:
            return {}
        parts = self._lod0_parts(include_site)
        out, used = {}, set()
        for grp in parts["order"]:
            if grp not in self.storeys:
                continue
            tag = storey_tag(grp)
            while tag in used:                    # 'L5' and 'L05' in one file: keep both
                tag += "_"
            used.add(tag)
            name, z = f"{stem}_int_{tag}", self.storeys[grp]
            W = glb.GlbWriter(extras={"frame": "block-local, glTF Y up", "source": self.path.name, "lod": 0,
                                      "chunk": "interior", "storey": grp, "elevation": z})
            root = W.node(name, extras={"storey": grp, "elevation": z})
            self._emit_storeys(W, root, parts, {}, storey=grp)
            path = Path(out_dir) / f"{name}.glb"
            stats = W.write(path, name)
            stats.update(glb.node_counts(n["name"] for n in W.nodes))
            out[grp] = dict(stats, path=path, tag=tag, elevation=z)
        return out

    def _group(self, g):
        """The LOD0 storey group of an element: its storey, or SITE for elements outside every storey."""
        st = self.meshes[g]["storey"]
        return st if st in self.storeys else "SITE"

    def repeated_furniture(self, guids) -> list:
        """The furniture worth instancing: elements whose triangles in their placement frame (and colours) equal
        another element's (geometry_key), as every occurrence of one furniture type does. Mirrored placements and
        one-off pieces stay in the static mesh. Returns guids in the given order."""
        by_key, key_of = {}, {}
        for g in guids:
            d, el = self.meshes[g], self.ent[g]
            if el.ObjectPlacement is None or not len(d["faces"]):
                continue
            M = self.placement(el)
            if abs(np.linalg.det(M[:3, :3]) - 1.0) > 1e-6:
                continue
            key_of[g] = geometry_key(xf(np.linalg.inv(M), d["verts"])[d["faces"]], d["colours"])
            by_key[key_of[g]] = by_key.get(key_of[g], 0) + 1
        return [g for g in guids if g in key_of and by_key[key_of[g]] > 1]

    def _element_node(self, W, d, M, name, parent, extras, shared=None, mesh_name=None):
        """A separately addressable element: mesh in its placement frame, node matrix = placement."""
        if abs(np.linalg.det(M[:3, :3]) - 1.0) > 1e-6:      # mirrored / scaled placement: bake into the parent frame
            M = np.eye(4)
        b = _Batch()
        b.add(d, 0, xf(np.linalg.inv(M), d["verts"])[d["faces"]])
        node, kept = b.emit(W, name, self.materials, parent=parent, matrix=M, extras=extras, shared=shared,
                            mesh_name=mesh_name)
        return node, [(xf(M, t), e) for t, e in kept]

    # ---- doors: frame + leaves
    def _door_node(self, W, g, name, parent, shared):
        """DOOR_ node (frame mesh, matrix = placement) with one child per leaf whose origin is its hinge / jamb
        point; records the leaves in self.door_leaves. Returns the kept triangles in the export frame."""
        d, el = self.meshes[g], self.ent[g]
        M = self.placement(el)
        extras = {"guid": g, "ifc_class": "IfcDoor", "storey": d["storey"]}
        self.door_leaves[g] = []
        if abs(np.linalg.det(M[:3, :3]) - 1.0) > 1e-6:      # mirrored placement: one baked mesh, no leaves
            return self._element_node(W, d, M, name, parent, extras)[1]
        loc = xf(np.linalg.inv(M), d["verts"])[d["faces"]]
        key = geometry_key(loc, d["colours"])
        if key not in self._door_specs:
            self._door_specs[key] = self._split_door(el, loc)
        labels, leaves = self._door_specs[key]
        t = uel.get_type(el)
        base = safe(t.Name, 50) if t is not None and t.Name else name
        frame, kept = _Batch(), []
        frame.add(d, 0, loc, sel=labels < 0)
        if frame.parts:
            node, k = frame.emit(W, name, self.materials, parent=parent, matrix=M, extras=extras, shared=shared,
                                 mesh_name=base)
            kept += [(xf(M, tr), e) for tr, e in k]
        else:
            node = W.node(name, matrix=glb.matrix_to_yup(M), parent=parent, extras=extras)
        for i, lf in enumerate(leaves):
            P = leaf_frame(lf["pivot"])
            b = _Batch()
            b.add(d, 0, loc - (*lf["pivot"], 0.0), sel=labels == i)
            sfx = leaf_suffix(i, len(leaves))
            _, k = b.emit(W, name + sfx, self.materials, parent=node, matrix=P, shared=shared, mesh_name=base + sfx,
                          extras={"leaf": i, "motion": lf["motion"]})
            kept += [(xf(M @ P, tr), e) for tr, e in k]
            self.door_leaves[g].append(dict(lf, node=name + sfx))
        return kept

    def _body_items(self, el):
        """Body representation items of an element with their item -> object 4x4 transforms (through mapped items)."""
        out = []
        reps = list(el.Representation.Representations) if el.Representation else []
        body = [r for r in reps if r.RepresentationIdentifier == "Body"] or reps

        def walk(items, T):
            for it in items:
                if it.is_a("IfcMappedItem"):
                    walk(it.MappingSource.MappedRepresentation.Items,
                         T @ np.asarray(upl.get_mappeditem_transformation(it), float))
                else:
                    out.append((it, T))
        if body:
            walk(body[0].Items, np.eye(4))
        return out

    def _item_tris(self, item):
        if item.id() not in self._items:
            sh = ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), item)
            v = np.array(sh.verts, float).reshape(-1, 3)
            self._items[item.id()] = v[np.array(sh.faces, dtype=np.int64).reshape(-1, 3)]
        return self._items[item.id()]

    def _split_door(self, el, loc):
        """Label every face of a door (``loc``: its triangles in the door frame) frame (-1) or leaf k by matching
        it to the separately tessellated items of the door's body representation. IfcOpenShell's parametric doors
        (add_door_representation) put panels and handles strictly between the jambs and above the threshold, and
        linings, casings and the threshold outside that box; panels are the tall items, handles go with the panel
        they sit on. A rolling door's curtain counts as a panel even where it reaches the floor (a shutter has no
        threshold). Returns (labels, leaf specs: nav_doorpose.leaf_spec); no leaves when a face cannot be matched
        to an item."""
        none = (np.full(len(loc), -1), [])
        try:
            items = [xf(T, self._item_tris(it).reshape(-1, 3)).reshape(-1, 3, 3) for it, T in self._body_items(el)]
        except Exception:  # noqa: BLE001  (an item the kernel cannot tessellate on its own: keep the door whole)
            return none
        items = [t for t in items if len(t)]
        if not items or not len(loc):
            return none
        lo = np.array([t.reshape(-1, 3).min(0) for t in items])
        hi = np.array([t.reshape(-1, 3).max(0) for t in items])
        w = float(el.OverallWidth or hi[:, 0].max())
        h = float(el.OverallHeight or (hi[:, 2].max() - lo[:, 2].min()))
        op = el.OperationType or ""
        rolling = "ROLLING" in op            # a shutter curtain has no threshold under it: it may reach the floor
        inner = [i for i in range(len(items))
                 if lo[i, 0] > LEAF_TOL and hi[i, 0] < w - LEAF_TOL
                 and (lo[i, 2] > LEAF_TOL or (rolling and hi[i, 2] - lo[i, 2] > 0.5 * h))]
        panels = sorted((i for i in inner if hi[i, 2] - lo[i, 2] > 0.5 * h), key=lambda i: (lo[i, 0], hi[i, 0]))
        if not panels:
            return none
        owner = np.full(len(items), -1)
        for k, p in enumerate(panels):
            owner[p] = k
        for i in inner:
            if owner[i] < 0:
                cx = (lo[i, 0] + hi[i, 0]) / 2
                owner[i] = min(range(len(panels)), key=lambda k: (not lo[panels[k], 0] <= cx <= hi[panels[k], 0],
                                                                  abs((lo[panels[k], 0] + hi[panels[k], 0]) / 2 - cx)))
        cent = np.concatenate([t.mean(axis=1) for t in items])
        item_of = np.concatenate([np.full(len(t), i) for i, t in enumerate(items)])
        c = loc.mean(axis=1)
        near, dist = np.empty(len(c), int), np.empty(len(c))
        for s in range(0, len(c), 256):
            d2 = ((c[s:s + 256, None, :] - cent[None]) ** 2).sum(-1)
            near[s:s + 256], dist[s:s + 256] = d2.argmin(1), np.sqrt(d2.min(1))
        if dist.max() > 1e-3:
            return none
        labels = owner[item_of[near]]
        return labels, [leaf_spec(op, lo[p], hi[p], k, len(panels)) for k, p in enumerate(panels)]

    # ---- site-only file: tiles
    def _site_lod0(self, W, root, stem, shared):
        """Tiles of TILE m. Ground, structure and furniture triangles are cut on the tile lines and merged per tile;
        every tree is a TREE_ node sharing one mesh per tree type. Fills self.tiles; returns the tree count."""
        batches, trees, box = {}, {}, {}

        def grow(ij, pts):
            lo, hi = pts.min(0), pts.max(0)
            b = box.get(ij)
            box[ij] = (lo, hi) if b is None else (np.minimum(b[0], lo), np.maximum(b[1], hi))

        for g, d in self.meshes.items():
            if d["cls"] == "IfcSpace":
                continue
            cat = self.site_category(g)
            if cat == "tree":
                M = self.placement(self.ent[g])
                ij = tile_index(M[:2, 3])
                trees.setdefault(ij, []).append((g, M))
                grow(ij, d["verts"])
                continue
            tris, src = split_grid(d["verts"][d["faces"]])
            ij_all = np.floor(tris.mean(axis=1)[:, :2] / TILE).astype(int)
            for key in np.unique(ij_all, axis=0):
                sel = (ij_all == key).all(axis=1)
                ij = (int(key[0]), int(key[1]))
                batches.setdefault((ij, cat), _Batch()).add_tris(tris[sel], d["colours"][src[sel]],
                                                                 np.zeros(int(sel.sum()), int))
                grow(ij, tris[sel].reshape(-1, 3))
        for ij in sorted(box):
            tn = f"{safe(stem)}_{ij[0]}_{ij[1]}"
            tnode = W.node(tn, parent=root, extras={"tile": list(ij), "size": TILE})
            for cat in ("ground", "structures", "furniture"):
                if (ij, cat) in batches:
                    batches[(ij, cat)].emit(W, f"{tn}_{cat}", self.materials, parent=tnode)
            for g, M in sorted(trees.get(ij, []), key=lambda t: t[0]):
                el = self.ent[g]
                t = uel.get_type(el)
                self._element_node(W, self.meshes[g], M, tree_node_name(el.Name, g), tnode,
                                   {"guid": g, "ifc_class": el.is_a(), "type": t.Name}, shared, safe(t.Name, 50))
            lo, hi = box[ij]
            self.tiles.append(dict(node=tn, index=list(ij), size=TILE, trees=len(trees.get(ij, [])),
                                   bounds=[r3(lo), r3(hi)]))
        return sum(len(v) for v in trees.values())

    def _flat_budget(self, rooms, kept_static, elems, kept_nodes):
        """LOD0 triangles per flat: triangles of elements named after the unit, then triangles whose centroid lies
        in the unit's rooms (grown 0.15 m to take in its walls), plus an equal share of the storey's common
        triangles (slab, lobby walls, ledges of no unit). Instanced door and window nodes count in full: the
        budget is what is drawn, not what is stored."""
        regions = {}
        for r in rooms:
            if r["flat"]:
                regions.setdefault(r["storey"], {}).setdefault(r["flat"], []).append(r["_poly"])
        regions = {s: [(u, unary_union(p).buffer(0.15, join_style="mitre")) for u, p in sorted(v.items())]
                   for s, v in regions.items()}
        unit_of_elem = np.array([unit_of(self.ent[g].Name) or "" for g in elems] + [""], dtype=object)
        per_flat, common = {}, {}
        for st, kept in kept_static.items():
            regs = regions.get(st, [])
            for tris, eids in kept:
                self._attribute(tris, unit_of_elem[eids], regs, per_flat, common, st)
        for g, kept in kept_nodes.items():
            st = self.meshes[g]["storey"]
            u = unit_of(self.ent[g].Name) or ""
            for tris, eids in kept:
                self._attribute(tris, np.full(len(tris), u, dtype=object), regions.get(st, []), per_flat, common, st)
        storey_of = {}
        for st, regs in regions.items():
            for u, _ in regs:
                storey_of[u] = st
        share = {u: common.get(st, 0) / len(regions[st]) for u, st in storey_of.items()}
        total = {u: per_flat.get(u, 0) + share[u] for u in storey_of}
        vals = list(total.values())
        over = sorted([u for u, v in total.items() if v > LOD0_BUDGET_PER_FLAT])
        return dict(limit=LOD0_BUDGET_PER_FLAT, flats=len(total),
                    max=round(max(vals), 1) if vals else 0, mean=round(float(np.mean(vals)), 1) if vals else 0,
                    min=round(min(vals), 1) if vals else 0, over_budget=over,
                    per_flat={u: dict(own=int(per_flat.get(u, 0)), shared=round(share[u], 1), total=round(total[u], 1))
                              for u in sorted(total)},
                    common_by_storey={k: int(v) for k, v in sorted(common.items())})

    @staticmethod
    def _attribute(tris, units, regs, per_flat, common, st):
        if not len(tris):
            return
        units = np.asarray(units, dtype=object)
        named = units != ""
        for u in set(units[named].tolist()):
            per_flat[u] = per_flat.get(u, 0) + int((units == u).sum())
        rest = ~named
        c = tris.mean(axis=1)
        for u, reg in regs:
            if not rest.any():
                break
            m = rest & shapely.contains_xy(reg, c[:, 0], c[:, 1])
            if m.any():
                per_flat[u] = per_flat.get(u, 0) + int(m.sum())
                rest &= ~m
        common[st] = common.get(st, 0) + int(rest.sum())

    # ---- LOD1
    def lod1(self, path, stem):
        W = glb.GlbWriter(extras={"frame": "block-local, glTF Y up", "source": self.path.name, "lod": 1})
        root = W.node(stem)
        if not self.has_storeys:
            kept_n, proxies = self._site_lod1(W, root, stem)
            stats = W.write(path, stem)
            stats.update(elements=kept_n, tree_proxies=proxies)
            return stats
        b, kept_n, openings = _Batch(), 0, []
        for g, d in self.meshes.items():
            if not self.lod1_keep(g):
                continue
            if d["cls"] in ("IfcDoor", "IfcWindow"):
                openings.append(g)
            else:
                b.add(d, kept_n)
            kept_n += 1
        if b.parts:
            b.emit(W, f"{safe(stem)}_shell", self.materials, parent=root)
        if openings:                 # instanced as in LOD0 (whole door, closed): one mesh per door / window type
            grp, shared = W.node(f"{safe(stem)}_openings", parent=root), {}
            for g in openings:
                d, el = self.meshes[g], self.ent[g]
                t = uel.get_type(el)
                nm = "L1" + (door_node_name if d["cls"] == "IfcDoor" else window_node_name)(el.Name, g)
                self._element_node(W, d, self.placement(el), nm, grp, {"guid": g, "ifc_class": d["cls"]}, shared,
                                   "L1_" + safe(t.Name, 47) if t is not None and t.Name else None)
        stats = W.write(path, stem)
        stats.update(elements=kept_n, opening_nodes=len(openings))
        return stats

    def _site_lod1(self, W, root, stem):
        """The LOD0 tiles without furniture and slender items (plan extent < SITE_LOD1_MIN: linkway columns, posts);
        each tree becomes an eight-triangle crown in its canopy colour, merged into its tile's shell. Returns
        (elements, crowns)."""
        batches, kept_n, crowns = {}, 0, 0
        for g, d in self.meshes.items():
            cat = self.site_category(g)
            if d["cls"] in LOD1_DROP or cat == "furniture" or np.ptp(d["verts"][:, :2], axis=0).max() < SITE_LOD1_MIN:
                continue
            if cat == "tree":
                v = d["verts"]
                lo, hi = v.min(0), v.max(0)
                lo = np.array([lo[0], lo[1], lo[2] + 0.4 * (hi[2] - lo[2])])
                tris = d["verts"][d["faces"]]
                top = tris.mean(axis=1)[:, 2] >= lo[2]
                keys, cnt = np.unique(np.round(clean_colours(d["colours"][top if top.any() else slice(None)]), 3),
                                      axis=0, return_counts=True)
                crown = octahedron(lo, hi)
                ij = tile_index(self.placement(self.ent[g])[:2, 3])
                batches.setdefault(ij, _Batch()).add_tris(crown, np.repeat(keys[cnt.argmax()][None], len(crown), 0),
                                                          np.zeros(len(crown), int))
                crowns += 1
                continue
            tris, src = split_grid(d["verts"][d["faces"]])
            ij_all = np.floor(tris.mean(axis=1)[:, :2] / TILE).astype(int)
            for key in np.unique(ij_all, axis=0):
                sel = (ij_all == key).all(axis=1)
                batches.setdefault((int(key[0]), int(key[1])), _Batch()).add_tris(
                    tris[sel], d["colours"][src[sel]], np.zeros(int(sel.sum()), int))
            kept_n += 1
        for ij in sorted(batches):
            tn = f"{safe(stem)}_{ij[0]}_{ij[1]}"
            tnode = W.node(tn, parent=root, extras={"tile": list(ij), "size": TILE})
            batches[ij].emit(W, f"{tn}_shell", self.materials, parent=tnode)
        return kept_n, crowns

    # ---- LOD2
    def massing(self):
        """(footprint polygons, z0, body top, [(roof structure polygon, top z)]) in the local frame."""
        bldg = [g for g in self.meshes if not self.is_site(g) and self.meshes[g]["cls"] not in LOD1_DROP]
        if not bldg:
            return [], 0.0, 0.0, []
        slab_polys = []
        for g in bldg:
            el = self.ent[g]
            if not el.is_a("IfcSlab") or g in self.stair_parts or getattr(el, "PredefinedType", None) == "USERDEFINED":
                continue                                    # AC ledges are USERDEFINED slabs
            d = self.meshes[g]
            tris = d["verts"][d["faces"]]
            n, ln = glb.face_normals(tris)
            up = tris[(n[:, 2] / (ln + 1e-12)) > 0.9]
            slab_polys += [p for p in (Polygon(t[:, :2]) for t in up) if p.area > 1e-8]
        allv = np.concatenate([self.meshes[g]["verts"] for g in bldg])
        if slab_polys:
            fp = shapely.union_all(slab_polys, grid_size=1e-3)
            fp = fp.buffer(0.05, join_style="mitre").buffer(-0.05, join_style="mitre")
        else:
            fp = MultiPoint(allv[:, :2]).convex_hull
        parts = [Polygon(p.exterior).simplify(0.005) for p in getattr(fp, "geoms", [fp])
                 if p.geom_type == "Polygon" and p.area > 0.5]
        z0 = float(allv[:, 2].min())
        roof = [g for g in bldg if self.meshes[g]["storey"] == self.top and self.ent[g].is_a("IfcSlab")
                and getattr(self.ent[g], "PredefinedType", None) == "ROOF"
                and abs(self.meshes[g]["verts"][:, 2].max() - self.storeys[self.top]) < 0.05]
        if roof:
            body = max(float(self.meshes[g]["verts"][:, 2].max()) for g in roof)
            for g in bldg:
                el = self.ent[g]
                parapet = self.wall_type.get(g, "").upper().startswith("PAR") or getattr(el, "PredefinedType", None) == "PARAPET"
                if self.meshes[g]["storey"] == self.top and el.is_a("IfcWall") and parapet:
                    body = max(body, float(self.meshes[g]["verts"][:, 2].max()))
        else:
            body = float(allv[:, 2].max())
        hulls = []
        for g in bldg:
            v = self.meshes[g]["verts"]
            if g in self.core_stair_parts or v[:, 2].max() <= body + 0.05:
                continue
            h = MultiPoint(v[:, :2]).convex_hull
            if h.area > 0.01:
                hulls.append((h, float(v[:, 2].max())))
        towers = []
        if hulls:
            u = unary_union([h for h, _ in hulls])
            for p in getattr(u, "geoms", [u]):
                top = max(z for h, z in hulls if h.intersects(p))
                towers.append((Polygon(p.exterior).simplify(0.005), top))
        return parts, z0, body, towers

    def lod2(self, path, stem):
        W = glb.GlbWriter(extras={"frame": "block-local, glTF Y up", "source": self.path.name, "lod": 2})
        root = W.node(stem)
        parts, z0, body, towers = self.massing()
        walls, roofs = [], []
        for p in parts:
            s, t, _ = prism(p, z0, body)
            walls.append(s)
            roofs.append(t)
        for p, top in sorted(towers, key=lambda t: (t[0].bounds, t[1])):
            s, t, _ = prism(p, body, top)
            walls.append(s)
            roofs.append(t)
        prims = []
        for tris, key in ((walls, "ext"), (roofs, "roof")):
            if tris:
                tr = np.concatenate(tris)
                name, rgb, a = PALETTE[key]
                prims.append((glb.soup(glb.to_yup(tr), glb.box_uvs(tr)), W.material(name, [*rgb, 1.0 - a])))
        if prims:
            W.node(f"{safe(stem)}_massing", W.mesh(f"{safe(stem)}_massing", prims), parent=root)
        stats = W.write(path, stem)
        stats.update(footprint_area=round(sum(p.area for p in parts), 2), body_z=[round(z0, 3), round(body, 3)],
                     roof_structures=len(towers))
        return stats, parts, (z0, body, towers)

    # ---- gameplay data
    def doors(self, door_nodes):
        out = []
        for d in sorted(self.f.by_type("IfcDoor"), key=lambda e: e.GlobalId):
            g = d.GlobalId
            pp = self.props.get(g, {})
            nav = pp.get("Navigation", {})
            M = self.placement(d)
            X, Y, O = M[:3, 0], M[:3, 1], M[:3, 3]
            w = float(d.OverallWidth or 0.0)
            op = d.OperationType or ""
            if "SWING" in op and "DOUBLE_DOOR" in op:
                side, hinge = "both", O
            elif "SWING" in op and op.endswith("RIGHT"):
                side, hinge = "right", O + X * w
            elif "SWING" in op:
                side, hinge = "left", O
            else:
                side, hinge = "none", O
            leaves = [leaf_record(M, lf) for lf in self.door_leaves.get(g, [])]
            swing = next((lf for lf in leaves if lf["motion"] == "swing"), None)
            if swing is not None:
                hinge = np.array(swing["pivot"])
            unit = unit_of(d.Name)
            out.append(dict(guid=g, name=d.Name, node=door_nodes.get(g), node_prefix=door_nodes.get(g),
                            leaf_node=leaves[0]["node"] if leaves else None,
                            storey=self.meshes.get(g, {}).get("storey") or _storey_of(d), flat=unit,
                            kind=nav.get("DoorKind"), passable=nav.get("Passable", True),
                            width=round(w, 3), height=round(float(d.OverallHeight or 0.0), 3),
                            clear_width=nav.get("ClearWidth"),
                            hinge=r3(hinge), hinge_side=side, origin=r3(O), along_wall=r3(X), swing_side=r3(Y),
                            swing=None if swing is None else dict(axis=swing["axis"], open_sign=swing["open_sign"],
                                                                  max_angle_deg=swing["max_angle_deg"],
                                                                  opens_towards=r3(Y)),
                            leaf_thickness=leaves[0]["thickness"] if leaves else None, leaves=leaves,
                            centre=r3(O + X * w / 2), operation=op, host_wall=self.host.get(g),
                            external=pp.get("Pset_DoorCommon", {}).get("IsExternal"),
                            from_face=nav.get("FromSpace"), to_face=nav.get("ToSpace")))
        return out

    def portals(self, doors, rooms):
        """Door openings as polylines for navmesh off-mesh links: threshold on the wall centre line, opening
        outline, and a link 0.3 m beyond each wall face, with the rooms found at either end.

        A link end that lies in no room polygon is flagged in ``outside`` when it is outside the storey's built
        outline, and otherwise takes its room from the door's Navigation FromSpace / ToSpace (IfcSpace names, or
        face ids that the storey prefix turns into one). Returns (portals, problems): a problem for every passable
        door with an end that is neither a room nor outside (a stair or service room without an IfcSpace)."""
        ops = {g: o for g, o in self.opening_of.items() if self.f.by_guid(g).is_a("IfcDoor")}
        geo = meshcache.tessellate(self.f, include=list(ops.values())) if ops else {}
        by_storey, by_name, name_of, outlines = {}, {}, {}, {}
        for r in rooms:
            by_storey.setdefault(r["storey"], []).append(r)
            by_name.setdefault(r["name"], []).append(r)
            name_of[r["guid"]] = r["name"]
        out, problems = [], []
        for dr in doors:
            op = ops.get(dr["guid"])
            if op is None or op.GlobalId not in geo:
                continue
            Mo = self.L @ upl.get_local_placement(op.ObjectPlacement)
            v = xf(np.linalg.inv(Mo), xf(self.L, geo[op.GlobalId]["verts"]))
            (x0, y0, z0), (x1, y1, z1) = v.min(0), v.max(0)
            ym = (y0 + y1) / 2
            P = lambda x, y, z: xf(Mo, np.array([x, y, z]))  # noqa: E731
            a, b = P((x0 + x1) / 2, y0 - 0.2, z0), P((x0 + x1) / 2, y1 + 0.2, z0)
            rooms_here = by_storey.get(dr["storey"], [])
            sides = [next((r["guid"] for r in rooms_here if r["_poly"].contains(Point(pt[0], pt[1]))), None) for pt in (a, b)]
            if None in sides:
                outline = self._outline(dr["storey"], rooms_here, outlines)
                inside = [outline.contains(Point(pt[0], pt[1])) for pt in (a, b)]
            else:
                inside = [True, True]
            sides, outside, names = self._resolve_sides(dr, sides, (a, b), inside, by_name, name_of)
            corners = [(x0, z0), (x1, z0), (x1, z1), (x0, z1), (x0, z0)]
            out.append(dict(door=dr["guid"], node=dr["node"], storey=dr["storey"], kind=dr["kind"], passable=dr["passable"],
                            width=round(float(x1 - x0), 3), floor_z=round(float(P(0, ym, z0)[2]), 3),
                            threshold=[r3(P(x0, ym, z0)), r3(P(x1, ym, z0))],
                            outline=[r3(P(x, ym, z)) for x, z in corners], link=[r3(a), r3(b)], spaces=sides,
                            space_names=names, outside=outside))
            if dr["passable"]:
                for i in (0, 1):
                    if sides[i] is None and not outside[i]:
                        problems.append(f"portal {dr['name']} ({dr['guid']}): passable door with no room at link end "
                                        f"{i} (FromSpace={dr['from_face']!r}, ToSpace={dr['to_face']!r})")
        return out, problems

    def _outline(self, storey, rooms_here, cache):
        """The storey's built outline: rooms, walls and columns, holes filled (a stair core counts as inside)."""
        if storey not in cache:
            u = unary_union([*(r["_poly"] for r in rooms_here), self._storey_footprints(storey)])
            u = u.buffer(0.05, join_style="mitre")
            cache[storey] = unary_union([Polygon(p.exterior) for p in getattr(u, "geoms", [u])
                                         if p.geom_type == "Polygon" and not p.is_empty])
        return cache[storey]

    @staticmethod
    def _resolve_sides(dr, sides, ends, inside, by_name, name_of):
        """Complete the link ends that lie in no room. An end outside the storey outline leads outside; an end
        inside it takes its room from the door's FromSpace / ToSpace (with one end known, the name that is not
        that room; with neither, the names go to the ends nearest their rooms). Returns (room guids, outside
        flags, IfcSpace names) per end."""
        def room(name):
            if name is None or str(name).lower() in OUTSIDE:
                return None
            rs = by_name.get(name) or by_name.get(f"{dr['storey']}-{name}") or []
            rs = [r for r in rs if r["storey"] == dr["storey"]] or rs
            return rs[0] if rs else None

        def dist(k, pt):
            return 50.0 if cand[k] is None else cand[k]["_poly"].distance(Point(pt[0], pt[1]))

        sides = list(sides)
        outside = [sides[i] is None and not inside[i] for i in (0, 1)]
        cand = [room(n) for n in (dr["from_face"], dr["to_face"])]
        real = [k for k in (0, 1) if cand[k] is not None and cand[k]["guid"] not in sides]
        missing = [i for i in (0, 1) if sides[i] is None and not outside[i]]
        if len(missing) == 2 and len(real) == 2:
            straight = dist(0, ends[0]) + dist(1, ends[1])
            if dist(1, ends[0]) + dist(0, ends[1]) < straight:
                real = real[::-1]
            sides = [cand[real[0]]["guid"], cand[real[1]]["guid"]]
        elif missing and real:
            for i in missing:
                if real:
                    k = min(real, key=lambda k: dist(k, ends[i]))
                    sides[i] = cand[k]["guid"]
                    real.remove(k)
        return sides, outside, ["outside" if o else name_of.get(s) for s, o in zip(sides, outside)]

    def _storey_footprints(self, storey, classes=("IfcWall", "IfcColumn")):
        hulls = [MultiPoint(d["verts"][:, :2]).convex_hull for g, d in self.meshes.items()
                 if d["storey"] == storey and d["cls"] in classes]
        return unary_union([h for h in hulls if h.area > 1e-6]) if hulls else Polygon()

    def lifts(self, doors):
        cars = [t for t in self.f.by_type("IfcTransportElement") if t.PredefinedType == "ELEVATOR"
                or self.props.get(t.GlobalId, {}).get("Navigation", {}).get("Elevator")]
        cars.sort(key=lambda t: _natural(t.Name or t.GlobalId))
        walls = self._storey_footprints(self.lowest, ("IfcWall",)) if self.lowest else Polygon()
        out = []
        for i, car in enumerate(cars, 1):
            g = car.GlobalId
            m = re.search(r"lift\s*(\w+)\s*car", car.Name or "", re.I)
            k = m.group(1) if m else str(i)
            nav = self.props.get(g, {}).get("Navigation", {})
            d = self.meshes.get(g)
            cb = [float(x) for x in (*d["verts"][:, :2].min(0), *d["verts"][:, :2].max(0))] if d else None
            shaft = self._shaft_bounds(cb, walls) if cb else None
            served = [s for s in (nav.get("ServesLevels") or "").split(",") if s in self.storeys] or \
                [s for s in self.storey_order if s != self.top]
            pat = re.compile(rf"\blift\s+{re.escape(k)}\b.*landing door", re.I)
            mine = [dr for dr in doors if dr["kind"] == "lift" and pat.search(dr["name"] or "")]
            if not mine and nav.get("Shaft"):
                mine = [dr for dr in doors if dr["kind"] == "lift" and nav["Shaft"] in (dr["from_face"], dr["to_face"])]
            if not mine and shaft:
                grow = Polygon.from_bounds(*shaft).buffer(0.4, join_style="mitre")
                mine = [dr for dr in doors if dr["kind"] == "lift" and grow.contains(Point(dr["centre"][:2]))]
            by_level = {dr["storey"]: dr["node"] for dr in mine}
            out.append(dict(name=f"Lift {k}", car_guid=g, car_node=lift_node_name(car.Name, g) if d else None,
                            car_xy_bounds=[round(x, 3) for x in cb] if cb else None,
                            car_z=r3([d["verts"][:, 2].min(), d["verts"][:, 2].max()]) if d else None,
                            shaft=nav.get("Shaft"), shaft_xy_bounds=[round(x, 3) for x in shaft] if shaft else None,
                            served_levels=served, levels={s: self.storeys[s] for s in served},
                            landing_doors=[by_level[s] for s in served if s in by_level],
                            landing_door_by_level={s: by_level.get(s) for s in served}))
        return out

    @staticmethod
    def _shaft_bounds(cb, walls, reach=6.0):
        """Shaft = the car box grown until it meets the enclosing walls (ray casts from the car centre)."""
        cx, cy = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
        if walls.is_empty:
            return [cb[0] - 0.25, cb[1] - 0.25, cb[2] + 0.25, cb[3] + 0.25]
        out = []
        for dx, dy, half in ((-1, 0, (cb[2] - cb[0]) / 2), (0, -1, (cb[3] - cb[1]) / 2),
                             (1, 0, (cb[2] - cb[0]) / 2), (0, 1, (cb[3] - cb[1]) / 2)):
            ray = LineString([(cx, cy), (cx + dx * reach, cy + dy * reach)])
            hit = ray.intersection(walls)
            dist = min((Point(cx, cy).distance(gm) for gm in getattr(hit, "geoms", [hit]) if not gm.is_empty),
                       default=None)
            dist = dist if dist is not None and dist >= half - 1e-6 else half + 0.25
            out.append(cx + dx * dist if dx else cy + dy * dist)
        return [out[0], out[1], out[2], out[3]]

    def entrance_spawns(self, entrances):
        """Spawn points given in the estate frame (manifest.entrance_spawns: the pedestrian entrances of the
        masterplan, each set out along the path that leaves it) moved into the export frame."""
        R = self.L[:3, :3]
        out = []
        for e in entrances:
            p, en = xf(self.L, [*e["position"][:2], 0.0]), xf(self.L, [*e["entrance"][:2], 0.0])
            out.append(dict(name=e["name"], position=r3(p), facing=r3(R @ np.array([*e["facing"][:2], 0.0])),
                            entrance=r3(en), source=e.get("source", "masterplan")))
        return out

    def spawn_rooms(self, spawns, rooms):
        """Record the ground-storey IfcSpace each spawn stands in (``room``, None outside every room). A spawn in an
        enclosed room (Pset_SpaceCommon.IsExternal not set: a hawker hall, a lobby) is not at an entrance: its
        'Entrance' / 'Void deck entrance' label becomes 'Interior'. Void decks, forecourts and walkways are
        external spaces and keep their labels."""
        here = [r for r in rooms if r["storey"] == self.lowest]
        for sp in spawns:
            pt = Point(*sp["position"][:2])
            r = next((r for r in here if r["_poly"].contains(pt)), None)
            sp["room"] = r["name"] if r else None
            if r is not None and not r["external"]:
                label = next((lb for lb in ENTRANCE_LABELS if sp["name"].startswith(lb)), None)
                if label:
                    sp["name"] = "Interior" + sp["name"][len(label):]
        return spawns

    def spawns(self, rooms, doors, footprint_parts, kind=None):
        """Derived entrances, for IFCs outside the masterplan: rays from the lift lobby along +-x / +-y that stay
        on the ground storey's walkable rooms and leave the building without crossing a wall or column; the spawn
        sits 1.5 m outside the edge, at ground level. Named 'Void deck entrance' for residential blocks (kind
        'block', or an open ground storey when the kind is unknown), otherwise 'Entrance'."""
        lo = self.lowest
        if lo is None:
            return []
        z = self.storeys[lo]
        deck = [r["_poly"] for r in rooms if r["storey"] == lo and r["external"]] or \
            [r["_poly"] for r in rooms if r["storey"] == lo]
        if not deck:
            return []
        walk = unary_union(deck).buffer(0.02, join_style="mitre")
        obstacles = self._storey_footprints(lo).buffer(-0.01, join_style="mitre")
        inside = unary_union([walk, self._storey_footprints(lo), *footprint_parts]).buffer(-0.05, join_style="mitre")
        lobby = []                                          # a point in front of every ground-floor lift door
        for dr in doors:
            if dr["storey"] == lo and dr["kind"] == "lift":
                c, y = np.array(dr["centre"]), np.array(dr["swing_side"])
                for s in (-1.0, 1.0):
                    p = c + s * y
                    if walk.contains(Point(p[0], p[1])):
                        lobby.append(p[:2])
                        break
        clusters = []                                       # one origin per lift lobby (doors within 8 m)
        for p in lobby:
            for cl in clusters:
                if np.linalg.norm(np.mean(cl, axis=0) - p) < 8.0:
                    cl.append(p)
                    break
            else:
                clusters.append([p])
        origins = []
        for cl in clusters:
            o = np.mean(cl, axis=0)
            origins.append(o if walk.contains(Point(*o)) else cl[0])
        if not origins:
            rp = walk.representative_point()
            origins = [np.array([rp.x, rp.y])]
        open_deck = any(r["external"] for r in rooms if r["storey"] == lo)
        label = "Void deck entrance" if (kind == "block" if kind else open_deck) else "Entrance"
        out = []
        for k, origin in enumerate(origins, 1):
            for compass, d in (("E", (1.0, 0.0)), ("W", (-1.0, 0.0)), ("N", (0.0, 1.0)), ("S", (0.0, -1.0))):
                d = np.array(d)
                ex = self._exit(walk, obstacles, inside, origin, d)
                if ex is None:
                    continue
                p = ex + d * 1.5
                if any(np.hypot(p[0] - q["position"][0], p[1] - q["position"][1]) < 4.0 for q in out):
                    continue
                name = f"{label} {compass}" + (f" (lobby {k})" if len(origins) > 1 else "")
                out.append(dict(name=name, position=r3([p[0], p[1], z]), facing=r3([-d[0], -d[1], 0.0]),
                                entrance=r3([ex[0], ex[1], z]), source="derived"))
        return out

    @staticmethod
    def _exit(walk, obstacles, inside, origin, d):
        """Where a ray from origin along d leaves the walkable deck, if it gets outside without crossing walls."""
        perp = np.array([-d[1], d[0]])
        for off in (0.0, 0.6, -0.6, 1.2, -1.2):
            o = origin + perp * off
            if not walk.contains(Point(*o)):
                continue
            inter = LineString([o, o + d * 500.0]).intersection(walk)
            piece = next((gm for gm in getattr(inter, "geoms", [inter]) if gm.geom_type == "LineString"
                          and gm.distance(Point(*o)) < 1e-6), None)
            if piece is None:
                continue
            cs = np.array(piece.coords)
            ex = cs[np.argmax(cs @ d)]
            if not (LineString([o, ex]).intersects(obstacles) or inside.contains(Point(*(ex + d * 0.3)))):
                return ex
        return None

    def climbable(self):
        out = []
        for g in sorted(self.props):
            nav = self.props[g].get("Navigation", {})
            if not nav.get("ClimbableTopEdge") or g not in self.meshes:
                continue
            d = self.meshes[g]
            loops, ztop = top_edges(d)
            if loops:
                el = self.ent[g]
                out.append(dict(guid=g, name=el.Name, ifc_class=el.is_a(), storey=d["storey"], top_z=round(ztop, 3),
                                floor_level=self.storeys.get(d["storey"]), polylines=loops))
        return out

    def flats(self, rooms, doors, budget):
        rooms_by_zone = {}
        for r in rooms:
            if r["zone"]:
                rooms_by_zone.setdefault(r["zone"], []).append(r)
        mains = {}
        for dr in doors:
            if dr["flat"] and (dr["kind"] == "main" or (dr["name"] or "").endswith("Main door")):
                mains.setdefault(dr["flat"], dr["guid"])
        out = []
        for z in sorted(self.f.by_type("IfcZone"), key=lambda z: z.Name or ""):
            if "SampleCity_Flat" not in self.props.get(z.GlobalId, {}):
                continue
            rs = rooms_by_zone.get(z.GlobalId, [])
            p = self.props[z.GlobalId]["SampleCity_Flat"]
            unit = p.get("UnitNumber") or z.Name
            out.append(dict(unit=unit, guid=z.GlobalId, name=z.LongName or z.Name, type=p.get("FlatType"),
                            template=p.get("Template"), variant=p.get("Variant"), signature=p.get("Signature"),
                            storey=p.get("Storey") or (rs[0]["storey"] if rs else None),
                            ifa=p.get("InternalFloorArea"), gross_area=p.get("GrossArea"),
                            rooms=[dict(guid=r["guid"], code=r["code"], name=r["room"]) for r in rs],
                            main_door=mains.get(unit),
                            lod0_triangles=budget["per_flat"].get(unit, {}).get("total")))
        return out

    def zones(self, rooms):
        """Zones that are not flats (car-park decks, shops, ...): name, rooms and their own property set."""
        count = {}
        for r in rooms:
            if r["zone"] and not r["flat"]:
                count.setdefault(r["zone"], []).append(r["guid"])
        out = []
        for z in sorted(self.f.by_type("IfcZone"), key=lambda z: z.Name or ""):
            pp = self.props.get(z.GlobalId, {})
            if "SampleCity_Flat" in pp:
                continue
            out.append(dict(guid=z.GlobalId, name=z.Name, long_name=z.LongName, rooms=count.get(z.GlobalId, []),
                            props=next((v for k, v in sorted(pp.items()) if k.startswith("SampleCity")), {})))
        return out


def _storey_of(el):
    c = uel.get_container(el)
    return c.Name if c is not None else None


def _natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _sha(path):
    return meshcache.sha256(path) if Path(path).exists() else None


def agent_settings():
    try:
        from estate import config
        a = config.load().get("agent", {})
    except Exception:  # noqa: BLE001  (no estate config: the rule constants)
        a = {}
    return dict(radius=float(a.get("radius", AGENT_RADIUS)), height=float(a.get("height", AGENT_HEIGHT)),
                max_step=float(a.get("step", AGENT_STEP)),
                note="nav3d walk test: every room, lobby and the roof reachable at r=0.30 m, h=1.8 m, step 0.4 m. "
                     "Interior doors are 0.8-0.9 m: keep the capsule / navmesh agent radius <= 0.30 m or use <= 10 cm "
                     "navmesh cells indoors.")


# ----------------------------------------------------------------------------- entry point
def export_building(ifc_path, out_dir=None, stem=None, local_matrix=None, spawns=None, include_site=True,
                    num_threads=None, cache=True, entrances=None, site_kind=None) -> dict:
    """Write <stem>_lod0/1/2.glb, the interior chunks <stem>_int_<tag>.glb and <stem>_engine.json for one building
    IFC; returns a report dict whose ``files`` holds every output (chunks under int_<tag>), so the export checks,
    the build state and its freshness test cover the chunks too. Chunk files of this stem that the export no
    longer writes (a storey removed) are deleted.

    local_matrix: 4x4 estate -> block-local (inverse of the building placement), "building" to read it from the
    IfcBuilding placement, or None to keep IFC world coordinates. spawns: optional [(x, y[, z])] in the local
    frame (or dicts with name/position). entrances: spawn dicts (name, position, facing, entrance) in the estate
    frame, from manifest.entrance_spawns; without either, entrances are derived from the ground storey and named
    after site_kind ('block', 'mscp', 'nc'). cache: reuse / write the tessellation cache in build/cache
    (meshcache). The report's ``problems`` lists passable portals with an unresolved side (also in the JSON)."""
    t0 = time.time()
    ifc_path = Path(ifc_path)
    out_dir = Path(out_dir) if out_dir else ifc_path.parent
    stem = stem or ifc_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    X = BuildingExport(ifc_path, local_matrix, num_threads, cache)
    lods = ("lod0", "lod1", "lod2") if X.has_storeys else ("lod0", "lod1")   # no massing for a site-only file
    files = {k: out_dir / f"{stem}_{k}.glb" for k in lods}
    files["engine"] = out_dir / f"{stem}_engine.json"
    rooms = X.rooms()
    stats = {}
    stats["lod0"], door_nodes, lift_nodes, budget = X.lod0(files["lod0"], stem, rooms, include_site)
    chunks = X.interior_chunks(out_dir, stem, include_site)
    X._parts.clear()                          # the static batches: not needed past LOD0 and the chunks
    for c in chunks.values():
        files[f"int_{c['tag']}"] = c["path"]
    mine, keep = re.compile(rf"{re.escape(stem)}_int_[A-Za-z0-9_\-]+\.glb"), {c["path"].name for c in chunks.values()}
    for p in sorted(out_dir.glob(f"{glob.escape(stem)}_int_*.glb")):
        if mine.fullmatch(p.name) and p.name not in keep:
            p.unlink()                        # a chunk of a storey this IFC no longer has
    stats["lod1"] = X.lod1(files["lod1"], stem)
    parts, z0, body, towers = [], 0.0, 0.0, []
    if "lod2" in files:
        stats["lod2"], parts, (z0, body, towers) = X.lod2(files["lod2"], stem)
    doors = X.doors(door_nodes)
    lifts = X.lifts(doors)
    if spawns:
        sp = [s if isinstance(s, dict) else dict(name=f"Spawn {i + 1}", position=r3([*s[:2], s[2] if len(s) > 2 else 0.0]))
              for i, s in enumerate(spawns)]
    elif entrances:
        sp = X.spawn_rooms(X.entrance_spawns(entrances), rooms)
    else:
        sp = X.spawn_rooms(X.spawns(rooms, doors, parts, site_kind), rooms)
    portals, problems = X.portals(doors, rooms)
    climb = X.climbable()
    flats = X.flats(rooms, doors, budget)
    zones = X.zones(rooms)
    allv = np.concatenate([d["verts"] for g, d in X.meshes.items() if not X.is_site(g)]) if X.meshes else np.zeros((1, 3))
    lo, hi = allv.min(0), allv.max(0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    ce = xf(X.T, corners)
    for r in rooms:
        r.pop("_poly", None)
    for k, s in stats.items():
        s["file"] = files[k].name
        s["sha256"] = _sha(files[k])
    info = dict(
        frame="Block-local IFC frame: metres, right-handed, Z up, origin at the building origin. estate = transform @ "
              "[x, y, z, 1]. The glbs hold the same geometry in glTF Y up: (x, y, z) -> (x, z, -y).",
        ifc=ifc_path.name, ifc_sha256=_sha(ifc_path), building=next((b.Name for b in X.f.by_type("IfcBuilding")), None),
        transform=mat_list(X.T), bounds_local=[r3(lo), r3(hi)],
        bounds_estate=[r3(ce.min(0)), r3(ce.max(0))],
        glb=stats,
        interior_chunks={s: dict(file=c["path"].name, elevation=c["elevation"], triangles=c["triangles"],
                                 nodes=c["nodes"], **{k: c[k] for k in CHUNK_COUNTS}, bytes=c["bytes"],
                                 sha256=_sha(c["path"])) for s, c in chunks.items()},
        spawn=sp[0] if sp else None, spawns=sp, verified_agent=agent_settings(),
        storeys=X.storeys, doors=doors, lifts=lifts, rooms=rooms, flats=flats, zones=zones, portals=portals,
        climbable_edges=climb, budget=budget, problems=problems,
        massing=dict(footprint=[[r3(c, 3)[:2] for c in p.exterior.coords] for p in parts], z0=round(z0, 3),
                     top=round(body, 3), roof_structures=[dict(polygon=[r3(c)[:2] for c in p.exterior.coords],
                                                               top=round(t, 3)) for p, t in towers]))
    if X.tiles:
        info["tiles"] = X.tiles
    tmp = files["engine"].with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=1), encoding="utf-8")
    tmp.replace(files["engine"])
    return dict(stem=stem, ifc=str(ifc_path), files={k: str(v) for k, v in files.items()},
                **stats, interior_chunks={s: dict(key=f"int_{c['tag']}", triangles=c["triangles"], bytes=c["bytes"])
                                          for s, c in chunks.items()},
                doors=len(doors), door_nodes=len(door_nodes), lift_nodes=len(lift_nodes),
                door_leaves=stats["lod0"]["door_leaves"], lifts=len(lifts), rooms=len(rooms), flats=len(flats),
                portals=len(portals), climbable=len(climb), spawns=len(sp), tiles=len(X.tiles), problems=problems,
                budget={k: budget[k] for k in ("limit", "flats", "max", "mean", "min", "over_budget")},
                seconds=round(time.time() - t0, 1))


# ----------------------------------------------------------------------------- round-trip checks
def check_glb(path) -> dict:
    """Re-read a glb with the in-house reader: spec problems + counts + Z-up world bounds."""
    s = glb.summary(path)
    s["problems"] = glb.validate(path)
    for k in ("names", "per_node", "node_bounds"):
        s.pop(k, None)
    return s


BLENDER_EXPR = r'''
import bpy, json, re, sys, numpy as np
leaf = re.compile(r"_leaf(_[A-Z0-9]+)?$")
paths = sys.argv[sys.argv.index("--") + 1:]
out = {}
for p in paths:
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.images):
        bpy.data.batch_remove(list(coll))
    bpy.ops.import_scene.gltf(filepath=p)
    objs = list(bpy.context.scene.objects)
    meshes = [o for o in objs if o.type == "MESH"]
    tris, loops, lo, hi, per = 0, 0, np.full(3, np.inf), np.full(3, -np.inf), {}
    for o in meshes:
        me = o.data
        tris += len(me.polygons)
        loops += sum(pl.loop_total for pl in me.polygons)
        n = len(me.vertices)
        if n:
            co = np.empty(n * 3)
            me.vertices.foreach_get("co", co)
            M = np.array(o.matrix_world)
            w = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
            per[o.name] = [len(me.polygons), w.min(0).tolist(), w.max(0).tolist()]
    names = [o.name for o in objs]
    leaves = sum(n.startswith("DOOR_") and bool(leaf.search(n)) for n in names)
    out[p] = dict(objects=len(objs), mesh_objects=len(meshes), triangles=tris, all_triangles=loops == 3 * tris,
                  door_nodes=sum(n.startswith("DOOR_") for n in names) - leaves, door_leaves=leaves,
                  window_nodes=sum(n.startswith("WIN_") for n in names),
                  lift_nodes=sum(n.startswith("LIFT_") for n in names), tree_nodes=sum(n.startswith("TREE_") for n in names),
                  furniture_nodes=sum(n.startswith("FURN_") for n in names), shared_meshes=len({o.data.name for o in meshes}),
                  bounds=[lo.tolist(), hi.tolist()] if meshes else None, names=sorted(names), per_object=per)
print("GLBCHECK " + json.dumps(out))
'''


def blender_roundtrip(paths, timeout=1800) -> dict:
    """Import each glb with Blender's glTF importer (headless, factory settings) and compare with the reader:
    object, mesh-object, triangle, DOOR_* (doors and leaves), WIN_*, LIFT_*, TREE_* and FURN_* counts and the number of distinct
    meshes (instancing survives the import) must match exactly, bounds within 1 mm."""
    paths = [str(Path(p).resolve()) for p in paths]
    cmd = [str(env.BLENDER_EXE), "-b", "--factory-startup", "--python-exit-code", "1", "--python-expr", BLENDER_EXPR,
           "--", *paths]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("GLBCHECK ")), None)
    if line is None:
        raise RuntimeError(f"Blender import failed (exit {r.returncode}): {r.stderr[-2000:] or r.stdout[-2000:]}")
    got = json.loads(line[len("GLBCHECK "):])
    report = {}
    for p in paths:
        mine = glb.summary(p)
        b = got[p]
        diffs = []
        for k_mine, k_bl in (("nodes", "objects"), ("mesh_nodes", "mesh_objects"), ("triangles", "triangles"),
                             ("door_nodes", "door_nodes"), ("door_leaves", "door_leaves"),
                             ("window_nodes", "window_nodes"), ("lift_nodes", "lift_nodes"), ("tree_nodes", "tree_nodes"),
                             ("furniture_nodes", "furniture_nodes"), ("meshes", "shared_meshes")):
            if mine[k_mine] != b[k_bl]:
                diffs.append(f"{k_mine}: wrote {mine[k_mine]}, Blender {b[k_bl]}")
        if not b["all_triangles"]:
            diffs.append("Blender faces are not all triangles")
        if sorted(mine["names"]) != b["names"]:
            missing = sorted(set(mine["names"]) - set(b["names"]))[:5]
            diffs.append(f"object names differ (e.g. {missing})")
        dev = None
        if mine["bounds"] and b["bounds"]:
            dev = float(np.abs(np.array(mine["bounds"]) - np.array(b["bounds"])).max())
            if dev > 1e-3:
                diffs.append(f"bounds deviate by {dev * 1000:.2f} mm")
        obj_dev, tri_bad = 0.0, []
        for name, (t, olo, ohi) in b["per_object"].items():
            if name not in mine["per_node"]:
                continue
            if mine["per_node"][name] != t:
                tri_bad.append(name)
            obj_dev = max(obj_dev, float(np.abs(np.array(mine["node_bounds"][name]) - np.array([olo, ohi])).max()))
        if tri_bad:
            diffs.append(f"{len(tri_bad)} objects with different triangle counts (e.g. {tri_bad[:3]})")
        if obj_dev > 1e-3:
            diffs.append(f"object bounds deviate by up to {obj_dev * 1000:.2f} mm")
        report[p] = dict(ok=not diffs, diffs=diffs,
                         wrote={k: mine[k] for k in ("nodes", "mesh_nodes", "meshes", "triangles", "door_nodes",
                                                     "door_leaves", "lift_nodes", "tree_nodes", "furniture_nodes",
                                                     "bounds")},
                         blender={k: b[k] for k in ("objects", "mesh_objects", "shared_meshes", "triangles",
                                                    "door_nodes", "door_leaves", "lift_nodes", "tree_nodes",
                                                    "furniture_nodes", "bounds")},
                         bounds_dev_mm=None if dev is None else round(dev * 1000, 4),
                         object_bounds_dev_mm=round(obj_dev * 1000, 4))
    return report
