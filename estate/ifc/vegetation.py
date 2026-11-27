"""Typed low-poly trees: one IfcGeographicElementType per species, a Bark trunk under a Foliage crown.

Each estate species has a crown silhouette of its own (`crown_parts`), so the street trees read apart at a glance:

- Rain tree (young): a wide, flattened umbrella with a slightly hollow underside.
- Angsana: a broad spreading dome whose rim droops below the crown's underside.
- Tembusu: a narrow, columnar cone.
- Yellow flame: three stacked, flattened tiers with the trunk showing between them.
- Sea apple: a dense cluster of four overlapping lobes (one crown, four closed shells).

Every crown is a lathe: a profile of (radius, height) from the top pole to the bottom pole swept round the trunk
axis, its radius optionally modulated by a gentle cos(k*theta) wobble. The wobble scales whole meridians, so every
quad between two rings stays planar (both its sides meet the axis at the same point) and the faceted breps are
valid IFC. The trunk is the original octagonal extrusion (radius 0.25 m, estate.site.builder.TRUNK_R), now run up
into the crown so no crown floats. Heights and canopy radii stay close to the species table: every crown keeps its
widest reach within 0.85-1.2 x the canopy radius the site plan draws.

A full tree is at most a few hundred triangles (`triangle_count`; the web pack's repeated-mesh limit is 600). The
original single-block toolkit (legacy/toolkit/hdb_block.py) keeps its subdivided-icosphere crown: `style="legacy"`.
"""
from __future__ import annotations

import math

import numpy as np
import ifcopenshell.api.geometry as geometry
import ifcopenshell.api.root as root

SPECIES = [("Rain tree (young)", 3.0, 2.6), ("Angsana", 3.6, 3.2)]
ESTATE_SPECIES = SPECIES + [("Yellow flame", 4.2, 3.0), ("Sea apple", 2.6, 2.0), ("Tembusu", 5.0, 2.4)]

TRUNK_R = 0.25
TRUNK_SIDES = 8
MAX_TREE_TRIANGLES = 600      # the web pack's repeatedMesh limit (one instanced mesh per species)


# ---------------------------------------------------------------- crown geometry (pure numpy, Z up, trunk at 0, 0)
def lathe(profile, n, centre=(0.0, 0.0, 0.0), scale=(1.0, 1.0), wobble=(0, 0.0), phase=0.0):
    """A closed shell swept from `profile` [(r, z)] (top pole first, bottom pole last, both r = 0) through n
    meridians. scale = (horizontal, vertical); wobble = (k, a) multiplies every radius by 1 + a*cos(k*theta).
    Returns (verts (V, 3), faces [tuple]) with outward-facing faces: triangles at the poles, planar quads between."""
    if profile[0][0] != 0 or profile[-1][0] != 0:
        raise ValueError("a lathe profile starts and ends on the axis")
    sh, sv = scale
    k, a = wobble
    cx, cy, cz = centre
    theta = phase + 2 * math.pi * np.arange(n) / n
    w = 1.0 + a * np.cos(k * theta)
    rings = profile[1:-1]
    verts = [(cx, cy, cz + profile[0][1] * sv)]
    for r, z in rings:
        rr = r * sh * w
        verts += list(zip(cx + rr * np.cos(theta), cy + rr * np.sin(theta), np.full(n, cz + z * sv)))
    verts.append((cx, cy, cz + profile[-1][1] * sv))
    top, bottom = 0, len(verts) - 1

    def at(i, j):
        return 1 + i * n + j % n
    faces = [(top, at(0, j), at(0, j + 1)) for j in range(n)]
    for i in range(len(rings) - 1):
        faces += [(at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)) for j in range(n)]
    last = len(rings) - 1
    faces += [(at(last, j), bottom, at(last, j + 1)) for j in range(n)]
    return np.array(verts, float), faces


def _ellipsoid(rings):
    """A sphere profile of unit radius with `rings` rings between the poles."""
    out = [(0.0, 1.0)]
    for i in range(1, rings + 1):
        phi = math.pi * i / (rings + 1)
        out.append((math.sin(phi), math.cos(phi)))
    return out + [(0.0, -1.0)]


def _rain_tree(h, rad):
    # wide flattened umbrella: widest at 1.12 rad, 1.0 rad deep, a shallow hollow under the canopy
    prof = [(0, 1.30), (0.42, 1.26), (0.78, 1.14), (1.02, 0.94), (1.12, 0.70), (1.04, 0.48), (0.66, 0.30),
            (0, 0.40)]
    return [lathe(prof, 14, (0, 0, h), (rad, rad), wobble=(5, 0.06))], h + 0.5 * rad


def _angsana(h, rad):
    # broad spreading dome; the rim hangs below the underside (the drooping branch tips)
    prof = [(0, 1.45), (0.40, 1.40), (0.76, 1.22), (0.98, 0.95), (1.08, 0.65), (1.04, 0.35), (0.92, 0.11),
            (0.56, 0.27), (0, 0.37)]
    return [lathe(prof, 14, (0, 0, h), (rad, rad), wobble=(4, 0.05), phase=0.3)], h + 0.5 * rad


def _tembusu(h, rad):
    # narrow columnar cone, widest low (0.78 rad), pointed crown, starting below the trunk height
    prof = [(0, 1.95), (0.24, 1.72), (0.48, 1.32), (0.66, 0.86), (0.78, 0.40), (0.72, 0.00), (0.46, -0.28),
            (0, -0.36)]
    return [lathe(prof, 12, (0, 0, h), (rad, rad), wobble=(3, 0.05))], h + 0.4 * rad


def _yellow_flame(h, rad):
    # three flattened tiers, smaller upwards, the trunk carried up through all three
    disc = [(0, 0.24), (0.50, 0.20), (0.88, 0.09), (1.0, -0.02), (0.66, -0.13), (0, -0.15)]
    tiers = [(1.00, 0.10), (0.76, 0.66), (0.50, 1.18)]
    parts = [lathe(disc, 12, (0, 0, h + dz * rad), (s * rad, rad), wobble=(5, 0.07), phase=1.1 * i)
             for i, (s, dz) in enumerate(tiers)]
    return parts, h + 1.18 * rad


def _sea_apple(h, rad):
    # a dense cluster: a top lobe over three side lobes, all overlapping; reach 1.0 rad
    sphere = _ellipsoid(4)
    parts = [lathe(sphere, 10, (0.04 * rad, -0.03 * rad, h + 0.98 * rad), (0.66 * rad, 0.58 * rad))]
    for i, (t, r_off, r_lobe) in enumerate([(0.4, 0.36, 0.64), (2.5, 0.38, 0.60), (4.6, 0.37, 0.62)]):
        parts.append(lathe(sphere, 10, (r_off * rad * math.cos(t), r_off * rad * math.sin(t), h + 0.52 * rad),
                           (r_lobe * rad, 0.54 * rad), phase=0.3 * i))
    return parts, h + 0.5 * rad


CROWNS = {"Rain tree": _rain_tree, "Angsana": _angsana, "Tembusu": _tembusu, "Yellow flame": _yellow_flame,
          "Sea apple": _sea_apple}


def species_key(name):
    """'Tembusu (young)' -> 'Tembusu': growth stages share their species' silhouette."""
    return name.split(" (")[0]


def crown_parts(name, h, rad):
    """([(verts, faces)] closed crown shells, trunk top z) for a species of trunk height h and canopy radius rad."""
    return CROWNS[species_key(name)](h, rad)


def _faces_tris(faces):
    return sum(len(f) - 2 for f in faces)


def triangle_count(name, h, rad):
    """Triangles of one full tree (crown shells + the octagonal trunk with its two caps)."""
    parts, _ = crown_parts(name, h, rad)
    return sum(_faces_tris(f) for _, f in parts) + 2 * TRUNK_SIDES + 2 * (TRUNK_SIDES - 2)


# ---------------------------------------------------------------- the legacy icosphere crown
_P = (1 + 5 ** 0.5) / 2
_ICO = [(-1, _P, 0), (1, _P, 0), (-1, -_P, 0), (1, -_P, 0), (0, -1, _P), (0, 1, _P), (0, -1, -_P), (0, 1, -_P),
        (_P, 0, -1), (_P, 0, 1), (-_P, 0, -1), (-_P, 0, 1)]
_ICO_FACES = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
              (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
              (6, 2, 10), (8, 6, 7), (9, 8, 1)]


def icosphere_crown(h, rad):
    """The toolkit's crown: a once-subdivided icosphere of radius rad centred at h + 0.7 rad (80 triangles)."""
    v = [np.array(x, float) / np.linalg.norm(x) for x in _ICO]
    cache, faces = {}, []

    def mid(a, b):
        k = (min(a, b), max(a, b))
        if k not in cache:
            c = (v[a] + v[b]) / 2
            v.append(c / np.linalg.norm(c))
            cache[k] = len(v) - 1
        return cache[k]
    for a, b, c in _ICO_FACES:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array([x * rad + np.array([0, 0, h + rad * 0.7]) for x in v]), faces


# ---------------------------------------------------------------- IFC
def _brep(m, verts, faces):
    pts = [m.createIfcCartesianPoint(tuple(float(q) for q in x)) for x in verts]
    return m.createIfcFacetedBrep(m.createIfcClosedShell([m.createIfcFace([m.createIfcFaceOuterBound(
        m.createIfcPolyLoop([pts[i] for i in f]), True)]) for f in faces]))


def tree_types(W, species=SPECIES, style="estate"):
    """One IfcGeographicElementType per (name, trunk height, canopy radius), its Body a Bark trunk and Foliage crown
    shells. style="legacy" draws every species with the toolkit's icosphere crown on a trunk of height h."""
    m = W.m
    out = []
    for tname, h, rad in species:
        if style == "legacy":
            parts, trunk_top = [icosphere_crown(h, rad)], h
        else:
            parts, trunk_top = crown_parts(tname, h, rad)
        crowns = [_brep(m, v, f) for v, f in parts]
        octo = [(TRUNK_R * math.cos(t), TRUNK_R * math.sin(t))
                for t in np.linspace(0, 2 * math.pi, TRUNK_SIDES, endpoint=False)]
        trunk = m.createIfcExtrudedAreaSolid(m.createIfcArbitraryClosedProfileDef("AREA", None, W._curve(octo)),
                                             W._place3d(), m.createIfcDirection((0.0, 0.0, 1.0)), float(trunk_top))
        m.createIfcStyledItem(trunk, [W.mat["bark"][1]], None)
        for crown in crowns:
            m.createIfcStyledItem(crown, [W.mat["foliage"][1]], None)
        pre, ot = ("VEGETATION", None) if W.schema == "IFC4X3" else ("USERDEFINED", "Vegetation")
        tt = root.create_entity(m, "IfcGeographicElementType", name=tname, predefined_type=pre)
        if hasattr(W, "stable_id"):
            W.stable_id(tt, f"IfcGeographicElementType/{tname}")
        if ot:
            tt.ElementType = ot
        geometry.assign_representation(m, product=tt, representation=m.createIfcShapeRepresentation(
            W.body, "Body", "SolidModel", [trunk] + crowns))
        out.append(tt)
    return out
