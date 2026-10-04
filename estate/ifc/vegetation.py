"""Typed low-poly trees (lifted from BlockBuilder.tree_types in legacy/toolkit/hdb_block.py)."""
from __future__ import annotations

import math

import numpy as np
import ifcopenshell.api.geometry as geometry
import ifcopenshell.api.root as root

SPECIES = [("Rain tree (young)", 3.0, 2.6), ("Angsana", 3.6, 3.2)]
ESTATE_SPECIES = SPECIES + [("Yellow flame", 4.2, 3.0), ("Sea apple", 2.6, 2.0), ("Tembusu", 5.0, 2.4)]


def tree_types(W, species=SPECIES):
    m = W.m
    out = []
    p = (1 + 5 ** 0.5) / 2
    base = [(-1, p, 0), (1, p, 0), (-1, -p, 0), (1, -p, 0), (0, -1, p), (0, 1, p), (0, -1, -p), (0, 1, -p),
            (p, 0, -1), (p, 0, 1), (-p, 0, -1), (-p, 0, 1)]
    faces0 = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
              (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
              (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for tname, h, rad in species:
        v = [np.array(x, float) / np.linalg.norm(x) for x in base]
        cache, faces = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                c = (v[a] + v[b]) / 2
                v.append(c / np.linalg.norm(c))
                cache[k] = len(v) - 1
            return cache[k]
        for a, b, c in faces0:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        pts = [m.createIfcCartesianPoint(tuple(float(q) for q in x * rad + np.array([0, 0, h + rad * 0.7]))) for x in v]
        crown = m.createIfcFacetedBrep(m.createIfcClosedShell([m.createIfcFace([m.createIfcFaceOuterBound(
            m.createIfcPolyLoop([pts[i] for i in f]), True)]) for f in faces]))
        octo = [(0.25 * math.cos(t), 0.25 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
        trunk = m.createIfcExtrudedAreaSolid(m.createIfcArbitraryClosedProfileDef("AREA", None, W._curve(octo)),
                                             W._place3d(), m.createIfcDirection((0.0, 0.0, 1.0)), h)
        m.createIfcStyledItem(trunk, [W.mat["bark"][1]], None)
        m.createIfcStyledItem(crown, [W.mat["foliage"][1]], None)
        pre, ot = ("VEGETATION", None) if W.schema == "IFC4X3" else ("USERDEFINED", "Vegetation")
        tt = root.create_entity(m, "IfcGeographicElementType", name=tname, predefined_type=pre)
        if hasattr(W, "stable_id"):
            W.stable_id(tt, f"IfcGeographicElementType/{tname}")
        if ot:
            tt.ElementType = ot
        geometry.assign_representation(m, product=tt, representation=m.createIfcShapeRepresentation(
            W.body, "Body", "SolidModel", [trunk, crown]))
        out.append(tt)
    return out
