"""Site works of the estate (SITE.ifc): roads, driveways, footpaths, covered linkways, bus stops, greens, trees.

The site is planned as plain geometry first (``builder.plan_site``) and then written to IFC in either schema
(``builder.write_site``), so the IFC4X3 model and its IFC4 copy describe exactly the same works. ``Piece`` is the
record every planning module emits: one physical element (or one part of an assembly) with its plan polygon,
vertical extent, material and where it belongs in the spatial structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Piece:
    kind: str                       # IFC mapping key (builder.KINDS)
    name: str
    geom: object = None             # plan Polygon / MultiPolygon (extruded from z0 by h)
    z0: float = 0.0
    h: float = 0.15
    mat: str = "paving"
    part: str | None = None         # spatial container key, e.g. "road:AVE5:SIDEWALK"; None = the IfcSite
    assembly: str | None = None     # element assembly key (linkway segment, shelter, porch, pavilion)
    object_type: str | None = None
    props: dict = field(default_factory=dict)
    solid: dict | None = None       # non-prismatic shapes: {"wedge": {...}} | {"boxes": [...]} | {"type": ...}
