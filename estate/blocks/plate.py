"""Floor plates: the 2D description of one storey that the IFC builder turns into walls, openings and spaces.

A plate is a set of rectilinear *faces* that tile the floor (flat rooms, corridor, lift shafts, stair
enclosures, refuse room, void deck, roof deck, balconies). Everything outside the faces is ``@out``. Walls are
not drawn by hand: geom/arrangement.py derives one centred wall per shared boundary from the face pair.
Coordinates are block-local metres, +x east, +y north.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import Polygon

OUT = "@out"

# face kinds
ROOM, CORRIDOR, LOBBY, LIFT, STAIR, REFUSE, BALCONY, VOID, AMENITY, DECK, PLANT = (
    "room", "corridor", "lobby", "lift", "stair", "refuse", "balcony", "void", "amenity", "deck", "plant")
CORE_KINDS = {LIFT, STAIR, REFUSE, PLANT}
COMMON_KINDS = {CORRIDOR, LOBBY}


@dataclass
class Face:
    id: str
    poly: Polygon
    kind: str
    flat: str | None = None       # stack id, e.g. "103"
    room: str | None = None       # room code, e.g. "MB"
    name: str = ""                # long name for the IfcSpace
    meta: dict = field(default_factory=dict)


@dataclass
class DoorSpec:
    a: str                        # face ids on either side
    b: str
    at: tuple                     # centre point on the shared boundary
    width: float
    kind: str
    into: str                     # face the leaf swings into
    hinge: tuple                  # a point at the hinge end of the opening
    name: str
    height: float = 2.10


@dataclass
class WindowSpec:
    face: str
    at: tuple
    width: float
    sill: float
    height: float
    name: str
    kind: str = "window"          # window | vent | opening
    required: bool = False        # True: an error if the window cannot be placed


@dataclass
class Ledge:
    poly: Polygon
    flat: str | None
    name: str
    rail: list = field(default_factory=list)   # railing band polygons along the open edges
    kind: str = "ac"                           # ac (AC ledge, aluminium railing) | bay (bay window, glazed guard)


@dataclass
class FlatInfo:
    stack: str
    template: str
    flat_type: str
    long_name: str
    variant: dict
    signature: str
    envelope: Polygon
    gross_area: float


@dataclass
class Plate:
    faces: dict = field(default_factory=dict)
    doors: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    open_pairs: set = field(default_factory=set)
    ledges: list = field(default_factory=list)
    flats: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def add(self, face: Face) -> Face:
        if face.id in self.faces:
            raise ValueError(f"duplicate face id {face.id}")
        self.faces[face.id] = face
        return face

    def open(self, a: str, b: str):
        self.open_pairs.add(frozenset((a, b)))

    def kind(self, fid: str) -> str:
        return "out" if fid == OUT else self.faces[fid].kind

    def extend(self, other: "Plate"):
        for f in other.faces.values():
            self.add(f)
        self.doors += other.doors
        self.windows += other.windows
        self.open_pairs |= other.open_pairs
        self.ledges += other.ledges
        self.flats.update(other.flats)
        self.notes += other.notes
