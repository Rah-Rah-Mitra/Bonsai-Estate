"""Design rules, materials and element kinds shared by every building.

Lifted from legacy/toolkit/hdb_block.py (Singapore Approved Document / BCA stair rules, HDB conventions) and
extended for the estate (party walls, 250 mm shelter walls, balcony / sliding / service door kinds).
Statutory values that still need confirming live with their thresholds in config/rules.toml.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------- storeys and stairs
SLAB = 0.20
FTF = 2.80                      # 2.6 m ceiling + slab
VOID_DECK_FTF = 3.60
RISER_MAX = 0.175               # Approved Document: riser <= 175 mm
GOING = 0.28                    # common stairs: going >= 275 mm
MAX_RISERS_PER_FLIGHT = 18
FLIGHT_W = 1.15                 # >= 1.0 m clear
WELL = 0.20
LANDING = 1.30                  # >= 1.0 m
WAIST = 0.15
DOOR_H = 2.10
PARAPET_H = 1.10                # barrier >= 1.0 m
BALUSTRADE_H = 1.00             # stair balustrade above the pitch line / landing guards
RAILING_H = 1.10

# legacy point-block dimensions (blocks/legacy.py and the PT4 core)
UNIT_W, UNIT_D = 11.0, 8.6
CORE_HALF = 2.9
LOBBY_HALF = 1.5
LIFT_D = 2.7
STAIR_D = 5.8
STAIR_W = 2.9                   # one stair enclosure (outer), flights 2 x 1.15 + well

# ----------------------------------------------------------------------------- materials
PALETTE = {  # key: (name, rgb, transparency)
    "ext": ("Painted RC - HDB off-white", (0.93, 0.91, 0.86), 0.0),
    "int": ("Lightweight panel - white", (0.96, 0.95, 0.92), 0.0),
    "hs": ("RC - household shelter", (0.80, 0.79, 0.76), 0.0),
    "core": ("Painted RC - core accent", (0.62, 0.70, 0.62), 0.0),
    "slab": ("RC slab", (0.70, 0.70, 0.68), 0.0),
    "roof": ("Roof waterproofing", (0.42, 0.43, 0.45), 0.0),
    "stair": ("RC stair - granolithic", (0.78, 0.76, 0.72), 0.0),
    "steel": ("Mild steel - painted", (0.25, 0.27, 0.30), 0.0),
    "timber": ("Timber door", (0.58, 0.40, 0.25), 0.0),
    "blast": ("Steel blast door", (0.55, 0.57, 0.58), 0.0),
    "lift": ("Stainless steel", (0.78, 0.80, 0.82), 0.0),
    "glass": ("Glass", (0.55, 0.74, 0.85), 0.6),
    "frame": ("Aluminium frame", (0.80, 0.81, 0.82), 0.0),
    "alu": ("Aluminium railing", (0.74, 0.76, 0.78), 0.0),
    "column": ("RC column", (0.85, 0.83, 0.78), 0.0),
    "grass": ("Grass", (0.42, 0.60, 0.33), 0.0),
    "paving": ("Concrete paving", (0.74, 0.72, 0.68), 0.0),
    "asphalt": ("Asphalt", (0.22, 0.22, 0.24), 0.0),
    "tank": ("Water tank - GRP", (0.45, 0.62, 0.78), 0.0),
    "space": ("Space", (0.95, 0.90, 0.55), 0.85),
    "bark": ("Bark", (0.40, 0.28, 0.18), 0.0),
    "foliage": ("Foliage", (0.24, 0.48, 0.22), 0.0),
    # estate additions
    "party": ("Painted RC - party wall", (0.90, 0.88, 0.84), 0.0),
    "accent": ("Painted RC - block accent", (0.80, 0.55, 0.42), 0.0),
    "tile": ("Homogeneous tile - wet area", (0.86, 0.87, 0.88), 0.0),
    "kerb": ("Precast concrete kerb", (0.66, 0.65, 0.62), 0.0),
    "marking": ("Road marking - white", (0.95, 0.95, 0.93), 0.0),
    "rubber": ("EPDM playground surface", (0.80, 0.36, 0.28), 0.0),
    "metal_roof": ("Metal deck roof - linkway", (0.56, 0.60, 0.64), 0.0),
    "shutter": ("Roller shutter - galvanised", (0.62, 0.64, 0.66), 0.0),
}
NO_MATERIAL = {"space", "bark", "foliage"}   # styles only

WALL_TYPES = {  # key: (type name, thickness, palette key, predefined type)
    "EXT": ("EXT-200 RC painted", 0.20, "ext", "SOLIDWALL"),
    "INT": ("INT-100 lightweight partition", 0.10, "int", "PARTITIONING"),
    "HS": ("HS-200 RC household shelter", 0.20, "hs", "SHEAR"),
    "CORE": ("CORE-200 RC lift / stair core", 0.20, "core", "SHEAR"),
    "PARAPET": ("PAR-200 RC parapet", 0.20, "ext", "PARAPET"),
    # estate additions
    "PARTY": ("PARTY-200 RC party wall", 0.20, "party", "SOLIDWALL"),
    "HS250": ("HS-250 RC household shelter", 0.25, "hs", "SHEAR"),
    "WET": ("WET-100 partition, tiled", 0.10, "tile", "PARTITIONING"),
}
DOOR_KINDS = {  # kind: (operation type, palette key, passable for people, fire rating)
    "main": ("SINGLE_SWING_LEFT", "timber", True, "0.5 hr"),   # HDB main doors are half-hour fire doors
    "internal": ("SINGLE_SWING_LEFT", "timber", True, None),
    "bath": ("SINGLE_SWING_LEFT", "timber", True, None),
    "shelter": ("SINGLE_SWING_LEFT", "blast", True, None),
    "stair": ("SINGLE_SWING_LEFT", "steel", True, "1 hr"),
    "roof": ("SINGLE_SWING_LEFT", "steel", True, None),
    "lift": ("DOUBLE_DOOR_SLIDING", "lift", False, "1 hr"),
    # estate additions
    "balcony": ("SLIDING_TO_LEFT", "frame", True, None),
    "folding": ("SINGLE_SWING_LEFT", "timber", True, None),    # kitchen / service-yard folding doors
    "service": ("SINGLE_SWING_LEFT", "steel", True, None),     # refuse chute, bin centre, plant rooms
    "locked": ("SINGLE_SWING_LEFT", "steel", False, "2 hr"),   # substation, switch room
    "shop": ("DOUBLE_DOOR_SINGLE_SWING", "frame", True, None),
    "shutter": ("ROLLINGUP", "shutter", True, None),
}

# ----------------------------------------------------------------------------- navigation agent
AGENT_RADIUS = 0.30
AGENT_HEIGHT = 1.80
AGENT_STEP = 0.40
RAMP_MAX = 1 / 12
