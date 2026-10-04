"""Wall joins and door / window parameters for Bonsai's parametric tools.

Bonsai regenerates a wall body (``ifcopenshell.api.geometry.regenerate_wall_representation``, run by its wall
tools and ``bim.recalculate_wall``) from three things only: the wall's reference line (the Plan/Axis/GRAPH_VIEW
polyline), its own IfcMaterialLayerSetUsage (offset and sense of the layer set from that line) and the
IfcRelConnectsPathElements at each end. A joined end is reset to the other wall's reference line and closed by
the layer faces of both walls; an unjoined end stops at its own reference line end. The generator draws bodies
with the corner extensions already applied (geom/arrangement.finish_runs), so the joins written here describe
the same corners:

- L corner: two walls end at the same node at right angles: end <-> end (ATSTART is the end at the smaller local
  x, ATEND the larger). Where a wall ends on two collinear walls that both end there, it turns the corner into
  one of them: the one of its own height where the other is lower (a lobby facade wall continued by the lobby
  parapet, a lift tower wall continued by the roof parapet), else the thicker one (a shelter wall continued by a
  thinner partition).
- T junction: a wall ends on a wall that passes through the node (a merged run's pass-through node), or on two
  collinear walls of its own height and thickness that both end there (a type change): the stem's end (relating)
  <-> the bar (related, ATPATH).
- A collinear continuation (a same-orientation wall ends at the node) and X crossings get no join: there the body
  ends at the node.

Bonsai rebuilds a corner in plan and extrudes each wall to its own height, so walls are joined only where that
gives back the generated corner at every height: walls of one height, or a T on a taller wall that is at least as
strong (the stem stops at its face, and the taller wall fills the corner square above the stem). A T on a lower
wall would cut the stem back to the lower wall's face and leave the corner square above it empty; an L with a
lower wall would mitre it. So every end that finish_runs extended (ext > 0) gets exactly one join unless it only
meets walls of another height (in the estate one lift tower corner, on the roof of BLK_510), free ends get none,
and the reference line runs node to node (the extensions stay in the body only). An extended end left unjoined
keeps its extension in the reference line too (``run_axis``), because Bonsai stops an unjoined end at its
reference line end. The only unextended ends with a join are the continued walls a corner turns into. Relating /
related priorities stay empty: non-empty priority lists crash Bonsai's regenerator; the layer priorities of the
wall types (``LAYER_PRIORITY``) decide which wall runs through.

Regenerating every wall of a storey in Bonsai gives back the generated wall bodies' union exactly, at every height
(tests/test_parametric.py compares sections at each height band). Regenerating only some walls can leave one gap:
where a corner turns into a continued wall (shelter-type and lobby-type corners) that wall's body ends at the
node, so when Bonsai regenerates the other wall of the corner but not this one (Recalculate Wall on a wall joined
to the other one's far end), the corner square is short of the triangle Bonsai's mitre gives the continued wall
until that wall is regenerated too.

Doors and windows carry Bonsai's ``BBIM_Door`` / ``BBIM_Window`` data (one IfcText "Data" property holding the
JSON its door / window tools edit). The representation of every type is built from that data through the same
snake_case -> API mapping Bonsai's ``update_door_modifier_representation`` / ``update_window_modifier_representation``
use, so the stored geometry and the data cannot drift apart: a Bonsai edit that changes nothing rebuilds the
same solids.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from estate.geom.arrangement import snap

ATSTART, ATEND, ATPATH = "ATSTART", "ATEND", "ATPATH"
TOL = 1e-6

# IfcMaterialLayer.Priority of the wall types (0..100): Bonsai's regenerator runs the higher-priority wall through
# a corner or junction and stops the other at its face. Structural RC walls outrank partitions.
LAYER_PRIORITY = {"CORE": 90, "HS": 80, "HS250": 80, "EXT": 70, "PARTY": 70, "PARAPET": 60, "INT": 30, "WET": 30}


@dataclass(frozen=True)
class Join:
    relating: object          # key of the relating wall (run idx, list index)
    relating_end: str         # ATSTART | ATEND
    related: object
    related_end: str          # ATSTART | ATEND | ATPATH


# ----------------------------------------------------------------------------- derived plates
def through_nodes(r) -> list:
    """Pass-through nodes of a merged run (boundaries between its parts), snapped like the run end points so they
    match other runs' end points exactly."""
    parts = r.meta.get("parts") or []
    u = r.u
    return [(snap(r.p0[0] + u[0] * b), snap(r.p0[1] + u[1] * b)) for _, b, *_ in parts[:-1]]


def priority(r) -> int:
    return LAYER_PRIORITY.get(r.wall, 0)


def plate_joins(runs, height) -> list[Join]:
    """Joins between the wall runs of a derived plate (after merge_collinear), keyed by run idx; height(run) is the
    height of the run's wall (all of a plate's walls stand on its floor). At most one join per wall end that
    finish_runs extended into a perpendicular wall, none elsewhere (bar the continued walls corners turn into).

    Bonsai rebuilds a joined corner in plan and extrudes it to each wall's own height, so a join is only written
    where that gives back the generated corner at every height: between walls of one height, or a T on a wall
    passing through that is at least as tall and at least as strong (the stem stops at its face, the taller wall
    fills the corner square above the stem). Every other extended end stays unjoined (run_axis keeps its body end)."""
    walls = [r for r in runs if r.wall and r.wall != "RAILING" and r.t > 0]
    hgt = {r.idx: float(height(r)) for r in walls}
    level = lambda o, r: abs(hgt[o.idx] - hgt[r.idx]) < TOL  # noqa: E731
    at = defaultdict(list)                    # node -> [(run, ATSTART | ATEND | ATPATH)]
    for r in walls:
        at[r.p0].append((r, ATSTART))
        at[r.p1].append((r, ATEND))
        for node in through_nodes(r):
            at[node].append((r, ATPATH))
    out, paired = [], set()
    for r in walls:
        for role, node in ((ATSTART, r.p0), (ATEND, r.p1)):
            here = [(o, k) for o, k in at[node] if o is not r]
            if any(o.horizontal == r.horizontal for o, _ in here):
                continue                      # collinear continuation: the body ends at the node
            bars = [o for o, k in here if k == ATPATH]
            if bars:                          # T on a wall passing through (merged runs: at most one)
                o = bars[0]
                if level(o, r) or (hgt[o.idx] > hgt[r.idx] and priority(o) >= priority(r)):
                    out.append(Join(r.idx, role, o.idx, ATPATH))
                continue
            ends = [(o, k) for o, k in here if level(o, r)]
            if not ends:
                # only walls of another height end here (a lift tower wall on two lower stair tower walls): any
                # join would let Bonsai cut the corner square down to a lower wall's top, or raise a lower one
                continue
            strength = lambda e: (e[0].t, priority(e[0]), -e[0].idx)  # noqa: E731
            if len(ends) == 2 and abs(ends[0][0].t - ends[1][0].t) < TOL:
                # two collinear walls of one thickness and of this wall's height end here (a type change along a
                # facade): a T on the stronger one; whichever face Bonsai stops this wall at, the two walls fill
                # the corner square to the full height
                out.append(Join(r.idx, role, max(ends, key=strength)[0].idx, ATPATH))
                continue
            # an L corner. Where two collinear walls end here, the corner turns into one of them and the other
            # continues unjoined from the node: into the one of this wall's height where the other is lower (a lobby
            # facade wall continued by the lobby parapet: a T would stop this wall at the facade wall's face and
            # leave the corner square above the parapet empty), into the thicker one where both are (a shelter wall
            # continued by a partition: finish_runs ran this end to the thicker wall's far face, which only an L
            # reproduces; a T would leave a notch beside the thinner wall)
            o, k = max(ends, key=strength)
            if o.t < max(p.t for p, _ in here) - TOL:
                continue                      # a thicker, lower wall: this end runs past the L to its far face
            pair = tuple(sorted(((r.idx, role), (o.idx, k))))
            if pair not in paired:            # written once, from the first of the two walls
                paired.add(pair)
                out.append(Join(r.idx, role, o.idx, k))
    return out


def joined_ends(wall_joins) -> set:
    """{(wall key, ATSTART | ATEND)} of the wall ends that a join resets (not the bar side of a T)."""
    out = {(j.relating, j.relating_end) for j in wall_joins}
    return out | {(j.related, j.related_end) for j in wall_joins if j.related_end != ATPATH}


def run_axis(r, joined=None) -> tuple:
    """Reference line of a run's wall in wall-local x (the body starts ext0 before the run's first node): node to
    node. With `joined` (joined_ends of the plate's joins), an extended end left without a join (it meets only walls
    of another height) runs to the body's end instead: Bonsai stops an unjoined end at its reference line end."""
    a0, a1 = r.ext0, r.ext0 + r.length
    if joined is not None:
        if r.ext0 > 0 and (r.idx, ATSTART) not in joined:
            a0 = 0.0
        if r.ext1 > 0 and (r.idx, ATEND) not in joined:
            a1 = r.ext0 + r.length + r.ext1
    return (a0, a1)


# ----------------------------------------------------------------------------- free-standing walls
def _centre_line(w):
    u = np.asarray(w.u, float)[:2]
    n = np.array([-u[1], u[0]])
    a = np.asarray(w.p0, float)[:2] + n * (w.lo + w.t / 2)
    return a, u, float(w.length)


def segment_joins(specs) -> list[Join]:
    """Joins between free-standing straight walls (WallSpec list; perimeter parapets, ramp lane walls), found from
    their geometry and keyed by list index. A wall end joins a perpendicular wall of the same base and height when
    the end lies within that wall's thickness band (butting into it or overlapping it) and the node (the crossing
    of the two centre lines) lies within the other wall's length: an L corner when the node is also at an end of
    the other wall, else a T on it. Ends with a collinear wall touching them get no join. Sets each spec's
    meta['axis'] (the reference line in wall-local x: joined ends at the node, free ends at the body end)."""
    lines = [_centre_line(w) for w in specs]

    def along(i, p):
        a, u, _ = lines[i]
        return float((np.asarray(p) - a) @ u)

    def across(i, p):
        a, u, _ = lines[i]
        return float((np.asarray(p) - a) @ np.array([-u[1], u[0]]))

    def end_point(i, role):
        a, u, L = lines[i]
        return a + u * (0.0 if role == ATSTART else L)

    def continued(i, role):
        """Another wall on the same centre line touches this end."""
        e = end_point(i, role)
        for j, (a, u, L) in enumerate(lines):
            if j == i or abs(abs(float(u @ lines[i][1])) - 1.0) > TOL or abs(across(j, e)) > TOL:
                continue
            s = along(j, e)
            if -TOL <= s <= L + TOL:
                return True
        return False

    cands = {}
    for i, w in enumerate(specs):
        a, u, L = lines[i]
        for role in (ATSTART, ATEND):
            if continued(i, role):
                continue
            e = end_point(i, role)
            found = []
            for j, v in enumerate(specs):
                b, uj, Lj = lines[j]
                if j == i or abs(float(u @ uj)) > TOL or abs(v.z0 - w.z0) > TOL or abs(v.h - w.h) > TOL:
                    continue
                d = across(j, e)                       # signed distance of the end from j's centre line
                if abs(d) > v.t / 2 + TOL:
                    continue
                node = e - u * (d / float(u @ np.array([-uj[1], uj[0]])))
                s = along(j, node)
                if s < -w.t / 2 - TOL or s > Lj + w.t / 2 + TOL:
                    continue
                if s <= w.t / 2 + TOL:
                    k = ATSTART
                elif s >= Lj - w.t / 2 - TOL:
                    k = ATEND
                else:
                    k = ATPATH
                found.append((abs(d), -v.t, j, k, along(i, node)))
            if found:
                cands[(i, role)] = sorted(found)
    out, joined, axis = [], set(), {}
    for (i, role) in sorted(cands, key=lambda x: (x[0], x[1] != ATSTART)):
        if (i, role) in joined:
            continue
        _, _, j, k, s_node = cands[(i, role)][0]
        if k != ATPATH and ((j, k) in joined or continued(j, k)
                            or not any(c[2] == i for c in cands.get((j, k), ()))):
            k = ATPATH                                 # the other end is taken or continued: a T on that wall
        if k != ATPATH:
            joined.add((j, k))
            axis[(j, k)] = next(c[4] for c in cands[(j, k)] if c[2] == i)
        joined.add((i, role))
        axis[(i, role)] = s_node
        out.append(Join(i, role, j, k))
    for i, w in enumerate(specs):
        node = {k: round(v, 6) + 0.0 for (j, k), v in axis.items() if j == i}
        w.meta["axis"] = (node.get(ATSTART, 0.0), node.get(ATEND, float(w.length)))
    return out


# ----------------------------------------------------------------------------- doors and windows
BONSAI_DOOR_TYPES = ("SINGLE_SWING_LEFT", "SINGLE_SWING_RIGHT", "DOUBLE_SWING_LEFT", "DOUBLE_SWING_RIGHT",
                     "DOUBLE_DOOR_SINGLE_SWING", "SLIDING_TO_LEFT", "SLIDING_TO_RIGHT", "DOUBLE_DOOR_SLIDING")
WINDOW_PANELS = {"SINGLE_PANEL": 1, "DOUBLE_PANEL_HORIZONTAL": 2, "DOUBLE_PANEL_VERTICAL": 2, "TRIPLE_PANEL_BOTTOM": 3,
                 "TRIPLE_PANEL_TOP": 3, "TRIPLE_PANEL_LEFT": 3, "TRIPLE_PANEL_RIGHT": 3,
                 "TRIPLE_PANEL_HORIZONTAL": 3, "TRIPLE_PANEL_VERTICAL": 3}
# Bonsai's BIMDoorProperties / BIMWindowProperties keys -> ifcopenshell.api.geometry lining / panel attributes
# (bonsai/bim/module/model/door.py and window.py, update_*_modifier_representation)
DOOR_LINING_API = {"lining_depth": "LiningDepth", "lining_thickness": "LiningThickness", "lining_offset": "LiningOffset",
                   "lining_to_panel_offset_x": "LiningToPanelOffsetX", "lining_to_panel_offset_y": "LiningToPanelOffsetY",
                   "transom_thickness": "TransomThickness", "transom_offset": "TransomOffset",
                   "casing_thickness": "CasingThickness", "casing_depth": "CasingDepth",
                   "threshold_thickness": "ThresholdThickness", "threshold_depth": "ThresholdDepth",
                   "threshold_offset": "ThresholdOffset"}
DOOR_PANEL_API = {"panel_depth": "PanelDepth", "panel_width_ratio": "PanelWidth", "frame_depth": "FrameDepth",
                  "frame_thickness": "FrameThickness"}
WINDOW_LINING_API = {"lining_depth": "LiningDepth", "lining_thickness": "LiningThickness", "lining_offset": "LiningOffset",
                     "lining_to_panel_offset_x": "LiningToPanelOffsetX", "lining_to_panel_offset_y": "LiningToPanelOffsetY",
                     "mullion_thickness": "MullionThickness", "first_mullion_offset": "FirstMullionOffset",
                     "second_mullion_offset": "SecondMullionOffset", "transom_thickness": "TransomThickness",
                     "first_transom_offset": "FirstTransomOffset", "second_transom_offset": "SecondTransomOffset"}


def dim(x) -> float:
    """Door / window sizes in the BBIM data and the type geometry: 0.1 mm, the plan grid (drops float noise such as
    0.8999999999999999 from opening widths measured along a run)."""
    return round(float(x) + 0.0, 4)


def door_data(operation_type: str, width: float, height: float) -> dict | None:
    """BBIM_Door data as Bonsai's door tool writes it (BIMDoorProperties.get_*_kwargs, default parameters), or None
    for an operation type Bonsai's door tool cannot rebuild (ROLLINGUP shutters keep their own geometry)."""
    if operation_type not in BONSAI_DOOR_TYPES:
        return None
    lining = {"lining_depth": 0.05, "lining_thickness": 0.05, "lining_offset": 0.0}
    if "SLIDING" not in operation_type:
        lining.update({"lining_to_panel_offset_x": 0.025, "lining_to_panel_offset_y": 0.025})
    lining.update({"transom_thickness": 0.0, "casing_thickness": 0.075, "casing_depth": 0.005,
                   "threshold_thickness": 0.025, "threshold_depth": 0.1, "threshold_offset": 0.0})
    return {"door_type": operation_type, "overall_height": dim(height), "overall_width": dim(width),
            "lining_properties": lining, "panel_properties": {"panel_depth": 0.035, "panel_width_ratio": 1.0}}


def window_data(width: float, height: float, window_type: str = "SINGLE_PANEL") -> dict:
    """BBIM_Window data as Bonsai's window tool writes it (BIMWindowProperties.get_*_kwargs, default parameters)."""
    return {"window_type": window_type, "overall_height": dim(height), "overall_width": dim(width),
            "lining_properties": {"lining_depth": 0.05, "lining_thickness": 0.05, "lining_offset": 0.05,
                                  "lining_to_panel_offset_x": 0.025, "lining_to_panel_offset_y": 0.025},
            "panel_properties": {"frame_depth": [0.035, 0.035, 0.035], "frame_thickness": [0.035, 0.035, 0.035]}}


def door_kwargs(data: dict) -> dict:
    """ifcopenshell.api.geometry.add_door_representation keyword arguments (without context) for BBIM_Door data;
    keys the data leaves out keep the API defaults, which equal Bonsai's property defaults."""
    lining, panel = data.get("lining_properties", {}), data.get("panel_properties", {})
    return {"operation_type": data["door_type"], "overall_height": data["overall_height"],
            "overall_width": data["overall_width"],
            "lining_properties": {api: lining[k] for k, api in DOOR_LINING_API.items() if k in lining},
            "panel_properties": {api: panel[k] for k, api in DOOR_PANEL_API.items() if k in panel}}


def window_kwargs(data: dict) -> dict:
    """ifcopenshell.api.geometry.add_window_representation keyword arguments (without context) for BBIM_Window
    data: one panel entry per panel of the window type, from the per-panel frame depth / thickness lists."""
    lining, panel = data.get("lining_properties", {}), data.get("panel_properties", {})
    n = WINDOW_PANELS[data["window_type"]]
    fd, ft = panel.get("frame_depth", [0.035] * 3), panel.get("frame_thickness", [0.035] * 3)
    return {"partition_type": data["window_type"], "overall_height": data["overall_height"],
            "overall_width": data["overall_width"],
            "lining_properties": {api: lining[k] for k, api in WINDOW_LINING_API.items() if k in lining},
            "panel_properties": [{"FrameDepth": fd[i], "FrameThickness": ft[i]} for i in range(n)]}


def data_text(data: dict) -> str:
    """The JSON stored in the pset's Data property (as Bonsai's tool.Pset.write_bbim_data serialises it)."""
    return json.dumps(data, default=list)
