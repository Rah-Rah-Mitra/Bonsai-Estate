"""Door leaves: where a leaf sits closed and how it opens (numpy only). The one home of the pose formula.

The engine export (estate/export/engine.py) splits a door into its frame and its leaves and records each leaf here:
``leaf_spec`` gives a panel's motion and its pivot in the door's own frame (x along the opening, y across the wall,
z up from the sill), ``leaf_frame`` the leaf node's matrix under the door node (its origin on the pivot) and
``leaf_record`` the block-local leaf the engine JSON lists. The web stage (estate/web) reads those records back:
``open_matrix`` is the opened pose and ``opened_footprint`` the plan of the opened leaf, so a viewer that bakes
the leaves open and the walk grid that blocks them use the same numbers.

- swing: the pivot is the hinge, on the swing-side face (the middle of the leaf for a double-swing door);
  turning the leaf about the door's up axis by ``open_sign`` * ``max_angle_deg`` opens it towards the swing side;
- slide: the pivot is on the jamb-side edge, and the leaf moves by ``travel`` (its width, along the wall);
- roll: a shutter; the pivot is its lower jamb-side corner, and the curtain rolls up by ``travel`` (its height);
- none: the leaf does not move.

This module is matched by the nav stage's ``estate/validate/nav*.py`` and listed in the glb stage's code
(estate/pipeline/state.py STAGE_DEPS), so a change to the pose formula rebuilds every stage that uses it.
"""
from __future__ import annotations

import math

import numpy as np

OPEN_DEG = 90.0              # a swing leaf's opened angle unless the leaf says otherwise


def _xf(M, v):
    v = np.asarray(v, float)
    return v @ M[:3, :3].T + M[:3, 3]


def _r3(v, nd=3):
    return [round(float(x), nd) for x in v]


# ----------------------------------------------------------------------------- the export side
def leaf_spec(op: str, lo, hi, k: int, n: int) -> dict:
    """Motion and pivot of panel k of n (its box lo..hi in the door frame) for a door of OperationType ``op``.

    Sliding leaves slide away from the middle of a pair (or to the left for a single LEFT door); swing leaves hinge
    on the outer jamb of a pair, or on the jamb that names the operation (RIGHT: at x1)."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    first, last = k == 0, k == n - 1
    spec = dict(thickness=round(float(y1 - y0), 4), width=round(float(x1 - x0), 4),
                height=round(float(z1 - z0), 4), z0=round(float(z0), 4))
    if "SLIDING" in op:
        sgn = -1.0 if op.endswith("LEFT") or (n > 1 and first) else 1.0
        spec.update(motion="slide", pivot=(float(x0 if sgn < 0 else x1), float((y0 + y1) / 2)),
                    travel=(sgn * float(x1 - x0), 0.0, 0.0))
    elif "ROLLING" in op:
        spec.update(motion="roll", pivot=(float(x0), float((y0 + y1) / 2)), travel=(0.0, 0.0, float(z1 - z0)))
    elif "SWING" in op:
        right = (n > 1 and last and not first) or (n == 1 and op.endswith("RIGHT"))
        both = "DOUBLE_SWING" in op
        spec.update(motion="swing", pivot=(float(x1 if right else x0), float((y0 + y1) / 2 if both else y1)),
                    open_sign=-1 if right else 1, both_ways=both, max_angle_deg=OPEN_DEG)
    else:
        spec.update(motion="none", pivot=(float(x0), float((y0 + y1) / 2)))
    return spec


def leaf_frame(pivot) -> np.ndarray:
    """The leaf node's matrix under its door node: a shift to the pivot (the leaf mesh is stored about it)."""
    P = np.eye(4)
    P[:2, 3] = pivot
    return P


def leaf_record(M, lf: dict) -> dict:
    """The engine JSON's record of a leaf (``lf``: a leaf_spec plus its node name) of a door placed at M (door frame
    -> block-local): pivot, axis and travel in block-local metres, Z up."""
    rec = dict(node=lf["node"], motion=lf["motion"], pivot=_r3(_xf(M, [*lf["pivot"], 0.0])),
               thickness=lf["thickness"], width=lf["width"], height=lf["height"])
    if lf["motion"] == "swing":
        rec.update(axis=_r3(M[:3, 2]), open_sign=lf["open_sign"], both_ways=lf["both_ways"],
                   max_angle_deg=lf["max_angle_deg"])
    elif "travel" in lf:
        rec["travel"] = _r3(M[:3, :3] @ np.array(lf["travel"]))
    return rec


# ----------------------------------------------------------------------------- the viewer / walk side
def _rotation(axis, angle) -> np.ndarray:
    k = np.asarray(axis, float)
    k = k / max(float(np.linalg.norm(k)), 1e-12)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def open_matrix(rec: dict, angle_deg: float | None = None) -> np.ndarray:
    """4x4 block-local (Z up) transform that takes a leaf of the engine JSON (``rec``) from closed to open: a turn
    about the hinge axis through the pivot for a swing leaf, a shift by ``travel`` for a sliding leaf or a shutter,
    the identity for a leaf that does not move."""
    O = np.eye(4)
    if rec["motion"] == "swing":
        deg = rec.get("max_angle_deg", OPEN_DEG) if angle_deg is None else angle_deg
        R = _rotation(rec.get("axis", (0.0, 0.0, 1.0)), math.radians(rec.get("open_sign", 1) * deg))
        p = np.asarray(rec["pivot"], float)
        O[:3, :3] = R
        O[:3, 3] = p - R @ p
    elif rec.get("travel") is not None:
        O[:3, 3] = rec["travel"]
    return O


def leaf_extent(rec: dict, along=None) -> tuple:
    """The closed leaf's box about its pivot, in the door frame: ((x0, x1), (y0, y1), (z0, z1)) along the opening,
    across the wall (towards the swing side) and up. leaf_spec in reverse: a swing leaf runs from its hinge into the
    opening (towards -x when it hinges on the right, open_sign -1) and lies behind the swing-side face (centred for
    a double-swing leaf); a sliding leaf runs from its jamb-side edge away from its travel; shutters and fixed
    leaves run from the pivot along +x. ``along`` (the door's along-wall axis, block-local) gives the travel's
    direction; the height starts at the pivot (the sill)."""
    w, t, h = float(rec["width"]), float(rec["thickness"]), float(rec["height"])
    mid = (-t / 2, t / 2)
    if rec["motion"] == "swing":
        xs = (0.0, w) if rec.get("open_sign", 1) > 0 else (-w, 0.0)
        ys = mid if rec.get("both_ways") else (-t, 0.0)
    elif rec["motion"] == "slide":
        sgn = float(np.dot(rec.get("travel", (0.0, 0.0, 0.0)), along if along is not None else (1.0, 0.0, 0.0)))
        xs, ys = ((0.0, w) if sgn < 0 else (-w, 0.0)), mid
    else:
        xs, ys = (0.0, w), mid
    return xs, ys, (0.0, h)


def leaf_corners(rec: dict, along, across, opened=True, angle_deg: float | None = None) -> np.ndarray:
    """The 8 corners (block-local, Z up) of a leaf's box, opened by open_matrix (to ``angle_deg``, a swing leaf's
    own angle when None) or closed. ``along`` and ``across`` are the door's axes (the engine JSON's along_wall and
    swing_side); up is their cross product."""
    X, Y = np.asarray(along, float), np.asarray(across, float)
    Z = np.cross(X, Y)
    Z = Z / max(float(np.linalg.norm(Z)), 1e-12)
    xs, ys, zs = leaf_extent(rec, X)
    p = np.asarray(rec["pivot"], float)
    C = np.array([p + x * X + y * Y + z * Z for x in xs for y in ys for z in zs])
    if not opened:
        return C
    O = open_matrix(rec, angle_deg)
    return C @ O[:3, :3].T + O[:3, 3]


def opened_footprint(rec: dict, along, across, angle_deg: float | None = None) -> list:
    """Plan of the opened leaf: the convex hull of its box corners (counter-clockwise, first point not repeated)."""
    pts = np.unique(np.round(leaf_corners(rec, along, across, True, angle_deg)[:, :2], 6), axis=0)
    if len(pts) < 3:
        return [list(map(float, q)) for q in pts]
    pts = sorted(map(tuple, pts))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 1e-12:
            lower.pop()
        lower.append(q)
    for q in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 1e-12:
            upper.pop()
        upper.append(q)
    return [list(q) for q in lower[:-1] + upper[:-1]]


def opened_z(rec: dict, along, across, angle_deg: float | None = None) -> tuple:
    """Lowest and highest point of the opened leaf, block-local: a rolled-up shutter clears the agent's head."""
    C = leaf_corners(rec, along, across, True, angle_deg)
    return float(C[:, 2].min()), float(C[:, 2].max())
