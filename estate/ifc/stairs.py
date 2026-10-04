"""Half-turn (dog-leg) common stairs (lifted from legacy/toolkit/hdb_block.py: flight, stair, flight_run).

``stair`` builds in the legacy frame (entry on the north side, flights running north-south) and rotates the
whole assembly by k * 90 degrees about the enclosure centre for other entry sides.
"""
from __future__ import annotations

import math

import numpy as np
import ifcopenshell.api.aggregate as aggregate
from shapely.geometry import box

from estate.geom.walls import rotate_z, translate
from estate.rules import BALUSTRADE_H, FLIGHT_W, GOING, LANDING, MAX_RISERS_PER_FLIGHT, RISER_MAX, WAIST


class RuleViolation(Exception):
    pass


ENTRY_TURNS = {"N": 0, "W": 1, "S": 2, "E": 3}


def flight_run(h):
    N = math.ceil(h / RISER_MAX - 1e-9)
    return (math.ceil(N / 2) - 1) * GOING


def flight(W, name, x_lo, x_hi, y_start, direction, z0, n, r, container):
    """Stepped RC flight: n risers of height r, (n-1) goings, running north (+1) or south (-1)."""
    m = W.m
    dv = WAIST * math.sqrt(1 + (r / GOING) ** 2)
    prof = [(0.0, 0.0), (0.0, r)]
    for i in range(1, n):
        prof += [(i * GOING, i * r), (i * GOING, (i + 1) * r)]
    prof += [((n - 1) * GOING, (n - 1) * r - dv), (dv * GOING / r, 0.0)]
    R = np.array([0.0, float(direction), 0.0])
    Z = np.cross(R, [0.0, 0.0, 1.0])
    x_side = x_lo if Z[0] > 0 else x_hi
    solid = m.createIfcExtrudedAreaSolid(
        m.createIfcArbitraryClosedProfileDef("AREA", None, W._curve(prof)),
        W._place3d(axis=Z, ref=R), m.createIfcDirection((0.0, 0.0, 1.0)), float(x_hi - x_lo))
    rep = m.createIfcShapeRepresentation(W.body, "Body", "SweptSolid", [solid])
    el = W.product("IfcStairFlight", name, rep, translate(x_side, y_start, z0), None, "STRAIGHT", "stair")
    el.NumberOfRisers, el.NumberOfTreads = n, n - 1
    el.RiserHeight, el.TreadLength = round(r, 4), GOING
    W.props(el, "Pset_StairFlightCommon", {"NumberOfRiser": n, "NumberOfTreads": n - 1,
                                           "RiserHeight": round(r, 4), "TreadLength": GOING})
    # balustrade on the well side, 1.0 m above the pitch line
    inner = x_lo if x_side == x_hi else x_hi
    t = 0.04
    rail_side = inner + t if Z[0] < 0 else inner - t
    rail_prof = [(0.0, r - 0.3), ((n - 1) * GOING, n * r - 0.3), ((n - 1) * GOING, n * r + BALUSTRADE_H),
                 (0.0, r + BALUSTRADE_H)]
    rsolid = m.createIfcExtrudedAreaSolid(
        m.createIfcArbitraryClosedProfileDef("AREA", None, W._curve(rail_prof)),
        W._place3d(axis=Z, ref=R), m.createIfcDirection((0.0, 0.0, 1.0)), t)
    rail = W.product("IfcRailing", name + " balustrade",
                     m.createIfcShapeRepresentation(W.body, "Body", "SweptSolid", [rsolid]),
                     translate(rail_side, y_start, z0), None, "BALUSTRADE", "steel")
    return el, rail


def _stair_north(W, name, inner, z_lo, z_hi, storey, run_next, guards=True):
    x0, x1, yN, yS = inner
    depth = yN - yS
    d_mid = depth - LANDING
    h = z_hi - z_lo
    N = math.ceil(h / RISER_MAX - 1e-9)
    r = h / N
    N1, N2 = math.ceil(N / 2), N - math.ceil(N / 2)
    if not (max(N1, N2) <= MAX_RISERS_PER_FLIGHT and (N1 - 1) * GOING <= d_mid - LANDING + 1e-6):
        raise RuleViolation(f"{name}: {N} risers of {r:.3f} m do not fit a {depth:.2f} m enclosure")
    run1, run2 = (N1 - 1) * GOING, (N2 - 1) * GOING
    st = W.product("IfcStair", name, None, translate(x0, yS, z_lo), storey, "HALF_TURN_STAIR")
    W.props(st, "Pset_StairCommon", {"NumberOfRiser": N, "RiserHeight": round(r, 4), "TreadLength": GOING,
                                     "RequiredHeadroom": 2.0, "FireExit": True, "IsExternal": False})
    parts = []
    z_mid = z_lo + N1 * r
    east, west = (x1 - FLIGHT_W, x1), (x0, x0 + FLIGHT_W)
    parts += flight(W, f"{name} flight 1", *east, yN - (d_mid - run1), -1, z_lo, N1, r, storey)
    parts.append(W.slab(box(x0, yS, x1, yN - d_mid), z_mid, f"{name} mid landing", None, "LANDING", "stair"))
    parts += flight(W, f"{name} flight 2", *west, yN - d_mid, 1, z_mid, N2, r, storey)
    reach_w = d_mid - run2
    reach_e = d_mid - run_next if run_next is not None else reach_w
    land = box(west[0], yN - reach_w, west[1], yN).union(box(east[0], yN - reach_e, east[1], yN)).union(
        box(x0, yN - min(reach_w, reach_e), x1, yN))
    parts.append(W.slab(land, z_hi, f"{name} floor landing", None, "LANDING", "stair"))
    # guards across the stair well at the mid landing and the floor landing; at the topmost landing (roof) the
    # open edge over flight 1 is guarded too
    bands = [(box(west[1], yN - d_mid, east[0], yN - d_mid + 0.05), z_mid, "mid landing well guard")]
    if run_next is None:
        bands.append((box(west[1], yN - reach_w - 0.05, x1, yN - reach_w), z_hi, "landing guard"))
    else:
        y_f = yN - min(reach_w, reach_e)
        bands.append((box(west[1], y_f - 0.05, east[0], y_f), z_hi, "floor landing well guard"))
    for band, zg, label in (bands if guards else []):
        c = band.centroid
        g = W.product("IfcRailing", f"{name} {label}", W.extrusion(band, BALUSTRADE_H + 0.1, (c.x, c.y)),
                      translate(c.x, c.y, zg), None, "GUARDRAIL", "steel")
        parts.append(g)
    aggregate.assign_object(W.m, products=parts, relating_object=st)
    return N, r


def stair(W, name, inner, z_lo, z_hi, storey, run_next, entry="N", guards=True):
    """Dog-leg stair in an enclosure. inner = (x0, x1, y_north, y_south) for entry 'N' (legacy frame);
    for other entries pass the world inner box the same way and the stair is rotated about its centre."""
    k = ENTRY_TURNS[entry]
    if k == 0:
        return _stair_north(W, name, inner, z_lo, z_hi, storey, run_next, guards)
    x0, x1, yN, yS = inner
    cx, cy = (x0 + x1) / 2, (yN + yS) / 2
    hw, hd = (x1 - x0) / 2, (yN - yS) / 2
    if k % 2:
        hw, hd = hd, hw
    with W.transformed(rotate_z(k, (cx, cy))):
        return _stair_north(W, name, (cx - hw, cx + hw, cy + hd, cy - hd), z_lo, z_hi, storey, run_next, guards)
