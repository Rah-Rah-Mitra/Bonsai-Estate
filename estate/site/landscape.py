"""Soft landscape: the grass terrain and the trees (deterministic positions, typed species instances).

The terrain is everything in the estate extent that is not a building, carriageway, kerb or paved surface, cut
into 100 m tiles so a game engine can cull it. Grass sits 50 mm below the paving (top -0.05) so paths read as
paths. Trees go in the road verges at about 9 m centres, along the park connector and on a jittered 9 m grid in
the greens; every trunk keeps clear of paths, canopies, play items, crossings and building faces. Positions and
species come from sha256-derived seeds, never from Python's hash().
"""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import Point, box
from shapely.ops import unary_union

from estate.flats.template import derive_seed
from estate.site import Piece

TERRAIN_TOP, TERRAIN_T = -0.05, 0.30
ROAD_SPACING = 9.0
GREEN_SPACING = 9.0
OPEN_SPACING = 13.0
MIN_TREE_GAP = 5.5
# species index into estate.ifc.vegetation.ESTATE_SPECIES
RAIN_TREE, ANGSANA, YELLOW_FLAME, SEA_APPLE, TEMBUSU = range(5)
ROAD_SPECIES = {"collector": ANGSANA, "local": RAIN_TREE, "access": YELLOW_FLAME}


def terrain_pieces(extent, hard, tile=100.0):
    soft = extent.difference(hard)
    x0, y0, x1, y1 = extent.bounds
    out = []
    ni, nj = int(math.ceil((x1 - x0) / tile)), int(math.ceil((y1 - y0) / tile))
    for j in range(nj):
        for i in range(ni):
            t = soft.intersection(box(x0 + i * tile, y0 + j * tile, min(x1, x0 + (i + 1) * tile), min(y1, y0 + (j + 1) * tile)))
            if t.is_empty or t.area < 1.0:
                continue
            out.append(Piece("terrain", f"Terrain {chr(65 + j)}{i + 1}", t, TERRAIN_TOP - TERRAIN_T, TERRAIN_T, "grass",
                             props={"Tile": f"{chr(65 + j)}{i + 1}", "SurfaceTop": TERRAIN_TOP}))
    return out


class Planter:
    """Accepts tree positions that keep clear of the exclusion geometry and of each other."""

    def __init__(self, keep_out):
        self.keep_out = keep_out
        shapely.prepare(self.keep_out)
        self.trees = []
        self._grid = {}

    def _near(self, x, y):
        k = (int(x // MIN_TREE_GAP), int(y // MIN_TREE_GAP))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for (tx, ty) in self._grid.get((k[0] + di, k[1] + dj), ()):
                    if math.hypot(tx - x, ty - y) < MIN_TREE_GAP:
                        return True
        return False

    def add(self, x, y, species, zone, allowed=None):
        p = Point(x, y)
        if allowed is not None and not allowed.contains(p):
            return False
        if self.keep_out.contains(p) or self._near(x, y):
            return False
        self.trees.append(dict(x=round(float(x), 2), y=round(float(y), 2), species=int(species), zone=zone))
        self._grid.setdefault((int(x // MIN_TREE_GAP), int(y // MIN_TREE_GAP)), []).append((x, y))
        return True


def plant(seed, net, greens, pcn, keep_out, verge_keep_out, open_area=None):
    """Tree list [dict(x, y, species, zone)] for road verges, the park connector and the greens."""
    pl = Planter(keep_out)
    vpl = Planter(unary_union([keep_out, verge_keep_out]))
    vpl._grid = pl._grid
    vpl.trees = pl.trees
    for r in net.roads:
        if r.verge < 1.4:
            continue
        species = ROAD_SPECIES.get(r.cls, RAIN_TREE)
        allowed = net.verges[r.id].buffer(-0.3)
        for side in (-1, 1):
            d = side * (r.cw / 2 + 0.15 + r.verge / 2)
            off = (derive_seed(seed, "trees", r.id, side) % 900) / 100.0
            s = 4.0 + off
            while s < r.length - 2.0:
                p = r.pt(s, d)
                vpl.add(p[0], p[1], species, f"road:{r.id}", allowed)
                s += ROAD_SPACING
    if pcn is not None:
        line, area = pcn["line"], pcn["area"]
        (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
        L = line.length
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        for side in (-1, 1):
            k = 0
            s = 6.0
            while s < L - 6.0:
                off = pcn["width"] / 2 + 2.5
                x, y = x0 + ux * s - uy * off * side, y0 + uy * s + ux * off * side
                pl.add(x, y, TEMBUSU if k % 3 else SEA_APPLE, "park_connector", area)
                s += ROAD_SPACING
                k += 1
    for gr in greens:
        rng = np.random.default_rng(derive_seed(seed, "trees", gr["id"]))
        x0, y0, x1, y1 = gr["rect"].bounds
        allowed = gr["rect"].buffer(-1.5)
        ys = np.arange(y0 + GREEN_SPACING / 2, y1, GREEN_SPACING)
        xs = np.arange(x0 + GREEN_SPACING / 2, x1, GREEN_SPACING)
        for y in ys:
            for x in xs:
                jx, jy = rng.uniform(-2.2, 2.2, 2)
                sp = int(rng.integers(0, 5))
                pl.add(x + jx, y + jy, sp, f"green:{gr['id']}", allowed)
    if open_area is not None and not open_area.is_empty:
        rng = np.random.default_rng(derive_seed(seed, "trees", "open"))
        shapely.prepare(open_area)
        x0, y0, x1, y1 = open_area.bounds
        for y in np.arange(y0 + OPEN_SPACING / 2, y1, OPEN_SPACING):
            for x in np.arange(x0 + OPEN_SPACING / 2, x1, OPEN_SPACING):
                jx, jy = rng.uniform(-3.0, 3.0, 2)
                sp = int(rng.choice(5, p=[0.3, 0.2, 0.2, 0.2, 0.1]))
                pl.add(x + jx, y + jy, sp, "open_space", open_area)
    return pl.trees


def tree_pieces(trees):
    return [Piece("tree", f"Tree {k + 1:04d}", None, props={"Zone": t["zone"]},
                  solid={"tree": t["species"], "at": (t["x"], t["y"], TERRAIN_TOP)}) for k, t in enumerate(trees)]
