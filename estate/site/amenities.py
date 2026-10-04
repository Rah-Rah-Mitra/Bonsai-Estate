"""Amenities on the site: greens (park paths, playground, fitness corner, pavilion, ball court, lawn), drop-off
porches at the blocks, bus shelters and service bays (geometry and element pieces).

A green is laid out on its long axis: a perimeter loop path inset from the edge, a spine path from edge to edge
through the middle (its ends are the gates the footpath router connects to), and the listed items in slots along
the spine, alternating sides, each touching the spine so it is entered from a path. Equipment is low-poly boxes:
enough to read as a playground in plan, in a render and to a navigation agent (it blocks walking).
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from estate.site import Piece
from estate.site.linkways import H, V

ITEM_SIZE = {"playground": (18.0, 14.0), "fitness_corner": (12.0, 9.0), "pavilion": (10.0, 8.0),
             "ball_court": (32.0, 19.0), "lawn": (24.0, 18.0)}
ITEM_NAME = {"playground": "Playground", "fitness_corner": "Fitness corner", "pavilion": "Pavilion",
             "ball_court": "Ball court", "lawn": "Lawn"}
LOOP_INSET, LOOP_W, SPINE_W, GAP = 3.0, 2.0, 2.4, 0.6
# drop-off porch (frame: a outward from the facade, b along it)
APRON_A, LANE_A, LANE_B, CANOPY_A, CANOPY_B, PORCH_SOFFIT = 3.0, 7.0, 9.0, 7.6, 5.5, 3.6
BAY = 8.0


class Frame:
    """2D frame: local (a, b) -> origin + a * u + b * n (n = u rotated +90 degrees)."""

    def __init__(self, origin, u):
        self.o = np.asarray(origin, float)[:2]
        u = np.asarray(u, float)[:2]
        self.u = u / np.linalg.norm(u)
        self.n = np.array([-self.u[1], self.u[0]])

    def p(self, a, b):
        q = self.o + self.u * a + self.n * b
        return (round(float(q[0]), 4), round(float(q[1]), 4))

    def rect(self, a0, b0, a1, b1):
        return shapely.normalize(Polygon([self.p(a0, b0), self.p(a1, b0), self.p(a1, b1), self.p(a0, b1)]))

    def sq(self, a, b, size):
        return self.rect(a - size / 2, b - size / 2, a + size / 2, b + size / 2)


def _box(poly, z0, h, mat):
    return {"box": poly, "z0": z0, "h": h, "mat": mat}


def bench(F: Frame, a, b, name, along=True, length=1.8):
    """Bench centred at (a, b); seat along the frame's u axis (or n when along=False), backrest on the +b side."""
    if along:
        seat = F.rect(a - length / 2, b - 0.22, a + length / 2, b + 0.22)
        back = F.rect(a - length / 2, b + 0.18, a + length / 2, b + 0.26)
    else:
        seat = F.rect(a - 0.22, b - length / 2, a + 0.22, b + length / 2)
        back = F.rect(a + 0.18, b - length / 2, a + 0.26, b + length / 2)
    return Piece("furniture", name, None, object_type="Bench", mat="timber",
                 solid={"boxes": [_box(seat, 0.0, 0.45, "timber"), _box(back, 0.45, 0.4, "timber")]})


# --------------------------------------------------------------------------- greens
def layout_green(g: dict, obstacles) -> dict:
    """Paths, item rectangles, gates and pieces of one green (config [[green]] table)."""
    x0, y0, x1, y1 = g["rect"]
    rect = box(x0, y0, x1, y1)
    usable = rect.difference(obstacles.buffer(1.0)) if obstacles is not None and not obstacles.is_empty else rect
    gid = g["id"]
    w, h = x1 - x0, y1 - y0
    along_x = w >= h
    Lg, Wg = (w, h) if along_x else (h, w)
    c = ((x0 + x1) / 2, (y0 + y1) / 2)
    F = Frame(c, (1, 0) if along_x else (0, 1))       # a along the long axis, b across
    out = dict(id=gid, name=g.get("name", gid), rect=rect, items=[], lines=[], gates=[], pieces=[], lawns=[],
               axis=H if along_x else V)
    # paths
    outer = rect.buffer(-LOOP_INSET, join_style="mitre")
    inner = rect.buffer(-(LOOP_INSET + LOOP_W), join_style="mitre")
    loop = outer.difference(inner)
    spine = F.rect(-Lg / 2, -SPINE_W / 2, Lg / 2, SPINE_W / 2)
    paths = unary_union([loop, spine]).intersection(usable)
    ring = rect.buffer(-(LOOP_INSET + LOOP_W / 2), join_style="mitre").exterior
    cs = list(ring.coords)
    for p, q in zip(cs, cs[1:]):
        out["lines"].append((LineString([p, q]), LOOP_W))
    out["lines"].append((LineString([F.p(-Lg / 2, 0), F.p(Lg / 2, 0)]), SPINE_W))
    out["gates"] = [(F.p(-Lg / 2, 0), tuple(-F.u)), (F.p(Lg / 2, 0), tuple(F.u))]
    # items in slots along the spine
    items = [i for i in g.get("items", []) if i in ITEM_SIZE]
    n = len(items)
    inner_len = Lg - 2 * (LOOP_INSET + LOOP_W) - 2.0
    half_avail = Wg / 2 - SPINE_W / 2 - GAP - (LOOP_INSET + LOOP_W) - 1.0
    slot = inner_len / max(n, 1)
    blocks = []
    for k, kind in enumerate(items):
        a_c = -inner_len / 2 + slot * (k + 0.5)
        side = 1 if k % 2 == 0 else -1
        sa, sb = ITEM_SIZE[kind]
        sa, sb = min(sa, slot - 2.0), min(sb, half_avail)
        if sa < 6.0 or sb < 5.0:
            continue
        b0 = side * (SPINE_W / 2 + GAP)
        b1 = side * (SPINE_W / 2 + GAP + sb)
        r = F.rect(a_c - sa / 2, min(b0, b1), a_c + sa / 2, max(b0, b1))
        if not usable.buffer(0.01).contains(r):
            continue
        IF = Frame(F.p(a_c, (b0 + b1) / 2), F.u * (1 if side > 0 else -1))    # +b of the item faces away from the spine
        name = f"{out['name']} {ITEM_NAME[kind].lower()}"
        item = dict(kind=kind, rect=r, name=name, frame=IF, size=(sa, sb))
        out["items"].append(item)
        out["pieces"] += ITEM_PIECES[kind](IF, sa, sb, name, f"{gid}:{kind}")
        if kind == "lawn":
            out["lawns"].append(r)
        else:
            blocks.append(r)
    paths = paths.difference(unary_union(blocks)) if blocks else paths
    out["paths"] = paths
    out["pieces"].insert(0, Piece("paving", f"{out['name']} paths", paths, -0.15, 0.15, "paving",
                                  object_type="Park path", props={"Kind": "park"}))
    out["block"] = unary_union(blocks) if blocks else Polygon()
    # benches along the spine between items
    for k in range(n + 1):
        a_c = -inner_len / 2 + slot * k
        side = -1 if k % 2 == 0 else 1
        bpt = F.p(a_c, side * (SPINE_W / 2 + 0.6))
        if usable.contains(shapely.Point(bpt)) and not any(it["rect"].buffer(0.5).contains(shapely.Point(bpt))
                                                            for it in out["items"]):
            out["pieces"].append(bench(F, a_c, side * (SPINE_W / 2 + 0.6), f"{out['name']} bench {k + 1}"))
    return out


def _playground(F: Frame, sa, sb, name, key):
    A, B = sa / 2, sb / 2
    pcs = [Piece("paving", f"{name} surface", F.rect(-A, -B, A, B), -0.05, 0.05, "rubber", object_type="Playground surface")]
    # play tower with slide
    ta, tb = -A * 0.45, 0.0
    tower = [_box(F.sq(ta + dx, tb + dy, 0.12), 0.0, 2.7, "steel") for dx in (-1.1, 1.1) for dy in (-1.1, 1.1)]
    tower += [_box(F.sq(ta, tb, 2.4), 1.2, 0.1, "timber"), _box(F.sq(ta, tb, 2.8), 2.6, 0.12, "accent")]
    steps = [_box(F.rect(ta + 1.2 + 0.6 * k, tb - 0.35, ta + 1.8 + 0.6 * k, tb + 0.35), 0.0, 1.0 - 0.25 * k, "accent")
             for k in range(4)]
    pcs.append(Piece("furniture", f"{name} play tower", None, object_type="Play equipment", mat="accent",
                     solid={"boxes": tower + steps}))
    # swing frame
    sa_ = A * 0.45
    sw = [_box(F.sq(sa_ - 1.6, tb + dy, 0.1), 0.0, 2.4, "steel") for dy in (-0.9, 0.9)]
    sw += [_box(F.sq(sa_ + 1.6, tb + dy, 0.1), 0.0, 2.4, "steel") for dy in (-0.9, 0.9)]
    sw += [_box(F.rect(sa_ - 1.65, tb - 0.05, sa_ + 1.65, tb + 0.05), 2.4, 0.1, "steel")]
    sw += [_box(F.rect(sa_ + dx - 0.25, tb - 0.12, sa_ + dx + 0.25, tb + 0.12), 0.45, 0.05, "rubber") for dx in (-0.8, 0.8)]
    pcs.append(Piece("furniture", f"{name} swings", None, object_type="Play equipment", mat="steel", solid={"boxes": sw}))
    for k, (da, db) in enumerate(((0.0, -B * 0.55), (0.0, B * 0.55))):
        pcs.append(Piece("furniture", f"{name} spring rider {k + 1}", None, object_type="Play equipment", mat="accent",
                         solid={"boxes": [_box(F.sq(da, db, 0.3), 0.0, 0.35, "steel"),
                                          _box(F.rect(da - 0.4, db - 0.15, da + 0.4, db + 0.15), 0.35, 0.3, "accent")]}))
    return pcs


def _fitness(F: Frame, sa, sb, name, key):
    A, B = sa / 2, sb / 2
    pcs = [Piece("paving", f"{name} surface", F.rect(-A, -B, A, B), -0.05, 0.05, "rubber", object_type="Fitness surface")]
    for k in range(4):
        a = -A + sa * (k + 0.5) / 4
        fr = [_box(F.sq(a - 0.6, 0.0, 0.12), 0.0, 2.0, "steel"), _box(F.sq(a + 0.6, 0.0, 0.12), 0.0, 2.0, "steel"),
              _box(F.rect(a - 0.65, -0.06, a + 0.65, 0.06), 1.9, 0.1, "steel"),
              _box(F.rect(a - 0.3, -0.3, a + 0.3, 0.3), 0.0, 0.5, "accent")]
        pcs.append(Piece("furniture", f"{name} station {k + 1}", None, object_type="Fitness station", mat="steel",
                         solid={"boxes": fr}))
    return pcs


def _pavilion(F: Frame, sa, sb, name, key):
    A, B = sa / 2, sb / 2
    asm = f"pav:{key}"
    pcs = [Piece("paving", f"{name} floor", F.rect(-A, -B, A, B), -0.15, 0.15, "paving", assembly=asm,
                 object_type="Pavilion floor")]
    for k, (a, b) in enumerate(((-A + 0.5, -B + 0.5), (0.0, -B + 0.5), (A - 0.5, -B + 0.5),
                                (-A + 0.5, B - 0.5), (0.0, B - 0.5), (A - 0.5, B - 0.5))):
        pcs.append(Piece("column", f"{name} column {k + 1}", F.sq(a, b, 0.3), 0.0, 3.0, "column", assembly=asm))
    pcs.append(Piece("roof", f"{name} roof", F.rect(-A - 0.6, -B - 0.6, A + 0.6, B + 0.6), 3.0, 0.15, "roof", assembly=asm))
    for k, b in enumerate((-B + 1.3, B - 1.3)):
        p = bench(F, 0.0, b, f"{name} bench {k + 1}", length=3.0)
        p.assembly = asm
        pcs.append(p)
    return pcs


def _court(F: Frame, sa, sb, name, key):
    A, B = sa / 2, sb / 2
    pcs = [Piece("paving", f"{name} surface", F.rect(-A, -B, A, B), -0.1, 0.1, "kerb", object_type="Ball court")]
    t, gate = 0.05, 2.0
    sides = [_box(F.rect(-A, B - t, A, B), 0.0, 3.0, "alu"),                   # far side
             _box(F.rect(-A, -B, -A + t, B), 0.0, 3.0, "alu"), _box(F.rect(A - t, -B, A, B), 0.0, 3.0, "alu"),
             _box(F.rect(-A, -B, -gate / 2, -B + t), 0.0, 3.0, "alu"),         # spine side with a gate
             _box(F.rect(gate / 2, -B, A, -B + t), 0.0, 3.0, "alu")]
    pcs.append(Piece("fence", f"{name} fence", None, mat="alu", object_type="Fence", solid={"boxes": sides},
                     props={"Height": 3.0, "GateWidth": gate}))
    for k, a in enumerate((-A + 1.2, A - 1.2)):
        sgn = 1 if a > 0 else -1
        hoop = [_box(F.sq(a + sgn * 0.6, 0.0, 0.2), 0.0, 3.0, "steel"),
                _box(F.rect(a - 0.03, -0.9, a + 0.03, 0.9), 2.9, 1.05, "alu")]
        pcs.append(Piece("furniture", f"{name} hoop {k + 1}", None, object_type="Basketball hoop", mat="steel",
                         solid={"boxes": hoop}))
    return pcs


def _lawn(F: Frame, sa, sb, name, key):
    return [bench(F, a, -sb / 2 + 0.6, f"{name} bench {k + 1}") for k, a in enumerate((-sa / 4, sa / 4))]


ITEM_PIECES = {"playground": _playground, "fitness_corner": _fitness, "pavilion": _pavilion, "ball_court": _court,
               "lawn": _lawn}


# --------------------------------------------------------------------------- porches and bays
def porch_geometry(E, outward):
    """Apron, drop-off lane, canopy and the points the routers attach to, for an entrance E on a facade."""
    F = Frame(E, outward)
    return dict(frame=F, apron=F.rect(0.0, -6.0, APRON_A, 6.0), lane=F.rect(APRON_A, -LANE_B, LANE_A, LANE_B),
                canopy=F.rect(0.0, -CANOPY_B, CANOPY_A, CANOPY_B), area=F.rect(0.0, -LANE_B, CANOPY_A + 0.5, LANE_B),
                drive_pt=F.p((APRON_A + LANE_A) / 2, 0.0), stub_pt=F.p(CANOPY_A, 0.0),
                along=tuple(F.n), out=tuple(F.u))


def porch_pieces(pg, name, key):
    F = pg["frame"]
    asm = f"porch:{key}"
    pcs = [Piece("paving", f"{name} drop-off apron", pg["apron"], -0.15, 0.15, "paving", assembly=asm,
                 object_type="Drop-off apron"),
           Piece("roof", f"{name} porch canopy", pg["canopy"], PORCH_SOFFIT, 0.15, "slab", assembly=asm)]
    for k, (a, b) in enumerate(((0.45, -CANOPY_B + 0.3), (0.45, CANOPY_B - 0.3), (CANOPY_A - 0.3, -CANOPY_B + 0.3),
                                (CANOPY_A - 0.3, CANOPY_B - 0.3))):
        pcs.append(Piece("column", f"{name} porch column {k + 1}", F.sq(a, b, 0.3), 0.0, PORCH_SOFFIT, "column",
                         assembly=asm))
    return pcs


def bay_geometry(P, outward):
    F = Frame(P, outward)
    return dict(frame=F, bay=F.rect(0.0, -BAY / 2, BAY, BAY / 2), drive_pt=F.p(BAY / 2, 0.0), out=tuple(F.u))


SHELTER_GAP = 1.8       # opening in the middle of the back panel, where the covered linkway lands


def bus_shelter_pieces(bs, part):
    """Shelter on a bus platform: roof, rear posts, a back panel in two halves either side of the opening the
    linkway arrives through (the shelter centre line, which the pedestrian graph walks), and a bench either side
    of it (one assembly in the BUS_STOP part)."""
    r, s, side, kd = bs.road, bs.s, bs.side, bs.kerb_d
    from estate.site.roads import SHELTER_LEN, SHELTER_REAR, SHELTER_SOFFIT
    asm = f"bs:{bs.id}"

    def P(ss, dd):
        q = r.pt(ss, kd + side * dd)
        return (float(q[0]), float(q[1]))

    def rect(s0, d0, s1, d1):
        return shapely.normalize(Polygon([P(s0, d0), P(s1, d0), P(s1, d1), P(s0, d1)]))
    pcs = [Piece("roof", f"Bus stop {bs.id} shelter roof", bs.shelter, SHELTER_SOFFIT, 0.1, "metal_roof",
                 part=part, assembly=asm)]
    g = SHELTER_GAP / 2
    for k, ds in enumerate((-SHELTER_LEN / 2 + 0.4, -g - 0.06, g + 0.06, SHELTER_LEN / 2 - 0.4)):
        pcs.append(Piece("column", f"Bus stop {bs.id} post {k + 1}",
                         rect(s + ds - 0.06, SHELTER_REAR - 0.36, s + ds + 0.06, SHELTER_REAR - 0.24),
                         0.0, SHELTER_SOFFIT, "steel", part=part, assembly=asm))
    panels = [_box(rect(s0, SHELTER_REAR - 0.33, s1, SHELTER_REAR - 0.27), 0.3, 1.9, "glass")
              for s0, s1 in ((s - SHELTER_LEN / 2 + 0.6, s - g - 0.12), (s + g + 0.12, s + SHELTER_LEN / 2 - 0.6))]
    pcs.append(Piece("furniture", f"Bus stop {bs.id} back panel", None, part=part, assembly=asm,
                     object_type="Shelter panel", mat="glass", solid={"boxes": panels}))
    for k, (s0, s1) in enumerate(((s - 3.9, s - g - 0.3), (s + g + 0.3, s + 3.9))):
        seat = rect(s0, SHELTER_REAR - 0.95, s1, SHELTER_REAR - 0.5)
        pcs.append(Piece("furniture", f"Bus stop {bs.id} bench {k + 1}", None, part=part, assembly=asm,
                         object_type="Bench", mat="timber", solid={"boxes": [_box(seat, 0.0, 0.45, "timber")]}))
    return pcs
