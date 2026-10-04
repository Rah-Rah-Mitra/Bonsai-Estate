"""Raster plans with PIL (no matplotlib in Blender's Python): floor plates, flat gallery tiles, site plans.

Drawn from the derived plate (the same walls and openings the IFC is built from), so the drawings show
exactly what the model contains: faces coloured by use, walls by type, door leaves with swing arcs,
glazing in blue, room codes and net areas.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from estate.blocks.plate import (AMENITY, BALCONY, CORRIDOR, DECK, LIFT, LOBBY, PLANT, REFUSE, ROOM, STAIR, VOID)
from estate.geom.arrangement import wall_footprint

FILL = {
    "habitable": (246, 239, 217), "wet": (214, 230, 240), "service": (226, 226, 222), "shelter": (200, 198, 192),
    "circulation": (240, 234, 210), "outdoor": (214, 236, 204),
    CORRIDOR: (227, 236, 242), LOBBY: (227, 236, 242), LIFT: (190, 196, 200), STAIR: (234, 223, 199),
    REFUSE: (210, 205, 195), PLANT: (210, 205, 195), VOID: (236, 236, 233), AMENITY: (226, 220, 240),
    DECK: (205, 206, 208),
}
WALL = {"EXT": (59, 59, 59), "PARTY": (40, 40, 40), "CORE": (70, 92, 74), "HS": (25, 25, 25), "HS250": (25, 25, 25),
        "INT": (120, 120, 120), "WET": (110, 130, 150), "PARAPET": (150, 150, 150)}
FLAT_EDGE = [(31, 59, 87), (139, 61, 35), (40, 110, 60), (110, 50, 120), (150, 110, 20), (20, 110, 120)]


def font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Canvas:
    def __init__(self, bounds, scale=40.0, margin=60, top=40):
        x0, y0, x1, y1 = bounds
        self.x0, self.y1 = x0, y1
        self.s = scale
        self.m = margin
        self.top = top
        W = int((x1 - x0) * scale + 2 * margin)
        H = int((y1 - y0) * scale + 2 * margin + top)
        self.img = Image.new("RGB", (W, H), "white")
        self.d = ImageDraw.Draw(self.img)

    def P(self, p):
        return (self.m + (p[0] - self.x0) * self.s, self.top + self.m + (self.y1 - p[1]) * self.s)

    def poly(self, g, fill=None, outline=None, width=1):
        for part in getattr(g, "geoms", [g]):
            if part.geom_type != "Polygon" or part.is_empty:
                continue
            self.d.polygon([self.P(c) for c in part.exterior.coords], fill=fill, outline=outline, width=width)
            for hole in part.interiors:
                self.d.polygon([self.P(c) for c in hole.coords], fill="white")

    def line(self, pts, fill, width=1):
        self.d.line([self.P(p) for p in pts], fill=fill, width=width)

    def text(self, p, s, size=11, fill=(40, 40, 40), anchor="mm"):
        self.d.multiline_text(self.P(p), s, fill=fill, font=font(size), anchor=anchor, align="center")

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.img.save(path)


def face_fill(f):
    if f.kind in (ROOM, BALCONY):
        return FILL.get(f.meta.get("category", "habitable"), FILL["habitable"])
    return FILL.get(f.kind, (240, 240, 240))


def draw_plate(cv: Canvas, dp, labels=True, flat_labels=True, small=False):
    plate = dp.plate
    for fid, f in plate.faces.items():
        cv.poly(f.poly, fill=face_fill(f))
    for led in plate.ledges:
        cv.poly(led.poly, fill=(225, 225, 225), outline=(150, 150, 150))
    runs = {r.idx: r for r in dp.runs}
    for r in dp.runs:
        if r.wall == "RAILING":
            cv.line([r.p0, r.p1], (120, 140, 120), max(1, int(cv.s * 0.06)))
        elif r.wall:
            cv.poly(wall_footprint(r), fill=WALL.get(r.wall, (60, 60, 60)))
    for o in dp.openings:
        r = runs[o.run]
        u = np.asarray(r.u)
        n = np.array([-u[1], u[0]])
        p0 = np.asarray(r.p0)
        a, b = p0 + u * o.s0, p0 + u * o.s1
        h = r.t / 2 + 0.01
        gap = __import__("shapely").Polygon([a - n * h, b - n * h, b + n * h, a + n * h])
        if o.kind == "window":
            cv.poly(gap, fill=(127, 179, 213))
            cv.line([a, b], (60, 120, 170), 1)
        else:
            cv.poly(gap, fill="white")
            w = o.s1 - o.s0
            if o.door_kind == "lift":
                cv.line([a, b], (150, 150, 150), max(2, int(cv.s * 0.05)))
                continue
            from estate.rules import DOOR_KINDS
            if DOOR_KINDS.get(o.door_kind, ("",))[0].startswith("SLIDING") or o.door_kind == "shutter":
                off = n * 0.04
                cv.line([a + off, a + (b - a) * 0.55 + off], (192, 57, 43), max(1, int(cv.s * 0.03)))
                cv.line([b - off, a + (b - a) * 0.45 - off], (192, 57, 43), max(1, int(cv.s * 0.03)))
                continue
            hinge = a if o.hinge == "start" else b
            closed_dir = (b - a) / w if o.hinge == "start" else (a - b) / w
            side = n * o.swing
            leaf_end = hinge + side * w
            cv.line([hinge, leaf_end], (192, 57, 43), max(1, int(cv.s * 0.03)))
            ang0 = math.atan2(closed_dir[1], closed_dir[0])
            ang1 = math.atan2(side[1], side[0])
            da = (ang1 - ang0 + math.pi) % (2 * math.pi) - math.pi
            arc = [hinge + w * np.array([math.cos(ang0 + da * t), math.sin(ang0 + da * t)]) for t in np.linspace(0, 1, 16)]
            cv.line(arc, (192, 57, 43), 1)
    if labels:
        fs = 9 if small else 11
        for fid, f in plate.faces.items():
            net = dp.spaces.get(fid)
            if net is None or net.is_empty or f.kind in (LIFT, STAIR):
                if f.kind in (LIFT, STAIR):
                    c = f.poly.representative_point()
                    cv.text((c.x, c.y), f.name.replace(" shaft", "").upper(), fs - 1, (90, 90, 90))
                continue
            c = net.representative_point()
            code = f.room if f.room else (f.name or fid)
            if f.kind in (CORRIDOR, LOBBY, VOID, DECK, AMENITY):
                cv.text((c.x, c.y), f"{f.name or fid}\n{net.area:.1f} m²", fs, (70, 80, 90))
            else:
                cv.text((c.x, c.y), f"{code}\n{net.area:.1f}", fs)
    if flat_labels:
        for k, (stack, fi) in enumerate(sorted(plate.flats.items())):
            env = fi.envelope
            cv.poly(env, outline=FLAT_EDGE[k % len(FLAT_EDGE)], width=2)
            c = env.centroid
            from shapely.ops import unary_union
            extra = [l.poly for l in plate.ledges if l.flat == stack] +                     [f.poly for f in plate.faces.values() if f.flat == stack and f.kind == BALCONY]
            b = unary_union([env] + extra).bounds
            from estate.flats.template import flat_ifa
            ifa = flat_ifa(fi, plate)
            label = f"{stack}  {fi.template}  ({fi.flat_type}, {ifa:.0f} m² IFA)"
            pos, anchor = _label_pos(plate, stack, env, b)
            cv.text(pos, label, 12 if not small else 10, FLAT_EDGE[k % len(FLAT_EDGE)], anchor=anchor)


def _label_pos(plate, stack, env, b, gap=0.9):
    """Beyond the flat's rear facade: the bounding-box side farthest from the common corridor / lobby, unless that
    lands on another flat or core, then the next best side; falls back to inside the envelope."""
    import shapely
    from shapely.ops import unary_union
    common = unary_union([f.poly for f in plate.faces.values() if f.kind in (CORRIDOR, LOBBY)])
    others = unary_union([f.poly for f in plate.faces.values() if f.flat != stack and f.kind not in (CORRIDOR, LOBBY)])
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    cands = [((cx, b[3] + gap), "mm"), ((cx, b[1] - gap), "mm"), ((b[0] - 0.4, cy), "rm"), ((b[2] + 0.4, cy), "lm")]

    def score(c):
        (x, y), anchor = c
        probe = shapely.Point(x - 2.0 if anchor == "rm" else x + 2.0 if anchor == "lm" else x, y)
        blocked = (not others.is_empty and others.buffer(0.6).contains(probe)) or env.contains(probe)
        return (blocked, -(common.distance(probe) if not common.is_empty else 0.0))
    best = min(cands, key=score)
    if score(best)[0]:
        rp = env.representative_point()
        return (rp.x, rp.y), "mm"
    return best


def render_plate(dp, path, title="", scale=40.0, labels=True, flat_labels=True, pad=2.5):
    from shapely.ops import unary_union
    allg = unary_union([f.poly for f in dp.plate.faces.values()] + [l.poly for l in dp.plate.ledges])
    x0, y0, x1, y1 = allg.bounds
    side = max(pad, 9.0) if flat_labels and dp.plate.flats else pad     # room for labels beside the flats
    cv = Canvas((x0 - side, y0 - pad, x1 + side, y1 + pad), scale)
    draw_plate(cv, dp, labels, flat_labels, small=scale < 30)
    if title:
        cv.d.text((cv.m, 12), title, fill=(30, 30, 30), font=font(16))
    # 5 m scale bar
    sb0 = (x1 - 5.0, y0 - pad + 0.6)
    cv.line([sb0, (sb0[0] + 5.0, sb0[1])], (0, 0, 0), 3)
    cv.text((sb0[0] + 2.5, sb0[1] + 0.45), "5 m", 11)
    cv.save(path)
    return path
