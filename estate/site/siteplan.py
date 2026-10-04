"""Site plan (reports/site_plan.png and reports/site_plan.svg): the estate drawn from the same layout SITE.ifc is
written from.

One drawing routine (draw) paints the plan through a backend with the same primitives: PngPlan rasterises it with
PIL, SvgPlan writes an editable vector drawing with svgwrite in the same pixel frame (viewBox = the PNG's size), so
the two never drift apart. About 2000 px wide for the 400 m estate (~0.2 m per pixel), with block numbers, road
names, a legend, a north arrow and a scale bar. Ground the site paves inside a footprint (an open forecourt, under
an overhang) shows as paving, and the pedestrian graph's legs through void decks and the centre are drawn dashed
across the buildings. Escape stairs that let out at ground level are marked with a green exit square at their
discharge and the path paved from it.

PNG: polygons are painted through per-polygon masks so holes stay transparent (ImageDraw alone cannot cut holes);
translucent layers (linkway canopies, tree crowns) are blended with an alpha mask.

SVG: one <g> per theme (LAYERS, also an Inkscape layer), drawn bottom to top so the groups stack exactly as the PNG
is painted; a polygon with holes or several parts is one <path> (fill-rule evenodd; footprints carry the site id),
RGBA alpha becomes fill-opacity, dashes a stroke-dasharray, text real <text> placed from PIL's own font metrics
(baselines, line spacing) with a white paint-order stroke for halos and a rotate transform for vertical names.
Coordinates are rounded to 0.01 px and nothing in the file depends on the run (no generated ids, no dates), so the
same layout gives the same bytes. One difference by design: tree crowns that overlap darken each other (each tree is
its own translucent circle, editable on its own; the PNG composites all crowns as one layer).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import Point

from estate.draw.raster import font
from estate.ifc.vegetation import ESTATE_SPECIES

C = {
    "bg": (246, 245, 240), "grass": (196, 222, 170), "green": (178, 214, 150), "verge": (166, 204, 138),
    "reserve": (205, 226, 186), "paving": (226, 220, 206), "sidewalk": (214, 210, 200), "path": (236, 226, 200),
    "park": (232, 222, 190), "carriage": (78, 80, 86), "drive": (120, 122, 128), "kerb": (236, 236, 232),
    "marking": (250, 250, 250), "ramp": (232, 160, 72), "rubber": (204, 92, 72), "court": (168, 166, 160),
    "block": (250, 250, 248), "block_edge": (40, 40, 44), "mscp": (232, 232, 240), "nc": (244, 236, 222),
    "linkway": (226, 112, 36), "roof": (226, 112, 36, 110), "porch": (120, 120, 140, 120), "shelter": (40, 96, 196),
    "tree": (52, 112, 52, 120), "tree_edge": (40, 90, 40), "text": (34, 34, 38), "foot": (150, 120, 70),
    "ground_route": (70, 70, 78), "exit": (30, 150, 70),
}
ROUTED = ("void_deck", "indoor")
# themes bottom to top, in the order draw() paints them (so the SVG's groups stack exactly as the PNG is painted):
# ground = every surface at grade (grass, the greens' lawns and park paths, the park connector, verges, sidewalks,
# bus stop platforms, footpaths, escape paths, aprons); roads = driveways, carriageways, kerbs, markings, zebras,
# kerb ramps; greens = the facilities on the greens (playgrounds, fitness corners, courts, pavilions) and their
# names; linkways = linkway paving and the footpath / escape path centrelines; footprints = buildings, the ground the
# site paves inside them and their outlines; graph = the pedestrian graph's legs through ground floors; canopies =
# porch and linkway roofs, bus shelters; trees; labels = block, road, green and bus stop names, entrance and stair
# exit markers, bus stop signs; legend = title block, legend, scale bar, north arrow
LAYERS = ("ground", "roads", "greens", "linkways", "footprints", "graph", "canopies", "trees", "labels", "legend")
FONT_FAMILY = "Arial, 'Segoe UI', 'DejaVu Sans', Helvetica, sans-serif"   # raster.font's search order, then sans
HALO = (255, 255, 255)
INKSCAPE = "http://www.inkscape.org/namespaces/inkscape"


class Plan:
    """Drawing backend interface: the estate-to-pixel frame and the primitives the site plan is made of.

    P maps estate metres to pixels (y flipped, s pixels per metre, a header above the estate for the title block and
    a footer below it for the scale bar). fill, line, dashed, text and circles take estate coordinates; ellipse,
    rect, polygon, pline, ptext and vtext take pixels of the same frame (markers, legend swatches, the north arrow).
    layer(name) sets the theme the next primitives belong to: the PNG ignores it, the SVG groups by it. Themes only
    go up (LAYERS order), so grouping by theme never changes what covers what."""

    def __init__(self, bounds, width=2000, margin=40, header=96, footer=44):
        x0, y0, x1, y1 = bounds
        self.x0, self.y1 = x0, y1
        self.s = (width - 2 * margin) / (x1 - x0)
        self.m, self.top = margin, header
        self.W, self.H = width, int(round((y1 - y0) * self.s)) + header + footer + margin
        self.theme = LAYERS[0]
        self._measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def P(self, p):
        return (self.m + (p[0] - self.x0) * self.s, self.top + (self.y1 - p[1]) * self.s)

    def layer(self, name):
        if name not in LAYERS:
            raise ValueError(f"unknown site plan layer {name!r}")
        if LAYERS.index(name) < LAYERS.index(self.theme):
            raise ValueError(f"site plan layer {name!r} after {self.theme!r}: themes are drawn bottom to top")
        self.theme = name

    def textlength(self, s, size):
        """Advance width in pixels as PIL lays the text out: the legend is spaced by it in both backends."""
        return self._measure.textlength(s, font=font(size))

    def dashed(self, a, b, color, width=1, dash=1.2, gap=0.8):
        """Dashed segment a-b (dash and gap in metres)."""
        L = math.dist(a, b)
        s = 0.0
        while s < L:
            e = min(L, s + dash)
            p = (a[0] + (b[0] - a[0]) * s / L, a[1] + (b[1] - a[1]) * s / L)
            q = (a[0] + (b[0] - a[0]) * e / L, a[1] + (b[1] - a[1]) * e / L)
            self.line([p, q], color, width)
            s = e + gap

    # estate coordinates
    def fill(self, geom, color, outline=None, width=1, name=None):
        """Polygons (with holes) of geom; color RGB or RGBA; name identifies the shape where a backend can."""
        raise NotImplementedError

    def line(self, pts, color, width=1):
        raise NotImplementedError

    def text(self, p, s, size=14, color=C["text"], anchor="mm", stroke=0):
        """Text at p, lines centred on each other; anchor as PIL's; stroke = halo width in pixels."""
        raise NotImplementedError

    def circles(self, pts, color, edge=None):
        """Translucent discs [(x, y, r)] (tree crowns)."""
        raise NotImplementedError

    # pixel coordinates
    def ellipse(self, box, fill, outline=None):
        raise NotImplementedError

    def rect(self, box, fill, outline=None, width=1):
        raise NotImplementedError

    def polygon(self, pts, fill):
        raise NotImplementedError

    def pline(self, pts, color, width=1):
        raise NotImplementedError

    def ptext(self, xy, s, size, color=C["text"], anchor="la"):
        raise NotImplementedError

    def vtext(self, xy, s, size, color, box):
        """One line of text centred on xy reading bottom to top; box = the length in pixels it is laid out in
        (the PNG draws it in a box x 24 px layer, rotated and pasted on whole pixels, and clips to it)."""
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


class PngPlan(Plan):
    """PIL raster backend (reports/site_plan.png)."""

    def __init__(self, bounds, width=2000, **kw):
        super().__init__(bounds, width, **kw)
        self.img = Image.new("RGB", (self.W, self.H), C["bg"])
        self.d = ImageDraw.Draw(self.img)

    def fill(self, geom, color, outline=None, width=1, name=None):
        if geom is None or geom.is_empty:
            return
        rgb = color[:3]
        alpha = color[3] if len(color) == 4 else 255
        for poly in getattr(geom, "geoms", [geom]):
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
            ext = [self.P(c) for c in poly.exterior.coords]
            xs, ys = [p[0] for p in ext], [p[1] for p in ext]
            bx0, by0 = int(math.floor(min(xs))) - 1, int(math.floor(min(ys))) - 1
            bx1, by1 = int(math.ceil(max(xs))) + 2, int(math.ceil(max(ys))) + 2
            w, h = max(1, bx1 - bx0), max(1, by1 - by0)
            mask = Image.new("L", (w, h), 0)
            md = ImageDraw.Draw(mask)
            md.polygon([(x - bx0, y - by0) for x, y in ext], fill=alpha)
            for hole in poly.interiors:
                md.polygon([(x - bx0, y - by0) for x, y in (self.P(c) for c in hole.coords)], fill=0)
            self.img.paste(rgb, (bx0, by0, bx0 + w, by0 + h), mask)
            if outline is not None:
                self.d.line(ext, fill=outline, width=width)
                for hole in poly.interiors:
                    self.d.line([self.P(c) for c in hole.coords], fill=outline, width=width)

    def line(self, pts, color, width=1):
        self.d.line([self.P(p) for p in pts], fill=color, width=width)

    def text(self, p, s, size=14, color=C["text"], anchor="mm", stroke=0):
        self.d.multiline_text(self.P(p), s, fill=color, font=font(size), anchor=anchor, align="center",
                              stroke_width=stroke, stroke_fill=HALO)

    def circles(self, pts, color, edge=None):
        layer = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        for x, y, r in pts:
            cx, cy = self.P((x, y))
            rr = r * self.s
            ld.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=color, outline=edge)
        self.img.paste(layer, (0, 0), layer)

    def ellipse(self, box, fill, outline=None):
        self.d.ellipse(box, fill=fill, outline=outline)

    def rect(self, box, fill, outline=None, width=1):
        self.d.rectangle(box, fill=fill, outline=outline, width=width)

    def polygon(self, pts, fill):
        self.d.polygon(pts, fill=fill)

    def pline(self, pts, color, width=1):
        self.d.line(pts, fill=color, width=width)

    def ptext(self, xy, s, size, color=C["text"], anchor="la"):
        self.d.text(xy, s, fill=color, font=font(size), anchor=anchor)

    def vtext(self, xy, s, size, color, box):
        lay = Image.new("RGBA", (box, 24), (0, 0, 0, 0))
        ImageDraw.Draw(lay).text((box / 2, 12), s, fill=color, font=font(size), anchor="mm")
        lay = lay.rotate(90, expand=True)
        x, y = xy
        self.img.paste(lay, (int(x - lay.width / 2), int(y - lay.height / 2)), lay)

    def save(self, path):
        self.img.save(path, "PNG")


def _f(v):
    """A coordinate or length rounded to 0.01 px, without trailing zeros ('-0' as '0')."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _paint(key, color):
    """SVG paint attributes for an RGB or RGBA colour: fill='#rrggbb' (+ fill-opacity when translucent)."""
    out = {key: "#%02x%02x%02x" % tuple(color[:3])}
    if len(color) == 4 and color[3] != 255:
        out[f"{key}_opacity"] = f"{color[3] / 255:.3f}".rstrip("0").rstrip(".")
    return out


def _stroke(color, width=1):
    return dict(_paint("stroke", color), stroke_width=_f(width)) if color is not None else {}


# newline before each drawing element and closing group, never inside a <text> (whitespace there is rendered)
_BREAKS = re.compile(r"><(?=(?:g|/g|defs|path|polyline|polygon|line|rect|circle|ellipse|text)[\s>/])")


class SvgPlan(Plan):
    """svgwrite vector backend (reports/site_plan.svg) in the PNG's pixel frame: one <g> per theme in LAYERS order,
    also marked as an Inkscape layer. Text positions come from PIL's metrics of the same font (raster.font), so the
    labels sit where the PNG puts them when the viewer has that font."""

    def __init__(self, bounds, width=2000, **kw):
        import svgwrite
        super().__init__(bounds, width, **kw)
        self.dwg = svgwrite.Drawing(size=(str(self.W), str(self.H)), viewBox=f"0 0 {self.W} {self.H}", debug=False)
        self.dwg["xmlns:inkscape"] = INKSCAPE
        self.groups = {k: self.dwg.g(id=k, **{"inkscape:groupmode": "layer", "inkscape:label": k}) for k in LAYERS}
        self.groups["ground"].add(self.dwg.rect(("0", "0"), (str(self.W), str(self.H)), **_paint("fill", C["bg"])))
        self._base = {}

    def _add(self, el):
        self.groups[self.theme].add(el)

    def _pts(self, pts):
        return [(_f(x), _f(y)) for x, y in pts]

    def _ring(self, coords):
        pts = [self.P(c) for c in coords]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        return ("M" + "L".join(f"{_f(x)} {_f(y)}" for x, y in pts) + "Z") if pts else ""

    def fill(self, geom, color, outline=None, width=1, name=None):
        if geom is None or geom.is_empty:
            return
        d = "".join(self._ring(r.coords) for poly in getattr(geom, "geoms", [geom])
                    if poly.geom_type == "Polygon" and not poly.is_empty for r in (poly.exterior, *poly.interiors))
        if not d:
            return
        attrs = dict(_paint("fill", color), fill_rule="evenodd", **_stroke(outline, width))
        if name:
            attrs["id"] = name
        self._add(self.dwg.path(d=d, **attrs))

    def line(self, pts, color, width=1):
        self.pline([self.P(p) for p in pts], color, width)

    def dashed(self, a, b, color, width=1, dash=1.2, gap=0.8):
        if math.dist(a, b) <= 0:
            return
        (x1, y1), (x2, y2) = self.P(a), self.P(b)
        self._add(self.dwg.line((_f(x1), _f(y1)), (_f(x2), _f(y2)), **_stroke(color, width),
                                stroke_dasharray=f"{_f(dash * self.s)} {_f(gap * self.s)}"))

    def circles(self, pts, color, edge=None):
        g = self.dwg.g(**_paint("fill", color), **_stroke(edge))
        for x, y, r in pts:
            cx, cy = self.P((x, y))
            g.add(self.dwg.circle((_f(cx), _f(cy)), _f(r * self.s)))
        self._add(g)

    def ellipse(self, box, fill, outline=None):
        x0, y0, x1, y1 = box
        c, rx, ry = (_f((x0 + x1) / 2), _f((y0 + y1) / 2)), _f((x1 - x0) / 2), _f((y1 - y0) / 2)
        el = self.dwg.circle(c, rx) if rx == ry else self.dwg.ellipse(c, (rx, ry))
        el.update(dict(_paint("fill", fill), **_stroke(outline)))
        self._add(el)

    def rect(self, box, fill, outline=None, width=1):
        x0, y0, x1, y1 = box
        self._add(self.dwg.rect((_f(x0), _f(y0)), (_f(x1 - x0), _f(y1 - y0)), **_paint("fill", fill),
                                **_stroke(outline, width)))

    def polygon(self, pts, fill):
        self._add(self.dwg.polygon(self._pts(pts), **_paint("fill", fill)))

    def pline(self, pts, color, width=1):
        self._add(self.dwg.polyline(self._pts(pts), fill="none", **_stroke(color, width)))

    def _baseline(self, size, v):
        """Pixels from a PIL vertical anchor ('a' ascender, 'm' middle, 's' baseline, 'd' descender) down to the
        baseline an SVG <text> is placed on."""
        k = (size, v)
        if k not in self._base:
            f = font(size)
            self._base[k] = f.getbbox("H", "L", anchor="l" + v)[1] - f.getbbox("H", "L", anchor="ls")[1]
        return self._base[k]

    def _text(self, xy, s, size, color, anchor, stroke=0, rotate=False):
        """<text> laid out as PIL lays it out: lines centred on each other, line spacing = 'A' height + stroke + 4,
        anchor as PIL's (h in l/m/r -> text-anchor start/middle/end, v -> baseline offset)."""
        h, v = anchor
        f = font(size)
        lines = s.split("\n")
        x, y = xy
        if len(lines) > 1:
            ls = f.getbbox("A", "L", stroke_width=stroke)[3] + stroke + 4
            y -= (len(lines) - 1) * ls * {"m": 0.5, "d": 1.0}.get(v, 0.0)
            ws = [self._measure.textlength(t, font=f) for t in lines]
            shift = {"l": 0.5, "m": 0.0, "r": -0.5}[h]
            pos = [(x + (max(ws) - w) * shift, y + i * ls) for i, w in enumerate(ws)]
        else:
            pos = [(x, y)]
        b = self._baseline(size, v)
        attrs = dict(font_family=FONT_FAMILY, font_size=_f(size), text_anchor={"l": "start", "m": "middle",
                                                                                  "r": "end"}[h], **_paint("fill", color))
        if stroke:
            attrs.update(_stroke(HALO, 2 * stroke), stroke_linejoin="round", paint_order="stroke")
        if rotate:
            attrs["transform"] = f"rotate(-90 {_f(x)} {_f(y)})"
        (x0, y0), rest = pos[0], pos[1:]
        el = self.dwg.text(lines[0] if not rest else "", insert=(_f(x0), _f(y0 + b)), **attrs)
        if rest:
            for t, (px, py) in zip(lines, pos):
                el.add(self.dwg.tspan(t, insert=(_f(px), _f(py + b))))
        self._add(el)

    def text(self, p, s, size=14, color=C["text"], anchor="mm", stroke=0):
        self._text(self.P(p), s, size, color, anchor, stroke)

    def ptext(self, xy, s, size, color=C["text"], anchor="la"):
        self._text(xy, s, size, color, anchor)

    def vtext(self, xy, s, size, color, box):
        x, y = xy                                    # the PNG's whole-pixel paste of its rotated 24 x box layer
        self._text((int(x - 12) + 12, int(y - box / 2) + box / 2), s, size, color, "mm", rotate=True)

    def save(self, path):
        for k in LAYERS:
            self.dwg.add(self.groups[k])
        xml = _BREAKS.sub(">\n<", self.dwg.tostring())
        Path(path).write_bytes(('<?xml version="1.0" encoding="utf-8"?>\n' + xml + "\n").encode("utf-8"))


def render(L, path, width=2000, fmt=None):
    """Draw the planned site layout L (builder.SiteLayout) to path: fmt 'png' or 'svg' (default: by its suffix,
    .svg for the vector plan, anything else a PNG)."""
    fmt = (fmt or ("svg" if Path(path).suffix.lower() == ".svg" else "png")).lower()
    if fmt not in ("png", "svg"):
        raise ValueError(f"site plan format {fmt!r}: png or svg")
    pl = (SvgPlan if fmt == "svg" else PngPlan)(L.mp["extent"].bounds, width)
    draw(pl, L)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pl.save(path)
    return path


def draw(pl: Plan, L):
    """The site plan, painted through backend pl in drawing order (later covers earlier in the PNG)."""
    mp, net, D = L.mp, L.net, L.draw
    ext = mp["extent"]
    pl.layer("ground")
    pl.fill(ext, C["grass"])
    for g in L.greens:
        pl.fill(g["rect"], C["green"])
    pl.fill(net.reserve, C["reserve"])
    for v in net.verges.values():
        pl.fill(v, C["verge"])
    for g in L.greens:
        pl.fill(D["green_paths"][L.greens.index(g)], C["park"])
    pl.fill(D["pcn"], C["park"])
    for s in D["sidewalks"].values():
        pl.fill(s, C["sidewalk"])
    for p in D["platforms"]:
        pl.fill(p, C["paving"])
    pl.fill(D["approaches"], C["sidewalk"])
    pl.fill(D["foot"], C["path"])
    pl.fill(D.get("escape"), C["path"])
    pl.fill(D["aprons"], C["paving"])
    pl.layer("roads")
    pl.fill(D["drives"], C["drive"])
    for co in net.crossovers:
        pl.fill(co["poly"], C["drive"])
    pl.fill(net.C, C["carriage"])
    for k in D["kerbs"].values():
        pl.fill(k, C["kerb"])
    from estate.site.builder import _centre_marks, _zebra
    for r in net.roads:
        for mk in _centre_marks(r, net):
            pl.fill(mk, C["marking"])
    for c in net.crossings:
        for bar in _zebra(c):
            pl.fill(bar, C["marking"])
    pl.fill(D["ramps"], C["ramp"])
    pl.layer("greens")
    for g in L.greens:
        for it in g["items"]:
            col = {"playground": C["rubber"], "fitness_corner": C["rubber"], "ball_court": C["court"],
                   "pavilion": C["paving"], "lawn": None}[it["kind"]]
            if col:
                pl.fill(it["rect"], col, outline=(90, 90, 90))
            if it["kind"] == "pavilion":
                pl.fill(it["rect"].buffer(0.6, join_style="mitre"), (110, 100, 96, 150))
            label = {"playground": "play", "fitness_corner": "fitness", "ball_court": "court", "pavilion": "pavilion",
                     "lawn": "lawn"}[it["kind"]]
            c = it["rect"].centroid
            pl.text((c.x, c.y), label, 13, (40, 40, 40))
    # linkway paths and footpath centrelines
    pl.layer("linkways")
    pl.fill(D["linkway_paths"], C["paving"])
    for ln in D["foot_lines"]:
        pl.line(list(ln.coords), C["foot"], 2)
    for ln in D.get("escape_lines", []):
        pl.line(list(ln.coords), C["exit"], 2)
    # buildings, the ground the site paves inside them, and the graph's legs through their ground floors
    pl.layer("footprints")
    for s, fp in D["footprints"]:
        col = C["block"] if s.kind == "block" else C["mscp"] if s.kind == "mscp" else C["nc"]
        pl.fill(fp, col, name=f"footprint-{s.id}")
    pl.fill(D.get("ground_paving"), C["paving"])
    for s, fp in D["footprints"]:
        for poly in getattr(fp, "geoms", [fp]):
            pl.line(list(poly.exterior.coords), C["block_edge"], 3)
    pl.layer("graph")
    if L.graph is not None:
        for u, v, d in sorted(L.graph.edges(data=True), key=lambda e: (min(e[0], e[1]), max(e[0], e[1]))):
            if d.get("via") in ROUTED:
                pl.dashed(u, v, C["ground_route"], 2)
    # canopies over everything on the ground
    pl.layer("canopies")
    pl.fill(D["porch_canopies"], C["porch"], outline=(80, 80, 100))
    pl.fill(D["linkway_roofs"], C["roof"], outline=C["linkway"], width=2)
    for b in net.bus_stops:
        pl.fill(b.shelter, C["shelter"])
    # trees
    pl.layer("trees")
    crowns = [(t["x"], t["y"], ESTATE_SPECIES[t["species"]][2]) for t in L.trees]
    pl.circles(crowns, C["tree"], C["tree_edge"])
    # labels and markers
    pl.layer("labels")
    for s, fp in D["footprints"]:
        c = fp.representative_point() if not fp.contains(fp.centroid) else fp.centroid
        if s.kind == "block":
            pl.text((c.x, c.y), f"{s.blk}\n{s.storeys} st", 22, stroke=0)
        else:
            pl.text((c.x, c.y), f"{'MSCP' if s.kind == 'mscp' else 'Neighbourhood centre'}\n{s.blk}", 20)
    for e in L.entrances:
        x, y = pl.P(e.E)
        pl.ellipse((x - 5, y - 5, x + 5, y + 5), (200, 30, 30), (255, 255, 255))
    for at, _ in D.get("discharges", []):
        x, y = pl.P(at)
        pl.rect((x - 5, y - 5, x + 5, y + 5), C["exit"], (255, 255, 255))
    for b in net.bus_stops:
        x, y = pl.P(b.node)
        pl.rect((x - 13, y - 13, x + 13, y + 13), (25, 70, 170), (255, 255, 255), width=2)
        pl.ptext((x, y), "B", 16, (255, 255, 255), anchor="mm")
        o = (b.node[0] + b.out[0] * 9, b.node[1] + b.out[1] * 9)
        pl.text(o, b.id, 15, (25, 70, 170), stroke=2)
    for r in net.roads:
        mid = r.pt(r.length * 0.62, 0.0)
        if not ext.buffer(-5).contains(Point(mid)):
            mid = r.pt(r.length * 0.5, 0.0)
        txt = r.name
        if r.axis == 1:
            pl.text(tuple(mid), txt, 15, (250, 250, 250))
        else:
            pl.vtext(pl.P(tuple(mid)), txt, 15, (250, 250, 250), int(len(txt) * 9) + 20)   # vertical road
    for g in L.greens:
        x0, y0, x1, y1 = g["rect"].bounds
        pl.text(((x0 + x1) / 2, y1 - 1.6), g["name"], 15, (30, 80, 30), stroke=2)
    if L.pcn:
        x0, y0, x1, y1 = L.pcn["rect"].bounds
        pl.vtext(pl.P(((x0 + x1) / 2 + 6.5, (y0 + y1) / 2)), L.pcn["name"], 15, (30, 80, 30), 200)
    pl.layer("legend")
    _furniture(pl, L)


def _furniture(pl, L):
    """Title block, legend (top right), scale bar and north arrow (bottom left), in pixels."""
    W = pl.W
    cfg = L.mp["cfg"]
    st = L.graph_stats
    title = f"{cfg['estate']['name']} - site plan"
    pl.ptext((pl.m, 14), title, 30)
    sub = (f"{len(L.sites)} buildings, {len(L.net.crossings)} crossings, {len(L.trees)} trees, "
           f"linkways {st.get('length_by_kind', {}).get('linkway', 0):.0f} m covered, "
           f"footpaths {st.get('length_by_kind', {}).get('footpath', 0):.0f} m")
    pl.ptext((pl.m, 54), sub, 17, (80, 80, 80))
    # legend (top right)
    items = [("carriageway", C["carriage"]), ("driveway", C["drive"]), ("sidewalk / paving", C["sidewalk"]),
             ("footpath", C["path"]), ("covered linkway", C["linkway"]), ("kerb ramp", C["ramp"]),
             ("playground / fitness", C["rubber"]), ("green", C["green"]), ("bus shelter", C["shelter"])]
    x = W - pl.m
    xs = []
    for label, col in reversed(items):
        tw = pl.textlength(label, 14)
        x -= tw + 34
        xs.append((x, label, col))
    for x, label, col in xs:
        pl.rect((x, 20, x + 20, 36), col[:3], (60, 60, 60))
        pl.ptext((x + 26, 28), label, 14, anchor="lm")
    for k in range(4):
        pl.pline(((W - pl.m - 520 + k * 9, 61), (W - pl.m - 514 + k * 9, 61)), C["ground_route"], 2)
    pl.ptext((W - pl.m - 482, 61), "walk through ground floor", 14, anchor="lm")
    pl.rect((W - pl.m - 640, 56, W - pl.m - 630, 66), C["exit"])
    pl.ptext((W - pl.m - 624, 61), "stair exit", 14, anchor="lm")
    pl.ellipse((W - pl.m - 290, 56, W - pl.m - 280, 66), (200, 30, 30))
    pl.ptext((W - pl.m - 274, 61), "entrance", 14, anchor="lm")
    pl.ellipse((W - pl.m - 190, 52, W - pl.m - 172, 70), (52, 112, 52), (40, 90, 40))
    pl.ptext((W - pl.m - 166, 61), "tree (crown)", 14, anchor="lm")
    # scale bar and north arrow (bottom left, in the footer)
    y = pl.H - pl.m - 10
    x0 = pl.m
    for k in range(5):
        a, b = x0 + k * 20 * pl.s, x0 + (k + 1) * 20 * pl.s
        pl.rect((a, y - 8, b, y), (30, 30, 30) if k % 2 == 0 else (255, 255, 255), (30, 30, 30))
    for k in (0, 50, 100):
        pl.ptext((x0 + k * pl.s, y + 12), f"{k} m", 14, anchor="mm")
    nx_, ny_ = x0 + 100 * pl.s + 70, y - 4
    pl.polygon([(nx_, ny_ - 30), (nx_ - 11, ny_ + 6), (nx_, ny_ - 2), (nx_ + 11, ny_ + 6)], (30, 30, 30))
    pl.ptext((nx_ + 24, ny_ - 12), "N", 20, anchor="mm")
    pl.ptext((nx_ + 60, y - 4), "Estate grid: metres from the south-west corner; EPSG:3414 map conversion in the IFC.",
             14, (90, 90, 90), anchor="lm")
