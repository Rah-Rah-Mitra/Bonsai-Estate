"""Site plan raster (reports/site_plan.png): the estate drawn from the same layout SITE.ifc is written from.

Polygons are painted through per-polygon masks so holes stay transparent (ImageDraw alone cannot cut holes);
translucent layers (linkway canopies, tree crowns) are blended with an alpha mask. About 2000 px wide for the
400 m estate (~0.2 m per pixel), with block numbers, road names, a legend, a north arrow and a scale bar.
Ground the site paves inside a footprint (an open forecourt, under an overhang) shows as paving, and the
pedestrian graph's legs through void decks and the centre are drawn dashed across the buildings. Escape stairs that
let out at ground level are marked with a green exit square at their discharge and the path paved from it.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

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


class Plan:
    def __init__(self, bounds, width=2000, margin=40, header=96, footer=44):
        x0, y0, x1, y1 = bounds
        self.x0, self.y1 = x0, y1
        self.s = (width - 2 * margin) / (x1 - x0)
        self.m, self.top = margin, header
        H = int(round((y1 - y0) * self.s)) + header + footer + margin
        self.img = Image.new("RGB", (width, H), C["bg"])
        self.d = ImageDraw.Draw(self.img)

    def P(self, p):
        return (self.m + (p[0] - self.x0) * self.s, self.top + (self.y1 - p[1]) * self.s)

    def fill(self, geom, color, outline=None, width=1):
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

    def dashed(self, a, b, color, width=1, dash=1.2, gap=0.8):
        """Dashed segment a-b (dash and gap in metres)."""
        L = math.dist(a, b)
        s = 0.0
        while s < L:
            e = min(L, s + dash)
            p = (a[0] + (b[0] - a[0]) * s / L, a[1] + (b[1] - a[1]) * s / L)
            q = (a[0] + (b[0] - a[0]) * e / L, a[1] + (b[1] - a[1]) * e / L)
            self.d.line([self.P(p), self.P(q)], fill=color, width=width)
            s = e + gap

    def text(self, p, s, size=14, color=C["text"], anchor="mm", stroke=0):
        self.d.multiline_text(self.P(p), s, fill=color, font=font(size), anchor=anchor, align="center",
                              stroke_width=stroke, stroke_fill=(255, 255, 255))

    def circles(self, pts, color, edge=None):
        layer = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        for x, y, r in pts:
            cx, cy = self.P((x, y))
            rr = r * self.s
            ld.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=color, outline=edge)
        self.img.paste(layer, (0, 0), layer)


def render(L, path, width=2000):
    """Draw the planned site layout L (builder.SiteLayout) to a PNG."""
    mp, net, D = L.mp, L.net, L.draw
    ext = mp["extent"]
    pl = Plan(ext.bounds, width)
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
    pl.fill(D["linkway_paths"], C["paving"])
    for ln in D["foot_lines"]:
        pl.line(list(ln.coords), C["foot"], 2)
    for ln in D.get("escape_lines", []):
        pl.line(list(ln.coords), C["exit"], 2)
    # buildings, the ground the site paves inside them, and the graph's legs through their ground floors
    for s, fp in D["footprints"]:
        col = C["block"] if s.kind == "block" else C["mscp"] if s.kind == "mscp" else C["nc"]
        pl.fill(fp, col)
    pl.fill(D.get("ground_paving"), C["paving"])
    for s, fp in D["footprints"]:
        for poly in getattr(fp, "geoms", [fp]):
            pl.line(list(poly.exterior.coords), C["block_edge"], 3)
    if L.graph is not None:
        for u, v, d in sorted(L.graph.edges(data=True), key=lambda e: (min(e[0], e[1]), max(e[0], e[1]))):
            if d.get("via") in ROUTED:
                pl.dashed(u, v, C["ground_route"], 2)
    # canopies over everything on the ground
    pl.fill(D["porch_canopies"], C["porch"], outline=(80, 80, 100))
    pl.fill(D["linkway_roofs"], C["roof"], outline=C["linkway"], width=2)
    for b in net.bus_stops:
        pl.fill(b.shelter, C["shelter"])
    # trees
    crowns = [(t["x"], t["y"], ESTATE_SPECIES[t["species"]][2]) for t in L.trees]
    pl.circles(crowns, C["tree"], C["tree_edge"])
    # labels
    for s, fp in D["footprints"]:
        c = fp.representative_point() if not fp.contains(fp.centroid) else fp.centroid
        if s.kind == "block":
            pl.text((c.x, c.y), f"{s.blk}\n{s.storeys} st", 22, stroke=0)
        else:
            pl.text((c.x, c.y), f"{'MSCP' if s.kind == 'mscp' else 'Neighbourhood centre'}\n{s.blk}", 20)
    for e in L.entrances:
        x, y = pl.P(e.E)
        pl.d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(200, 30, 30), outline=(255, 255, 255))
    for at, _ in D.get("discharges", []):
        x, y = pl.P(at)
        pl.d.rectangle((x - 5, y - 5, x + 5, y + 5), fill=C["exit"], outline=(255, 255, 255))
    for b in net.bus_stops:
        x, y = pl.P(b.node)
        pl.d.rectangle((x - 13, y - 13, x + 13, y + 13), fill=(25, 70, 170), outline=(255, 255, 255), width=2)
        pl.d.text((x, y), "B", fill=(255, 255, 255), font=font(16), anchor="mm")
        o = (b.node[0] + b.out[0] * 9, b.node[1] + b.out[1] * 9)
        pl.text(o, b.id, 15, (25, 70, 170), stroke=2)
    for r in net.roads:
        mid = r.pt(r.length * 0.62, 0.0)
        if not ext.buffer(-5).contains(__import__("shapely").Point(mid)):
            mid = r.pt(r.length * 0.5, 0.0)
        txt = r.name
        if r.axis == 1:
            pl.text(tuple(mid), txt, 15, (250, 250, 250))
        else:
            # vertical road: draw rotated text
            tw = int(len(txt) * 9) + 20
            lay = Image.new("RGBA", (tw, 24), (0, 0, 0, 0))
            ImageDraw.Draw(lay).text((tw / 2, 12), txt, fill=(250, 250, 250), font=font(15), anchor="mm")
            lay = lay.rotate(90, expand=True)
            x, y = pl.P(tuple(mid))
            pl.img.paste(lay, (int(x - lay.width / 2), int(y - lay.height / 2)), lay)
    for g in L.greens:
        x0, y0, x1, y1 = g["rect"].bounds
        pl.text(((x0 + x1) / 2, y1 - 1.6), g["name"], 15, (30, 80, 30), stroke=2)
    if L.pcn:
        x0, y0, x1, y1 = L.pcn["rect"].bounds
        lay = Image.new("RGBA", (200, 24), (0, 0, 0, 0))
        ImageDraw.Draw(lay).text((100, 12), L.pcn["name"], fill=(30, 80, 30), font=font(15), anchor="mm")
        lay = lay.rotate(90, expand=True)
        x, y = pl.P(((x0 + x1) / 2 + 6.5, (y0 + y1) / 2))
        pl.img.paste(lay, (int(x - lay.width / 2), int(y - lay.height / 2)), lay)
    _furniture(pl, L)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pl.img.save(path)
    return path


def _furniture(pl, L):
    d = pl.d
    W = pl.img.width
    cfg = L.mp["cfg"]
    st = L.graph_stats
    title = f"{cfg['estate']['name']} - site plan"
    d.text((pl.m, 14), title, fill=C["text"], font=font(30))
    sub = (f"{len(L.sites)} buildings, {len(L.net.crossings)} crossings, {len(L.trees)} trees, "
           f"linkways {st.get('length_by_kind', {}).get('linkway', 0):.0f} m covered, "
           f"footpaths {st.get('length_by_kind', {}).get('footpath', 0):.0f} m")
    d.text((pl.m, 54), sub, fill=(80, 80, 80), font=font(17))
    # legend (top right)
    items = [("carriageway", C["carriage"]), ("driveway", C["drive"]), ("sidewalk / paving", C["sidewalk"]),
             ("footpath", C["path"]), ("covered linkway", C["linkway"]), ("kerb ramp", C["ramp"]),
             ("playground / fitness", C["rubber"]), ("green", C["green"]), ("bus shelter", C["shelter"])]
    x = W - pl.m
    xs = []
    for label, col in reversed(items):
        tw = d.textlength(label, font=font(14))
        x -= tw + 34
        xs.append((x, label, col))
    for x, label, col in xs:
        d.rectangle((x, 20, x + 20, 36), fill=col[:3], outline=(60, 60, 60))
        d.text((x + 26, 28), label, fill=C["text"], font=font(14), anchor="lm")
    for k in range(4):
        d.line((W - pl.m - 520 + k * 9, 61, W - pl.m - 514 + k * 9, 61), fill=C["ground_route"], width=2)
    d.text((W - pl.m - 482, 61), "walk through ground floor", fill=C["text"], font=font(14), anchor="lm")
    d.rectangle((W - pl.m - 640, 56, W - pl.m - 630, 66), fill=C["exit"])
    d.text((W - pl.m - 624, 61), "stair exit", fill=C["text"], font=font(14), anchor="lm")
    d.ellipse((W - pl.m - 290, 56, W - pl.m - 280, 66), fill=(200, 30, 30))
    d.text((W - pl.m - 274, 61), "entrance", fill=C["text"], font=font(14), anchor="lm")
    d.ellipse((W - pl.m - 190, 52, W - pl.m - 172, 70), fill=(52, 112, 52), outline=(40, 90, 40))
    d.text((W - pl.m - 166, 61), "tree (crown)", fill=C["text"], font=font(14), anchor="lm")
    # scale bar and north arrow (bottom left, in the footer)
    y = pl.img.height - pl.m - 10
    x0 = pl.m
    for k in range(5):
        a, b = x0 + k * 20 * pl.s, x0 + (k + 1) * 20 * pl.s
        d.rectangle((a, y - 8, b, y), fill=(30, 30, 30) if k % 2 == 0 else (255, 255, 255), outline=(30, 30, 30))
    for k in (0, 50, 100):
        d.text((x0 + k * pl.s, y + 12), f"{k} m", fill=C["text"], font=font(14), anchor="mm")
    nx_, ny_ = x0 + 100 * pl.s + 70, y - 4
    d.polygon([(nx_, ny_ - 30), (nx_ - 11, ny_ + 6), (nx_, ny_ - 2), (nx_ + 11, ny_ + 6)], fill=(30, 30, 30))
    d.text((nx_ + 24, ny_ - 12), "N", fill=C["text"], font=font(20), anchor="mm")
    d.text((nx_ + 60, y - 4), "Estate grid: metres from the south-west corner; EPSG:3414 map conversion in the IFC.",
           fill=(90, 90, 90), font=font(14), anchor="lm")
