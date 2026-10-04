"""Flat templates: grid-based room layouts (config/flats/*.toml), seeded variants and placement into plates.

Template frame: x runs along the access facade (0 = the side next to the core / party wall), y runs from the
access side (y = 0, facing the lift lobby or common corridor, where the main door is) to the rear facade.
Rooms are unions of axis-aligned rectangles whose edges lie on the template grid, so the rooms tile the
envelope by construction and the stretch solver can move grid lines without breaking adjacency.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import box
from shapely.ops import unary_union

from estate.blocks.plate import BALCONY, OUT, ROOM, DoorSpec, Face, FlatInfo, Ledge, Plate, WindowSpec

ENTRY = "@entry"

# code: (long name, category). Categories drive wall types, checks and engine tags.
ROOM_CODES = {
    "LD": ("Living / Dining", "habitable"),
    "LDK": ("Living / Dining / Kitchen", "habitable"),
    "K": ("Kitchen", "wet"),
    "SY": ("Service Yard", "service"),
    "F": ("Foyer", "circulation"),
    "H": ("Hall", "circulation"),
    "HS": ("Household Shelter", "shelter"),
    "BED": ("Bedroom", "habitable"),
    "MB": ("Master Bedroom", "habitable"),
    "MBa": ("Master Bathroom", "wet"),
    "B2": ("Bedroom 2", "habitable"),
    "B3": ("Bedroom 3", "habitable"),
    "B4": ("Bedroom 4", "habitable"),
    "B4a": ("Bedroom 4 Bathroom", "wet"),
    "B2a": ("Bedroom 2 Bathroom", "wet"),
    "BA": ("Bathroom", "wet"),
    "CB": ("Common Bathroom", "wet"),
    "ST": ("Study", "habitable"),
    "UT": ("Utility / Store", "service"),
    "BAL": ("Balcony", "outdoor"),
}
FLAT_TYPES = {
    "2RF": "2-room Flexi flat", "3R": "3-room flat", "4R": "4-room flat", "5R": "5-room flat",
    "3G": "3Gen flat", "EA": "Executive apartment",
}


def category(code: str) -> str:
    return ROOM_CODES[code][1]


# ----------------------------------------------------------------------------- local (unplaced) flat
@dataclass
class LocalFlat:
    template: str
    flat_type: str
    rooms: dict                      # code -> Polygon (local)
    doors: list                      # dicts: a, b, at, width, kind, into, hinge (point), name
    windows: list                    # dicts: room, at, width, sill, height, name, kind, required
    ledges: list                     # dicts: poly, rail [(p, q)], name
    open: list                       # [(a, b)]
    variant: dict
    width: float
    depth: float
    shelter_wall: str = "HS250"

    @property
    def envelope(self):
        return unary_union([p for c, p in self.rooms.items() if category(c) != "outdoor"])

    def topology(self, canonical=True) -> str:
        """What the layout *is*, independent of dimensions and handing: the rooms, which rooms connect (by a door
        or an opening; an open kitchen and a closed one count as the same plan) and on which facade each room has
        its windows (F = access side, R = rear, A/B = the two flanks; a mirror swaps A and B)."""
        W, D = self.width, self.depth

        def facade(pt, mirror):
            x, y = pt
            d = {"F": abs(y), "R": abs(y - D), "A": abs(x), "B": abs(x - W)}
            s = min(d, key=d.get)
            return {"A": "B", "B": "A"}.get(s, s) if mirror else s

        def key(mirror):
            rooms = sorted(self.rooms)
            edges = sorted({tuple(sorted((d["a"], d["b"]))) for d in self.doors} |
                           {tuple(sorted(p)) for p in self.open})
            wins = sorted({(w["room"], facade(w["at"], mirror)) for w in self.windows})
            return json.dumps([self.flat_type, rooms, edges, wins])
        k = key(False)
        if canonical:
            k = min(k, key(True))
        return hashlib.sha256(k.encode()).hexdigest()[:12]

    def signature(self, canonical=False) -> str:
        def key(mirror):
            def X(x):
                return round(self.width - x if mirror else x, 2)

            def pt(p):
                return (X(p[0]), round(p[1], 2))

            def poly(pl):
                b = pl.bounds
                return (min(X(b[0]), X(b[2])), round(b[1], 2), max(X(b[0]), X(b[2])), round(b[3], 2), round(pl.area, 2))
            rooms = sorted((c, poly(p)) for c, p in self.rooms.items())
            doors = sorted((d["a"], d["b"], pt(d["at"]), round(d["width"], 2)) for d in self.doors)
            wins = sorted((w["room"], pt(w["at"]), round(w["width"], 2)) for w in self.windows)
            ledges = sorted(poly(l["poly"]) for l in self.ledges)
            return json.dumps([self.flat_type, rooms, doors, wins, ledges, sorted(map(sorted, self.open))])
        k = key(False)
        if canonical:
            k = min(k, key(True))
        return hashlib.sha256(k.encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------- template
@dataclass
class Template:
    id: str
    flat_type: str
    title: str
    fits: list
    xs: list
    ys: list
    rooms: dict                      # code -> [[x0, y0, x1, y1], ...]
    open: list
    doors: list
    windows: list
    ledges: list
    variants: dict
    shelter_wall: str = "HS250"
    source: str = ""
    notes: str = ""
    known_issues: list = field(default_factory=list)   # issue codes accepted for this template (legacy reference)
    estate: bool = True                                 # False: reference only, never placed by the masterplan

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: Path) -> "Template":
        d = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        rooms = dict(d["rooms"])
        open_ = [tuple(p) for p in rooms.pop("open", [])] + [tuple(p) for p in d.get("open", [])]
        t = cls(id=d["id"], flat_type=d["type"], title=d.get("title", d["id"]), fits=d.get("fits", []),
                xs=[float(v) for v in d["x"]], ys=[float(v) for v in d["y"]], rooms=rooms, open=open_,
                doors=d.get("doors", []), windows=d.get("windows", []), ledges=d.get("ledges", []),
                variants=d.get("variants", {}), shelter_wall=d.get("shelter_wall", "HS250"), source=str(path),
                notes=d.get("notes", ""), known_issues=list(d.get("known_issues", [])), estate=bool(d.get("estate", True)))
        t.validate_grid()
        return t

    @property
    def width(self):
        return self.xs[-1] - self.xs[0]

    @property
    def depth(self):
        return self.ys[-1] - self.ys[0]

    def validate_grid(self):
        gx, gy = set(self.xs), set(self.ys)
        for code, rects in self.rooms.items():
            if code not in ROOM_CODES:
                raise ValueError(f"{self.id}: unknown room code {code}")
            for r in rects:
                x0, y0, x1, y1 = r
                outdoor = category(code) == "outdoor"
                if not outdoor and not ({x0, x1} <= gx and {y0, y1} <= gy):
                    raise ValueError(f"{self.id}: room {code} rect {r} is not on the grid")

    # ------------------------------------------------------------------ variants
    def variant_space(self) -> dict:
        v = self.variants
        return {"mirror": [False, True] if v.get("mirror", False) else [False],
                "kitchen": list(v.get("kitchen", ["closed"])),
                "outdoor": list(v.get("outdoor", ["ac_ledge" if self.ledges else "none"]))}

    def pick_variant(self, seed: int, pins: dict | None = None) -> dict:
        rng = random.Random(seed)
        out = {}
        for k, opts in self.variant_space().items():
            out[k] = rng.choice(opts)
        for k, val in (pins or {}).items():
            if k in ("width", "depth"):
                out[k] = val
            elif k in out:
                if val not in self.variant_space()[k]:
                    raise ValueError(f"{self.id}: variant {k}={val} not allowed {self.variant_space()[k]}")
                out[k] = val
        return out

    def _stretch(self, axis: str, target: float | None):
        lines = self.xs if axis == "x" else self.ys
        if target is None or abs(target - (lines[-1] - lines[0])) < 1e-6:
            return list(lines)
        lims = self.variants.get("stretch", {}).get(axis)
        if not lims:
            raise ValueError(f"{self.id}: no stretch limits for {axis}, cannot reach {target}")
        sizes = np.diff(lines)
        lo = np.array([l[0] for l in lims], float)
        hi = np.array([l[1] for l in lims], float)
        if len(lo) != len(sizes):
            raise ValueError(f"{self.id}: stretch.{axis} needs {len(sizes)} [min, max] pairs")
        delta = target - sizes.sum()
        new = sizes.copy()
        for _ in range(len(sizes) + 1):   # water-fill proportionally to the remaining capacity
            cap = (hi - new) if delta > 0 else (new - lo)
            free = cap > 1e-9
            if abs(delta) < 1e-9 or not free.any():
                break
            share = cap[free] / cap[free].sum() * delta
            share = np.clip(share, -cap[free], cap[free]) if delta < 0 else np.minimum(share, cap[free])
            new[free] += share
            delta = target - new.sum()
        if abs(delta) > 1e-6:
            raise ValueError(f"{self.id}: cannot stretch {axis} to {target:.3f} (limits {lo.sum():.2f}-{hi.sum():.2f})")
        new = np.round(new, 3)
        new[-1] += round(target - new.sum(), 3)
        return [round(lines[0] + float(v), 4) for v in np.concatenate([[0.0], np.cumsum(new)])]

    def realize(self, variant: dict) -> LocalFlat:
        variant = dict(variant)
        if self.variants.get("mechanical_vent"):
            variant.setdefault("mechanical_vent", list(self.variants["mechanical_vent"]))
        xs2 = self._stretch("x", variant.get("width"))
        ys2 = self._stretch("y", variant.get("depth"))

        def mapc(v, src, dst):
            if v <= src[0]:
                return dst[0] + (v - src[0])
            if v >= src[-1]:
                return dst[-1] + (v - src[-1])
            i = max(0, np.searchsorted(src, v, side="right") - 1)
            i = min(i, len(src) - 2)
            f = (v - src[i]) / (src[i + 1] - src[i])
            return dst[i] + f * (dst[i + 1] - dst[i])

        def P(p):
            return (round(mapc(p[0], self.xs, xs2), 4), round(mapc(p[1], self.ys, ys2), 4))

        def R(r):
            a, b = P((r[0], r[1])), P((r[2], r[3]))
            return box(a[0], a[1], b[0], b[1])

        rooms = {c: unary_union([R(r) for r in rects]) for c, rects in self.rooms.items()}
        doors = [dict(d) for d in self.doors]
        windows = [dict(w) for w in self.windows]
        ledges = [dict(l) for l in self.ledges]
        open_ = [tuple(p) for p in self.open]

        if variant.get("kitchen") == "open":
            ko = self.variants.get("kitchen_open", {})
            drop = set(ko.get("remove_doors", []))
            doors = [d for d in doors if d.get("name") not in drop]
            open_ += [tuple(p) for p in ko.get("open", [])]
        outdoor = variant.get("outdoor", "none")
        if outdoor != "ac_ledge":
            ledges = []
        if outdoor == "balcony":
            bal = self.variants["balcony"]
            rooms["BAL"] = R(bal["rect"])
            dd = dict(bal["door"])
            dd.setdefault("between", [dd.get("from", "LD"), "BAL"])
            dd.setdefault("name", "Balcony door")
            doors.append(dd)
            drop = set(bal.get("replace_windows", []))
            windows = [w for w in windows if w.get("name") not in drop]
            ledges = [dict(l) for l in bal.get("ledges", [])]

        out_doors = []
        for d in doors:
            a, b = d["between"]
            at = P(d["at"])
            w = float(d["width"])
            horiz = self._edge_is_horizontal(rooms, a, b, at)
            off = np.array([w / 2, 0.0]) if horiz else np.array([0.0, w / 2])
            hinge = np.array(at) - off if d.get("hinge", "lo") == "lo" else np.array(at) + off
            out_doors.append(dict(a=a, b=b, at=at, width=w, kind=d.get("kind", "internal"), into=d.get("into", a),
                                  hinge=tuple(hinge), name=d.get("name") or f"{ROOM_CODES.get(a, (a,))[0]} door"))
        out_windows = [dict(room=w["room"], at=P(w["at"]), width=float(w["width"]), sill=float(w.get("sill", 1.0)),
                            height=float(w.get("height", 1.3)), name=w.get("name") or f"{ROOM_CODES[w['room']][0]} window",
                            kind=w.get("kind", "window"), required=bool(w.get("required", False))) for w in windows]
        out_ledges = []
        for l in ledges:
            poly = R(l["rect"])
            out_ledges.append(dict(poly=poly, name=l.get("name", "AC ledge"), rail=l.get("rail", "outer")))
        lf = LocalFlat(template=self.id, flat_type=self.flat_type, rooms=rooms, doors=out_doors, windows=out_windows,
                       ledges=out_ledges, open=open_, variant=dict(variant), width=xs2[-1] - xs2[0],
                       depth=ys2[-1] - ys2[0], shelter_wall=self.shelter_wall)
        if variant.get("mirror"):
            lf = mirror_x(lf)
        return lf

    @staticmethod
    def _edge_is_horizontal(rooms, a, b, at):
        """Orientation of the shared edge at a door point (ENTRY / outdoor: from the room boundary)."""
        pa = rooms.get(a)
        pb = rooms.get(b) if b in rooms else None
        if pb is not None and pa is not None:
            line = pa.boundary.intersection(pb.boundary)
        else:
            line = (pa if pa is not None else pb).boundary
        segs = []
        for g in getattr(line, "geoms", [line]):
            if g.geom_type in ("LineString", "LinearRing"):
                c = list(g.coords)
                segs += list(zip(c[:-1], c[1:]))
        for p, q in segs:
            ls = shapely.LineString([p, q])
            if ls.distance(shapely.Point(at)) < 1e-3:
                return abs(p[1] - q[1]) < 1e-9
        raise ValueError(f"door {a}-{b} at {at} is not on their shared edge")


def mirror_x(lf: LocalFlat) -> LocalFlat:
    W = lf.width
    A = [-1, 0, 0, 1, W, 0]
    out = copy.copy(lf)
    out.rooms = {c: shapely.affinity.affine_transform(p, A) for c, p in lf.rooms.items()}

    def P(p):
        return (round(W - p[0], 4), p[1])
    out.doors = [dict(d, at=P(d["at"]), hinge=P(d["hinge"])) for d in lf.doors]
    out.windows = [dict(w, at=P(w["at"])) for w in lf.windows]
    out.ledges = [dict(l, poly=shapely.affinity.affine_transform(l["poly"], A)) for l in lf.ledges]
    return out


# ----------------------------------------------------------------------------- placement
class Affine2:
    """2D map p -> A p + t with A in {0, +-1} (rotations by 90 degrees and mirrors)."""

    def __init__(self, a=1, b=0, d=0, e=1, xoff=0.0, yoff=0.0):
        self.A = np.array([[a, b], [d, e]], float)
        self.t = np.array([xoff, yoff], float)

    def __call__(self, p):
        q = self.A @ np.asarray(p, float) + self.t
        return (round(float(q[0]), 4), round(float(q[1]), 4))

    def poly(self, g):
        (a, b), (d, e) = self.A
        return shapely.affinity.affine_transform(g, [a, b, d, e, self.t[0], self.t[1]])

    @property
    def det(self):
        return float(np.linalg.det(self.A))


def flat_ifa(fi, plate) -> float:
    """Internal floor area: the envelope inside the half-thickness of its external / party walls, plus AC ledges."""
    return fi.envelope.buffer(-0.1, join_style="mitre").area + sum(l.poly.area for l in plate.ledges if l.flat == fi.stack)


def place_flat(plate: Plate, lf: LocalFlat, stack: str, xf: Affine2, entry_face: str, long_name=None) -> FlatInfo:
    """Add a realized flat to a plate. Face ids are 'U<stack>:<room code>'."""
    def fid(code):
        if code == ENTRY:
            return entry_face
        if code == "@out":
            return OUT
        return f"U{stack}:{code}"
    for code, poly in lf.rooms.items():
        kind = BALCONY if category(code) == "outdoor" else ROOM
        plate.add(Face(fid(code), xf.poly(poly), kind, flat=stack, room=code, name=ROOM_CODES[code][0],
                       meta={"category": category(code), "shelter_wall": lf.shelter_wall}))
    for a, b in lf.open:
        plate.open(fid(a), fid(b))
    for d in lf.doors:
        plate.doors.append(DoorSpec(fid(d["a"]), fid(d["b"]), xf(d["at"]), d["width"], d["kind"], fid(d["into"]),
                                    xf(d["hinge"]), d["name"]))
    for w in lf.windows:
        plate.windows.append(WindowSpec(fid(w["room"]), xf(w["at"]), w["width"], w["sill"], w["height"], w["name"],
                                        w["kind"], w["required"]))
    env = xf.poly(lf.envelope)
    for l in lf.ledges:
        lp = xf.poly(l["poly"])
        rails = []
        if l.get("rail", "outer") in ("outer", "glazed"):
            # one U-shaped 50 mm railing band on every ledge edge that does not touch the flat envelope
            band = lp.difference(lp.buffer(-0.05, join_style="mitre")).difference(env.buffer(0.06, join_style="mitre"))
            rails = [g for g in getattr(band, "geoms", [band]) if g.area > 1e-4]
        kind = "bay" if l.get("rail") == "glazed" else "ac"
        plate.ledges.append(Ledge(lp, stack, l["name"], rails, kind))
    info = FlatInfo(stack=stack, template=lf.template, flat_type=lf.flat_type,
                    long_name=long_name or FLAT_TYPES[lf.flat_type], variant=lf.variant,
                    signature=lf.signature(), envelope=env, gross_area=round(env.area, 2))
    info.variant = dict(lf.variant, canonical=lf.signature(canonical=True), topology=lf.topology())
    plate.flats[stack] = info
    return info


# ----------------------------------------------------------------------------- catalogue
_CATALOGUES: dict = {}


def catalogue(folder: Path | None = None, reload: bool = False) -> dict:
    """Templates in a folder (default config/flats), cached per folder so fixtures never replace the estate set."""
    from estate.env import CONFIG
    folder = Path(folder or (CONFIG / "flats")).resolve()
    if reload or folder not in _CATALOGUES:
        cat = {}
        for p in sorted(folder.glob("*.toml")):
            t = Template.load(p)
            cat[t.id] = t
        _CATALOGUES[folder] = cat
    return _CATALOGUES[folder]


def derive_seed(*parts) -> int:
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:15], 16)
