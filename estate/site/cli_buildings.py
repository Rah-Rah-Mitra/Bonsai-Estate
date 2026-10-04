"""CLI for the non-residential buildings (MSCP 513, neighbourhood centre NC 514) and a plan drawing read back
from the written IFC.

The plan is drawn from the tessellated IFC (estate/export/meshcache.py), not from the generator's layout, so it
shows exactly what a viewer or the navigation check will see: elements of one storey cut at ``cut`` metres above
its floor, coloured by class, spaces outlined and labelled, in estate coordinates.

Register with ``estate.site.cli_buildings.register(sub)``; the command is ``nonres``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from estate import config, env

STYLE = {  # class: (paint order, fill, outline)
    "IfcSlab": (0, (226, 226, 222), None),
    "IfcGeographicElement": (1, (150, 196, 128), None),
    "IfcRampFlight": (3, (214, 200, 168), (130, 110, 80)),
    "IfcKerb": (4, (150, 150, 140), None),
    "IfcBuildingElementProxy": (4, (205, 205, 205), None),
    "IfcStairFlight": (5, (234, 223, 199), (130, 110, 80)),
    "IfcFurniture": (6, (176, 124, 72), None),
    "IfcColumn": (7, (85, 85, 85), None),
    "IfcWall": (8, (55, 55, 55), None),
    "IfcRailing": (9, (95, 120, 95), None),
    "IfcDoor": (10, (192, 57, 43), None),
    "IfcWindow": (10, (90, 150, 205), None),
    "IfcTransportElement": (11, (150, 162, 175), None),
}
SPACE_FILL = {"lot": (250, 245, 222), "moto": (242, 232, 214), "aisle": (232, 238, 244), "other": (232, 226, 244),
              "external": (224, 238, 220)}


def _space_kind(el):
    pre = getattr(el, "PredefinedType", None)
    long_name = (el.LongName or "").lower()
    if "motorcycle lot" in long_name:
        return "moto"
    if pre == "PARKING":
        return "aisle" if ("aisle" in long_name or "landing" in long_name or "entrance" in long_name) else "lot"
    if pre == "EXTERNAL":
        return "external"
    return "other"


def _outline(cv, geom, colour, width=1):
    """Ring outlines only (Canvas.poly paints holes white, which would cover what was drawn inside them)."""
    for part in getattr(geom, "geoms", [geom]):
        if part.geom_type != "Polygon" or part.is_empty:
            continue
        for ring in [part.exterior] + list(part.interiors):
            cv.line(list(ring.coords), colour, width)


def plan_png(ifc_path, storey: str, out_png, title: str | None = None, scale: float = 10.0, cut: float = 1.6,
             labels: bool = True, ramp_band: float = 1.5):
    """Top-down plan of one storey from the IFC itself (elements below the cut plane, painter-sorted by class)."""
    import ifcopenshell.util.element as uel
    from estate.draw.raster import Canvas, font
    from estate.export.meshcache import load

    f, meshes = load(ifc_path)
    st = next(s for s in f.by_type("IfcBuildingStorey") if s.Name == storey)
    items, spaces = [], []
    for guid, d in meshes.items():
        if not len(d["faces"]):
            continue
        v = d["verts"]
        zmin, zmax = float(v[:, 2].min()), float(v[:, 2].max())
        cls = d["cls"]
        if cls == "IfcSpace":
            if d["storey"] == storey:
                spaces.append((guid, d))
            continue
        if cls not in STYLE:
            continue
        in_storey = d["storey"] == storey
        items.append((guid, d, zmin, zmax, in_storey))
    import ifcopenshell.util.placement as upl
    z0 = float(upl.get_local_placement(st.ObjectPlacement)[2, 3])     # world z of the storey
    keep = []
    for guid, d, zmin, zmax, in_storey in items:
        if in_storey and zmin < z0 + cut:
            keep.append((guid, d))
        elif d["cls"] in ("IfcRampFlight", "IfcKerb") and zmax > z0 - ramp_band and zmin < z0 + cut:
            keep.append((guid, d))
        elif d["cls"] == "IfcSlab" and abs(zmax - z0) < 0.05:      # landings of stairs / ramps from below
            keep.append((guid, d))
    allxy = np.vstack([d["verts"][:, :2] for _, d in keep] + [d["verts"][:, :2] for _, d in spaces])
    x0, y0 = allxy.min(axis=0) - 3.0
    x1, y1 = allxy.max(axis=0) + 3.0
    cv = Canvas((x0, y0, x1, y1), scale, margin=50, top=40)

    def tris_of(d):
        v, fc = d["verts"], d["faces"]
        return v[fc][:, :, :2]

    def paint(items):
        for guid, d in items:
            order, fill, outline = STYLE[d["cls"]]
            for t in tris_of(d):
                cv.d.polygon([cv.P(p) for p in t], fill=fill)
            if outline is not None:
                _outline(cv, unary_union([Polygon(t) for t in tris_of(d) if Polygon(t).area > 1e-6]).buffer(0), outline)

    # slabs and terrain, then spaces (fill + outline), then the other elements on top, labels last
    ordered = sorted(keep, key=lambda gd: (STYLE[gd[1]["cls"]][0], gd[0]))
    paint([gd for gd in ordered if STYLE[gd[1]["cls"]][0] <= 1])
    space_polys = []
    for guid, d in spaces:
        el = f.by_guid(guid)
        poly = unary_union([Polygon(t) for t in tris_of(d) if Polygon(t).area > 1e-6]).buffer(0)
        kind = _space_kind(el)
        cv.poly(poly, fill=SPACE_FILL[kind])
        _outline(cv, poly, (150, 160, 190))
        space_polys.append((el, poly, kind))
    paint([gd for gd in ordered if STYLE[gd[1]["cls"]][0] > 1])
    for el, poly, kind in space_polys:
        if not labels or poly.is_empty:
            continue
        c = poly.representative_point()
        if kind in ("lot", "moto"):
            if scale >= 9:
                txt = el.Name.split("-")[-1]
                cv.text((c.x, c.y), txt, 7 if kind == "lot" else 6, (120, 110, 80))
        elif poly.area < 15.0:          # stalls, small rooms: number only
            cv.text((c.x, c.y), el.Name.split("-", 1)[-1] if el.Name.startswith("#") else el.Name, 8, (60, 70, 90))
        else:
            bx0, by0, bx1, by1 = poly.bounds
            label = el.LongName or el.Name
            if len(label) * 6.5 > min(bx1 - bx0, by1 - by0) * scale and el.Name.startswith("#"):
                label = el.Name                 # narrow rooms (shop units): unit number only
            cv.text((c.x, c.y), f"{label}\n{poly.area:.0f} m²", 10, (60, 70, 90))
    # estate grid ticks every 10 m
    for gx in range(int(np.ceil(x0 / 10)) * 10, int(x1) + 1, 10):
        cv.line([(gx, y0), (gx, y0 + 0.8)], (120, 120, 120), 1)
        cv.text((gx, y0 - 0.9), str(gx), 9, (110, 110, 110))
    for gy in range(int(np.ceil(y0 / 10)) * 10, int(y1) + 1, 10):
        cv.line([(x0, gy), (x0 + 0.8, gy)], (120, 120, 120), 1)
        cv.text((x0 - 1.6, gy), str(gy), 9, (110, 110, 110))
    sb = (x1 - 12.0, y0 + 1.2)
    cv.line([sb, (sb[0] + 10.0, sb[1])], (0, 0, 0), 3)
    cv.text((sb[0] + 5.0, sb[1] + 0.9), "10 m", 10)
    cv.line([(x1 - 1.5, y1 - 6.0), (x1 - 1.5, y1 - 2.0)], (0, 0, 0), 2)
    cv.text((x1 - 1.5, y1 - 1.2), "N", 12)
    cv.d.text((cv.m, 10), title or f"{Path(ifc_path).stem}  {storey} (z = {z0:.2f} m, cut +{cut} m)", fill=(30, 30, 30),
              font=font(16))
    cv.save(out_png)
    return out_png


def build(which: str, cfg: dict, schema: str, out_dir=None) -> dict:
    """Build one building ('mscp' or 'nc'); writes <id>.ifc and (from the builder) <id>.build.json under
    model/<id>/ or out_dir."""
    from estate.site.centre import build_nc_ifc
    from estate.site.mscp import build_mscp_ifc
    blk = str(cfg[which]["blk"])
    tid = config.target_id(which, blk)
    folder = Path(out_dir) if out_dir else env.MODEL / tid
    suffix = "" if schema == cfg["estate"].get("schema", "IFC4X3") else f"_{schema.lower()}"
    path = folder / f"{tid}{suffix}.ifc"
    fn = build_mscp_ifc if which == "mscp" else build_nc_ifc
    return fn(cfg, path, schema)


def validate(path, express_rules: bool = True) -> list[str]:
    import ifcopenshell
    import ifcopenshell.validate
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(ifcopenshell.open(str(path)), logger, express_rules=express_rules)
    return [f"{s.get('type', '')}: {s.get('message', '')}"[:300] for s in logger.statements]


def cmd_nonres(a):
    cfg = config.load()
    which = [a.only] if a.only else ["mscp", "nc"]
    bad = 0
    for w in which:
        t = time.time()
        info = build(w, cfg, a.schema, a.out_dir)
        n = info["counts"]["IfcElement"]
        print(f"{info['id']}: {env.rel(info['path'])} in {time.time() - t:.1f} s, {n} IfcElement"
              f"{'  <-- over budget' if n >= cfg['budget']['max_ifc_elements_per_file'] else ''}")
        bad += n >= cfg["budget"]["max_ifc_elements_per_file"]
        if w == "mscp":
            print(f"  car lots {info['car_lots']} ({', '.join(f'{k} {v}' for k, v in info['car_lots_per_deck'].items())}), "
                  f"motorcycle lots {info['motorcycle_lots']}, stair doors {len(info['stair_doors'])}")
        else:
            print(f"  stalls {info['stalls']}, seats {info['seats']}, shops {info['shops']}, "
                  f"forecourt {info['forecourt_m2']} m2, stair doors {len(info['stair_doors'])}")
        print(f"  entrances (estate): {json.dumps(info['entrances'])}")
        if a.validate:
            errs = validate(info["path"])
            print(f"  ifcopenshell.validate: {len(errs)} issue(s)")
            for e in errs[:10]:
                print(f"    {e}")
            bad += bool(errs)
        if a.draw:
            for storey in ("L1", "L2"):
                out = Path(info["path"]).parent / "plans" / f"{info['id']}_{storey}.png"
                plan_png(info["path"], storey, out, scale=10.0 if w == "mscp" else 9.0)
                print(f"  plan: {env.rel(out)}")
    return 1 if bad else 0


def register(sub):
    p = sub.add_parser("nonres", help="build the MSCP and the neighbourhood centre IFCs (model/MSCP_513, model/NC_514)")
    p.add_argument("--schema", default="IFC4X3", choices=["IFC4X3", "IFC4"])
    p.add_argument("--only", choices=["mscp", "nc"])
    p.add_argument("--out-dir", default=None, help="write here instead of model/<id>/")
    p.add_argument("--validate", action="store_true", help="run ifcopenshell.validate on the written files")
    p.add_argument("--draw", action="store_true", help="render L1 and L2 plan PNGs next to the IFC")
    p.set_defaults(fn=cmd_nonres)
