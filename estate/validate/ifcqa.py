"""IFC-level quality checks of one building file, and of the federation of building files.

The generator writes IfcOpenShell files that Bonsai must load unchanged, so this module checks what Bonsai and
downstream tools rely on rather than what the generator meant: schema validity (ifcopenshell.validate, optionally
with the EXPRESS rules), unique GlobalIds, spatial containment, the void/fill graph of openings, openings per
wall (Bonsai's void_limit), zones and property sets, flat areas against config/rules.toml, the element budget
(Bonsai's load guard), right-handed placements and mapped items (mirrored flats must be explicit geometry), the
georeference, and the IDS in ids/estate.ids. ``check_federation`` adds what only a set of files can show:
GlobalIds unique across files except the shared IfcProject / IfcSite, which must be identical, and one
IfcMapConversion / IfcProjectedCRS everywhere. The programme of the flats (rooms, areas, shelter, glazing, vents,
entrances) is checked from the same file by estate/validate/programme.py.

Every check reports {check, level, severity, count, examples}: ``severity`` is what a failure means (error / warn /
info), ``level`` is the effective level (``info`` when the count is zero).
"""
from __future__ import annotations

import time
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import ifcopenshell
import ifcopenshell.util.element as uel
import ifcopenshell.util.placement as upl

from estate import env

IDS_PATH = Path(__file__).resolve().parent / "ids" / "estate.ids"
FLAT_TYPES = ("2RF", "3R", "4R", "5R", "3G", "EA")
REQUIRED_PSETS = {          # class: property sets / quantity sets every instance must carry
    "IfcWall": ("Pset_WallCommon",),
    "IfcDoor": ("Pset_DoorCommon", "Navigation"),
    "IfcWindow": ("Pset_WindowCommon",),
    "IfcSpace": ("Qto_SpaceBaseQuantities",),
    "IfcZone": ("SampleCity_Flat",),
}
DEFAULT_BUDGET = {"max_ifc_elements_per_file": 25000, "max_openings_per_wall": 20}
FLAT_ALIASES = {        # normalised FlatType label (lower case, letters and digits only, no trailing 'flat') -> type
    "2rf": "2RF", "2room": "2RF", "2roomflexi": "2RF", "flexi": "2RF", "3r": "3R", "3room": "3R", "4r": "4R",
    "4room": "4R", "5r": "5R", "5room": "5R", "3g": "3G", "3gen": "3G", "3generation": "3G", "ea": "EA",
    "executive": "EA", "executiveapartment": "EA", "executiveflat": "EA"}


# ----------------------------------------------------------------------------- report
def _plain(x):
    if isinstance(x, ifcopenshell.entity_instance):
        return f"#{x.id()} {x.is_a()} {getattr(x, 'Name', '') or ''}".strip()
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


class Report:
    """Collects check results in the shared {check, level, severity, count, examples} format."""

    def __init__(self):
        self.results = []

    def add(self, check, severity, failures=(), *, count=None, value=None, note=None, limit=6):
        failures = list(failures)
        n = len(failures) if count is None else int(count)
        r = {"check": check, "level": severity if n else "info", "severity": severity, "count": n,
             "examples": [_plain(x) for x in failures[:limit]]}
        if value is not None:
            r["value"] = _plain(value)
        if note:
            r["note"] = note
        self.results.append(r)
        return r


def summarize(results) -> dict:
    return {"checks": len(results),
            "failed_errors": sum(1 for r in results if r["level"] == "error"),
            "failed_warnings": sum(1 for r in results if r["level"] == "warn"),
            "error_items": sum(r["count"] for r in results if r["level"] == "error"),
            "warning_items": sum(r["count"] for r in results if r["level"] == "warn")}


# ----------------------------------------------------------------------------- config
def rules_config() -> dict:
    try:
        return tomllib.loads((env.CONFIG / "rules.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def budget_config() -> dict:
    try:
        cfg = tomllib.loads((env.CONFIG / "estate.toml").read_text(encoding="utf-8"))
        return dict(DEFAULT_BUDGET, **cfg.get("budget", {}))
    except (OSError, tomllib.TOMLDecodeError):
        return dict(DEFAULT_BUDGET)


def expected_identity() -> dict | None:
    try:
        from estate import config
        return config.identity(config.load())
    except Exception:  # noqa: BLE001 - identity is optional for test files
        return None


# ----------------------------------------------------------------------------- helpers
def canonical_flat_type(label) -> str | None:
    """The HDB flat type of a SampleCity_Flat.FlatType label: '4R', and legacy labels such as '4-Room' or
    '4-room flat' (hdb_block.ifc), map to 4R; None when the label names no flat type."""
    if label in FLAT_TYPES:
        return label
    if not isinstance(label, str):
        return None
    key = "".join(ch for ch in label.lower() if ch.isalnum())
    return FLAT_ALIASES.get(key) or FLAT_ALIASES.get(key.removesuffix("flat"))


def flat_pset(zone) -> dict | None:
    """SampleCity_Flat of a flat zone; None for other zones, including a zone that carries the set without a FlatType
    or UnitNumber (e.g. common property written with the writer's default set name)."""
    p = uel.get_pset(zone, "SampleCity_Flat")
    return p if p and (p.get("FlatType") is not None or p.get("UnitNumber") is not None) else None


def pset_index(f) -> dict:
    """element id -> set of property / quantity set names (occurrence and type)."""
    out = defaultdict(set)
    for rel in f.by_type("IfcRelDefinesByProperties"):
        pd = rel.RelatingPropertyDefinition
        names = [p.Name for p in pd] if isinstance(pd, tuple) else [pd.Name]
        for o in rel.RelatedObjects:
            out[o.id()].update(names)
    for rel in f.by_type("IfcRelDefinesByType"):
        names = [p.Name for p in (rel.RelatingType.HasPropertySets or ())]
        for o in rel.RelatedObjects:
            out[o.id()].update(names)
    return out


def _is_contained(el, memo) -> bool:
    i = el.id()
    if i in memo:
        return memo[i]
    memo[i] = False                       # cycle guard
    ok = bool(getattr(el, "ContainedInStructure", None))
    if not ok:
        for rel in getattr(el, "Decomposes", ()) or ():
            whole = rel.RelatingObject
            if whole.is_a("IfcSpatialElement") or (whole.is_a("IfcElement") and _is_contained(whole, memo)):
                ok = True
        for rel in getattr(el, "Nests", ()) or ():
            if rel.RelatingObject.is_a("IfcElement") and _is_contained(rel.RelatingObject, memo):
                ok = True
    memo[i] = ok
    return ok


def _zone_of(space):
    for rel in getattr(space, "HasAssignments", ()) or ():
        if rel.is_a("IfcRelAssignsToGroup") and rel.RelatingGroup.is_a("IfcZone"):
            return rel.RelatingGroup
    return None


def _det_info(M):
    R = np.asarray(M, float)[:3, :3]
    det = float(np.linalg.det(R))
    ortho = float(np.abs(R.T @ R - np.eye(3)).max()) if np.isfinite(R).all() else float("inf")
    return det, ortho


def georef(f) -> dict | None:
    mc = f.by_type("IfcMapConversion")
    if not mc:
        return None
    m = mc[0]
    crs = m.TargetCRS
    return {"Eastings": m.Eastings, "Northings": m.Northings, "OrthogonalHeight": m.OrthogonalHeight,
            "XAxisAbscissa": m.XAxisAbscissa, "XAxisOrdinate": m.XAxisOrdinate, "Scale": m.Scale,
            "CRS": getattr(crs, "Name", None), "GeodeticDatum": getattr(crs, "GeodeticDatum", None),
            "MapProjection": getattr(crs, "MapProjection", None), "count": len(mc)}


def placement_issues(f) -> list:
    """Object placements that are not proper rotations (det +1, orthonormal) and mirrored mapped items."""
    bad = []
    with np.errstate(all="ignore"):
        for lp in f.by_type("IfcLocalPlacement"):
            try:
                det, ortho = _det_info(upl.get_local_placement(lp))
            except Exception as e:  # noqa: BLE001
                bad.append(f"#{lp.id()} IfcLocalPlacement: {e!r}")
                continue
            if not (abs(det - 1.0) <= 1e-6 and ortho <= 1e-6):     # also catches NaN from degenerate axes
                users = list(lp.PlacesObject or ())
                bad.append(f"#{lp.id()} det {det:+.4f} ortho {ortho:.1e} {_plain(users[0]) if users else ''}")
        for mi in f.by_type("IfcMappedItem"):
            if not mi.MappingTarget.is_a("IfcCartesianTransformationOperator3D"):
                continue
            try:
                det, _ = _det_info(upl.get_mappeditem_transformation(mi))
            except Exception as e:  # noqa: BLE001
                bad.append(f"#{mi.id()} IfcMappedItem: {e!r}")
                continue
            if not det > 0:
                bad.append(f"#{mi.id()} IfcMappedItem mirrored (det {det:+.3f})")
    return bad


def shared_representations(f) -> list:
    """IfcShapeRepresentations used by more (or fewer) than one product shape / representation map / shape aspect
    (the IfcShapeModel.WR11 rule, checked directly: the full EXPRESS rule run costs ~30 s on a large block)."""
    bad = []
    for r in f.by_type("IfcShapeRepresentation"):
        n = sum(len(getattr(r, a, None) or ()) for a in ("OfProductRepresentation", "RepresentationMap", "OfShapeAspect"))
        if n != 1:
            users = [s for pd in (r.OfProductRepresentation or ()) for s in (getattr(pd, "ShapeOfProduct", None) or ())]
            bad.append(f"#{r.id()} {r.RepresentationIdentifier} used {n}x" + (f" (e.g. {_plain(users[0])})" if users else ""))
    return bad


# ----------------------------------------------------------------------------- schema
def schema_errors(f, path=None, express_rules=False) -> list:
    import ifcopenshell.validate as validate
    lg = validate.json_logger()
    validate.validate(str(path) if path is not None else f, lg, express_rules=False)
    out = [dict(kind="schema", **{k: str(v)[:240] for k, v in s.items() if k != "level"})
           for s in lg.statements if s.get("level") in ("error", "critical")]
    rules = []
    if express_rules:
        lg = validate.json_logger()
        import ifcopenshell.express.rule_executor as rex
        try:
            rex.run(f, lg)
        except Exception as e:  # noqa: BLE001 - a rule crash is reported, not raised
            lg.statements.append({"level": "error", "message": f"rule executor failed: {e!r}"})
        rules = [{k: str(v)[:240] for k, v in s.items() if k != "level"} for s in lg.statements
                 if s.get("level") in ("error", "critical")]
    return out, rules


# ----------------------------------------------------------------------------- IDS
def run_ids(f, ids_path=IDS_PATH) -> list:
    """Run the IDS with ifctester; one result row per specification."""
    import xml.etree.ElementTree as ET
    from ifctester import ids as ids_mod
    spec_set = ids_mod.open(str(ids_path))
    spec_set.validate(f)
    ns = "{http://standards.buildingsmart.org/IDS}"
    idents = {s.get("name"): s.get("identifier") for s in ET.parse(ids_path).getroot().iter(f"{ns}specification")}
    rows = []
    for s in spec_set.specifications:
        s.identifier = s.identifier or idents.get(s.name)
        failures = []
        for req in s.requirements:
            for fl in req.failures:
                failures.append(f"{_plain(fl['element'])}: {fl['reason']}")
        if not s.applicable_entities and s.minOccurs != 0:
            failures.append("no applicable entity in a required specification")
        rows.append(dict(name=s.name, identifier=s.identifier, status=bool(s.status),
                         applicable=len(s.applicable_entities), failed=len(s.failed_entities), failures=failures))
    return rows


# ----------------------------------------------------------------------------- per-file checks
def check(f, path=None, *, schema=True, express_rules=False, ids=True, rules=None, budget=None) -> list:
    """All IFC-level checks of one opened file; returns the list of result dicts."""
    rules = rules_config() if rules is None else rules
    budget = budget_config() if budget is None else budget
    rep = Report()

    if schema:
        errs, rule_errs = schema_errors(f, path, express_rules)
        rep.add("schema", "error", errs, note="ifcopenshell.validate (types, cardinalities, inverses, header)")
        if express_rules:
            c = Counter(r.get("attribute") or r.get("message", "")[:60] for r in rule_errs)
            rep.add("express_rules", "error", rule_errs, value=dict(c), note="ifcopenshell.express rule executor")

    # GlobalIds
    seen, dups = {}, []
    for e in f.by_type("IfcRoot"):
        g = e.GlobalId
        if g in seen:
            dups.append(f"{g}: {_plain(seen[g])} / {_plain(e)}")
        else:
            seen[g] = e
    rep.add("unique_guids", "error", dups)

    # containment
    memo, loose = {}, []
    for e in f.by_type("IfcElement"):
        if e.is_a("IfcFeatureElement"):
            continue
        if not _is_contained(e, memo):
            loose.append(e)
    rep.add("containment", "error", loose, note="every IfcElement in a spatial structure or aggregated into a contained element")
    loose_sp = [s for s in f.by_type("IfcSpace") if not (s.Decomposes or getattr(s, "ContainedInStructure", None))]
    rep.add("space_structure", "error", loose_sp, note="every IfcSpace aggregated into the spatial structure")

    # openings and fillings
    bad_void, empty = [], []
    for o in f.by_type("IfcOpeningElement"):
        n = len(o.VoidsElements or ())
        if n != 1:
            bad_void.append(f"{_plain(o)} voids {n} elements")
        if not o.HasFillings:
            empty.append(o)
    rep.add("opening_voids", "error", bad_void, note="every IfcOpeningElement voids exactly one element")
    bad_fill = []
    for cls in ("IfcDoor", "IfcWindow"):
        for e in f.by_type(cls):
            n = len(e.FillsVoids or ())
            if n != 1:
                bad_fill.append(f"{_plain(e)} fills {n} openings")
    rep.add("filling", "error", bad_fill, note="every IfcDoor / IfcWindow fills exactly one opening")
    rep.add("empty_openings", "warn", empty, note="openings with no door / window")
    lim = int(budget.get("max_openings_per_wall", 20))
    many = [f"{_plain(w)}: {len(w.HasOpenings)} openings" for w in f.by_type("IfcElement")
            if len(getattr(w, "HasOpenings", ()) or ()) > lim]
    rep.add("openings_per_wall", "error", many, value=lim, note="Bonsai void_limit headroom")
    rep.add("representation_sharing", "error", shared_representations(f),
            note="IfcShapeModel.WR11: a shape representation belongs to exactly one product shape or one type map")

    # zones, psets
    psets = pset_index(f)
    unzoned = []
    for s in f.by_type("IfcSpace"):
        if "SampleCity_Room" in psets.get(s.id(), ()) and _zone_of(s) is None:
            unzoned.append(s)
    rep.add("space_in_zone", "error", unzoned, note="residential spaces (SampleCity_Room) grouped in an IfcZone")
    missing = []
    from estate.flats.template import ROOM_CODES
    flat_zones = {z.id() for s in f.by_type("IfcSpace") if "SampleCity_Room" in psets.get(s.id(), ())
                  and (uel.get_pset(s, "SampleCity_Room") or {}).get("RoomCode") in ROOM_CODES
                  for z in [_zone_of(s)] if z is not None}
    for cls, names in REQUIRED_PSETS.items():
        for e in f.by_type(cls):
            if cls == "IfcZone" and e.id() not in flat_zones:
                continue    # car park / centre zones are not flats
            have = psets.get(e.id(), set())
            lack = [n for n in names if n not in have]
            if lack:
                missing.append(f"{_plain(e)}: {', '.join(lack)}")
    rep.add("required_psets", "error", missing, value={k: list(v) for k, v in REQUIRED_PSETS.items()})

    bands = rules.get("flat", {}).get("ifa_band", {})
    bad_ifa = []
    for z in f.by_type("IfcZone"):
        p = flat_pset(z)
        if not p:
            continue
        label, t = p.get("FlatType"), canonical_flat_type(p.get("FlatType"))
        ifa, what = p.get("InternalFloorArea"), "IFA"
        if not isinstance(ifa, (int, float)):        # legacy zones carry only GrossArea
            ifa, what = p.get("GrossArea"), "gross area"
        if t not in bands:
            bad_ifa.append(f"{z.Name}: unknown flat type {label!r}")
        elif not isinstance(ifa, (int, float)):
            bad_ifa.append(f"{z.Name}: no InternalFloorArea or GrossArea")
        elif not (bands[t][0] - 1e-6 <= ifa <= bands[t][1] + 1e-6):
            bad_ifa.append(f"{z.Name}: {t} {what} {ifa:.2f} outside {bands[t]}")
    rep.add("zone_ifa", "error", bad_ifa, value=bands,
            note="config/rules.toml [flat.ifa_band] on InternalFloorArea (GrossArea when missing); legacy FlatType labels "
                 "such as '4-Room' map to 4R")

    n_el = len(f.by_type("IfcElement"))
    cap = int(budget.get("max_ifc_elements_per_file", 25000))
    rep.add("element_budget", "error", [f"{n_el} IfcElement >= {cap}"] if n_el >= cap else [], value=n_el)

    # placements
    bad_pl = placement_issues(f)
    rep.add("placements_right_handed", "error", bad_pl, note="det(rotation) = +1 for every placement and mapped item")

    # georeference and project identity
    g = georef(f)
    rep.add("georef", "error", [] if g and g["CRS"] else ["no IfcMapConversion / IfcProjectedCRS"], value=g)
    ident = expected_identity()
    if ident:
        pr, si = f.by_type("IfcProject"), f.by_type("IfcSite")
        bad = []
        if not pr or pr[0].GlobalId != ident["project_guid"]:
            bad.append(f"IfcProject {pr[0].GlobalId if pr else None} != {ident['project_guid']}")
        if not si or si[0].GlobalId != ident["site_guid"]:
            bad.append(f"IfcSite {si[0].GlobalId if si else None} != {ident['site_guid']}")
        rep.add("project_identity", "warn", bad, note="IfcProject / IfcSite GUIDs from config.identity()")

    if ids:
        try:
            rows = run_ids(f)
        except Exception as e:  # noqa: BLE001
            rep.add("ids", "error", [f"IDS run failed: {e!r}"])
        else:
            for r in rows:
                rep.add(f"ids:{r['identifier'] or r['name']}", "error", r["failures"],
                        count=r["failed"] + (1 if r["applicable"] == 0 and r["failures"] else 0),
                        value={"applicable": r["applicable"]}, note=r["name"])
    return rep.results


def check_file(path, *, schema=True, express_rules=False, ids=True, f=None) -> dict:
    t = time.time()
    f = f if f is not None else ifcopenshell.open(str(path))
    res = check(f, path, schema=schema, express_rules=express_rules, ids=ids)
    return {"file": env.rel(path), "schema": f.schema, "elements": len(f.by_type("IfcElement")),
            "runtime_s": round(time.time() - t, 2), "summary": summarize(res), "results": res}


# ----------------------------------------------------------------------------- federation
def check_federation(paths_or_files) -> dict:
    """GUID uniqueness across files (IfcProject / IfcSite shared and identical) and one georeference."""
    t = time.time()
    rep = Report()
    files = []
    for p in paths_or_files:
        if isinstance(p, ifcopenshell.file):
            files.append((f"<memory {len(files)}>", p))
        else:
            files.append((env.rel(p), ifcopenshell.open(str(p))))
    shared = defaultdict(set)
    owner = {}
    dups = []
    for name, f in files:
        for cls in ("IfcProject", "IfcSite"):
            for e in f.by_type(cls):
                shared[cls].add(e.GlobalId)
        skip = {e.id() for c in ("IfcProject", "IfcSite") for e in f.by_type(c)}
        for e in f.by_type("IfcRoot"):
            if e.id() in skip:
                continue
            g = e.GlobalId
            if g in owner and owner[g][0] != name:
                dups.append(f"{g} {e.is_a()} in {owner[g][0]} and {name}")
            else:
                owner.setdefault(g, (name, e.is_a()))
    rep.add("federation_guids", "error", dups, note="GlobalIds unique across files")
    ident = []
    for cls in ("IfcProject", "IfcSite"):
        if len(shared[cls]) != 1:
            ident.append(f"{cls} GUIDs differ: {sorted(shared[cls])}")
    names = {f.by_type("IfcProject")[0].Name for _, f in files if f.by_type("IfcProject")}
    if len(names) > 1:
        ident.append(f"IfcProject names differ: {sorted(names)}")
    rep.add("federation_identity", "error", ident, note="one IfcProject / IfcSite shared by every file")
    geo = {name: georef(f) for name, f in files}
    keys = ("Eastings", "Northings", "OrthogonalHeight", "XAxisAbscissa", "XAxisOrdinate", "Scale", "CRS")
    variants = defaultdict(list)
    for name, g in geo.items():
        variants[None if g is None else tuple((k, g[k]) for k in keys)].append(name)
    bad = []
    if len(variants) > 1 or None in variants:
        for k, names_ in variants.items():
            bad.append(f"{'no georeference' if k is None else dict(k)}: {', '.join(names_)}")
    rep.add("federation_georef", "error", bad, note="identical IfcMapConversion / IfcProjectedCRS")
    units = defaultdict(list)
    for name, f in files:
        u = tuple(sorted((x.UnitType, getattr(x, "Prefix", None), getattr(x, "Name", None))
                         for x in f.by_type("IfcSIUnit")))
        units[u].append(name)
    rep.add("federation_units", "error", [", ".join(v) for v in units.values()] if len(units) > 1 else [])
    return {"files": [n for n, _ in files], "runtime_s": round(time.time() - t, 2),
            "summary": summarize(rep.results), "results": rep.results}
