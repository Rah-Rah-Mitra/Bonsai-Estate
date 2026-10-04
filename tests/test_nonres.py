"""Non-residential building tests: the car park (MSCP 513) and the neighbourhood centre (NC 514) are built once into
build/nonres_tests/unittest and checked for the review fixes: the NC forecourt closes the ground surface of the
whole [nc] rect, both upper-floor escape routes of the shop block, the car park's third stair, IfcSpaces behind every
passable door and Navigation FromSpace / ToSpace naming them, door types with materials and styled items (roller
shutters as their own type), the published <id>.build.json, the element budget and ifcopenshell.validate.

Run: "<blender python>" -I -B tests/test_nonres.py -v
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estate.env import bootstrap  # noqa: E402

bootstrap()

import ifcopenshell  # noqa: E402
import ifcopenshell.util.element as uel  # noqa: E402
import ifcopenshell.util.placement as upl  # noqa: E402
from shapely.geometry import box  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from estate import config, env  # noqa: E402
from estate.site import centre, mscp  # noqa: E402
from estate.site.cli_buildings import validate  # noqa: E402
from estate.validate.programme import solid_footprint  # noqa: E402

OUT = env.BUILD / "nonres_tests" / "unittest"
_FILES = {}


def built(which):
    """Build each building once per run (about 5 s each) and keep the open file and the info dict."""
    if which not in _FILES:
        cfg = config.load()
        OUT.mkdir(parents=True, exist_ok=True)
        fn = mscp.build_mscp_ifc if which == "mscp" else centre.build_nc_ifc
        tid = config.target_id(which, cfg[which]["blk"])
        info = fn(cfg, OUT / f"{tid}.ifc", "IFC4X3")
        _FILES[which] = (ifcopenshell.open(info["path"]), info, cfg)
    return _FILES[which]


def storey_of(el):
    """Storey name of an element (contained) or a space (aggregated)."""
    s = uel.get_container(el) or uel.get_aggregate(el)
    return s.Name if s is not None else None


def nav(el):
    return uel.get_pset(el, "Navigation") or {}


def door_xy(d):
    """Estate xy of a door's centre (leaf origin plus half its width along the leaf's x axis)."""
    M = upl.get_local_placement(d.ObjectPlacement)
    p = M[:3, 3] + M[:3, 0] * float(d.OverallWidth) / 2
    return float(p[0]), float(p[1])


class TestCentre(unittest.TestCase):
    def test_forecourt_closes_the_rect(self):
        """Hawker hall, shop block and forecourt slabs tile the [nc] rect with no gap or overlap, all at 0.0."""
        f, info, cfg = built("nc")
        r = cfg["nc"]["rect"]
        polys = []
        for s in f.by_type("IfcSlab"):
            if s.PredefinedType != "BASESLAB":
                continue
            poly, z0, z1 = solid_footprint(s)
            self.assertAlmostEqual(z1, 0.0, places=3, msg=s.Name)
            polys.append((s.Name, poly))
        names = {n for n, _ in polys}
        self.assertIn("NC forecourt paving", names)
        union = unary_union([p for _, p in polys])
        rect = box(*r)
        self.assertLess(rect.symmetric_difference(union).area, 0.01)
        overlap = sum(p.area for _, p in polys) - union.area
        self.assertLess(overlap, 0.01)
        self.assertGreater(info["forecourt_m2"], 2000.0)
        fc = next(s for s in f.by_type("IfcSpace") if s.Name == "L1-FORECOURT")
        self.assertEqual(fc.PredefinedType, "EXTERNAL")

    def test_shop_block_has_two_escape_stairs(self):
        """Two IfcStairs; every upper-floor shop door lies between the two stair doors along the corridor."""
        f, info, cfg = built("nc")
        self.assertEqual(len(f.by_type("IfcStair")), 2)
        l2 = [d for d in f.by_type("IfcDoor") if storey_of(d) == "L2"]
        stairs = [d for d in l2 if nav(d).get("DoorKind") == "stair"]
        self.assertEqual(len(stairs), 2)
        for d in stairs:
            self.assertTrue({"L2-CORR", "L2-LOBBY"} & {nav(d)["FromSpace"], nav(d)["ToSpace"]}, d.Name)
            self.assertTrue(any(nav(d)[k].startswith("L2-STAIR") for k in ("FromSpace", "ToSpace")), d.Name)
        xs = sorted(door_xy(d)[0] for d in stairs)
        shop_doors = [d for d in l2 if nav(d).get("DoorKind") == "shop"]
        self.assertEqual(len(shop_doors), 10)
        for d in shop_doors:
            self.assertTrue(xs[0] <= door_xy(d)[0] <= xs[1], d.Name)
        # the layout keeps every unit at least 4 m wide
        sl = centre.shop_layout(cfg, ((cfg["nc"]["rect"][0] + cfg["nc"]["rect"][2]) / 2,
                                      (cfg["nc"]["rect"][1] + cfg["nc"]["rect"][3]) / 2))
        self.assertGreaterEqual(sl["unit_w"], 4.0)

    def test_shutters_and_shop_doors(self):
        """Roller shutters are their own ROLLINGUP type (curtain + coil box, not a hinged leaf); shop doors are
        glazed aluminium (two glass leaves)."""
        f, info, cfg = built("nc")
        shutters = [d for d in f.by_type("IfcDoor") if nav(d).get("DoorKind") == "shutter"]
        self.assertEqual(len(shutters), 48)
        for d in shutters:
            t = uel.get_type(d)
            self.assertEqual((d.OperationType, t.OperationType, t.ElementType), ("ROLLINGUP", "ROLLINGUP", "Roller shutter"))
            self.assertEqual(uel.get_material(d).Name, "Roller shutter - galvanised")
        t = uel.get_type(next(d for d in f.by_type("IfcDoor") if nav(d).get("DoorKind") == "shop"))
        styles = [s.Name for rm in t.RepresentationMaps for it in rm.MappedRepresentation.Items
                  for si in it.StyledByItem for s in si.Styles]
        self.assertEqual(styles.count("Glass"), 2)
        self.assertEqual(uel.get_material(t).Name, "Aluminium frame")


class TestCarPark(unittest.TestCase):
    def test_three_stairs_reach_the_decks(self):
        """Stair 3 at the south-west corner: three stair doors per deck, every car lot of L2 within 45 m (straight
        line) of one, and stair 3 discharges into the pedestrian bay at L1."""
        f, info, cfg = built("mscp")
        self.assertEqual(len(info["stair_doors"]), 3)
        decks = [n for n in info["levels"] if n != "RF"]
        doors = [d for d in f.by_type("IfcDoor") if nav(d).get("DoorKind") == "stair"]
        for lv in decks:
            self.assertEqual(sum("fire door" in d.Name and storey_of(d) == lv for d in doors), 3, lv)
        pts = [door_xy(d) for d in doors if storey_of(d) == "L2"]
        far = 0.0
        for sp in f.by_type("IfcSpace"):
            if storey_of(sp) == "L2" and (sp.LongName or "").startswith("Car lot"):
                c = upl.get_local_placement(sp.ObjectPlacement)[:2, 3]
                far = max(far, min(math.dist(c, p) for p in pts))
        self.assertGreater(far, 10.0)
        self.assertLess(far, 45.0)
        dis = next(d for d in doors if d.Name == "L1 stair 3 discharge door")
        self.assertEqual({nav(dis)["FromSpace"], nav(dis)["ToSpace"]}, {"L1-STAIR3", "L1-S3BAY"})
        self.assertEqual(len(info["stair_discharge"]), 1)


class TestBoth(unittest.TestCase):
    def test_spaces_behind_doors(self):
        """Every door names IfcSpaces of the file (or 'outside' / a shaft) and every stair enclosure is a space."""
        for which in ("mscp", "nc"):
            f, info, cfg = built(which)
            names = {s.Name for s in f.by_type("IfcSpace")}
            for d in f.by_type("IfcDoor"):
                n = nav(d)
                for k in ("FromSpace", "ToSpace"):
                    v = n.get(k)
                    ok = v in names or v == "outside" or (n.get("DoorKind") in ("lift", "locked") and v.endswith("(shaft)"))
                    self.assertTrue(ok, f"{which} {d.Name} {k}={v}")
                if n.get("Passable"):
                    self.assertTrue(n["FromSpace"] in names or n["ToSpace"] in names, d.Name)
            stair_spaces = [s for s in f.by_type("IfcSpace") if "-STAIR" in s.Name]
            self.assertEqual(len(stair_spaces), 24 if which == "mscp" else 4)

    def test_door_types_have_material_and_style(self):
        for which in ("mscp", "nc"):
            f, info, cfg = built(which)
            for t in f.by_type("IfcDoorType"):
                self.assertIsNotNone(uel.get_material(t), t.Name)
                items = [it for rm in t.RepresentationMaps for it in rm.MappedRepresentation.Items]
                self.assertTrue(all(it.StyledByItem for it in items), t.Name)

    def test_build_json_budget_and_schema(self):
        for which in ("mscp", "nc"):
            f, info, cfg = built(which)
            bj = Path(info["path"]).with_name(f"{info['id']}.build.json")
            pub = json.loads(bj.read_text(encoding="utf-8"))
            self.assertEqual(pub["entrances"], json.loads(json.dumps(info["entrances"])))
            self.assertTrue(pub["lift_lobbies"] and pub["stair_doors"])
            self.assertLess(info["counts"]["IfcElement"], cfg["budget"]["max_ifc_elements_per_file"])
            self.assertEqual(validate(info["path"]), [], which)


if __name__ == "__main__":
    unittest.main()
