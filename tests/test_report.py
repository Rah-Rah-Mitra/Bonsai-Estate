"""Report tests: collect() covers every section of report.html and report.json is its exact JSON round trip; the
page renders with no artefacts at all; the BCF writer (estate/report/bcf_out.py) on a small synthetic record set
re-opens with the bcf library, one topic per failing item with the item's GlobalIds selected and fixed GUIDs / dates,
byte-identical on a second write; validation items carry real GlobalIds and points (validate_one on the PT4 block and
injected defects of the mutation suite: ifcqa entity failures, programme rooms and doors, a door / wall clash).

Run: "<blender python>" -I -B tests/test_report.py -v
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estate.env import bootstrap  # noqa: E402

bootstrap()

import ifcopenshell  # noqa: E402

from estate import env  # noqa: E402
from estate.env import BUILD  # noqa: E402
from estate.report import bcf_out  # noqa: E402
from estate.report import html as report  # noqa: E402

T_PT4 = BUILD / "t_pt4.ifc"
_BLK = BUILD.parent / "model" / "BLK_501" / "BLK_501.ifc"
if _BLK.exists():    # prefer a real estate point block, as tests/test_validate.py does
    T_PT4 = _BLK

G = ["0aAaAaAaAaAaAaAaAaAaA1", "0aAaAaAaAaAaAaAaAaAaA2", "0aAaAaAaAaAaAaAaAaAaA3", "0aAaAaAaAaAaAaAaAaAaA4"]


def _res(check, level, items, **kw):
    return dict(check=check, level=level, severity=level, count=len(items), examples=[i["text"] for i in items[:6]],
                items=items, **kw)


RECORDS = [{
    "file": "model/BLK_900/BLK_900.ifc", "ifc_project": "1dZ33uP0MTqrdmSLFx2C$g",
    "ifcqa": {"results": [
        _res("filling", "error", [{"guids": [G[0]], "xyz": [1.0, 2.0, 3.0], "text": "#5 IfcDoor D1 fills 0 openings"},
                                  {"guids": [G[1]], "xyz": None, "text": "#6 IfcDoor D2 fills 0 openings"}]),
        _res("schema", "error", [])]},             # passing: level info after Report.add, no topic
    "geometry": {"results": [
        _res("wall_overlap", "error", [{"guids": [G[2], G[3]], "xyz": [5.0, 6.0, 1.3],
                                        "text": "L2 IfcWall W1 / W2: 0.0400 m2 > 0.0101 m2"}],
             value={"worst_excess_m2": 0.03}, note="junction tolerance"),
        _res("falling_edge_gaps", "warn", [{"guids": [G[0]], "xyz": [0.0, 0.0, 3.6], "text": "same gap"},
                                           {"guids": [G[0]], "xyz": [0.0, 0.5, 3.6], "text": "same gap"}])]},
}]
RECORDS[0]["ifcqa"]["results"][1]["level"] = "info"
N_ITEMS = 5


def _guids_in(f, guids):
    bad = []
    for g in guids:
        try:
            f.by_guid(g)
        except RuntimeError:
            bad.append(g)
    return bad


# ----------------------------------------------------------------------------- report.html / report.json
class TestCollect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.data = report.collect()
        cls.page_path = report.write_report(cls.dir / "report.html")
        cls.page = cls.page_path.read_text(encoding="utf-8")
        cls.json_text = (cls.dir / "report.json").read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_section_has_a_key(self):
        keys = list(self.data)
        self.assertEqual(keys[:2], ["report", "files"])
        self.assertEqual([k for k in keys if k in report.SECTIONS], list(report.SECTIONS))
        self.assertEqual(set(keys) - set(report.SECTIONS), {"report", "files"})

    def test_every_heading_comes_from_its_data(self):
        heads = re.findall(r"<h2>(.*?)(?:<span|</h2>)", self.page)
        for key, title in report.SECTIONS.items():
            if title is None:
                continue
            shown = any(h.strip() == title for h in heads)
            self.assertEqual(shown, bool(self.data[key]), (key, title))
        self.assertEqual(len(heads), sum(1 for k, t in report.SECTIONS.items() if t and self.data[k]))

    def test_files_list_what_was_read(self):
        files = self.data["files"]
        self.assertEqual(len({x["path"] for x in files}), len(files))
        for x in files:
            p = env.ROOT / x["path"]
            self.assertTrue(p.is_file(), x["path"])
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), x["sha256"], x["path"])
            self.assertTrue(set(x["sections"]) <= set(report.SECTIONS) | {"report"}, x)
            self.assertEqual(len(set(x["sections"])), len(x["sections"]), x)
        if (env.MODEL / "masterplan.json").exists():
            self.assertIn("model/masterplan.json", {x["path"] for x in files})
            used = {s for x in files for s in x["sections"]}       # every non-empty section names its artefacts
            self.assertEqual([k for k in report.SECTIONS if self.data[k] and k not in used], [])

    def test_shared_inputs_list_every_section(self):
        """masterplan / plan_check / .state feed several sections each: every one is listed, in page order."""
        old = env.MODEL, env.REPORTS, env.BUILD
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            env.MODEL, env.REPORTS, env.BUILD = d / "model", d / "reports", d / "build"
            for p, obj in [(env.MODEL / "masterplan.json", {"estate": {"name": "Test Town"},
                                                            "sites": [{"id": "BLK_900", "kind": "block"}]}),
                           (env.REPORTS / "plan_check.json", {"mix": {"4R": 2}, "errors": ["e1"], "raw": 3,
                                                              "canonical": 2, "per_type": {"4R": {"raw": 3}}}),
                           (env.MODEL / ".state.json", {"BLK_900": {"glb": {"lod0_triangles": 10,
                                                                            "max_tris_per_flat": 5}}})]:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(obj), encoding="utf-8")
            try:
                data = report.collect()
            finally:
                env.MODEL, env.REPORTS, env.BUILD = old
        secs = {Path(x["path"]).name: x["sections"] for x in data["files"]}
        self.assertEqual(secs, {"masterplan.json": ["report", "kpi", "buildings"],
                                "plan_check.json": ["kpi", "mix", "plan_errors", "layout_variety"],
                                ".state.json": ["buildings", "engine"]})
        self.assertEqual(data["report"]["estate"], "Test Town")
        self.assertTrue(data["mix"] and data["plan_errors"] and data["layout_variety"] and data["engine"])

    def test_json_round_trip(self):
        loaded = json.loads(self.json_text)
        self.assertEqual(json.dumps(loaded, indent=1, ensure_ascii=False) + "\n", self.json_text)
        self.assertEqual(loaded, json.loads(json.dumps(self.data, default=str)))     # the same dict, no clock

    def test_collect_is_deterministic(self):
        self.assertEqual(json.dumps(report.collect(), default=str), json.dumps(self.data, default=str))

    def test_page_links_json_and_bcf(self):
        import bcf.bcfxml
        self.assertIn('href="report.json"', self.page)
        info = (self.data.get("validation") or {}).get("bcf")
        path = self.dir / "issues.bcf"
        self.assertTrue(path.exists())
        doc = bcf.bcfxml.load(path)
        try:
            n = len(doc.topics)
        finally:
            doc.close()
        if info is not None:
            self.assertIn('href="issues.bcf"', self.page)
            self.assertIn(f"has {info['topics']} topic(s)", self.page)
            self.assertEqual(n, info["topics"])

    def test_render_with_no_artefacts(self):
        old = env.MODEL, env.REPORTS, env.BUILD
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            env.MODEL, env.REPORTS, env.BUILD = d / "model", d / "reports", d / "build"
            try:
                data = report.collect()
                page = report.write_report(d / "out" / "report.html")
            finally:
                env.MODEL, env.REPORTS, env.BUILD = old
            self.assertEqual(data["files"], [])
            self.assertIsNone(data["validation"])
            self.assertEqual([r["id"] for r in data["buildings"]], ["SITE"])
            self.assertTrue(page.exists() and (d / "out" / "report.json").exists())
            with zipfile.ZipFile(d / "out" / "issues.bcf") as z:
                self.assertFalse([n for n in z.namelist() if n.endswith("markup.bcf")])


# ----------------------------------------------------------------------------- BCF
class TestBcf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import bcf.bcfxml
        cls.tmp = tempfile.TemporaryDirectory()
        cls.a, cls.b = Path(cls.tmp.name) / "a.bcf", Path(cls.tmp.name) / "b.bcf"
        cls.n = bcf_out.write_bcf(cls.a, RECORDS)
        bcf_out.write_bcf(cls.b, json.loads(json.dumps(RECORDS)))
        cls.doc = bcf.bcfxml.load(cls.a)
        cls.topics = sorted(cls.doc.topics.values(), key=lambda t: t.topic.index)

    @classmethod
    def tearDownClass(cls):
        cls.doc.close()
        cls.tmp.cleanup()

    def test_one_topic_per_item(self):
        self.assertEqual(self.n, N_ITEMS)
        self.assertEqual(len(self.topics), N_ITEMS)
        self.assertEqual(self.doc.version.version_id, "2.1")
        self.assertEqual([t.topic.index for t in self.topics], list(range(1, N_ITEMS + 1)))

    def test_byte_identical(self):
        self.assertEqual(self.a.read_bytes(), self.b.read_bytes())
        with zipfile.ZipFile(self.a) as z:
            self.assertEqual({i.date_time for i in z.infolist()}, {(2026, 10, 4, 0, 0, 0)})

    def test_selected_guids_and_cameras(self):
        items = [i for r in RECORDS[0]["ifcqa"]["results"] + RECORDS[0]["geometry"]["results"] for i in r["items"]]
        for t, it in zip(self.topics, items):
            vps = list(t.viewpoints.values())
            if it["xyz"] is None:
                self.assertEqual(vps, [])
                self.assertIn(", ".join(it["guids"]), t.topic.description)
                continue
            self.assertEqual(len(vps), 1)
            self.assertEqual(vps[0].get_selected_guids(), it["guids"])
            cam = vps[0].visualization_info.perspective_camera
            d = cam.camera_direction
            p = cam.camera_view_point          # the camera looks at the item's point
            for k, axis in enumerate("xyz"):
                self.assertAlmostEqual(getattr(p, axis) + 5 * 3 ** 0.5 * getattr(d, axis), it["xyz"][k], places=6)

    def test_topic_fields(self):
        t = self.topics[0]
        self.assertEqual(t.topic.title, "BLK_900 filling: #5 IfcDoor D1 fills 0 openings")
        self.assertEqual((t.topic.topic_type, t.topic.topic_status, t.topic.priority), ("Error", "Open", "High"))
        self.assertEqual(t.topic.labels, ["ifcqa", "filling", "BLK_900"])
        self.assertEqual(t.topic.creation_author, "estate validate")
        self.assertEqual(str(t.topic.creation_date), "2026-10-04T00:00:00Z")
        self.assertEqual(t.topic.guid, bcf_out._uuid("bcf/topic/BLK_900/ifcqa/filling/#5 IfcDoor D1 fills 0 openings"))
        h = t.header.file[0]
        self.assertEqual((h.filename, h.reference, h.ifc_project), ("BLK_900.ifc", "model/BLK_900/BLK_900.ifc",
                                                                    "1dZ33uP0MTqrdmSLFx2C$g"))
        w = self.topics[-1]
        self.assertEqual((w.topic.topic_type, w.topic.priority), ("Warning", "Normal"))
        self.assertIn("Check: geometry / falling_edge_gaps (warn, 2 item(s)", w.topic.description)
        self.assertEqual(len({t.topic.guid for t in self.topics}), N_ITEMS)       # the repeated text still differs
        self.assertIn('"worst_excess_m2": 0.03', self.topics[2].topic.description)
        self.assertEqual(self.doc.project.project_id, bcf_out._uuid("bcf/project/Sample Town N5"))
        self.assertEqual(self.doc.extensions.topic_types.topic_type, ["Error", "Warning"])

    def test_records_without_items_fall_back_to_examples(self):
        rec = {"file": "model/BLK_901/BLK_901.ifc", "geometry": {"results": [
            {"check": "falling_edge_gaps", "level": "warn", "severity": "warn", "count": 9, "examples": ["a", "b"]}]}}
        found = bcf_out.issues([rec])
        self.assertEqual([i["text"] for i in found][:2], ["a", "b"])
        self.assertEqual(len(found), 3)
        self.assertIn("7 more item(s)", found[2]["text"])
        self.assertEqual(found[0]["file_id"], "BLK_901")


# ----------------------------------------------------------------------------- items in validation records
@unittest.skipUnless(T_PT4.exists(), "needs build/t_pt4.ifc (estate test builds it)")
class TestItems(unittest.TestCase):
    def assert_items(self, f, r, n_guids=1):
        self.assertEqual(len(r["items"]), r["count"], r["check"])
        self.assertEqual([i["text"] for i in r["items"][:6]], [str(x) for x in r["examples"]], r["check"])
        for i in r["items"]:
            self.assertGreaterEqual(len(i["guids"]), n_guids, (r["check"], i))
            self.assertEqual(_guids_in(f, i["guids"]), [], (r["check"], i))
            self.assertTrue(i["xyz"] is not None and len(i["xyz"]) == 3, (r["check"], i))

    def test_validate_one_items_have_real_guids(self):
        from estate.validate import cli
        rec = cli.validate_one(T_PT4, schema=False, ids=False)
        f = ifcopenshell.open(str(T_PT4))
        self.assertEqual(rec["ifc_project"], f.by_type("IfcProject")[0].GlobalId)
        failing = []
        for fam in cli.FAMILIES:
            for r in rec[fam]["results"]:
                self.assertIn("items", r, r["check"])
                if r["level"] in ("error", "warn"):
                    failing.append(r)
                    self.assert_items(f, r)
                else:
                    self.assertEqual(r["items"], [])
        self.assertTrue(failing, "expected at least one failing check (falling_edge_gaps / programme) to look at")

    def test_injected_ifcqa_and_programme_defects(self):
        from estate.validate import ifcqa, mutations, programme
        f = ifcopenshell.open(str(T_PT4))
        for m in (mutations.m_remove_filling, mutations.m_space_out_of_zone, mutations.m_ifa_out_of_band):
            m(f)
        qa = {r["check"]: r for r in ifcqa.check(f, schema=False, ids=False)}
        for c in ("filling", "empty_openings", "space_in_zone", "zone_ifa"):
            self.assertGreater(qa[c]["count"], 0, c)
            self.assert_items(f, qa[c])
        g = ifcopenshell.open(str(T_PT4))
        base = {r["check"]: r["count"] for r in programme.check(g)}
        mutations.m_remove_bath_vent(g)
        mutations.m_stale_door_spaces(g)
        pr = {r["check"]: r for r in programme.check(g)}
        for c, n in (("bath_vent", 1), ("door_spaces", 3)):     # the door plus the spaces on either side
            self.assertGreater(pr[c]["count"], base[c], c)
            self.assert_items(g, pr[c], n)

    def test_injected_clash_has_both_guids(self):
        from estate.export import meshcache
        from estate.validate import geometry as geom
        from estate.validate import mutations
        _, meshes = meshcache.load(T_PT4)
        f = ifcopenshell.open(str(T_PT4))
        info = mutations.m_door_clash(f)
        meshes = dict(meshes)
        meshes.update(meshcache.tessellate(f, include=info["changed"]))
        res = {r["check"]: r for r in geom.check(f, meshes)}
        r = res["opening_clash"]
        self.assertGreater(r["count"], 0)
        self.assert_items(f, r, 2)
        door = info["changed"][0].GlobalId
        self.assertTrue(all(i["guids"][0] == door for i in r["items"]))


if __name__ == "__main__":
    unittest.main()
