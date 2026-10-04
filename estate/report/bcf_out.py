"""reports/issues.bcf: every failing validation item as a BCF 2.1 topic, for Bonsai's BCF panel or any BCF viewer.

The check stage writes model/<id>/<id>_validation.json (and model/SITE_validation.json) with one structured item
per failure (estate/validate/ifcqa.py Report.add: {guids, xyz, text}). Every item of an error or warning check becomes
one topic: title '<file> <check>: <text>', the family / check / level / value / note in the description, labels
[family, check, file], type and priority from the level (Error / High, Warning / Normal), status Open, author
'estate validate', and one viewpoint that selects the item's GlobalIds with the camera looking at its point (no
viewpoint when the check knows no point). The markup header names the building IFC and its IfcProject GlobalId, so a
viewer resolves the GlobalIds against the right model. A failing federation check (reports/validation_summary.json)
adds topics under the file id FEDERATION with no header file.

BCF 2.1 (bcf.v2 of the bcf library in Bonsai's site-packages) is used for the widest tool support. The library stamps
the project, topics and viewpoints with uuid4 and datetime.now(), and its zip entries with the wall clock. Here every
GUID is estate.guids.from_text(<file>/<family>/<check>/<text>) written as a UUID, so an issue keeps its topic GUID
from build to build while its text stays the same (a viewer that re-imports the file updates instead of duplicating),
every date is FIXED_DATE, topics follow the record order (file, family, check, item) and the zip is rewritten with
fixed entry times: two writes of the same records are byte-identical.
"""
from __future__ import annotations

import json
import uuid
import zipfile
from pathlib import Path
from xml.sax.saxutils import quoteattr

from estate import env, guids
from estate.validate.cli import FAMILIES, FIXED_DATE

AUTHOR = "estate validate"
LEVELS = {"error": ("Error", "High"), "warn": ("Warning", "Normal")}     # level -> (TopicType, Priority)
TITLE_MAX = 120
ZIP_TIME = tuple(int(x) for x in FIXED_DATE.split("-")) + (0, 0, 0)
FEDERATION = "FEDERATION"


def _uuid(key: str) -> str:
    """A BCF GUID (UUID text) for a key: estate.guids.from_text expanded back to its 128 bits."""
    import ifcopenshell.guid
    return str(uuid.UUID(hex=ifcopenshell.guid.expand(guids.from_text(key))))


def _load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------- reading
def validation_paths(model=None) -> list[Path]:
    """The per-file records the check stage wrote, buildings by id then SITE: model/<id>/<id>_validation.json and
    model/SITE_validation.json (those that exist)."""
    model = Path(model or env.MODEL)
    paths = sorted(p for p in model.glob("*/*_validation.json") if p.name == f"{p.parent.name}_validation.json")
    return paths + [p for p in [model / "SITE_validation.json"] if p.is_file()]


def validation_records(model=None) -> list[dict]:
    """The records of validation_paths(), each with an "id" (the file id the topics are labelled with)."""
    out = []
    for p in validation_paths(model):
        rec = _load(p)
        if isinstance(rec, dict):
            rec.setdefault("id", p.name.removesuffix("_validation.json"))
            out.append(rec)
    return out


def federation_record(summary=None) -> dict | None:
    """The federation check of reports/validation_summary.json as a record of the family 'federation'."""
    s = _load(summary or env.REPORTS / "validation_summary.json")
    fed = s.get("federation") if isinstance(s, dict) else None
    if not isinstance(fed, dict):
        return None
    return {"id": FEDERATION, "file": None, "federation": {"results": fed.get("results", [])}}


def issues(records) -> list[dict]:
    """One issue per failing item, in record order, then family (ifcqa, programme, geometry, federation), check and
    item order. A result written before items existed falls back to its examples (text only); a check whose items
    were capped gets one more issue that counts the rest."""
    out, seen = [], {}
    for rec in records:
        fid = rec.get("id") or Path(str(rec.get("file") or "file")).stem
        for fam in FAMILIES + ("federation",):
            for r in (rec.get(fam) or {}).get("results", []):
                level = r.get("level")
                if level not in LEVELS:
                    continue
                items = r.get("items")
                if items is None:
                    items = [{"guids": [], "xyz": None, "text": x if isinstance(x, str) else json.dumps(x, default=str)}
                             for x in r.get("examples", [])]
                    more = int(r.get("count", 0)) - len(items)
                else:
                    more = int(r.get("truncated", 0) or 0)
                if more > 0:
                    items = items + [{"guids": [], "xyz": None,
                                      "text": f"{more} more item(s) not listed in {rec.get('file') or 'the record'}"}]
                for it in items:
                    text = str(it.get("text", ""))
                    key = f"{fid}/{fam}/{r['check']}/{text}"
                    seen[key] = seen.get(key, 0) + 1
                    if seen[key] > 1:                          # the same text twice in one check
                        key = f"{key}/{seen[key]}"
                    out.append(dict(key=key, file_id=fid, file=rec.get("file"), ifc_project=rec.get("ifc_project"),
                                    family=fam, check=r["check"], level=level, count=r.get("count"),
                                    value=r.get("value"), note=r.get("note"), text=text,
                                    guids=[g for g in it.get("guids") or [] if g], xyz=it.get("xyz")))
    return out


def summary(found) -> dict:
    """Topic counts of a list of issues (for the report): total, by level, with a viewpoint, by file."""
    by_file = {}
    for i in found:
        by_file[i["file_id"]] = by_file.get(i["file_id"], 0) + 1
    return {"topics": len(found), "errors": sum(1 for i in found if i["level"] == "error"),
            "warnings": sum(1 for i in found if i["level"] == "warn"),
            "with_viewpoint": sum(1 for i in found if i["xyz"] is not None), "by_file": by_file}


# ----------------------------------------------------------------------------- writing
def _title(i) -> str:
    t = " ".join(f"{i['file_id']} {i['check']}: {i['text']}".split())
    return t if len(t) <= TITLE_MAX else t[:TITLE_MAX - 3].rstrip() + "..."


def _description(i) -> str:
    lines = [i["text"], "", f"File: {i['file'] or '-'}",
             f"Check: {i['family']} / {i['check']} ({i['level']}, {i['count']} item(s) in this check)"]
    if i["value"] is not None:
        v = json.dumps(i["value"], default=str)
        lines.append("Value: " + (v if len(v) <= 400 else v[:397] + "..."))
    if i["note"]:
        lines.append(f"Note: {i['note']}")
    if i["guids"]:
        lines.append("Elements: " + ", ".join(i["guids"]))
    if i["xyz"] is not None:
        lines.append("Point (m): " + ", ".join(f"{float(v):.3f}" for v in i["xyz"]))
    return "\n".join(lines)


def _extensions_xsd(labels) -> bytes:
    """extensions.xsd: the topic types, statuses, priorities, labels and author this file uses (BCF 2.1 'redefine'
    form), so viewers offer them as the project's lists."""
    rows = ['<?xml version="1.0" encoding="UTF-8"?>', '<schema xmlns="http://www.w3.org/2001/XMLSchema">',
            '  <redefine schemaLocation="markup.xsd">']
    for name, values in (("TopicType", [t for t, _ in LEVELS.values()]), ("TopicStatus", ["Open"]),
                         ("Priority", [p for _, p in LEVELS.values()]), ("TopicLabel", labels),
                         ("UserIdType", [AUTHOR])):
        rows += [f'    <simpleType name="{name}">', f'      <restriction base="{name}">']
        rows += [f"        <enumeration value={quoteattr(v)}/>" for v in values]
        rows += ["      </restriction>", "    </simpleType>"]
    rows += ["  </redefine>", "</schema>", ""]
    return "\n".join(rows).encode("utf-8")


def _normalise_zip(src: Path, dst: Path) -> None:
    """Copy a zip entry by entry with a fixed time, DOS attributes and deflate, so equal content is equal bytes."""
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(buf, "w") as zout:
        for info in zin.infolist():
            zi = zipfile.ZipInfo(info.filename, date_time=ZIP_TIME)
            zi.create_system = 0
            zi.external_attr = info.external_attr
            zi.compress_type = zipfile.ZIP_STORED if info.is_dir() else zipfile.ZIP_DEFLATED
            zout.writestr(zi, zin.read(info.filename))
    dst.write_bytes(buf.getvalue())


def write_bcf(path, records=None, *, project_name="Sample Town N5", federation=None) -> int:
    """Write the issues of validation records (default: validation_records() plus the federation record) as a
    BCF 2.1 file; returns the number of topics."""
    import numpy as np
    from xsdata.models.datatype import XmlDateTime

    import bcf.v2.model as mdl
    from bcf.v2.bcfxml import BcfXml
    from bcf.v2.visinfo import VisualizationInfoHandler, build_viewpoint_from_position_and_guids
    from bcf.xml_parser import XmlParserSerializer

    if records is None:
        records = validation_records()
        federation = federation_record() if federation is None else federation
    records = list(records) + ([federation] if federation else [])
    found = issues(records)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    date = XmlDateTime(*ZIP_TIME, 0, 0)
    xml = XmlParserSerializer()                 # one handler: each new one rebuilds xsdata's class metadata
    doc = BcfXml.create_new(project_name, xml)
    doc.project_info.project.project_id = _uuid(f"bcf/project/{project_name}")
    labels = []
    for n, i in enumerate(found, 1):
        ttype, prio = LEVELS[i["level"]]
        th = doc.add_topic(_title(i), _description(i), AUTHOR, ttype, "Open")
        tmp = th.guid
        guid = _uuid(f"bcf/topic/{i['key']}")
        t = th.topic
        t.guid, t.creation_date, t.priority, t.index = guid, date, prio, n
        t.labels = [i["family"], i["check"], i["file_id"]]
        th._topic_dir = Path(guid)              # save() names the folder from the GUID, the markup from _topic_dir
        doc.topics[guid] = doc.topics.pop(tmp)
        if i["file"]:
            th.markup.header = mdl.Header(file=[mdl.HeaderFile(
                filename=Path(i["file"]).name, date=date, reference=i["file"], ifc_project=i["ifc_project"],
                is_external=True)])
        if i["xyz"] is not None:
            vi = build_viewpoint_from_position_and_guids(np.array(i["xyz"], float), *i["guids"])
            vi.guid = _uuid(f"bcf/viewpoint/{i['key']}")
            th.add_visinfo_handler(VisualizationInfoHandler(vi, xml_handler=xml))
        labels += [x for x in t.labels if x not in labels]
    doc.project_info.extension_schema = "extensions.xsd"
    doc.extension_schema = _extensions_xsd(labels)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        doc.save(tmp_path)
        _normalise_zip(tmp_path, path)
    finally:
        doc.close()
        tmp_path.unlink(missing_ok=True)
    return len(found)
