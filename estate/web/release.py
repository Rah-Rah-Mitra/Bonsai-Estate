"""``estate release --tag vX.Y --out DIR``: the release zips of a finished build, deterministic and leak-scanned.

Writes DIR/SampleTownN5_<tag>_model.zip, DIR/SampleTownN5_<tag>_reports.zip and DIR/release_manifest.json
({tag, commit, zips: {name: {sha256, bytes}}, entries: {path: sha256}}). Nothing is uploaded.

Only allowlisted files go in (MODEL_ROOT and BUILDING under model/, REPORTS under reports/): no .blend, no Bonsai
link caches (*.ifc.cache.*), no .state.json, no nav/ maps and no plans/. Entries are sorted, every one has the same
fixed date (bcf_out.ZIP_TIME), Unix mode 0644 and deflate level 9, so equal files make equal zips.

The release is refused when:
- model/export_info.json is missing, says the generator was dirty, or names another commit than HEAD;
- the manifest or a walk grid / web JSON no longer has the hash export_info recorded, or a file the manifest lists
  no longer has the manifest's hash (a stale or edited export);
- a required file is missing (the IFC, LOD glbs, engine JSON, walk grid and web JSON of every site the manifest
  lists, and the site files);
- the leak scan (estate/leaks.py) finds a machine path or the username in any entry, or cannot read one.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import zipfile
from pathlib import Path

from estate import env

PREFIX = "SampleTownN5"
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MODE = 0o100644 << 16               # a regular file, rw-r--r--
MODEL_ROOT = {"estate_manifest.json": True, "export_info.json": True, "masterplan.json": True, "SITE.ifc": True,
              "SITE_ifc4.ifc": False, "SITE_lod0.glb": True, "SITE_lod1.glb": True, "SITE_engine.json": True,
              "SITE_graph.json": True}                                         # name: required
BUILDING = {"{id}.ifc": True, "{id}_ifc4.ifc": False, "{id}_lod0.glb": True, "{id}_lod1.glb": True,
            "{id}_lod2.glb": True, "{id}_engine.json": True, "{id}_flats.json": False, "{id}_nav.json": False,
            "{id}_validation.json": False, "{id}_walk.bin": True, "{id}_web.json": True}
CHUNK = "{id}_int_*.glb"
REPORTS = ("renders/**/*.png", "renders/**/*_views.json", "*.json")     # relative to reports/
NEVER = ("*.blend", "*.blend1", "*.ifc.cache.*", ".state.json", "*/nav/*", "*/plans/*", "*.tmp*")


class Refused(RuntimeError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _files_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_entries(model_dir: Path) -> dict:
    """{zip path: file} of the model zip: the allowlisted files of the site and of every site in the manifest."""
    man = json.loads((model_dir / "estate_manifest.json").read_text(encoding="utf-8"))
    out, missing = {}, []
    for name, required in MODEL_ROOT.items():
        p = model_dir / name
        if p.is_file():
            out[f"model/{name}"] = p
        elif required:
            missing.append(f"model/{name}")
    for s in man.get("sites", []):
        sid = s["id"]
        folder = model_dir / sid
        for pattern, required in BUILDING.items():
            p = folder / pattern.format(id=sid)
            if p.is_file():
                out[f"model/{sid}/{p.name}"] = p
            elif required:
                missing.append(f"model/{sid}/{p.name}")
        chunk = re.compile(re.escape(sid) + r"_int_[A-Za-z0-9_\-]+\.glb")
        for p in sorted(folder.glob(CHUNK.format(id=sid))):
            if chunk.fullmatch(p.name):
                out[f"model/{sid}/{p.name}"] = p
    if missing:
        raise Refused("missing from the build: " + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
    return out


def report_entries(reports_dir: Path) -> dict:
    out = {}
    for pattern in REPORTS:
        for p in sorted(reports_dir.glob(pattern)):
            if p.is_file():
                out["reports/" + p.relative_to(reports_dir).as_posix()] = p
    return out


def check_export(model_dir: Path, root=None) -> dict:
    """export_info.json, checked against HEAD and against the files it hashes; returns it."""
    from estate.web.info import git, sha256
    p = model_dir / "export_info.json"
    if not p.is_file():
        raise Refused("model/export_info.json missing: build first (./estate.sh build)")
    info = json.loads(p.read_text(encoding="utf-8"))
    if info.get("dirty") is not False:
        raise Refused(f"export_info.json: the generator was dirty at build time ({info.get('dirty_scope')})")
    head = git("rev-parse", "HEAD", root=root)
    if not head or info.get("commit") != head:
        raise Refused(f"export_info.json names commit {info.get('commit')}, HEAD is {head}: rebuild on HEAD")
    man = model_dir / "estate_manifest.json"
    if not man.is_file() or sha256(man) != info.get("manifest_sha256"):
        raise Refused("estate_manifest.json is not the manifest export_info.json was written for")
    for rel, rec in info.get("files", {}).items():
        f = model_dir / rel
        if not f.is_file() or f.stat().st_size != rec["bytes"] or sha256(f) != rec["sha256"]:
            raise Refused(f"model/{rel} differs from export_info.json")
    return info


def manifest_hashes(model_dir: Path) -> dict:
    """{zip path: sha256} of every file the estate manifest lists with a hash."""
    man = json.loads((model_dir / "estate_manifest.json").read_text(encoding="utf-8"))
    out = {}

    def take(rec):
        if isinstance(rec, dict) and "path" in rec and "sha256" in rec:
            out["model/" + rec["path"]] = rec["sha256"]
        elif isinstance(rec, list):
            for r in rec:
                take(r)
    for files in [s.get("files", {}) for s in man.get("sites", [])] + [man.get("site", {}).get("files", {})]:
        for rec in files.values():
            take(rec)
    take(man.get("pedestrian_graph_file"))
    return out


def write_zip(path: Path, entries: dict) -> None:
    """A zip of ``entries`` ({zip path: file}) in name order, fixed time and mode, deflate level 9."""
    from estate.report.bcf_out import ZIP_TIME
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w") as z:
        for name in sorted(entries):
            zi = zipfile.ZipInfo(name, date_time=ZIP_TIME)
            zi.create_system = 3
            zi.external_attr = MODE
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, entries[name].read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    tmp.replace(path)


def release(tag: str, out, model_dir=None, reports_dir=None, root=None, log=print) -> dict:
    """Write the two zips and release_manifest.json into ``out``; returns the release manifest. Raises Refused."""
    from estate import leaks
    if not TAG_RE.match(tag or ""):
        raise Refused(f"tag {tag!r}: letters, digits, '.', '_' and '-' only")
    model_dir = Path(model_dir) if model_dir else env.MODEL
    reports_dir = Path(reports_dir) if reports_dir else env.REPORTS
    info = check_export(model_dir, root)
    zips = {f"{PREFIX}_{tag}_model.zip": model_entries(model_dir),
            f"{PREFIX}_{tag}_reports.zip": report_entries(reports_dir)}
    listed = manifest_hashes(model_dir)
    entries, findings = {}, []
    for files in zips.values():
        for name in sorted(files):
            if any(fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(files[name].name, pat) for pat in NEVER):
                raise Refused(f"{name} is never released")
            data = files[name].read_bytes()
            entries[name] = _sha(data)
            if name in listed and listed[name] != entries[name]:
                raise Refused(f"{name} differs from estate_manifest.json: rebuild")
            findings += [str(f) for f in leaks.scan_bytes(data, name)]
    if findings:
        for f in findings[:40]:
            log(f"  leak: {f}")
        raise Refused(f"leak scan: {len(findings)} finding(s) in the release entries")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"tag": tag, "commit": info["commit"], "zips": {}, "entries": dict(sorted(entries.items()))}
    for name, files in zips.items():
        write_zip(out / name, files)
        sha, size = _files_sha(out / name), (out / name).stat().st_size
        manifest["zips"][name] = {"sha256": sha, "bytes": size}
        log(f"  {name}: {len(files)} entries, {size / 1e6:.1f} MB, sha256 {sha[:16]}")
    (out / "release_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def cmd_release(a):
    try:
        m = release(a.tag, a.out)
    except Refused as e:
        print(f"release refused: {e}")
        return 1
    print(f"release {m['tag']} of {m['commit'][:12]}: {len(m['entries'])} entries -> "
          f"{env.rel(Path(a.out) / 'release_manifest.json')}")
    return 0


def register(sub):
    p = sub.add_parser("release", help="deterministic, leak-scanned release zips of the current build (model and "
                                       "reports) plus release_manifest.json; uploads nothing")
    p.add_argument("--tag", required=True, help="release tag, e.g. v1.2 (names the zips)")
    p.add_argument("--out", required=True, help="folder for the zips and release_manifest.json")
    p.set_defaults(fn=cmd_release)
