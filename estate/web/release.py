"""``estate release --tag vX.Y --out DIR``: the release zips of a finished build, deterministic and leak-scanned.

Writes DIR/SampleTownN5_<tag>_model.zip, DIR/SampleTownN5_<tag>_reports.zip and DIR/release_manifest.json
({tag, commit, head, zips: {name: {sha256, bytes}}, entries: {path: sha256}}): ``commit`` is the commit the export
was built on (export_info.json), ``head`` the checkout it was released from. Nothing is uploaded.

Only allowlisted files go in. Under model/: MODEL_ROOT, and for every site the estate manifest lists, BUILDING plus
the interior chunks its manifest entry lists (files.glb_int). Under reports/: the JSON reports at its top
(REPORTS) and the renders and camera views (RENDERS) that model/.state.json records as written by the current code
for those sites, the site and the estate; an image no current render wrote stays out. Never a .blend, a Bonsai link
cache (*.ifc.cache.*), .state.json, the nav/ maps or plans/. Entries are sorted, every one has the same fixed date
(bcf_out.ZIP_TIME), Unix mode 0644 and deflate level 9, so equal files make equal zips: two releases of one build
are the same bytes. Two builds of one commit are not: the nav and validation reports and reports/*.json record how
long their work took.

The release is refused when:
- model/export_info.json is missing or says the generator was dirty, or names a commit that is neither HEAD nor an
  ancestor of HEAD with the same generator (``git diff <commit> HEAD -- estate config estate.py estate.sh
  estate.cmd`` empty: committing the reports a build rewrote, as the release procedure does, is fine);
- the manifest or a walk grid / web JSON no longer has the hash export_info recorded, or a file the manifest lists
  no longer has the manifest's hash (a stale or edited export);
- a released file is not an output of the current code: the record of the stage that writes it in model/.state.json
  is missing or carries another code hash (state.code_hash), as after a partial build on older code; or the last
  build had failures (reports/build_failures.json not empty);
- a required file is missing (the IFC, LOD glbs, engine JSON, walk grid and web JSON of every site the manifest
  lists, the site files, and REQUIRED_REPORTS: the estate's camera views and the aerial a viewer's poster is made
  from), or an interior chunk on disk is not one the manifest lists;
- a stair path of a web JSON cannot be walked on the walk grid beside it as a viewer decodes it, or a floor of the
  grid lies outside its band (estate/web/walkcheck.py: every point within 0.1 m of a walkable cell of its storey
  band, a floor under every 0.05 m, the first and last points on the floors of their storeys);
- --out lies inside model/ or reports/ (a second release would take in the first one's output);
- the leak scan (estate/leaks.py) finds a machine path or the username in an entry, an entry's name or a zip
  written, or cannot read an entry.

Not attested beyond the leak scan: what reports/*.json say; and a stage that a partial build left on unchanged code
but older inputs. Release from a full build (./estate.sh build --force).
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

from estate import env

PREFIX = "SampleTownN5"
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MODE = 0o100644 << 16               # a regular file, rw-r--r--
# name: (required, the SITE stage that writes it, or None for the files cmd_build's tail writes)
MODEL_ROOT = {"estate_manifest.json": (True, None), "export_info.json": (True, None),
              "masterplan.json": (True, "ifc"), "SITE.ifc": (True, "ifc"), "SITE_ifc4.ifc": (False, "ifc4"),
              "SITE_lod0.glb": (True, "glb"), "SITE_lod1.glb": (True, "glb"), "SITE_engine.json": (True, "glb"),
              "SITE_graph.json": (True, "ifc")}
# per site of the manifest: (required, the stage that writes it)
BUILDING = {"{id}.ifc": (True, "ifc"), "{id}_ifc4.ifc": (False, "ifc4"), "{id}_lod0.glb": (True, "glb"),
            "{id}_lod1.glb": (True, "glb"), "{id}_lod2.glb": (True, "glb"), "{id}_engine.json": (True, "glb"),
            "{id}_flats.json": (False, "ifc"), "{id}_nav.json": (False, "nav"), "{id}_validation.json": (False, "check"),
            "{id}_walk.bin": (True, "web"), "{id}_web.json": (True, "web")}
REPORTS = ("*.json",)                                                   # relative to reports/
RENDERS = ("renders/*/*.png", "renders/*/*_views.json")                 # ... the render stage's outputs among these
REQUIRED_REPORTS = ("renders/ESTATE/ESTATE_views.json", "renders/ESTATE/ESTATE_aerial_NE.png")
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


def _manifest(model_dir: Path) -> dict:
    return json.loads((model_dir / "estate_manifest.json").read_text(encoding="utf-8"))


def model_entries(model_dir: Path) -> tuple[dict, dict]:
    """({zip path: file}, {zip path: (target, stage)}) of the model zip: the allowlisted files of the site and of
    every site in the manifest, with the stage that writes each one."""
    man = _manifest(model_dir)
    out, made_by, missing = {}, {}, []
    for name, (required, stage) in MODEL_ROOT.items():
        p = model_dir / name
        if p.is_file():
            out[f"model/{name}"] = p
            if stage:
                made_by[f"model/{name}"] = ("SITE", stage)
        elif required:
            missing.append(f"model/{name}")
    for s in man.get("sites", []):
        sid = s["id"]
        folder = model_dir / sid
        for pattern, (required, stage) in BUILDING.items():
            p = folder / pattern.format(id=sid)
            if p.is_file():
                out[f"model/{sid}/{p.name}"] = p
                made_by[f"model/{sid}/{p.name}"] = (sid, stage)
            elif required:
                missing.append(f"model/{sid}/{p.name}")
        chunk = re.compile(re.escape(sid) + r"_int_[A-Za-z0-9_\-]+\.glb")
        listed = set()
        for c in s.get("files", {}).get("glb_int", []):
            rel = PurePosixPath(c["path"])
            if rel.parent.as_posix() != sid or not chunk.fullmatch(rel.name):
                raise Refused(f"estate_manifest.json lists an interior chunk {c['path']!r} outside {sid}/")
            listed.add(rel.name)
            if (folder / rel.name).is_file():
                out[f"model/{rel}"] = folder / rel.name
                made_by[f"model/{rel}"] = (sid, "glb")
            else:
                missing.append(f"model/{rel}")
        stray = sorted(p.name for p in folder.glob(f"{sid}_int_*.glb") if p.name not in listed)
        if stray:
            raise Refused(f"interior chunks of {sid} that estate_manifest.json does not list (left by an older or "
                          f"failed export): {', '.join(stray[:8])}")
    if missing:
        raise Refused("missing from the build: " + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
    return out, made_by


def report_entries(reports_dir: Path, state: dict, targets) -> tuple[dict, dict]:
    """({zip path: file}, {zip path: (target, 'render')}) of the reports zip: the JSON reports, and the renders and
    camera views that the render records of ``targets`` in ``state`` (model/.state.json) list."""
    out, made_by = {}, {}
    for pattern in REPORTS:
        for p in sorted(reports_dir.glob(pattern)):
            if p.is_file():
                out["reports/" + p.relative_to(reports_dir).as_posix()] = p
    missing = []
    for target in targets:
        for o in (state.get(target, {}).get("render") or {}).get("outputs", []):
            rel = PurePosixPath(o)
            if rel.parts[:1] != ("reports",):
                continue
            rel = PurePosixPath(*rel.parts[1:])
            if not any(fnmatch.fnmatch(rel.as_posix(), pat) for pat in RENDERS):
                continue
            p = reports_dir.joinpath(*rel.parts)
            if p.is_file():
                out[f"reports/{rel}"] = p
                made_by[f"reports/{rel}"] = (target, "render")
            else:
                missing.append(f"reports/{rel}")
    missing += [f"reports/{r}" for r in REQUIRED_REPORTS if f"reports/{r}" not in out]
    if missing:
        raise Refused("missing from the build (render the estate: ./estate.sh build --stages estate): "
                      + ", ".join(sorted(set(missing))[:20]))
    return out, made_by


def check_export(model_dir: Path, root=None) -> dict:
    """export_info.json, checked against the checkout and against the files it hashes; returns it with ``head``."""
    from estate.web.info import SCOPE, git, sha256
    p = model_dir / "export_info.json"
    if not p.is_file():
        raise Refused("model/export_info.json missing: build first (./estate.sh build)")
    info = json.loads(p.read_text(encoding="utf-8"))
    if info.get("dirty") is not False:
        raise Refused(f"export_info.json: the generator was dirty at build time ({info.get('dirty_scope')})")
    head, commit = git("rev-parse", "HEAD", root=root), info.get("commit")
    if not head:
        raise Refused("not a git checkout: the release names the commit it was built on")
    if commit != head:
        if not commit or git("merge-base", "--is-ancestor", commit, "HEAD", root=root) is None:
            raise Refused(f"export_info.json names commit {commit}, which is not HEAD ({head}) or an ancestor of "
                          f"it: rebuild on HEAD")
        changed = git("diff", "--name-only", commit, "HEAD", "--", *SCOPE, root=root)
        if changed is None or changed:
            raise Refused(f"the generator changed between the export's commit {commit[:12]} and HEAD "
                          f"({(changed or '?').splitlines()[:5]}): rebuild on HEAD")
    man = model_dir / "estate_manifest.json"
    if not man.is_file() or sha256(man) != info.get("manifest_sha256"):
        raise Refused("estate_manifest.json is not the manifest export_info.json was written for")
    for rel, rec in info.get("files", {}).items():
        f = model_dir / rel
        if not f.is_file() or f.stat().st_size != rec["bytes"] or sha256(f) != rec["sha256"]:
            raise Refused(f"model/{rel} differs from export_info.json")
    return dict(info, head=head)


def check_current(made_by: dict, state: dict, reports_dir: Path) -> None:
    """Every released file comes from the current code (its stage's record in ``state`` carries the stage's
    current code hash), and the last build had no failures."""
    from estate.pipeline import state as st
    p = reports_dir / "build_failures.json"
    if not p.is_file():
        raise Refused("reports/build_failures.json missing: build first (./estate.sh build)")
    failures = json.loads(p.read_text(encoding="utf-8"))
    if failures:
        raise Refused(f"the last build had {len(failures)} failure(s) (reports/build_failures.json): fix and rebuild")
    stale = {}
    for name, (target, stage) in sorted(made_by.items()):
        rec = state.get(target, {}).get(stage)
        why = ("no record" if not rec else "older code" if rec.get("code") != st.code_hash(stage) else None)
        if why:
            stale.setdefault(f"{target} {stage} ({why})", []).append(name)
    if stale:
        raise Refused(f"{sum(map(len, stale.values()))} file(s) not made by the current code, per model/.state.json: "
                      + "; ".join(f"{k}: {v[0]}{' ...' if len(v) > 1 else ''}" for k, v in list(stale.items())[:12])
                      + " -- rebuild (./estate.sh build --force)")


def check_walks(model_dir: Path, sites) -> None:
    """Every stair path of every building's web JSON can be walked on the walk grid shipped beside it, read as a
    viewer reads it, and every floor of the grid lies in its band (estate/web/walkcheck.py site_errors: each point
    within 0.1 m of a walkable cell of its storey band, a floor under every 0.05 m of it, first and last points on
    their storeys' floors)."""
    from estate.web import walkcheck
    bad = []
    for sid in sites:
        folder = model_dir / sid
        try:
            web = json.loads((folder / f"{sid}_web.json").read_text(encoding="utf-8"))
            errs = walkcheck.site_errors((folder / f"{sid}_walk.bin").read_bytes(), web, limit=3)
        except (OSError, ValueError, KeyError, TypeError) as e:
            errs = [f"walk grid / web JSON unreadable ({type(e).__name__}: {e})"]
        bad += [f"{sid} {e}" for e in errs]
    if bad:
        raise Refused(f"stair paths a viewer cannot walk on the shipped walk grids: {'; '.join(bad[:8])}"
                      f"{' ...' if len(bad) > 8 else ''} -- rebuild the web stage")


def manifest_hashes(model_dir: Path) -> dict:
    """{zip path: sha256} of every file the estate manifest lists with a hash."""
    man = _manifest(model_dir)
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


def _inside(path: Path, folder: Path) -> bool:
    path, folder = path.resolve(), folder.resolve()
    return path == folder or folder in path.parents


def release(tag: str, out, model_dir=None, reports_dir=None, root=None, log=print) -> dict:
    """Write the two zips and release_manifest.json into ``out``; returns the release manifest. Raises Refused."""
    from estate import leaks
    if not TAG_RE.match(tag or ""):
        raise Refused(f"tag {tag!r}: letters, digits, '.', '_' and '-' only")
    model_dir = Path(model_dir) if model_dir else env.MODEL
    reports_dir = Path(reports_dir) if reports_dir else env.REPORTS
    out = Path(out)
    for folder in (model_dir, reports_dir):
        if _inside(out, folder):
            raise Refused(f"--out {out} lies inside {folder.name}/: a later release would take this one in")
    info = check_export(model_dir, root)
    sp = model_dir / ".state.json"
    state = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else {}
    model, made_by = model_entries(model_dir)
    targets = [s["id"] for s in _manifest(model_dir).get("sites", [])] + ["SITE", "ESTATE"]
    reports, rendered = report_entries(reports_dir, state, targets)
    check_current(dict(made_by, **rendered), state, reports_dir)
    zips = {f"{PREFIX}_{tag}_model.zip": model, f"{PREFIX}_{tag}_reports.zip": reports}
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
            findings += [str(f) for f in leaks.find(name, f"entry name {name}") + leaks.scan_bytes(data, name)]
    if findings:
        for f in findings[:40]:
            log(f"  leak: {f}")
        raise Refused(f"leak scan: {len(findings)} finding(s) in the release entries")
    check_walks(model_dir, targets[:-2])
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"tag": tag, "commit": info["commit"], "head": info["head"], "zips": {},
                "entries": dict(sorted(entries.items()))}
    for name, files in zips.items():
        write_zip(out / name, files)
        findings = [str(f) for f in leaks.scan_bytes((out / name).read_bytes(), name)]
        if findings:                          # the zip as written: names, comments and every member once more
            for z in zips:
                (out / z).unlink(missing_ok=True)
            for f in findings[:40]:
                log(f"  leak: {f}")
            raise Refused(f"leak scan of {name} as written: {len(findings)} finding(s)")
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
    print(f"release {m['tag']} of {m['commit'][:12]} (from {m['head'][:12]}): {len(m['entries'])} entries -> "
          f"{env.rel(Path(a.out) / 'release_manifest.json')}")
    return 0


def register(sub):
    p = sub.add_parser("release", help="deterministic, leak-scanned release zips of the current build (model and "
                                       "reports) plus release_manifest.json; uploads nothing")
    p.add_argument("--tag", required=True, help="release tag, e.g. v1.2 (names the zips)")
    p.add_argument("--out", required=True, help="folder for the zips and release_manifest.json (outside model/ and "
                                                "reports/)")
    p.set_defaults(fn=cmd_release)
