"""Command line: estate.cmd <command> [options]."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from estate import env

BASELINE_CLASSES = ("IfcWall", "IfcDoor", "IfcWindow", "IfcSpace", "IfcZone", "IfcSlab", "IfcStair", "IfcStairFlight",
                    "IfcRailing", "IfcColumn", "IfcBuildingStorey", "IfcOpeningElement", "IfcTransportElement")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd_doctor(a):
    import platform
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {label}{(': ' + str(detail)) if detail else ''}")

    print("estate doctor")
    check("python 3.13", sys.version_info[:2] == env.PINS["python"], platform.python_version())
    check("isolated mode", sys.flags.isolated == 1)
    check("Blender executable", env.BLENDER_EXE.exists(), env.BLENDER_EXE)
    check("Bonsai site-packages", env.BONSAI_SITE.exists(), env.BONSAI_SITE)
    try:
        import ifcopenshell
        check("ifcopenshell pin", ifcopenshell.version == env.PINS["ifcopenshell"], ifcopenshell.version)
    except Exception as e:  # noqa: BLE001
        check("ifcopenshell import", False, e)
    for lib in ("shapely", "numpy", "networkx", "PIL", "ifctester", "ifcpatch", "svgwrite", "tomli_w"):
        try:
            mod = __import__(lib)
            check(lib, True, getattr(mod, "__version__", "ok"))
        except Exception as e:  # noqa: BLE001
            check(lib, False, e)
    try:
        import scipy  # noqa: F401
        print(f"  [ok] scipy (optional) {scipy.__version__}")
    except Exception:  # noqa: BLE001
        print("  [--] scipy (optional) not installed: nav3d uses the numpy fallback")
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    t = time.time()
    with ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as ex:
        r = list(ex.map(pow, [2, 3], [10, 10]))
    check("spawn process pool", r == [1024, 59049], f"{time.time() - t:.1f} s")
    return 0 if ok else 1


def count_classes(f, classes=BASELINE_CLASSES):
    return {c: len(f.by_type(c)) for c in classes}


def cmd_legacy(a):
    import ifcopenshell
    from estate.blocks.legacy import build_legacy
    out_dir = env.BUILD / "legacy"
    out_dir.mkdir(parents=True, exist_ok=True)
    shas = []
    for run in range(2 if a.twice else 1):
        t = time.time()
        W, names, ffl = build_legacy(storeys=a.storeys, schema=a.schema)
        out = (out_dir if run == 0 else out_dir / "run2") / "hdb_block_legacy.ifc"   # same name: it is in the header
        out.parent.mkdir(parents=True, exist_ok=True)
        W.write(out)
        W.close()
        shas.append(_sha(out))
        print(f"built {env.rel(out)} in {time.time() - t:.1f} s, sha {shas[-1][:12]}")
    new = count_classes(ifcopenshell.open(str(out_dir / "hdb_block_legacy.ifc")))
    base = count_classes(ifcopenshell.open(str(env.ROOT / "hdb_block.ifc")))
    bad = 0
    print(f"  {'class':<22}{'baseline':>9}{'rebuilt':>9}")
    for c in BASELINE_CLASSES:
        flag = "" if new[c] == base[c] else "   <-- differs"
        bad += bool(flag)
        print(f"  {c:<22}{base[c]:>9}{new[c]:>9}{flag}")
    if a.twice:
        same = shas[0] == shas[1]
        print(f"  deterministic: {'yes' if same else 'NO'}")
        bad += not same
        (out_dir / "run2" / "hdb_block_legacy.ifc").unlink(missing_ok=True)
    (out_dir / "legacy_counts.json").write_text(json.dumps({"baseline": base, "rebuilt": new, "sha": shas[0]}, indent=1))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="estate", description="Sample Town N5 estate pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="check versions, paths and the process pool").set_defaults(fn=cmd_doctor)
    p = sub.add_parser("legacy", help="rebuild the baseline Blk 123 and compare with hdb_block.ifc")
    p.add_argument("--storeys", type=int, default=12)
    p.add_argument("--schema", default="IFC4X3")
    p.add_argument("--twice", action="store_true", help="build twice and require identical bytes")
    p.set_defaults(fn=cmd_legacy)
    from estate import commands
    commands.register(sub)
    a = ap.parse_args(argv)
    return a.fn(a)
