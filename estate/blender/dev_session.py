"""Open a building .blend (or an IFC) in the Blender GUI for manual editing with Bonsai.

Usage, from the project root:
    estate.cmd dev model/BLK_507/BLK_507.blend      (Git Bash: ./estate.sh dev ...)
    estate.cmd dev model/BLK_507/BLK_507.ifc        opens the sibling .blend if there is one, else loads the IFC
or directly:
    blender.exe --python estate/blender/dev_session.py -- --json "{\"path\": \"model/BLK_507/BLK_507.blend\"}"

Blender starts with the user's own preferences (no --factory-startup); Bonsai is enabled if it is not already.
The file is opened from a timer once the window exists, because operators that read or replace the file need a
full UI context.

Hand-edit workflow (the pipeline must not overwrite the edits):
  1. Edit with Bonsai's tools (they write to the IFC in memory; Blender-only edits are lost on reload).
  2. Bonsai > Project > Save Project (bim.save_project): writes the IFC in place, path stays relative.
  3. File > Save (Ctrl+S): saves the .blend. Bonsai relinks objects by STEP id when the .blend reopens, so
     always save both together.
  4. Set ``frozen = true`` on that [[block]] in config/estate.toml so ``estate build`` never regenerates it,
     ``estate build`` then follows the saved IFC: the IFC4 copy, IFC-cut drawings, checks, walk test, export,
     .blend and the site's ground routing are all keyed on the IFC's bytes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True         # Blender ignores -B; keep the project free of __pycache__
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from estate.blender import _boot  # noqa: E402

HELP = __doc__.split("Hand-edit workflow")[1]


def resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (_boot.ROOT / p) if not p.exists() else p.resolve()
    return p.resolve()


def open_target(a: dict):
    import bpy
    _boot.boot(set_prefs=False)
    target = resolve(a["path"])
    if target.suffix.lower() == ".ifc" and target.with_suffix(".blend").exists() and not a.get("ifc_only"):
        target = target.with_suffix(".blend")
    if not target.exists():
        print(f"[estate dev] no such file: {target}")
        return None
    if target.suffix.lower() == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(target))
    else:
        bpy.ops.bim.load_project(filepath=str(target), skip_recent=False)
    print(f"[estate dev] opened {target}\n[estate dev] hand-edit workflow:{HELP}")
    return None       # unregister the timer


def main():
    import bpy
    a = _boot.args()
    if not a.get("path"):
        print(__doc__)
        return
    bpy.app.timers.register(lambda: open_target(a), first_interval=0.5)


if __name__ == "__main__":
    main()
