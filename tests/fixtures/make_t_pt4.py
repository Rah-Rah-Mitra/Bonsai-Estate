"""Build the representative PT4 test block build/t_pt4.ifc (12 storeys, four 4R-PT-A stacks, RC centre).

Several suites (validate, drawings, export, nav, blender) read this file; ``estate test`` builds it when it is missing.
Run on its own with: estate.cmd's python -I -B tests/fixtures/make_t_pt4.py [out.ifc]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from estate.env import bootstrap  # noqa: E402

bootstrap()

from estate import config, env  # noqa: E402

SPEC = dict(blk="900", storeys=12, stacks={s: {"t": "4R-PT-A"} for s in ("101", "103", "105", "107")},
            void_deck=["rc_centre"])
OUT = env.BUILD / "t_pt4.ifc"


def build(out: Path = OUT) -> Path:
    from estate.blocks.builder import build_building, derive_plan, plan_errors
    from estate.blocks.point import plan_point
    from estate.ifc.writer import IfcWriter
    plan = plan_point(dict(SPEC))
    dp = derive_plan(plan)
    errs = plan_errors(dp)
    if errs:
        raise SystemExit("t_pt4 plan errors: " + "; ".join(errs[:10]))
    W = IfcWriter(guid_key="test/900", typed_openings=True, **config.identity(config.load()))
    build_building(W, plan, dp)
    out.parent.mkdir(parents=True, exist_ok=True)
    W.write(out)
    W.close()
    return out


if __name__ == "__main__":
    t = time.time()
    p = build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
    print(f"built {env.rel(p)} in {time.time() - t:.1f} s")
