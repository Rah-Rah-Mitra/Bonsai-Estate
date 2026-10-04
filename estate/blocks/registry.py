"""Typology registry: typology name -> planner(spec, estate_seed, cat) -> BuildingPlan.

Planners return a BuildingPlan in block-local coordinates (see blocks/builder.py). Conventions every planner
follows so the masterplan, site works and engine export can rely on them:
  plan.meta["lift_lobbies"]  list of (x, y) block-local points at the void deck where the lifts are reached
  plan.meta["entrances"]     list of (x, y) points on the footprint edge where linkways should land
  plan.footprint             typical-floor outline (walls included)
"""
from __future__ import annotations

import importlib

PLANNERS = {
    "PT4": ("estate.blocks.point", "plan_point"),
    "SL": ("estate.blocks.slab", "plan_slab"),
    "LB": ("estate.blocks.lblock", "plan_lblock"),
}


def planner(typology: str):
    mod, fn = PLANNERS[typology]
    return getattr(importlib.import_module(mod), fn)


def available(typology: str) -> bool:
    try:
        planner(typology)
        return True
    except (ImportError, AttributeError):
        return False
