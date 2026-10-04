# Sample Town N5 — a walkable HDB neighbourhood as BIM

A generated Singapore HDB neighbourhood (about 400 x 400 m) authored as federated IFC4X3 with IfcOpenShell 0.9 (the
library underneath Bonsai) and delivered as Bonsai `.blend` files in this folder. It contains:

- 12 residential blocks with 1,206 flats in three typologies: 6 point blocks (PT4), 4 corridor slab blocks (SL) and 2 L-blocks (LB).
- A multi-storey car park (MSCP 513) and a neighbourhood centre with a hawker centre and shops (NC 514).
- Roads, covered linkways, greens and bus stops.

Every room of every flat is modelled and walkable for a person-sized agent. The original single-block toolkit
(`hdb_block*.*`, `hdb_block_toolkit.zip`, unpacked in `legacy/toolkit/`) is kept unchanged as the regression baseline.

## Run it

Everything runs on Blender 5.2's bundled Python (`estate.cmd` on Windows, `./estate.sh` in Git Bash):

```bash
./estate.sh doctor          # versions, paths, Bonsai, process pool
./estate.sh catalogue       # check all flat templates + render reports/flat_gallery/
./estate.sh plan            # masterplan, unit mix, layout variety, siting, plan-level checks (no IFC)
./estate.sh build           # all stages: ifc, drawings, check, nav, glb, blend, render, estate (incremental)
./estate.sh report          # reports/report.html (also written at the end of every full build)
./estate.sh test            # unit tests
./estate.sh clean           # prune caches and temp files (--all-caches, --link-caches)
```

`build` options:
- `--only BLK_507 SITE` rebuilds only the named targets.
- `--stages ifc,check` runs only the named stages.
- `--force` rebuilds stages that are up to date.
- `--jobs N` and `--blender-jobs N` set the parallelism.

A full cold build takes about 15 minutes on this machine. A no-change rebuild skips every stage except the site walk
test. Stages switched off in `config/estate.toml` `[outputs]` are left out unless named with `--stages`. A failing
target or a crashed worker fails its stage, not the build; `reports/build_failures.json` lists the failures and the
report shows them first.

The area commands can also be run on their own: `site`, `nonres`, `validate`, `mutate`, `nav`, `export`, `blend`,
`estate-blend`, `render`, `dev` (opens a `.blend` or IFC in the Blender GUI for hand edits with Bonsai), and
`legacy` (rebuilds the original block and compares it with `hdb_block.ifc`).

## What gets built (`model/`)

| File | What it is |
|---|---|
| `BLK_5xx/BLK_5xx.ifc`, `MSCP_513/…`, `NC_514/…` | One IFC4X3 file per building, placed in estate coordinates. All files share the IfcProject/IfcSite GUIDs and an EPSG:3414 map conversion. |
| `*_ifc4.ifc` | IFC4 copies for Unreal Datasmith. Buildings are migrated from their IFC4X3 master (`ifcpatch` Migrate), so every GlobalId and name is identical. The site is authored again in IFC4, and its IFC4 federation links the `_ifc4` copies. |
| `*.blend` | A Bonsai project per building. Each stores a relative link to its IFC, which remains the source of truth. |
| `SITE.ifc`, `SITE.blend` | External works. Also holds LINKED_MODEL references to every building. |
| `ESTATE.blend` | The site with all 14 buildings linked through Bonsai (`bim.load_link`). The cache files `*.ifc.cache.*` sit next to each IFC. |
| `*_lod0/1/2.glb`, `*_engine.json`, `estate_manifest.json` | Game-engine export, block-local and Y-up, with the transform in the manifest. Doors (a frame plus a leaf child pivoting on the hinge), windows and lift cars are separate nodes that share one mesh per type. SITE is cut into 100 m tiles with instanced trees. The JSON holds doors with leaf and swing, lifts, rooms, flats, door portals with both sides resolved, climbable edges, and spawns on the pedestrian entrances. Repeated furniture (hawker tables, stools, counters) is instanced as `FURN_` nodes. LOD1 keeps the exterior doors and windows as instanced closed `L1DOOR_` / `L1WIN_` nodes, so it is smaller than LOD0. Spawns carry their room; a spawn inside an enclosed room is labelled 'Interior'. Bus-stop spawns stand on the platform graph node. |
| `*_validation.json`, `*_nav.json`, `nav/*.png` | Check results and walk-test maps. |
| `plans/<id>_{ground,typical,roof}.png` | Plans of the generator's floor plates. |
| `plans/<id>_<storey>_ifc.png`, `plans/<id>_stair_<n>.png` | L1, typical and roof plans cut from the IFC at FFL + 1.2 m, and one section per stair enclosure. These are also drawn for frozen (hand-edited) blocks, the MSCP and the NC. |
| `masterplan.json`, `SITE_graph.json` | Resolved footprints and entrances, and the pedestrian network. |

`reports/` holds `report.html`, `unit_schedule.csv` (one row per flat), `mix.csv`, `site_plan.png`, the flat
gallery and the renders.

## How it is put together (`estate/`)

- `config/estate.toml` is the masterplan: blocks, stacks and sequences, roads, greens, bus stops and the expected mix.
- `config/flats/*.toml` holds 18 flat templates: 2-room Flexi, 3-room, 4-room, 5-room, 3Gen and Executive. Each is a grid layout with seeded variants (mirror, open kitchen, AC ledge / balcony / none, width stretch).
- `config/rules.toml` holds the design-rule thresholds.
- `blocks/`: each typology (`point.py`, `slab.py`, `lblock.py`) describes ground, typical and roof *plates* as rooms, cores and corridors. `builder.py` derives the walls from what sits on either side of each boundary (`geom/arrangement.py`), checks every door and window against them, and writes IFC through `ifc/writer.py` and `ifc/stairs.py`.
- `site/`: SITE.ifc, roads, linkways, landscape, the MSCP and the NC. The site reads every building's ground floor from its IFC, so linkways and indoor routes go around walls, columns and furniture. Build the buildings first; `estate build` does this in the right order. A paved escape
  path leads from each published stair discharge (the MSCP's stair 3) to the nearest footway, shown as an 'escape'
  edge in SITE_graph.json.
- `draw/`: `raster.py` draws the generator's plates. `sections.py` and `iplans.py` draw stair sections and floor plans cut from the IFC.
- `validate/`:
  - `ifcqa.py`: schema and EXPRESS rules, IDS, containment, openings, psets, IFA bands.
  - `programme.py`: the flat programme read back from the IFC alone: required rooms, net areas and widths, shelter by IFA band, glazing ≥ 10%, bathroom vents or declared mechanical ventilation, door widths, and a main door to a common space.
  - `geometry.py`: stairs re-measured from the solids, clashes, shafts, wet stacks, falling edges.
  - `nav3d.py`: the voxel walk test per building. Door linings are kept as obstacles, so 0.8 m doors are tested at 0.70 m clear, and every door is re-tested on 0.025 m voxels. It also checks step-free access by lift into each flat's entry room.
  - `nav2d.py` and `navgraph.py`: bus stop to every flat. The pedestrian graph is used only if it was written together with the current `SITE.ifc` (sha256 check). It is checked against the IFC: crossings need a zebra and kerb ramps, and covered edges must lie under a roof. Otherwise the test falls back to a grid on `SITE.ifc`.
  - `mutations.py`: injected defects (IFC, programme and geometry families) that the checks must catch.
- `export/`: tessellation cache, a glTF writer (no trimesh), engine JSON and the manifest.
- `blender/`: headless Bonsai scripts (`make_blend`, `make_estate`, `verify_blend`, `render`) driven by `run.py`.
- `pipeline/`: targets, incremental state (`model/.state.json`) and the build stages.

Builds are deterministic and IDs are stable. Every named element takes its GlobalId from a stable key (building + class
+ name, or a wall's storey, type and endpoints), so an element keeps its GUID when unrelated parts of the config change.
The GUID is also the same in the IFC4 copy. Set iteration is made stable, so the same config produces byte-identical IFC.

## Editing in Bonsai

Run `./estate.sh dev model/BLK_507/BLK_507.blend` and edit the block in Bonsai. Save with Bonsai's *Save Project*,
then set `frozen = true` on that `[[block]]` in `config/estate.toml`.

From then on `estate build` no longer regenerates that block's IFC4X3 master. Everything downstream still follows your
edits:
- the IFC4 copy (migrated from the master),
- the IFC-cut plans and stair sections,
- validation, the walk test and the engine export,
- the `.blend` and the site's ground-floor routing.

Only the config-plate plans are skipped. Any other change belongs in the config or a template, followed by
`./estate.sh build`.

## Not delivered (planned extras)

These were in the plan but are not built:
- per-storey interior glb chunks;
- a BCF issue file and a JSON twin of the report;
- an SVG site plan (the site plan is a PNG);
- a street-level render from a bus stop.

Bonsai's parametric editing data is also not written: wall joins (IfcRelConnectsPathElements) and the BBIM_Door /
BBIM_Window parameters. The walls, doors and windows are plain IFC elements: Bonsai edits them as such, but not
through its parametric wall and door tools.
