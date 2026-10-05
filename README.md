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
./estate.sh report          # reports/report.html, report.json and issues.bcf (also written at the end of every full build)
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
`legacy` (rebuilds the original block and compares it with `hdb_block.ifc`). `./estate.sh site` writes both site
plans (`--svg PATH` / `--no-svg` work like `--png PATH` / `--no-png`), and `./estate.sh render <blend> --street BS1`
renders an eye-level view from a bus stop.

## What gets built (`model/`)

| File | What it is |
|---|---|
| `BLK_5xx/BLK_5xx.ifc`, `MSCP_513/…`, `NC_514/…` | One IFC4X3 file per building, placed in estate coordinates. All files share the IfcProject/IfcSite GUIDs and an EPSG:3414 map conversion. |
| `*_ifc4.ifc` | IFC4 copies for Unreal Datasmith. Buildings are migrated from their IFC4X3 master (`ifcpatch` Migrate), so every GlobalId and name is identical. The site is authored again in IFC4, and its IFC4 federation links the `_ifc4` copies. |
| `*.blend` | A Bonsai project per building. Each stores a relative link to its IFC, which remains the source of truth. |
| `SITE.ifc`, `SITE.blend` | External works. Also holds LINKED_MODEL references to every building. |
| `ESTATE.blend` | The site with all 14 buildings linked through Bonsai (`bim.load_link`). The cache files `*.ifc.cache.*` sit next to each IFC. |
| `*_lod0/1/2.glb`, `*_engine.json`, `estate_manifest.json` | Game-engine export, block-local and Y-up, with the transform in the manifest. Doors (a frame plus a leaf child pivoting on the hinge), windows and lift cars are separate nodes that share one mesh per type. SITE is cut into 100 m tiles with instanced trees. The JSON holds doors with leaf and swing, lifts, rooms, flats, door portals with both sides resolved, climbable edges, and spawns on the pedestrian entrances. Repeated furniture (hawker tables, stools, counters) is instanced as `FURN_` nodes. LOD1 keeps the exterior doors and windows as instanced closed `L1DOOR_` / `L1WIN_` nodes, so it is smaller than LOD0. Spawns carry their room; a spawn inside an enclosed room is labelled 'Interior'. Bus-stop spawns stand on the platform graph node. |
| `*_int_<storey>.glb` | Per-storey interior chunks for engines that stream interiors floor by floor, next to each building's LOD files (`BLK_501_int_L05.glb` … `BLK_501_int_RF.glb`; storey numbers are zero-padded so the files sort bottom up). Each chunk holds exactly that storey's LOD0 nodes: the static batch, doors with their frame and leaves on the hinge, windows and furniture, with the same names and matrices, in the same block-local frame, so a chunk overlays LOD0/LOD1 with the manifest transform. Lift cars stay in LOD0 only, and SITE has no chunks. Each chunk is self-contained, so the chunks of a block take about 1.6x its LOD0 bytes. The engine JSON lists them under `interior_chunks`, and the manifest under each building's `files.glb_int` (storey, elevation, path, sha256, bytes), bottom up. |
| `*_validation.json`, `*_nav.json`, `nav/*.png` | Check results and walk-test maps. Each check lists its first examples and `items`: one {guids, xyz, text} per failure, giving the GlobalIds involved and a world point in metres (the element's placement, a point in the room at floor level, the door leaf, the middle of an unguarded edge at the slab top, or the centre of a clash). |
| `plans/<id>_{ground,typical,roof}.png` | Plans of the generator's floor plates. |
| `plans/<id>_<storey>_ifc.png`, `plans/<id>_stair_<n>.png` | L1, typical and roof plans cut from the IFC at FFL + 1.2 m, and one section per stair enclosure. These are also drawn for frozen (hand-edited) blocks, the MSCP and the NC. |
| `masterplan.json`, `SITE_graph.json` | Resolved footprints and entrances, and the pedestrian network. |

`reports/` holds:
- `report.html` and its JSON twin `report.json` (every section's data in page order, plus a `files` list of the
  artefacts it was read from, with size, sha256 and the sections each one feeds);
- `issues.bcf`, a BCF 2.1 issue file for Bonsai's BCF panel or any BCF viewer: one topic per failing validation item,
  titled `<file> <check>: <text>`, labelled with family / check / file, typed Error or Warning, with a viewpoint that
  selects the item's GlobalIds and looks at the defect. Topic GUIDs and dates are fixed, so re-importing after a
  rebuild updates topics instead of duplicating them;
- `site_plan.png` and `site_plan.svg`: the same plan, the SVG as an editable vector drawing in the PNG's pixel frame
  with one Inkscape/Illustrator layer per theme (ground, roads, greens, linkways, footprints, graph, canopies, trees,
  labels, legend), polygons with holes as single paths and every label as real text;
- `unit_schedule.csv` (one row per flat), `mix.csv` and the flat gallery;
- the renders (`renders/<target>/`). `renders/ESTATE/ESTATE_street_BS1.png` is an eye-level view from bus stop BS1:
  1.6 m above the platform just past the shelter, looking down Avenue 5 at Blk 501, with a level 24 mm camera and lens
  shift so verticals stay vertical. It is re-rendered whenever ESTATE.blend, the masterplan or the site graph moves
  the camera.

Generated outputs (`model/`, the renders and the gallery) are not tracked in git; each GitHub release carries them as
zips.

## How it is put together (`estate/`)

- `config/estate.toml` is the masterplan: blocks, stacks and sequences, roads, greens, bus stops and the expected mix.
- `config/flats/*.toml` holds 18 flat templates: 2-room Flexi, 3-room, 4-room, 5-room, 3Gen and Executive. Each is a grid layout with seeded variants (mirror, open kitchen, AC ledge / balcony / none, width stretch).
- `config/rules.toml` holds the design-rule thresholds.
- `blocks/`: each typology (`point.py`, `slab.py`, `lblock.py`) describes ground, typical and roof *plates* as rooms, cores and corridors. `builder.py` derives the walls from what sits on either side of each boundary (`geom/arrangement.py`), checks every door and window against them, and writes IFC through `ifc/writer.py`, `ifc/stairs.py` and `ifc/joins.py` (wall joins and door / window parameters for Bonsai).
- `site/`: SITE.ifc, roads, linkways, landscape, the MSCP and the NC. The site reads every building's ground floor from its IFC, so linkways and indoor routes go around walls, columns and furniture. Build the buildings first; `estate build` does this in the right order. A paved escape
  path leads from each published stair discharge (the MSCP's stair 3) to the nearest footway, shown as an 'escape'
  edge in SITE_graph.json. `siteplan.py` draws the site plan once, through a backend interface: PIL writes
  `site_plan.png` and svgwrite writes `site_plan.svg`.
- `draw/`: `raster.py` draws the generator's plates. `sections.py` and `iplans.py` draw stair sections and floor plans cut from the IFC.
- `validate/`:
  - `ifcqa.py`: schema and EXPRESS rules, IDS, containment, openings, psets, IFA bands.
  - `programme.py`: the flat programme read back from the IFC alone: required rooms, net areas and widths, shelter by IFA band, glazing ≥ 10%, bathroom vents or declared mechanical ventilation, door widths, and a main door to a common space.
  - `geometry.py`: stairs re-measured from the solids, clashes, shafts, wet stacks, falling edges.
  - `nav3d.py`: the voxel walk test per building. Door linings are kept as obstacles, so 0.8 m doors are tested at 0.70 m clear, and every door is re-tested on 0.025 m voxels. It also checks step-free access by lift into each flat's entry room.
  - `nav2d.py` and `navgraph.py`: bus stop to every flat. The pedestrian graph is used only if it was written together with the current `SITE.ifc` (sha256 check). It is checked against the IFC: crossings need a zebra and kerb ramps, and covered edges must lie under a roof. Otherwise the test falls back to a grid on `SITE.ifc`.
  - `mutations.py`: injected defects (IFC, programme and geometry families) that the checks must catch.
- `export/`: tessellation cache, a glTF writer (no trimesh), engine JSON with LOD0-2 and per-storey interior chunks, and the manifest.
- `blender/`: headless Bonsai scripts (`make_blend`, `make_estate`, `verify_blend`, `render`) driven by `run.py`. `render` also takes placed perspective cameras (`cameras: {view: {eye, target, lens, clip, shift, res, sky}}`).
- `report/`: `html.py` reads every section's data from the artefacts (`collect()`, fixed key order, no clock) and renders `report.html`; the same dict is written as `report.json`. `bcf_out.py` turns the validation items into `issues.bcf` with the bcf library's v2 API; GUIDs come from `estate.guids.from_text`, and dates and zip entry times are fixed, so two writes are byte-identical. `street.py` (no bpy) places the bus-stop camera from `masterplan.json` and `SITE_graph.json`. It sits outside every stage's code hash, so tuning it re-renders only ESTATE (the camera is in that render key).
- `pipeline/`: targets, incremental state (`model/.state.json`) and the build stages.

Builds are deterministic and IDs are stable. Every named element takes its GlobalId from a stable key (building + class
+ name, or a wall's storey, type and endpoints), so an element keeps its GUID when unrelated parts of the config change.
Wall joins take theirs from the relating wall's GlobalId and end, and each wall's material usage from its wall.
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

Walls, doors and windows are written for Bonsai's parametric tools (`estate/ifc/joins.py`):
- Every wall has its own `IfcMaterialLayerSetUsage` (AXIS2, layers centred on the reference line) and a
  `Plan/Axis/GRAPH_VIEW` reference line between the plan nodes. The body keeps its corner extensions.
- Wall joins are `IfcRelConnectsPathElements`. L corners join end to end, and T junctions join the stem's end to the
  bar (ATPATH). Collinear continuations and crossings have no join. Bonsai rebuilds a corner in plan and extrudes each
  wall to its own height, so only walls of one height are joined; the exception is a T on a taller wall, such as a
  roof parapet on a stair or lift tower. Where a wall is continued by a lower parapet (lobby corners, tower walls on
  the roof), the corner is an L with the wall of the same height. Layer priorities on the wall types (core 90,
  shelter 80, external / party 70, parapet 60, partitions 30) decide which wall runs through a corner.
- Every door and window type has a `BBIM_Door` / `BBIM_Window` pset, and its geometry is built from it. The hawker
  stalls' roller shutters have none, because Bonsai's door tool cannot rebuild a rolling shutter.

So Bonsai's wall tools (*Recalculate Wall*, extend, join) and its door and window editors work on the generated
elements. When Bonsai regenerates every wall of a storey, every wall section comes back as generated, at every height
(checked on all blocks, the car park and the centre). *Recalculate Wall* rebuilds a wall and the walls joined to it;
at a corner where a wall is continued by a thinner or lower one (household shelter and lobby corners), it can leave a
small triangle (up to 0.005 m² in plan) until the continued wall is recalculated too. Bonsai's door and window editors
replace the type's representation, so the generated per-item glass / frame colours give way to the type material.
The legacy baseline (`estate legacy`) is written without this data and stays byte-identical.

## Licence

The code is MIT licensed ([`LICENSE`](LICENSE)); that includes `legacy/` and `hdb_block_toolkit.zip`. The generated
model data (`model/`, the renders and artefacts under `reports/`, the release zips, and the baseline block's model
files and drawings at the root) is CC BY 4.0, credited as "Sample Town N5 © Rahul Mitra, CC BY 4.0, generated by
Bonsai-Estate"; [`LICENSE-DATA.md`](LICENSE-DATA.md) lists exactly what it covers.
