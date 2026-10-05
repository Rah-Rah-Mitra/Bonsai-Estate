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
./estate.sh build           # all stages: ifc, drawings, check, nav, glb, blend, render, estate, web (incremental)
./estate.sh report          # reports/report.html, report.json and issues.bcf (also written at the end of every full build)
./estate.sh release --tag v1.2 --out DIR   # the release zips of the current build (see "Web export"); uploads nothing
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
| `*_walk.bin`, `*_web.json`, `export_info.json` | The web export for a browser viewer: per-storey walk grids (SN5W), stairs with walking paths, and the opened pose of every passable door leaf, per building; and the provenance of the whole export. See "Web export". |

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

Generated outputs (`model/`, the renders and the gallery) are not tracked in git. From v1.2 on, each GitHub release
carries them as zips; the v1.0 and v1.1 zips were withdrawn because they held absolute paths of the build machine.

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
- `web/`: the `web` stage (walk grids, stairs, door poses: `walk.py`, `sn5w.py`, `stairs.py`, `export.py`, which
  with nav3d, the door poses and the mesh cache are its code key), `export_info.json` (`info.py`) and the `release`
  command (`release.py`), which rerun nothing when edited. The door-leaf pose formula has one home,
  `validate/nav_doorpose.py`, which the engine export imports too.
- `pipeline/`: targets, incremental state (`model/.state.json`) and the build stages.

Builds are deterministic and IDs are stable. Every named element takes its GlobalId from a stable key (building + class
+ name, or a wall's storey, type and endpoints), so an element keeps its GUID when unrelated parts of the config change.
Wall joins take theirs from the relating wall's GlobalId and end, and each wall's material usage from its wall.
The GUID is also the same in the IFC4 copy. Set iteration is made stable, so the same config produces byte-identical IFC.

## Web export

The `web` stage runs after `estate` (`[outputs] web = true`, settings in `[web]` of `config/estate.toml`) and writes
two files per building from its IFC and engine JSON; the site has none (a viewer samples its ground).

**`<ID>_walk.bin`: SN5W walk grids.** nav3d's walk test (`estate/validate/nav3d.py`, unchanged) run again for a
0.20 m agent on 0.10 m cells: the same voxeliser and solid, the same walkability test and fine door test. Then:
doorways the door test passes but the 0.1 m grid misses are stamped walkable; every passable leaf is blocked in its
opened pose, dilated by the radius (a leaf that would cut its doorway goes through, `leaf_passthrough`; one whose
full swing would cut off part of a room, as judged from the way through its opening, opens less, `leaf_narrowed`,
e.g. a main door swinging over a household shelter door, or a void-deck stair door over the foot of the flight);
only cells reachable from the street are kept. Storey s owns the floors from FFL_s - 0.25 m to FFL_s+1 - 0.25 m
(RF everything above), decided in whole millimetres; in each cell the floor closest to FFL_s is the layer's raster
value and any other floor in that band an overflow record. Coordinates are block-local, heights int16 mm above the
storey's FFL. The format (little-endian, `estate/web/sn5w.py` has the full description,
`tests/fixtures/web/sn5w_sample.bin` is a 304-byte example decoded field by field in `tests/test_web.py`: raw, delta
and same layers, an all-blocked layer, a layer with no overflow records, a three-character tag, a floor on a band
edge and the coarse flag):

```
0  "SN5W"  4 u16 version 1  6 u16 flags (bit 0 delta layers, bit 1 coarse 0.2 m fallback)
8  f32 cell  12 f32 radius  16 f32 step  20 f32 origin_x  24 f32 origin_y (block-local corner of cell 0,0)
28 u16 nx  30 u16 ny  32 u16 n_layers  34 u16 ref_layer  36..63 reserved 0
64 n_layers x 32 B: char[4] tag ("L1\0\0", "L12\0", "RF\0\0") f32 ffl u8 mode (0 raw, 1 delta, 2 same) u8 0 u16 0
                    u32 raster_off u32 raster_len u32 overflow_count u32 overflow_off (0 with no records) u32 0
raster int16[nx*ny] (iy*nx+ix; 0x7FFF = no floor; delta = value - reference, wrapping); overflow {u16 ix, u16 iy,
i16 mm, u16 0} sorted by (iy, ix, mm); the sections follow the table in layer order with no gaps
```

The reference layer is the typical storey with the most walkable cells; a layer equal to it is stored as `same`
(no raster). A file over `max_walk_gz` (128 KB gzipped) falls back to 0.2 m cells, walkable where all four 0.1 m
cells are. Read the cell size from the header.

**`<ID>_web.json`** (`sample-town-n5/web/1`, block-local, Z up, metres): `walk` (the grid's figures), `stairs`
(per IfcStair: name, room, storey and the storey it reaches, both FFLs, its flights with start, end, width, risers,
riser and going from the IFC, its landings, and a walking `path` from the floor landing tread by tread to the next
floor, never rising more than a riser between points) and `doors` (per leaf of every passable door; lift landing
doors stay closed and are absent): `leaf_node`, `storey`, `motion`, `angle` (swing leaves), `open`, `blocked` and
`grid` (`blocked`, `passthrough` or `overhead` for a rolled-up shutter).
- `path` stands on the walk grid all the way: every cell each of its segments crosses has a floor within 0.4 m of
  it. Where a landing's centre has none (in an opened leaf's sweep, beside a column) or the line to it crosses a
  cell without one, if only at a corner, the point moves to the nearest cell of that landing from which both lines
  are clear, by 1 cm where the landing allows (`walk.stair_points_moved`; `stair_off_grid` counts the cells still
  crossed without a floor, 0 on every building). Before writing, the stage decodes the grid, fits the paths to it
  and holds every path to `estate/web/walkcheck.py`, which reads it as a viewer's floorAt and nearestWalkable do:
  every point within 0.1 m of a walkable cell of its own storey band; every cell a segment crosses (an exact
  supercover, corner touches and cells within 0.1 mm included, not samples: samples every 0.05 m missed a 36 mm
  corner clip on NC_514) with a floor within 0.4 m of the path all the way across it; no rise of more than 0.4 m
  between points; the first point on its storey's floor and the last on the next storey's, within 5 cm of their
  FFLs; and every floor inside its band. A building that fails is not written, and the release checks the files
  again.
- `open` is the row-major 3 x 4 matrix (`validate/nav_doorpose.py`) that takes the closed leaf to its opened pose
  in block-local, Z-up coordinates: it premultiplies the leaf's block-local world transform. In a glb (Y up, with
  C: (x, y, z) -> (x, z, -y)) apply C O C^-1 to the leaf node's world matrix, then the inverse of its parent's world
  matrix (the `DOOR_` node, which carries the door's placement) for its local matrix; applied to the local matrix
  directly it is wrong.
- `blocked` is the plan of the opened leaf's box, without its lever handles (they stand about 4 cm proud).
- A door's two sides are best probed in front of its opening (within its clear width, 0.1 to 0.8 m off the
  threshold, at its floor), not at the engine JSON portal's `link` points, which sit mid-edge of each room and can
  fall on furniture: a hawker stall's counter fills the middle of its shutter opening (NC_514).

**`model/export_info.json`** (`sample-town-n5/export-info/1`), written by `build` right after the manifest: the
commit (`git rev-parse HEAD`), whether the generator was dirty (`git status --porcelain -- estate config estate.py
estate.sh estate.cmd`), the seed, the tool versions, the walk figures summed over the buildings, the sha256 of
`estate_manifest.json`, and sha256 and size of every walk grid and web JSON (`files`, keyed by path relative to
`model/`, as the manifest's paths are; inside the model zip every entry carries the `model/` prefix). It holds no
date or time, and the manifest lists each `.blend` by path alone (Blender saves one scene as different bytes each
time), so two builds of one commit give the same bytes. The manifest lists each building's `files.walk` and
`files.web`.

Renders also write `<prefix>_views.json` beside the images, e.g. `reports/renders/ESTATE/ESTATE_views.json`: per
shot the camera's `matrix_world` (row-major 4 x 4, estate frame, Z up; the camera looks along its local -Z with +Y
up), `lens` and `sensor_width` (mm), `sensor_fit` (AUTO: the sensor width spans the larger image side, so the
vertical field of view of a landscape shot is 2 atan(sensor_width h / (2 lens w))), `shift_x`/`shift_y` (in units of
the larger image side), `res` [w, h], `clip`, and `ortho_scale` for orthographic shots.

**`./estate.sh release --tag vX.Y --out DIR`** writes `SampleTownN5_<tag>_model.zip`,
`SampleTownN5_<tag>_reports.zip` and `release_manifest.json` (`{tag, commit, head, zips: {name: {sha256, bytes}},
entries: {path: sha256}}`: `commit` is the commit the export was built on, `head` the one released from) and
uploads nothing. The zips hold only:
- `model/`: `estate_manifest.json`, `export_info.json`, `masterplan.json`, `SITE.ifc`, `SITE_ifc4.ifc`,
  `SITE_lod0.glb`, `SITE_lod1.glb`, `SITE_engine.json`, `SITE_graph.json`, and per building `<ID>.ifc`,
  `<ID>_ifc4.ifc`, `<ID>_lod{0,1,2}.glb`, the `<ID>_int_*.glb` its manifest entry lists, `<ID>_engine.json`,
  `<ID>_flats.json`, `<ID>_nav.json`, `<ID>_validation.json`, `<ID>_walk.bin`, `<ID>_web.json`;
- `reports/`: `reports/*.json`, and the renders (`*.png`) and camera views (`*_views.json`) under `renders/` that
  `model/.state.json` records the current render code wrote; `renders/ESTATE/ESTATE_views.json` and
  `ESTATE_aerial_NE.png` are required.

Never a `.blend`, a Bonsai link cache (`*.ifc.cache.*`), `.state.json`, `nav/` or `plans/`. Entries are sorted and
have a fixed time and mode and deflate level 9, so two releases of one build give identical zips. Two builds of
one commit do not: `*_nav.json`, `*_validation.json` and `reports/*.json` record how long their work took. The
release is refused when:
- `export_info.json` says the generator was dirty, or names a commit that is neither HEAD nor an ancestor of HEAD
  with the same generator (no change under `estate config estate.py estate.sh estate.cmd` in between: build on M,
  commit the reports the build rewrote as R, release on R);
- a file no longer has the hash the manifest or `export_info.json` recorded;
- a released file's stage record in `model/.state.json` is missing or was written by other code than the checkout's
  (a partial build on older code), or the last build had failures (`reports/build_failures.json`);
- a required file is missing, or an interior chunk on disk is not one the manifest lists;
- a stair path cannot be walked on the walk grid shipped beside it, or a floor lies outside its band
  (`estate/web/walkcheck.py`, as above);
- `--out` lies inside `model/` or `reports/`;
- the leak scan (`estate/leaks.py`, every entry, entry name and zip written: JSON strings, PNG text chunks, GLB JSON,
  IFC text, walk-grid tags) finds a machine path or the username, or cannot read an entry.

Release from a full build (`./estate.sh build --force`): a stage a partial build reran on unchanged code but older
inputs is not detected.

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
