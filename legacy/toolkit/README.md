# HDB block toolkit (realistic, walkable BIM → game engine)

Requirements: `pip install ifcopenshell shapely numpy scipy trimesh matplotlib pillow`

| Step | Script | Output |
|---|---|---|
| 1. Generate | `python hdb_block.py -o hdb_block.ifc` (`--storeys 16`, `--schema IFC4` for Unreal Datasmith) | IFC4X3 / IFC4 model |
| 2. Draw | `python draw_block.py hdb_block.ifc` | `plan_L5.png`, `stair_section.png`, `cutaway_L5.png` |
| 3. Walk test | `python nav_check.py hdb_block.ifc --radius 0.3 0.4` | reachability per room, `nav_L5_r*.png`, `nav_report.json` |
| 4. Engine export | `python export_engine.py hdb_block.ifc` | `hdb_block.glb`, `hdb_block_engine.json` |

`ifc_mesh.py` tessellates the IFC once (IfcOpenShell) and caches meshes next to the file; steps 2-4 share it.

## What the model contains
- 12 storeys (void deck + 11 residential) + roof; 4 mirrored 4-room flats per floor (#02-101 ... #12-107), 11 rooms each as `IfcSpace`, grouped per flat as `IfcZone`.
- Lift lobby, 2 lift shafts with closed landing doors, 2 dog-leg common stairs per floor (risers ≤ 175 mm, goings 280 mm,
  ≤ 18 risers per flight, 1.15 m flights, 1.3 m landings), roof access, parapets 1.1 m, AC ledges with railings, water tank.
- Typed walls with material layer sets (editable in Bonsai), doors/windows in real openings, property sets incl. `Navigation`
  (passable doors, climbable top edges).

## Engine notes
- The glb merges static architecture per storey + material; door leaves are separate nodes (`DOOR_*`) to swap for interactive doors.
- `hdb_block_engine.json` (metres, Z up) lists door hinges / swing sides, lift shafts + landing doors per level, room polygons
  (trigger volumes), climbable edge polylines, a spawn point and the agent settings that passed the walk test.
- Walk test: every room and the roof are reachable for an agent of radius 0.30 m, height 1.8 m, step 0.4 m. At 0.40 m the
  0.8-0.9 m interior doors close off. Keep the capsule / navmesh agent radius ≤ 0.30 m or use ≤ 10 cm navmesh cells indoors.
