# Headless scene CLI

This repo’s `gipc` binary is a headless CLI that loads a scene from JSON, runs the simulation for a fixed number of frames, and writes one `.obj` per frame.

## Build

```bash
cmake --preset debug
cmake --build build/debug -j20
```

## Offline sort mode

Sort all `.msh` files in a directory and write sorted meshes + `.part` files:

```bash
./build/debug/gipc --sort examples --output output/sorted_mesh
```

outputs (per input mesh):
- `output/sorted_mesh/<name>_sorted.16.msh`
- `output/sorted_mesh/<name>_sorted.16.part`

## Run

```bash
./build/debug/gipc -j examples/free_fall.json -o output/free_fall_cli
```

## JSON schema (all fields required)

Top-level:
- `settings` (object): simulation parameters (matches `Assets/scene/parameterSetting.txt` fields)
- `simulation` (object): run controls
- `ground` (object): ground plane used by ground collision
- `objects` (array): one or more tetrahedral objects

### `settings`
- `volume_mesh_density` (number)
- `poisson_rate` (number)
- `friction_rate` (number)
- `gd_friction_rate` (number)
- `triangle_mesh_thickness` (number)
- `triangle_mesh_youngs_modulus` (number)
- `triangle_bend_youngs_modulus` (number)
- `triangle_mesh_density` (number)
- `strain_rate` (number)
- `motion_stiffness` (number)
- `collision_detection_buff_scale` (number)
- `motion_rate` (number)
- `ipc_time_step` (number)
- `pcg_solver_threshold` (number)
- `Newton_solver_threshold` (number)
- `IPC_ralative_dHat` (number)

### `simulation`
- `frames` (int): number of frames to simulate
- `preconditioner_type` (int): `0` disables MAS, non-zero enables MAS

### `ground`
- `normal` (vec3): plane normal
- `offset` (number): plane offset used by the ground collision kernel

### `objects[]`
- `is_obstacle` (bool): `true` for collision-only (fixed vertices), `false` for dynamic
- `mesh_msh` (string): path to a Gmsh `.msh` tet mesh
- `part_file` (string): path to a `.part` file (required; ignored when `simulation.preconditioner_type == 0`)
- `young_modulus` (number): per-object Young’s modulus
- `transform` (object)
  - `scale` (number): uniform scale
  - `translation` (vec3): world translation
- `initial_velocity` (vec3): initial velocity for all vertices
- `pin_boxes` (array): each entry pins vertices inside an axis-aligned box
  - `min` (vec3)
  - `max` (vec3)

If `simulation.preconditioner_type != 0`, every object (including obstacles) must provide a valid `part_file`.
