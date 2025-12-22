# Headless scene CLI

This repo’s `gipc` binary is a headless CLI that loads a scene from JSON, runs the simulation for a fixed number of frames, and writes one `.obj` per frame.

## Build

```bash
cmake --preset debug
cmake --build build/debug -j20
```

## Run (free fall)

```bash
./build/debug/gipc -j examples/free_fall.json -o Output/free_fall_cli
```

Outputs:
- `Output/free_fall_cli/frame_00000.obj` (one `.obj` per frame)

Each `.obj` contains all objects (combined).

Only tetrahedral objects are written to `.obj`. The analytic ground plane (`ground.normal`/`ground.offset`) is collision-only and is not written; the example scene models the floor as a real tetrahedral obstacle mesh (`examples/ground_plane.msh`) so it is included.

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
- `preconditioner_type` (int): only `0` is supported by this CLI

### `ground`
- `normal` (vec3): plane normal
- `offset` (number): plane offset used by the ground collision kernel

### `objects[]`
- `is_obstacle` (bool): `true` for collision-only (fixed vertices), `false` for dynamic
- `mesh_msh` (string): path to a Gmsh `.msh` tet mesh
- `young_modulus` (number): per-object Young’s modulus
- `transform` (object)
  - `scale` (number): uniform scale
  - `translation` (vec3): world translation
- `initial_velocity` (vec3): initial velocity for all vertices
- `pin_boxes` (array): each entry pins vertices inside an axis-aligned box
  - `min` (vec3)
  - `max` (vec3)
