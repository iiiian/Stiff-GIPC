# Headless scene CLI

## CLI arguments

- `-o, --output <dir>`: required in both modes.
- Offline sort mode: `--sort <input_dir>` (ignores `--json` if also provided).
- Run mode: `-j, --json <scene.json>` (required unless `--sort` is used).
- Linear system solve mode: `--A <A.mtx> --b <b.mtx>` (requires `--json`, solves once and exits).

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

## Solve a provided linear system (Ax=b)

When invoked with both `--A` and `--b`, the program loads the scene JSON (all fields required, same as run mode) to configure PCG tolerances and preconditioner settings, then solves the provided linear system once and exits.

Requirements:
- `--A`: MatrixMarket **coordinate** matrix, **real**, square, with `M % 3 == 0`.
- `--b`: MatrixMarket **array** matrix, **real**, shape `(M x 1)` (required).

Example:
```bash
./build/debug/gipc -j examples/free_fall.json -o output/solve_once --A output/linear_system/frame_00000_A.mtx --b output/linear_system/frame_00000_b.mtx
```

Outputs:
- `output/free_fall_cli/frame_00000.obj` (initial/pre-sim), `frame_00001.obj`, ... (one per solver frame)
- `output/free_fall_cli/stats.json` (overwritten each frame with accumulated stats/timers)

## `stats.json` (status output format)

The headless CLI writes per-frame solver status + timing information to `<output_dir>/stats.json` (some older notes may call this `status.json`; the schema is the same).

Top-level schema:
```json
{
  "frames": [
    {
      "newton": [
        { "pcg": { "iterations": 70 } }
      ],
      "timer": {
        "name": "GlobalTimer",
        "duration": 0.0,
        "count": 1,
        "parent": "",
        "children": []
      }
    }
  ]
}
```

Notes:
- `frames[i].newton` is a list of Newton iterations for frame `i`; each entry currently records `pcg.iterations` for that Newton step.
- `frames[i].timer` is a merged timer tree from `gipc::GlobalTimer::report_merged_as_json()`:
  - `duration` is in seconds (multiply by `1000` for ms).
  - `count` is how many times that timer node was hit within the frame.
  - `parent` is the full parent path (e.g. `/GlobalTimer/IPC_Solver`); the root has `parent == ""`.
  - `children` are nested timer nodes (sorted by total `duration`, descending).

## JSON schema (all fields required)

The loader uses `json.at(...)` for all fields below, so every key must be present (arrays can be empty). In particular, `objects[].part_file`, `objects[].pin_boxes`, `objects[].density`, and `objects[].poisson_ratio` must exist even when not used.

Top-level:
- `settings` (object): simulation parameters
- `simulation` (object): run controls
- `ground` (object): ground plane collision settings
- `animation` (array): animated soft constraints (target motion)
- `objects` (array): one or more tetrahedral objects

---

## `settings`

### `volume_mesh_density`
- **Type:** number

  ```cpp
  mesh.masses[vertex] += tetrahedron_volume * density / 4;
  ```
- **Effect:** Legacy global density. FEM masses are now driven by `objects[].density`; this value is kept for compatibility.

### `poisson_rate`
- **Type:** number
- **Effect:** Legacy global Poisson ratio. FEM is now driven by `objects[].poisson_ratio`; this value is kept for compatibility.

### `friction_rate`
- **Type:** number
- **Used in:** `GIPC.cu:9128, 9265, 10443`
- **Formula:** at `GIPC.cu:10443`
  ```cpp
  auto fric = frictionRate * Energy_Add_Reduction_Algorithm(5, TetMesh);
  ```
- **Effect:** Friction coefficient for object-object collisions. Multiplies friction energy/force contribution.

### `gd_friction_rate`
- **Type:** number
- **Stored in:** `ipc.gd_frictionRate`
- **Effect:** Friction coefficient for ground collisions. Only used when `ground.enabled == true`.

### `triangle_mesh_thickness`
- **Type:** number
- **Effect:** **UNUSED** - cloth simulation is disabled in headless CLI. Value required but has no effect.

### `triangle_mesh_youngs_modulus`
- **Type:** number
- **Effect:** **UNUSED** - cloth simulation is disabled in headless CLI. Value required but has no effect.

### `triangle_bend_youngs_modulus`
- **Type:** number
- **Effect:** **UNUSED** - cloth simulation is disabled in headless CLI. Value required but has no effect.

### `triangle_mesh_density`
- **Type:** number
- **Effect:** **UNUSED** - cloth simulation is disabled in headless CLI. Value required but has no effect.

### `strain_rate`
- **Type:** number
- **Effect:** **UNUSED** - cloth simulation is disabled in headless CLI. Value required but has no effect.

### `motion_stiffness`
- **Type:** number
- **Stored in:** `ipc.softMotionRate`
- **Used in:** `GIPC.cu:8671, 8819, 10363`
- **Effect:** Stiffness for soft constraint targets (animation targets). Higher = vertices track targets more rigidly.

### `gravity`
- **Type:** vec3 (array of 3 numbers)
- **Stored in:** `ipc.gravity`
- **Used in:** `GIPC.cu` -> `GIPC::computeXTilta()` (adds `gravity * dt^2` for free vertices)
- **Effect:** Global gravity acceleration vector. Use `[0, 0, 0]` to disable gravity.

### `collision_detection_buff_scale`
- **Type:** number
  ```cpp
  MAX_CCD_COLLITION_PAIRS_NUM = 1 * scale * (surface_Num*15 + edge_Num*10) * max(IPC_dt/0.01, 2.0);
  MAX_COLLITION_PAIRS_NUM = (surf_vertexNum*3 + edge_Num*2) * 3 * scale;
  ```
- **Effect:** Scales GPU buffer size for collision pairs. Increase if simulation crashes with "too many collision pairs".

### `motion_rate`
- **Type:** number
- **Stored in:** `ipc.animation_subRate = 1.0 / motion_rate` at `main.cu:524`
- **Used in:** `GIPC.cu:11009, 11045`
- **Effect:** Controls animation playback speed. Higher = faster animation target interpolation.

### `ipc_time_step`
- **Type:** number
- **Used in:** Multiple locations at `GIPC.cu:8029, 9123, 10766, 10881`
- **Effect:** Simulation timestep. Smaller = more stable but slower. Larger = faster but may diverge.

### `pcg_rel_threshold`
- **Type:** number
- **Stored in:** `ipc.pcg_rel_threshold`
- **Used in:** `gipc/gipc.cu`
  ```cpp
  cfg.rel_tol = pcg_rel_threshold;
  ```
- **Effect:** Relative tolerance for PCG convergence. Stops when `||r|| <= pcg_rel_threshold * ||r0||` (if `pcg_rel_threshold > 0`).

### `pcg_abs_threshold`
- **Type:** number
- **Stored in:** `ipc.pcg_abs_threshold`
- **Used in:** `gipc/gipc.cu`
  ```cpp
  cfg.abs_tol = pcg_abs_threshold;
  ```
- **Effect:** Absolute tolerance for PCG convergence. Stops when `||r|| <= pcg_abs_threshold` (if `pcg_abs_threshold > 0`).

### `pcg_max_iter`
- **Type:** int
- **Stored in:** `ipc.pcg_max_iter`
- **Used in:** `gipc/gipc.cu`
  ```cpp
  cfg.max_iter = pcg_max_iter;
  ```
- **Effect:** Maximum number of PCG iterations. `0` = unlimited (falls back to dof count).

### `pcg_use_preconditioned_norm`
- **Type:** bool
- **Stored in:** `ipc.pcg_use_preconditioned_norm`
- **Used in:** `gipc/gipc.cu`
  ```cpp
  cfg.use_preconditioned_residual_norm = pcg_use_preconditioned_norm;
  ```
- **Effect:** Chooses the norm used in the stop test:
  - `true`: `||r|| = sqrt(r^T M^{-1} r)` (preconditioned residual norm)
  - `false`: `||r|| = sqrt(r^T r)` (plain residual norm)

### `abs_xdelta_tol`
- **Type:** number
- **Stored in:** `ipc.abs_xdelta_tol`
- **Effect:** Absolute Newton termination threshold based on `||dx||_inf` scaled by the scene size (union AABB diagonal length over all `objects[]` with `is_obstacle == false`): stop when `||dx||_inf < abs_xdelta_tol * scene_scale`.

### `rel_xdelta_tol`
- **Type:** number
- **Stored in:** `ipc.rel_xdelta_tol`

---

## `simulation`

### `export_linear_system_frames`
- **Type:** array of int
- **Effect:** For each listed solver frame index, export the first Newton iteration’s linear system `A x = b` (the one passed to PCG) in MatrixMarket format.
- **Output:** written under `<output_dir>/linear_system/` as:
  - `frame_XXXXX_A.mtx` (coordinate, real, general; full matrix)
  - `frame_XXXXX_b.mtx` (array, real, general; `M x 1`)
- **Effect:** Relative Newton termination threshold based on `||dx||_inf`: stop when `||dx||_inf < rel_xdelta_tol * ||dx0||_inf`, where `dx0` is the Newton direction from the first iteration.

Newton stops when **either** condition is satisfied.

### `IPC_ralative_dHat`
- **Type:** number
- **Stored in:** `ipc.relative_dhat`
- **Formula:**
  ```cpp
  bboxDiagSize2 = squaredNorm(maxCorner - minCorner);  // Scene bounding box diagonal^2
  dHat = relative_dhat^2 * bboxDiagSize2;              // Actual collision distance threshold
  ```

### `armijo_c1`
- **Type:** number
- **Stored in:** `ipc.armijo_c1`
- **Effect:** Armijo sufficient decrease constant `c1` (typical `1e-4`). Used in line search acceptance test.

### `armijo_beta`
- **Type:** number
- **Stored in:** `ipc.armijo_beta`
- **Effect:** Backtracking shrink factor `beta` in `(0, 1)` (typical `0.5`). Each failed Armijo test reduces `alpha := beta * alpha`.

### `armijo_alpha_min`
- **Type:** number
- **Stored in:** `ipc.armijo_alpha_min`
- **Effect:** Lower bound on `alpha` during Armijo backtracking (typical `1e-12`).
- **Effect:** Collision activation distance relative to scene size. Collisions activate when `distance^2 < dHat`. Larger = earlier collision response, smoother but more expensive.

---

## `simulation`

### `frames`
- **Type:** int
  ```cpp
  write_obj(..., 0); // initial/pre-sim
  for(int frame = 0; frame < frames; ++frame) {
      ipc.IPC_Solver(d_tetMesh);
      write_obj(..., frame + 1);
  }
  ```
- **Effect:** Number of solver frames to run. Writes one extra `.obj` at frame 0 before running the solver.

### `preconditioner_type`
- **Type:** int
- **Stored in:** `ipc.pcg_data.P_type` at `main.cu:297`
- **Used in:** `gipc/gipc.cu:60-69`
  ```cpp
  if(pcg_data.P_type == 1) {
      m_global_linear_system->create<MAS_Preconditioner>(...);
  } else {
      m_global_linear_system->create<DiagPreconditioner>();
  }
  ```
- **Effect:**
  - `0` = Diagonal preconditioner (simple, no `.part` files needed)
  - `1` = MAS (Multi-level Additive Schwarz) preconditioner (faster convergence, requires `.part` files)

### `write_obj_frames`
- **Type:** bool
- **Effect:** Controls whether `.obj` mesh files are written each frame.
  - `true`: Write `frame_00000.obj`, `frame_00001.obj`, etc.
  - `false`: Skip OBJ output (only `stats.json` is written)

---

## `ground`

### `enabled`
- **Type:** bool
- **Stored in:** `ipc.useGround`
- **Effect:** Enable or disable ground plane collision.
  - `true`: Enables collision detection and response against the ground plane.
  - `false`: Ground collision is disabled.

### `normal`
- **Type:** vec3 (array of 3 numbers)
- **Effect:** The ground plane normal direction. Typically `[0, 1, 0]` for a horizontal floor.

### `offset`
- **Type:** number
- **Effect:** The ground plane offset from the origin along the normal direction. For example, with `normal: [0, 1, 0]` and `offset: 0`, the ground is at `y = 0`.

---

## `animation[]`

`animation` is a list of animated soft constraints. To disable animation, set `animation: []`.

### `animation[].object`
- **Type:** int
- **Effect:** Index into `objects[]` to select which object's vertices are tested against `boxes`.

### `animation[].boxes[]`
- **Type:** array of `{ "min": vec3, "max": vec3 }`
- **Effect:** Vertex selection AABBs in world coordinates (same style as `objects[].pin_boxes`).

### `animation[].rot_origin`
- **Type:** vec3
- **Effect:** Rotation origin in world coordinates.

### `animation[].rot_axis`
- **Type:** vec3
- **Effect:** Rotation axis in world coordinates (will be normalized).

### `animation[].rot_velocity`
- **Type:** number
- **Effect:** Rotation rate in radians per second. Each frame uses `theta = rot_velocity * ipc_time_step`.

---

## `objects[]`

### `is_obstacle`
- **Type:** bool
- **Used in:** `main.cu:335-344`
  ```cpp
  if(is_obstacle) {
      for(int i = v_begin; i < v_end; ++i) {
          tetMesh.boundaryTypies[i] = 1;  // Mark as fixed
      }
  }
  ```
- **Effect:** `true` = all vertices fixed (boundary type 1), no gravity/forces applied. `false` = dynamic vertices subject to physics.

### `mesh_msh`
- **Type:** string
- **Used in:** `main.cu:305, 318-319`
  ```cpp
  tetMesh.load_tetrahedraMesh(mesh_path, transform, young_modulus, density, poisson_ratio, ...);
  ```
- **Effect:** Path to Gmsh `.msh` file containing tetrahedral mesh.

### `part_file`
- **Type:** string
- **Used in:** `main.cu:306, 324-331`
- **Effect:** Path to METIS partition file for MAS preconditioner. `preconditioner_type == 0` ignores its contents, but the key must still be present in the JSON.

### `young_modulus`
- **Type:** number
- **Used in:** `main.cu:308, 318-319` -> stored per-tetrahedron in `mesh.vert_youngth_modules`
- **Flow:** `load_tetrahedraMesh()` -> `initFEM()` -> compute `lengthRate`/`volumeRate` per tet
- **Final usage:** `femEnergy.cu:1032-1050` (Stable Neo-Hookean energy)
  ```cpp
  // Stable Neo-Hookean energy formulation
  double Jminus1 = I3 - 1 - lenRate/volRate;
  return 0.5 * (lenRate*(I2-3) + volRate*(Jminus1^2)) * volume;
  ```
  Where `I2 = ||F||^2` (Frobenius norm squared) and `I3 = det(F)` (deformation gradient determinant).
- **Effect:** Per-object stiffness. Higher = stiffer material.

### `density`
- **Type:** number
- **Used in:** `main.cu` -> stored per-tetrahedron in `mesh.tet_densities` -> used in `initFEM()` to accumulate vertex masses
- **Formula:**
  ```cpp
  mesh.masses[vertex] += tetrahedron_volume * density / 4;
  ```
- **Effect:** Determines mass. Higher density = heavier mesh = slower motion under gravity.

### `poisson_ratio`
- **Type:** number
- **Used in:** `main.cu` -> stored per-tetrahedron in `mesh.tet_poisson_ratios` -> used in `initFEM()` to compute per-tet Lamé parameters
- **Effect:** Controls compressibility. Values near 0.5 = nearly incompressible (rubber-like). Values near 0 = easily compressible.

### `transform`

Transforms are applied in the following order:
1. Rotate around mesh bounding box center
2. Scale
3. Translate

#### `rotation`
- **Type:** vec3 (array of 3 numbers)
- **Used in:** `main.cu`, `load_mesh.cpp`
- **Effect:** Euler angles in degrees (X, Y, Z). Rotation is applied around the mesh's bounding box center, using XYZ rotation order (rotate X, then Y, then Z).

#### `scale`
- **Type:** number
- **Used in:** `main.cu`, `load_mesh.cpp`
- **Effect:** Uniform scale applied to mesh vertices during loading (after rotation).

#### `translation`
- **Type:** vec3 (array of 3 numbers)
- **Used in:** `main.cu`, `load_mesh.cpp`
- **Effect:** World-space translation applied to mesh vertices during loading (after rotation and scale).

### `initial_velocity`
- **Type:** vec3 (array of 3 numbers)
- **Used in:** `main.cu:310, 346-349`
  ```cpp
  for(int i = v_begin; i < v_end; ++i) {
      tetMesh.velocities[i] = init_vel;
  }
  ```
- **Effect:** Initial velocity for all vertices of the object.

### `pin_boxes[]`

#### `min`
- **Type:** vec3 (array of 3 numbers)
- **Effect:** Minimum corner of axis-aligned bounding box for vertex pinning.

#### `max`
- **Type:** vec3 (array of 3 numbers)
- **Used in:** `main.cu:351-365`
  ```cpp
  if(p.x >= bmin.x && p.x <= bmax.x &&
     p.y >= bmin.y && p.y <= bmax.y &&
     p.z >= bmin.z && p.z <= bmax.z) {
      tetMesh.boundaryTypies[i] = 1;  // Pin vertex
  }
  ```
- **Effect:** Vertices inside the axis-aligned bounding box are fixed (boundary type 1 = no movement).
