# MAS Preconditioner Codebase Guide

This guide summarizes how the connectivity‑enhanced Multilevel Additive Schwarz (MAS) preconditioner from the StiffGIPC paper is implemented in this repository, and how it fits into the overall code structure and solver pipeline.

The implementation closely follows Section 4 of the paper (“Connectivity‑Enhanced MAS Preconditioner with GPU Optimization”), plus the supplemental METIS‑based reordering description.

---

## 1. Repository overview (MAS‑relevant parts)

- `StiffGIPC/`
  - `gl_main.cu`: main application / demo entry point; loads meshes, builds IPC system, sets up MAS partitions and preconditioner, runs simulation.
  - `GIPC.cu`, `GIPC.cuh`: high‑level GIPC/IPC driver, wraps GPU data and the linear system.
  - `PCG_SOLVER.cu`, `PCG_SOLVER.cuh`: low‑level PCG support (`PCG_Data`) and MAS preconditioner ownership.
  - `MASPreconditioner.cu`, `MASPreconditioner.cuh`: core GPU implementation of the MAS hierarchy, matrix assembly, and application.
  - `load_mesh.cpp`, `load_mesh.h`: FEM/ABD mesh loader, neighbor graph construction, and storage for METIS partition IDs.
  - `eigen_data.h`: numerical types and MAS matrix storage; defines `BANKSIZE` (MAS subdomain size) and MAS block data structures.
  - `gipc/`: higher‑level scene, subsystem, and solver abstractions.
    - `gipc/gipc.cu`: builds global linear system, chooses and wires the preconditioner.
    - `gipc/utils/simple_scene_importer.cpp`: imports scenes, calls METIS‑based mesh partitioner, and loads per‑vertex partition IDs.
  - `linear_system/`: generic block sparse linear system infrastructure.
    - `linear_system/preconditioner/fem_mas_preconditioner.{h,cu}`: thin bridge between the FEM subsystem and `MASPreconditioner`.
    - `linear_system/solver/pcg_solver.{h,cu}`: generic PCG solver that calls the preconditioner.
- `MeshProcess/`
  - `metis_partition/`: offline METIS‑based mesh partitioner used to construct connectivity‑enhanced MAS subdomains.
    - `metis_sort.{h,cpp}`: runs METIS, builds partitions, and exports sorted meshes + `.part` files that encode per‑vertex partition IDs.

The MAS preconditioner lives primarily in `StiffGIPC/MASPreconditioner.*`, but it relies on mesh preprocessing (`MeshProcess/metis_partition`), mesh loading (`load_mesh.*`), and the linear system wiring (`gipc/*`, `linear_system/*`).

---

## 2. Solver pipeline and where MAS fits

High‑level flow per simulation step / Newton iteration:

1. **Scene and mesh loading**
   - `gl_main.cu:initScene()` chooses a scenario and uses `gipc::SimpleSceneImporter` to load ABD and FEM bodies (`tetrahedra_obj`) from assets.
   - For FEM bodies, when MAS is enabled, the importer uses `metis_sort` (Section 3) to obtain a METIS‑partitioned and vertex‑sorted mesh.

2. **Neighbor graph and MAS partition setup**
   - `tetrahedra_obj::getVertNeighbors()` (in `load_mesh.cpp`) builds a FEM vertex adjacency graph from tetrahedra and surface triangles.
   - `gl_main.cu:setMAS_partition()` converts METIS partition IDs into the MAS mapping and remapping arrays (Section 3.3).
   - `gl_main.cu:initFEM()` configures the IPC solver, builds BVHs, and calls
     `ipc.pcg_data.MP.initPreconditioner_Neighbor(...)` followed by `initPreconditioner_Matrix()` on the `MASPreconditioner` instance.

3. **Global linear system and PCG**
   - `GIPC::create_LinearSystem()` (`GIPC.cu:34`) creates:
     - an ABD linear subsystem,
     - a FEM linear subsystem,
     - a `PCGSolver`,
     - and either `MAS_Preconditioner` or `DiagPreconditioner` depending on `P_type`.
   - The linear system framework (`linear_system/*`) assembles the global block sparse matrix in BCOO form and connects it to the PCG solver.

4. **MAS preconditioner assembly (per Newton iteration)**
   - The FEM subsystem owns a `gipc::MAS_Preconditioner` (wrapper) which holds a reference to `MASPreconditioner`.
   - At each assemble call (`MAS_Preconditioner::assemble()` in `linear_system/preconditioner/fem_mas_preconditioner.cu:15`):
     - It computes local BCOO indices (`calculate_subsystem_bcoo_indices`).
     - It calls `MASPreconditioner::setPreconditioner_bcoo(...)`, which:
       1. Rebuilds the MAS hierarchy given the current collision graph (`ReorderRealtime(cpNum)`).
       2. Assembles the MAS matrices from the current global Hessian (`PrepareHessian_bcoo`).
       3. Inverts each MAS block to form the actual preconditioner (`__inverse6_P96x96`).

5. **MAS preconditioner application inside PCG**
   - The PCG solver (`linear_system/solver/pcg_solver.cu`) calls `apply_preconditioner(z, r)` from within `PCGSolver::pcg`.
   - For FEM DOFs, this dispatches to `MAS_Preconditioner::apply(...)`, which calls
     `MASPreconditioner::preconditioning((double3*)r.data(), (double3*)z.data())`.
   - `preconditioning` carries out a full multilevel MAS application:
     - builds the multilevel residual `R` (`BuildMultiLevelR`),
     - applies the block preconditioner on all levels (`SchwarzLocalXSym_*`),
     - and collects the final preconditioned vector `Z` on the fine mesh (`CollectFinalZ`).

---

## 3. METIS‑based node reordering and MAS partitions

### 3.1 Offline METIS partitioning (`MeshProcess/metis_partition`)

Files:
- `MeshProcess/metis_partition/metis_sort.{h,cpp}`
- `MeshProcess/metis_partition/node_edge_model.{h,cpp}`
- `MeshProcess/metis_partition/mesh.{h,cpp}`

Key ideas:

- `metis_sort(std::string obj_path, int dimension)`:
  - For **3D FEM** (`dimension == 3`):
    - Loads a tet mesh (`gipc::TetMesh`).
    - Builds a dual graph (nodes = vertices, edges = adjacency) using METIS.
    - Chooses block size `block_size = 16` (matching `BANKSIZE` in `eigen_data.h:17`).
    - Computes `nPart ≈ ceil(V / (block_size - 1))` and calls `mesh.k_way_partition(nPart, part)`.
    - Packs per‑partition vertex indices into contiguous blocks, producing a **sorted vertex order**.
    - Exports:
      - a **sorted mesh** file `*_sorted.16.<ext>` (vertices reordered to group partitions),
      - a **sorted partition file** `*_sorted.16.part` (per‑vertex partition ID in the sorted order).
  - For **2D cloth** (`dimension == 2`), it does the analogous work on triangle meshes.

This implements the “METIS‑based node reordering” that the paper describes (supplemental Section “METIS‑based Node Reordering”).

### 3.2 Loading partitions into `tetrahedra_obj`

Files:
- `gipc/utils/simple_scene_importer.cpp`
- `load_mesh.{h,cpp}`

Important pieces:

- `SimpleSceneImporter::load_geometry(...)` (`simple_scene_importer.cpp:39`):
  - For FEM and `preconditionerType != 0`, it calls
    ```cpp
    auto paths = metis_sort(meth_path, Dimensions);
    tetras.load_tetrahedraMesh(paths[0], transform, YoungthM, bodyType, body_boundary_type);
    tetras.load_parts(paths[1]);
    ```
  - Thus the simulation mesh is **already sorted** according to METIS partitions.

- `tetrahedra_obj::load_parts(const std::string& filename)` (`load_mesh.cpp:447`):
  - Reads a `*.part` file into `tetrahedra_obj::partId` (one partition ID per FEM vertex).
  - `part_offset` accumulates partition IDs across multiple FEM meshes/bodies:
    ```cpp
    while(ifs >> id) {
        partId.push_back(id + part_offset);
    }
    part_offset = partId.back() + 1;
    ```
  - Sanity check ensures the number of partition IDs matches the number of FEM vertices (`vertexes.size() - abd_vertexOffset`).

At this point, **every FEM vertex has a “partition ID” from METIS**, and vertices are already grouped in the file by partition.

### 3.3 Building MAS mapping/remapping arrays (`setMAS_partition`)

File:
- `gl_main.cu: setMAS_partition()`

Data structures in `tetrahedra_obj` (`load_mesh.h:42`):
- `std::vector<uint32_t> partId;` — METIS partition ID per FEM vertex.
- `std::vector<int> partId_map_real;` — MAS “mapping array” (size `M * BANKSIZE`).
- `std::vector<int> real_map_partId;` — MAS “remapping array”.
- `uint32_t part_offset;` — total number of METIS partitions across all FEM meshes.

Construction:

1. **Allocate mapping array**
   ```cpp
   tetMesh.partId_map_real.resize(tetMesh.part_offset * BANKSIZE, -1);
   tetMesh.real_map_partId.resize(tetMesh.partId.size());
   ```
   - `BANKSIZE` (subdomain size) is 16 (`eigen_data.h:17`).
   - `part_offset` is the number of METIS partitions across FEM meshes.
   - So `partId_map_real` has size `M * N` in paper notation (`M = part_offset`, `N = BANKSIZE`).

2. **Fill mapping array (`Map`)**
   ```cpp
   int index = 0;
   for(int i = 0; i < tetMesh.partId.size(); i++)
   {
       tetMesh.partId_map_real[BANKSIZE * tetMesh.partId[i] + index] = i;
       index++;
       if(i <= tetMesh.partId.size() - 2)
       {
           if(tetMesh.partId[i + 1] != tetMesh.partId[i])
           {
               index = 0;
           }
       }
   }
   ```
   - Vertices are already sorted by `partId`, so the loop fills each partition’s **contiguous block** of `BANKSIZE` entries.
   - Within each partition:
     - real FEM vertex indices are stored in slots `[p * BANKSIZE, p * BANKSIZE + size(p) - 1]`,
     - remaining slots are left as `-1` (padding).
   - This is exactly the “mapping array” from Figure 7 in the paper: it aligns each METIS partition to one MAS subdomain of size `N` with zero‑filled padding.

3. **Build remapping array (`ReMap`)**
   ```cpp
   index = 0;
   for(int i = 0; i < tetMesh.partId_map_real.size(); i++)
   {
       if(tetMesh.partId_map_real[i] == index)
       {
           tetMesh.real_map_partId[index] = i;
           index++;
       }
   }
   ```
   - `real_map_partId[v]` gives the **mapped slot index** (0..`M*N-1`) corresponding to real FEM vertex `v`.
   - Together, `partId_map_real` and `real_map_partId` implement the mapping and remapping arrays of Figure 7 in the paper.

4. **Upload to GPU and initialize MASPreconditioner**
   - In `initFEM` (`gl_main.cu:1080+`):
     ```cpp
     int neighborListSize = tetMesh.getVertNeighbors();
     ipc.pcg_data.MP.initPreconditioner_Neighbor(
         ipc.vertexNum - tetMesh.abd_vertexOffset,
         tetMesh.abd_vertexOffset,
         neighborListSize,
         ipc._collisonPairs,
         tetMesh.part_offset * BANKSIZE);

     // copy neighbor and mapping arrays to device
     cudaMemcpy(MP.d_neighborListInit, tetMesh.neighborList.data(), ...);
     cudaMemcpy(MP.d_neighborStart,    tetMesh.neighborStart.data(), ...);
     cudaMemcpy(MP.d_neighborNumInit,  tetMesh.neighborNum.data(), ...);
     cudaMemcpy(MP.d_partId_map_real,  tetMesh.partId_map_real.data(), ...);
     cudaMemcpy(MP.d_real_map_partId,  tetMesh.real_map_partId.data(), ...);

     MP.initPreconditioner_Matrix();
     ```
   - `initPreconditioner_Neighbor` records:
     - FEM vertex count and ABD offsets,
     - neighbor counts and collision pair arrays,
     - `partMapSize = part_offset * BANKSIZE = totalMapNodes`.
   - `initPreconditioner_Matrix` triggers an initial MAS hierarchy construction (no contacts) via `ReorderRealtime(0)` and allocates MAS matrices.

---

## 4. MASPreconditioner class structure (`MASPreconditioner.cuh`)

File:
- `StiffGIPC/MASPreconditioner.cuh`

Key members:

- Scalar meta:
  - `int totalNodes;` — number of FEM vertices (fine nodes).
  - `int totalMapNodes;` — size of the mapping array (`M * BANKSIZE`).
  - `int levelnum;` — number of MAS levels (computed by `computeNumLevels`).
  - `int collision_node_Offset;` — global index offset between ABD and FEM vertices.
  - `int totalNumberClusters;` — total MAS nodes across all levels (used to size multilevel arrays).
  - `int2 h_clevelSize;` — host copy of level size metadata (`(clusterCount, startOffset)`).

- Device arrays for hierarchy topology:
  - `int2* d_levelSize;` — per level `l`, `d_levelSize[l] = (count, offset)`; level 0 is fine DOFs, levels 1..L are increasingly coarser.
  - `int* d_coarseSpaceTables;` — for each level and node, maps fine node indices to coarse node indices.
  - `int* d_prefixOriginal;`, `int* d_prefixSumOriginal;` — number and prefix sums of clusters at level 0 (per warp).
  - `int* d_goingNext;` — for each node at any level, stores its “next” node in the aggregation chain (used to walk up the levels).
  - `int* d_denseLevel;` — auxiliary array for dense level info (used in `AggregationKernel`).
  - `__GEIGEN__::itable* d_coarseTable;` — for each fine node, stores the list of cluster IDs along the MAS hierarchy (up to 6 levels).
  - `unsigned int* d_fineConnectMask;` — connectivity bitmask for level 0 (1 bit per node in a local subdomain).
  - `unsigned int* d_nextConnectMask;`, `d_nextPrefix;`, `d_nextPrefixSum;` — work arrays for building higher levels.

- MAS matrices and multilevel vectors:
  - `__GEIGEN__::MasMatrixT* d_MatMas;` — (unused in current build, kept for non‑symmetric experiments).
  - `__GEIGEN__::MasMatrixSymT* d_inverseMatMas;` — MAS block matrices (double precision), before inverting to form the preconditioner.
  - `__GEIGEN__::MasMatrixSymf* d_precondMatMas;` — final MAS **preconditioner blocks** (float precision).
  - `Eigen::Vector3f* d_multiLevelR;` — residual on all MAS nodes (fine + coarser).
  - `Precision_T3* d_multiLevelZ;` — preconditioned solution on MAS nodes.

- Contact and adjacency:
  - `unsigned int* d_neighborList;`, `d_neighborStart;`, `d_neighborStartTemp;`, `d_neighborNum;`
  - `unsigned int* d_neighborListInit;`, `d_neighborNumInit;` — base copy of neighbor graph for reordering.
  - `int4* _collisonPairs;` — contact pairs; used to add collision‑based edges into the MAS connectivity.

- METIS mapping:
  - `int* d_partId_map_real;` — mapping array (slot → real vertex).
  - `int* d_real_map_partId;` — remapping array (real vertex → slot).

Public interface:

- Setup:
  - `void initPreconditioner_Neighbor(int vertNum, int mCollision_node_offset, int totalNeighborNum, int4* m_collisonPairs, int partMapSize);`
  - `void computeNumLevels(int vertNum);`
  - `void initPreconditioner_Matrix();`
- Hierarchy building:
  - `int ReorderRealtime(int cpNum);`
  - `void BuildConnectMaskL0();`
  - `void PreparePrefixSumL0();`
  - `void BuildLevel1();`
  - `void BuildConnectMaskLx(int level);`
  - `void NextLevelCluster(int level);`
  - `void PrefixSumLx(int level);`
  - `void ComputeNextLevel(int level);`
  - `void AggregationKernel();`
  - `void BuildCollisionConnection(unsigned int* connectionMsk, int* coarseTableSpace, int level, int cpNum);`
- Matrix assembly:
  - `void setPreconditioner_bcoo(Eigen::Matrix3d* triplet_values, int* row_ids, int* col_ids, uint32_t* indices, int offset, int triplet_num, int cpNum);`
  - `void PrepareHessian_bcoo(...);`
- Application:
  - `void preconditioning(const double3* R, double3* Z);`
  - `void BuildMultiLevelR(const double3* R);`
  - `void SchwarzLocalXSym();`
  - `void SchwarzLocalXSym_block3();` (currently used)
  - `void SchwarzLocalXSym_sym();`
  - `void CollectFinalZ(double3* Z);`
- Cleanup:
  - `void FreeMAS();`

Two compile‑time flags influence the MAS behavior:
- `#define BANKSIZE 16` (`eigen_data.h`): MAS subdomain size (`N` in the paper).
- `#define GROUP` and `#define SYME` (`MASPreconditioner.cu`): enable METIS grouping and symmetric MAS storage.

---

## 5. MAS hierarchy construction (`ReorderRealtime`)

File:
- `StiffGIPC/MASPreconditioner.cu`

### 5.1 Level count (`computeNumLevels`)

- `computeNumLevels(int vertNum)` (`MASPreconditioner.cu:1724`) computes how many MAS levels to build for a given number of map nodes:
  - Starts with `levelSz = ceil(vertNum / BANKSIZE) * BANKSIZE`.
  - Repeatedly divides `levelSz` by `BANKSIZE` (rounding up to multiples of `BANKSIZE`) until `levelSz <= BANKSIZE`.
  - Adds one extra level and caps the total at `levelnum <= 6`.
  - This matches the multi‑level additive Schwarz hierarchy described in the paper, with a hard upper bound on level count for efficiency.

### 5.2 Entry (`ReorderRealtime`)

`int MASPreconditioner::ReorderRealtime(int cpNum)` (`MASPreconditioner.cu:1772`) builds or rebuilds the MAS hierarchy, optionally including contact connectivity:

1. Reset level size metadata (`cudaMemset(d_levelSize, 0, levelnum * sizeof(int2));`).
2. Build level‑0 connectivity from the FEM neighbor graph and METIS mapping:
   - `BuildConnectMaskL0()`:
     - With `GROUP` enabled, launches `_buildCML0_new` over `totalMapNodes` slots.
     - For each mapped slot `tdx`:
       - Computes `warpId = tdx / BANKSIZE`, `laneId = tdx % BANKSIZE`.
       - Finds the corresponding real vertex `idx = _partId_map_real[tdx]` (possibly `-1` for padding).
       - Builds a bitmask of neighbors within the same warp by mapping each neighbor with `_real_map_partId`.
       - Neighbors outside the current warp are kept in `d_neighborList` for higher‑level processing.
     - Writes:
       - `d_fineConnectMask[idx]` — local connectivity bitmask inside a MAS subdomain.
       - `d_neighborNum[idx]` — filtered neighbor count (outside‑subdomain connections).
   - Optional contact connectivity:
     - If `cpNum > 0`, `BuildCollisionConnection(d_fineConnectMask, nullptr, -1, cpNum)` augments `d_fineConnectMask` using `_buildCollisionConnection_new`, adding edges induced by contact pairs (using either `real_map_partId` or `coarseSpaceTables` depending on `level`).

3. Compute level‑0 clusters and level‑1 mapping:
   - `PreparePrefixSumL0()`:
     - With `GROUP` enabled, `_preparePrefixSumL0_new` treats each warp’s 16 nodes as a small graph.
     - Using repeated bitmask expansion, it computes the transitive closure of `d_fineConnectMask[idx]` inside the warp.
     - For each warp:
       - `d_prefixOriginal[warpId]` = number of connected components (clusters).
       - `d_fineConnectMask[idx]` now encodes the full cluster membership bitmask for node `idx`.
   - `BuildLevel1()`:
     - `_buildLevel1_new` uses `d_prefixOriginal` and its exclusive scan `d_prefixSumOriginal` to assign contiguous cluster IDs at level 1.
     - For each fine node `idx`:
       - Finds the “representative lane” in its cluster (`elected_lane = ffs(connMsk) - 1`).
       - Reads that lane’s cluster prefix to get the cluster ID.
       - Stores this in:
         - `d_coarseSpaceTables[idx]` — level‑0 → level‑1 mapping.
         - `d_goingNext[idx]` — global index in the MAS hierarchy for the node’s level‑1 representative.
     - Writes `d_levelSize[1] = (cluster_count, level1_offset)`; `level1_offset` is the starting global index for level‑1 nodes.

4. Build higher levels (`level = 1 .. levelnum - 1`):

For each `level`:

1. Clear `d_nextConnectMask`.
2. `BuildConnectMaskLx(level)`:
   - With `GROUP` enabled, `_buildConnectMaskLx_new`:
     - Reads level‑0 `d_fineConnectMask` and previous level’s `d_coarseSpaceTables`.
     - Compresses neighbor connections through the current level’s cluster mapping, producing coarser connectivity masks in `d_nextConnectMask`.
3. If `cpNum > 0`, call `BuildCollisionConnection(d_nextConnectMask, d_coarseSpaceTables, level, cpNum)`:
   - Adds collision‑induced edges at the current level, using `coarseSpaceTables[(level - 1) * vertNum + idx]` as coarse node IDs.
4. Copy `d_levelSize[level]` to host (`h_clevelSize`), then:
   - `NextLevelCluster(level)`:
     - `_nextLevelCluster` performs transitive closure on `d_nextConnectMask` to form connected components at this level.
     - Stores per‑warp cluster counts in `d_nextPrefix`.
   - `PrefixSumLx(level)`:
     - `_prefixSumLx` runs a similar procedure to `BuildLevel1`, computing:
       - `d_levelSize[level+1]` (`cluster_count`, `offset` for the next level),
       - `d_nextConnectMsk[idx]` as the cluster index for each node,
       - `d_goingNext[idx + levelBegin]` as the global index of the node’s representative at the next level.
   - `ComputeNextLevel(level)`:
     - `_computeNextLevel` populates `d_coarseSpaceTables[level * number + idx]` using the new cluster indices in `d_nextConnectMsk`.

5. After the loop:
   - Reads `d_levelSize[levelnum]` to get `totalNumberClusters = h_clevelSize.y;`.
   - `AggregationKernel()` (`_aggregationKernel`) builds `d_coarseTable`:
     - For each fine node `idx`, walks `d_goingNext` across all levels and stores the chain of representatives into an `__GEIGEN__::itable`.
     - This table is later used in `CollectFinalZ` to propagate coarse‑level corrections back to the fine mesh.

The result of `ReorderRealtime` is a multi‑level MAS hierarchy informed by:
- the FEM mesh connectivity (`getVertNeighbors`),
- METIS partitions (via `partId_map_real` / `real_map_partId`),
- and current contact pairs (`_collisonPairs`) when `cpNum > 0`.

---

## 6. MAS matrix assembly (`setPreconditioner_bcoo`, `PrepareHessian_bcoo`)

### 6.1 Entry (`setPreconditioner_bcoo`)

File:
- `MASPreconditioner::setPreconditioner_bcoo(...)` (`MASPreconditioner.cu:2165`)

Called from:
- `gipc::MAS_Preconditioner::assemble()` (`linear_system/preconditioner/fem_mas_preconditioner.cu:15`).

Steps:

1. Ensure `totalNodes > 0`.
2. Reset working neighbor graph from the initial copy:
   - `d_neighborList` ← `d_neighborListInit`
   - `d_neighborNum`  ← `d_neighborNumInit`
3. Rebuild MAS hierarchy with current collisions:
   - `ReorderRealtime(cpNum);` where `cpNum = *cpNum` from FEM subsystem (number of contact pairs).
4. Zero MAS block matrices:
   - With `SYME` defined:
     ```cpp
     cudaMemset(d_inverseMatMas, 0, totalNumberClusters / BANKSIZE * sizeof(MasMatrixSymT));
     ```
5. Build MAS matrices from the global Hessian:
   - `PrepareHessian_bcoo(triplet_values, row_ids, col_ids, indices, offset, triplet_num);`

### 6.2 Mapping global Hessian blocks into MAS blocks (`PrepareHessian_bcoo`)

File:
- `MASPreconditioner::PrepareHessian_bcoo(...)` (`MASPreconditioner.cu:1830+`)

Inputs:
- `triplet_values[I]` — `Eigen::Matrix3d` 3×3 block (Hessian block).
- `row_ids[I], col_ids[I]` — global row and column vertex indices.
- `indices` — an index indirection array (BCOO structure).
- `offset` — subsystem DOF offset (so that FEM subsystem’s rows are contiguous).

Phase 1: **Assign each 3×3 block to a MAS block matrix**

- Device lambda (first `ParallelFor`):
  - For each `I`:
    1. Resolve actual index: `int index = indices[I];`
    2. Read and shift global vertex indices:
       ```cpp
       auto vertRid_real = row_ids[index] - offset;
       auto vertCid_real = col_ids[index] - offset;
       ```
    3. Map to MAS slot indices:
       ```cpp
       int vertCid = _real_map_partId[vertCid_real];
       int vertRid = _real_map_partId[vertRid_real];
       int cPid    = vertCid / BANKSIZE; // MAS subdomain id
       ```
    4. If `vertCid / BANKSIZE == vertRid / BANKSIZE` (same level‑0 MAS subdomain):
       - For `vertCid >= vertRid`, compute local triangular index:
         ```cpp
         int bvRid = vertRid % BANKSIZE;
         int bvCid = vertCid % BANKSIZE;
         int index = BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
         ```
       - Write `H` directly into `d_inverseMatMas[cPid].M[index]`.
    5. Else (block spans multiple subdomains):
       - Climb the MAS hierarchy using `_goingNext`:
         - For `level = 1..levelnum-1`:
           - On level 1, map from real vertices:
             ```cpp
             vertCid = _goingNext[vertCid_real];
             vertRid = _goingNext[vertRid_real];
             ```
           - On higher levels, reuse `vertCid`, `vertRid` as MAS indices:
             ```cpp
             vertCid = _goingNext[vertCid];
             vertRid = _goingNext[vertRid];
             ```
           - Once `vertCid / BANKSIZE == vertRid / BANKSIZE`, accumulate into the corresponding MAS block using `atomicAdd` on each `H(i,j)` entry; for diagonal blocks, also add `H^T`.

This implements the assignment and “prolongation” of global Hessian blocks into coarser MAS blocks, as described in Algorithm 1 (lines 1–16) in the paper.

Phase 2: **Aggregate level‑0 blocks upwards using warp reduction**

- Second `ParallelFor` (`tripletNum = totalMapNodes * BANKSIZE`):
  - For each `(MRid, MCid)` pair corresponding to level‑0 MAS slots in a given subdomain:
    1. Decode local and global slot indices:
       ```cpp
       int HSIZE = BANKSIZE * BANKSIZE;
       int Hid   = idx / HSIZE; // MAS block id
       int LMRid = (idx % HSIZE) / BANKSIZE;
       int LMCid = (idx % HSIZE) % BANKSIZE;
       int MRid  = Hid * BANKSIZE + LMRid;
       int MCid  = Hid * BANKSIZE + LMCid;
       int rdx   = _partId_map_real[MRid]; // real row vertex
       int cdx   = _partId_map_real[MCid]; // real col vertex
       ```
    2. Load the symmetric 3×3 block for this pair (`mat3`).
    3. If both `rdx` and `cdx` are valid (non‑padding):
       - For subdomains where `prefix == 1` (single cluster in warp):
         - Perform a warp‑level reduction over `mat3` entries, aggregating contributions across all pairs in the cluster.
         - For each boundary thread, walk up the MAS hierarchy using `_goingNext[nextId]`, and `atomicAdd` the aggregated `mat3` into every level’s diagonal block.
       - Otherwise:
         - Walk up the hierarchy separately for each `(rdx, cdx)` pair until they enter the same subdomain, then `atomicAdd` `mat3` into that MAS block (similar to Phase 1 logic).

This phase corresponds to the GPU warp‑reduction optimization in Section 4.3 of the paper, which reduces the number of atomic operations when building coarser MAS matrices from dense level‑0 data.

### 6.3 Block inversion (`__inverse6_P96x96`)

File:
- `__inverse6_P96x96` (`MASPreconditioner.cu:620+`)

After `PrepareHessian_bcoo`:

1. Compute `number2 = totalNumberClusters * 3;`.
2. Launch `__inverse6_P96x96` with block size `32 * 3` over all blocks.
3. For each block:
   - Reconstruct the full `(BANKSIZE*3) × (BANKSIZE*3)` symmetric matrix from compressed storage in `d_inverseMatMas`.
   - Perform an in‑place Gauss–Jordan style inversion in shared memory.
   - Write the inverted block back into `d_precondMatMas` (float‑precision matrices).

These inverted MAS blocks approximate `M^{-1}` in the preconditioned system `M^{-1} A x = M^{-1} b`.

---

## 7. Applying the MAS preconditioner (`preconditioning`)

File:
- `MASPreconditioner::preconditioning(const double3* R, double3* Z)` (`MASPreconditioner.cu:2206`)

Called from:
- `gipc::MAS_Preconditioner::apply(...)` (`fem_mas_preconditioner.cu:33`).

Steps:

1. Early exit if `totalNodes < 1`.
2. Zero multilevel work arrays:
   - `d_multiLevelR[totalMapNodes .. totalNumberClusters)` is set to zero (coarser levels).
   - `d_multiLevelZ[0 .. totalNumberClusters)` is set to zero.
3. Build multilevel residual:
   - `BuildMultiLevelR(R);`
   - With `GROUP` enabled, `__buildMultiLevelR_optimized_new`:
     - For each mapped slot `pdx`:
       - Fetches real vertex index `idx = _partId_map_real[pdx]`.
       - Loads residual `r` from `R[idx]` (or zeros if padding).
       - Uses `d_fineConnectMask` and `d_prefixOriginal` to:
         - Reduce residuals across each connectivity cluster using warp‑level reductions.
         - For boundary slots, propagate aggregated residuals upwards along `_goingNext`, accumulating into the appropriate entries of `d_multiLevelR`.
   - This produces `d_multiLevelR` populated on level‑0 and coarser MAS nodes, matching the structure used when building `d_precondMatMas`.

4. Apply block preconditioner (local solves on MAS blocks):
   - `SchwarzLocalXSym_block3();`
   - `_schwarzLocalXSym6`:
     - For each MAS block and local pair `(vrid, vcid)`:
       - Loads `smR` (subdomain slice of `d_multiLevelR`).
       - Computes `rdata = Pred[Hid].M[index] * smR[lvcid];` or its transpose product.
       - Uses warp‑level segmented reduction to sum contributions for each row index `vrid`.
       - Atomically adds the result into `d_multiLevelZ[vrid]`.
   - This effectively computes `Z = M^{-1} R` on the MAS hierarchy using the precomputed block inverses.

5. Collect final preconditioned vector on fine FEM vertices:
   - `CollectFinalZ(Z);`
   - With `GROUP` enabled, `__collectFinalZ_new`:
     - For each fine FEM vertex `idx`:
       - Uses `real_map_partId[idx]` to get its level‑0 MAS slot.
       - Uses `d_coarseTable[idx]` to walk through all levels’ representatives.
       - Accumulates contributions from `d_multiLevelZ` along the hierarchy chain into the output vector `Z[idx]`.

This full sequence implements the MAS preconditioner application described in the paper: restrict residual to subdomains, solve local systems (using inverted MAS blocks), and prolongate corrections back to the global FEM DOFs.

---

## 8. Runtime configuration and key knobs

- **Subdomain size (`BANKSIZE`)**
  - Defined as `#define BANKSIZE 16` in `eigen_data.h`.
  - Must match `block_size` in the METIS preprocess (`metis_sort`) and the MAS data structure assumptions.
  - Compared with the original GPU MAS [Wu et al. 2022] using `BANKSIZE = 32`, this implementation uses **smaller subdomains (16)**, as described in the paper, made viable by the connectivity‑enhanced METIS partitioning.

- **Number of levels (`levelnum`)**
  - Computed in `computeNumLevels(maxNodes)`.
  - Upper‑bounded at 6, which limits hierarchy depth and memory usage.

- **Preconditioner choice (`P_type`)**
  - `PCG_Data::P_type` (`PCG_SOLVER.cuh:16`) controls whether MAS is used:
    - `P_type == 1` → `MAS_Preconditioner` is created (`gipc::GIPC::create_LinearSystem`).
    - Else → `DiagPreconditioner` is used.

- **Contact coupling**
  - `cpNum` (collision pair count) is passed from the FEM subsystem into `setPreconditioner_bcoo`.
  - When `cpNum > 0`, collision connectivity is incorporated at each level via `BuildCollisionConnection`, so MAS captures both FEM stiffness and contact constraints.

---

## 9. Practical entry points and reading order

If you want to trace or modify the MAS preconditioner, a practical reading order is:

1. `mas_codebase_guide.md` (this file) and the MAS section of the StiffGIPC paper.
2. High‑level wiring:
   - `gl_main.cu: initScene`, `setMAS_partition`, `initFEM`.
   - `gipc/gipc.cu: GIPC::create_LinearSystem`.
   - `linear_system/preconditioner/fem_mas_preconditioner.{h,cu}`.
3. Mesh preprocessing and partitioning:
   - `MeshProcess/metis_partition/metis_sort.{h,cpp}`.
   - `load_mesh.{h,cpp}` (`load_tetrahedraMesh`, `load_parts`, `getVertNeighbors`).
4. MAS core:
   - `MASPreconditioner.cuh` (data layout).
   - `MASPreconditioner.cu`:
     - `computeNumLevels`, `initPreconditioner_*`.
     - `ReorderRealtime` and its helper kernels.
     - `setPreconditioner_bcoo`, `PrepareHessian_bcoo`, `__inverse6_P96x96`.
     - `preconditioning`, `BuildMultiLevelR`, `SchwarzLocalXSym_*`, `CollectFinalZ`.

With these references, you can navigate from the conceptual MAS design (paper) down to the exact dataflow and kernels in this implementation.

