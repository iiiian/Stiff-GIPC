//
// main.cu
// StiffGIPC headless CLI
//

#include <GIPC.cuh>
#include <argparse/argparse.hpp>
#include <gipc/utils/json.h>
#include <metis_sort.h>

#include "cuda_tools/cuda_tools.h"
#include "femEnergy.cuh"
#include "fem_parameters.h"
#include "gpu_eigen_libs.cuh"
#include "load_mesh.h"

#include <Eigen/Geometry>
#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// MAS preconditioner setup code copied verbatimly from the original gl_main.cu.
// TODO: understand how this works.
void setMAS_partition(tetrahedra_obj& tetMesh)
{
    tetMesh.partId_map_real.assign(tetMesh.part_offset * BANKSIZE, -1);
    tetMesh.real_map_partId.resize(tetMesh.partId.size());

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

    index = 0;
    for(int i = 0; i < tetMesh.partId_map_real.size(); i++)
    {
        if(tetMesh.partId_map_real[i] == index)
        {
            tetMesh.real_map_partId[index] = i;
            index++;
        }
    }
}

int run_sort_dir(const fs::path& input_dir, const fs::path& output_dir)
{
    if(!fs::exists(input_dir) || !fs::is_directory(input_dir))
    {
        throw std::runtime_error("Sort input_dir is not a directory: "
                                 + input_dir.string());
    }

    fs::create_directories(output_dir);

    std::vector<fs::path> msh_files;
    for(const auto& entry : fs::directory_iterator(input_dir))
    {
        if(entry.is_regular_file() && entry.path().extension() == ".msh")
        {
            msh_files.push_back(entry.path());
        }
    }
    std::sort(msh_files.begin(), msh_files.end());

    if(msh_files.empty())
    {
        throw std::runtime_error("No .msh files found in: " + input_dir.string());
    }

    for(const auto& mesh_path : msh_files)
    {
        metis_sort(mesh_path.string(), 3, output_dir.string());
    }

    return 0;
}

void Init_CUDA()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if(cudaStatus != cudaSuccess)
    {
        throw std::runtime_error("cudaSetDevice failed");
    }
}

double3 read_vec3(const gipc::Json& j)
{
    const auto a = j.get<std::vector<double>>();
    if(a.size() != 3)
    {
        throw std::runtime_error("Expected a vec3 array");
    }
    return make_double3(a[0], a[1], a[2]);
}

Eigen::Matrix4d read_transform(const gipc::Json& j)
{
    const double scale = j.at("scale").get<double>();
    const auto   t     = j.at("translation").get<std::vector<double>>();
    if(t.size() != 3)
    {
        throw std::runtime_error("Expected transform.translation vec3");
    }

    Eigen::Matrix4d transform   = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * scale;
    transform.block<3, 1>(0, 3) = Eigen::Vector3d{t[0], t[1], t[2]};
    return transform;
}

void apply_settings(GIPC&             ipc,
                    double&           collision_detection_buff_scale,
                    double&           motion_rate,
                    const gipc::Json& s)
{
    ipc.density           = s.at("volume_mesh_density").get<double>();
    ipc.PoissonRate       = s.at("poisson_rate").get<double>();
    ipc.frictionRate      = s.at("friction_rate").get<double>();
    ipc.gd_frictionRate   = s.at("gd_friction_rate").get<double>();
    ipc.clothThickness    = s.at("triangle_mesh_thickness").get<double>();
    ipc.clothYoungModulus = s.at("triangle_mesh_youngs_modulus").get<double>();
    ipc.bendYoungModulus  = s.at("triangle_bend_youngs_modulus").get<double>();
    ipc.clothDensity      = s.at("triangle_mesh_density").get<double>();
    ipc.strainRate        = s.at("strain_rate").get<double>();
    ipc.softMotionRate    = s.at("motion_stiffness").get<double>();

    collision_detection_buff_scale =
        s.at("collision_detection_buff_scale").get<double>();
    motion_rate           = s.at("motion_rate").get<double>();
    ipc.IPC_dt            = s.at("ipc_time_step").get<double>();
    ipc.pcg_rel_threshold = s.at("pcg_rel_threshold").get<double>();
    ipc.pcg_abs_threshold = s.at("pcg_abs_threshold").get<double>();
    ipc.pcg_use_preconditioned_norm = s.at("pcg_use_preconditioned_norm").get<bool>();
    ipc.Newton_solver_threshold = s.at("Newton_solver_threshold").get<double>();
    ipc.relative_dhat           = s.at("IPC_ralative_dHat").get<double>();

    // As far as I am awared, below lame parameters are not used anywhere.
    // For FEM simulation, the true lame parameter is computed in initFEM().
    ipc.lengthRateLame = ipc.YoungModulus / (2 * (1 + ipc.PoissonRate));
    ipc.volumeRateLame = ipc.YoungModulus * ipc.PoissonRate
                         / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
    ipc.lengthRate = 4 * ipc.lengthRateLame / 3;
    ipc.volumeRate = ipc.volumeRateLame + 5 * ipc.lengthRateLame / 6;

    // For cloth.
    ipc.stretchStiff = ipc.clothYoungModulus / (2 * (1 + ipc.PoissonRate));
    ipc.bendStiff    = ipc.bendYoungModulus * pow(ipc.clothThickness, 3)
                    / (24 * (1 - ipc.PoissonRate * ipc.PoissonRate));
    ipc.shearStiff = 0.03 * ipc.stretchStiff * ipc.strainRate;
}

void initFEM(tetrahedra_obj& mesh, const GIPC& ipc)
{
    double massSum   = 0;
    double volumeSum = 0;

    for(int i = 0; i < mesh.tetrahedraNum; i++)
    {
        __GEIGEN__::Matrix3x3d DM;
        __calculateDms3D_double(mesh.vertexes.data(), mesh.tetrahedras[i], DM);

        __GEIGEN__::Matrix3x3d DM_inverse;
        __GEIGEN__::__Inverse(DM, DM_inverse);

        double vlm = calculateVolum(mesh.vertexes.data(), mesh.tetrahedras[i]);

        mesh.masses[mesh.tetrahedras[i].x] += vlm * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].y] += vlm * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].z] += vlm * ipc.density / 4;
        mesh.masses[mesh.tetrahedras[i].w] += vlm * ipc.density / 4;

        massSum += vlm * ipc.density;
        volumeSum += vlm;
        mesh.DM_inverse.push_back(DM_inverse);
        mesh.volum.push_back(vlm);


        double lengthRateLame =
            mesh.vert_youngth_modules[i] / (2 * (1 + ipc.PoissonRate));
        double volumeRateLame = mesh.vert_youngth_modules[i] * ipc.PoissonRate
                                / ((1 + ipc.PoissonRate) * (1 - 2 * ipc.PoissonRate));
        // Since we are no longer using SNK energy, remove the lame parameter offsets.
        double lengthRate = lengthRateLame;
        double volumeRate = volumeRateLame;

        mesh.lengthRate.push_back(lengthRate);
        mesh.volumeRate.push_back(volumeRate);
    }

    // Cloth mesh setup is removed here.
    // Needs to add it back if we want to support 2D mesh in the future.

    mesh.meanMass  = massSum / mesh.vertexNum;
    mesh.meanVolum = volumeSum / mesh.vertexNum;
}

void write_obj(const fs::path& path, const tetrahedra_obj& mesh)
{
    std::ofstream out(path);
    if(!out)
    {
        throw std::runtime_error("Failed to open output obj: " + path.string());
    }

    for(const auto& v : mesh.vertexes)
    {
        out << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }
    for(const auto& f : mesh.surface)
    {
        out << "f " << f.x + 1 << " " << f.y + 1 << " " << f.z + 1 << "\n";
    }
}

fs::path frame_obj_path(const fs::path& dir, int frame)
{
    std::ostringstream ss;
    ss << "frame_";
    ss << std::setw(5) << std::setfill('0') << frame;
    ss << ".obj";
    return dir / ss.str();
}

int main(int argc, char** argv)
{
    argparse::ArgumentParser program("gipc");
    program.add_argument("-j", "--json");
    program.add_argument("--sort").help(
        "offline metis sort mode: --sort <input_dir> --output <output_dir>");
    program.add_argument("-o", "--output").required();
    try
    {
        program.parse_args(argc, argv);
    }
    catch(const std::exception& err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    const auto output_dir = program.get<std::string>("--output");

    if(program.is_used("--sort"))
    {
        const auto input_dir = program.get<std::string>("--sort");
        return run_sort_dir(input_dir, output_dir);
    }

    if(!program.is_used("--json"))
    {
        std::cerr << "Missing required argument: --json\n";
        std::cerr << program;
        return 1;
    }

    const auto scene_path = program.get<std::string>("--json");

    gipc::Json scene = gipc::Json::parse(std::ifstream(scene_path));

    Init_CUDA();

    // This is the aggregated mesh class that contains all objects.
    tetrahedra_obj   tetMesh;
    device_TetraData d_tetMesh;
    GIPC             ipc;

    // Apply parameters in json.
    double collision_detection_buff_scale = 1.0;
    double motion_rate                    = 1.0;
    apply_settings(ipc, collision_detection_buff_scale, motion_rate, scene.at("settings"));

    const int frames = scene.at("simulation").at("frames").get<int>();
    const int preconditioner_type =
        scene.at("simulation").at("preconditioner_type").get<int>();
    ipc.pcg_data.P_type = preconditioner_type;

    // This call stores a reference to d_tetMesh.
    ipc.build_gipc_system(d_tetMesh);

    const auto& objects = scene.at("objects");
    for(const auto& obj : objects)
    {
        const auto  mesh_path     = obj.at("mesh_msh").get<std::string>();
        const auto  part_file     = obj.at("part_file").get<std::string>();
        const bool  is_obstacle   = obj.at("is_obstacle").get<bool>();
        const auto  young_modulus = obj.at("young_modulus").get<double>();
        const auto  transform     = read_transform(obj.at("transform"));
        const auto  init_vel      = read_vec3(obj.at("initial_velocity"));
        const auto& pin_boxes     = obj.at("pin_boxes");

        // This is for ABD system only and has no effect on FEM object.
        const BodyBoundaryType boundary = BodyBoundaryType::Free;

        const int v_begin = static_cast<int>(tetMesh.vertexes.size());

        tetMesh.load_tetrahedraMesh(
            mesh_path, transform, young_modulus, gipc::BodyType::FEM, boundary);

        // If precondition type !=0, we need to load part file for MAS preconditioner.
        // if type == 0, this is a simple diagonal preconditioner.
        // see gipc.cu.
        if(preconditioner_type != 0)
        {
            // This file map vertices to MAS parts.
            if(!tetMesh.load_parts(part_file))
            {
                throw std::runtime_error("Failed to load part file: " + part_file);
            }
        }

        const int v_end = static_cast<int>(tetMesh.vertexes.size());

        if(is_obstacle)
        {
            for(int i = v_begin; i < v_end; ++i)
            {
                // By default this is 0 (free).
                // Set to 1 (fixed).
                // 2 should be motor?
                tetMesh.boundaryTypies[i] = 1;
            }
        }

        for(int i = v_begin; i < v_end; ++i)
        {
            tetMesh.velocities[i] = init_vel;
        }

        // Pin vertices inside box selection.
        for(const auto& box : pin_boxes)
        {
            const double3 bmin = read_vec3(box.at("min"));
            const double3 bmax = read_vec3(box.at("max"));
            for(int i = v_begin; i < v_end; ++i)
            {
                const auto& p = tetMesh.vertexes[i];
                if(p.x >= bmin.x && p.x <= bmax.x && p.y >= bmin.y
                   && p.y <= bmax.y && p.z >= bmin.z && p.z <= bmax.z)
                {
                    tetMesh.boundaryTypies[i] = 1;
                }
            }
        }
    }

    // MAS preconditioner prep.
    if(preconditioner_type != 0)
    {
        setMAS_partition(tetMesh);
    }

    tetMesh.getSurface();
    initFEM(tetMesh, ipc);

    d_tetMesh.Malloc_DEVICE_MEM(tetMesh.vertexNum,
                                tetMesh.tetrahedraNum,
                                tetMesh.triangleNum,
                                tetMesh.softNum,
                                tetMesh.tri_edges.size(),
                                tetMesh.abd_fem_count_info.total_body_num());

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.masses,
                              tetMesh.masses.data(),
                              tetMesh.vertexNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.lengthRate,
                              tetMesh.lengthRate.data(),
                              tetMesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.volumeRate,
                              tetMesh.volumeRate.data(),
                              tetMesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.volum,
                              tetMesh.volum.data(),
                              tetMesh.tetrahedraNum * sizeof(double),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.vertexes,
                              tetMesh.vertexes.data(),
                              tetMesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.o_vertexes,
                              tetMesh.vertexes.data(),
                              tetMesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tetrahedras,
                              tetMesh.tetrahedras.data(),
                              tetMesh.tetrahedraNum * sizeof(uint4),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.DmInverses,
                              tetMesh.DM_inverse.data(),
                              tetMesh.tetrahedraNum * sizeof(__GEIGEN__::Matrix3x3d),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.BoundaryType,
                              tetMesh.boundaryTypies.data(),
                              tetMesh.vertexNum * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.velocities,
                              tetMesh.velocities.data(),
                              tetMesh.vertexNum * sizeof(double3),
                              cudaMemcpyHostToDevice));

    // TriMesh setup is removed here.

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.body_id_to_boundary_type,
                              tetMesh.body_id_to_is_fixed.data(),
                              tetMesh.body_id_to_is_fixed.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.point_id_to_body_id,
                              tetMesh.point_id_to_body_id.data(),
                              tetMesh.point_id_to_body_id.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.tet_id_to_body_id,
                              tetMesh.tet_id_to_body_id.data(),
                              tetMesh.tet_id_to_body_id.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

    ipc.vertexNum      = tetMesh.vertexNum;
    ipc.tetrahedraNum  = tetMesh.tetrahedraNum;
    ipc._vertexes      = d_tetMesh.vertexes;
    ipc._rest_vertexes = d_tetMesh.rest_vertexes;
    ipc.surf_vertexNum = tetMesh.surfVerts.size();
    ipc.surface_Num    = tetMesh.surface.size();
    ipc.edge_Num       = tetMesh.surfEdges.size();
    ipc.tri_edge_num   = tetMesh.tri_edges.size();

    ipc.MAX_CCD_COLLITION_PAIRS_NUM =
        1 * collision_detection_buff_scale
        * (((double)(ipc.surface_Num * 15 + ipc.edge_Num * 10))
           * std::max((ipc.IPC_dt / 0.01), 2.0));
    ipc.MAX_COLLITION_PAIRS_NUM = (ipc.surf_vertexNum * 3 + ipc.edge_Num * 2)
                                  * 3 * collision_detection_buff_scale;

    ipc.triangleNum        = tetMesh.triangleNum;
    ipc.targetVert         = d_tetMesh.targetVert;
    ipc.targetInd          = d_tetMesh.targetIndex;
    ipc.softNum            = tetMesh.softNum;
    ipc.abd_fem_count_info = tetMesh.abd_fem_count_info;

    ipc.MALLOC_DEVICE_MEM();

    // Ground collision is disabled (ipc.useGround = false by default)

    CUDA_SAFE_CALL(cudaMemcpy(
        ipc._faces, tetMesh.surface.data(), ipc.surface_Num * sizeof(uint3), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(
        ipc._edges, tetMesh.surfEdges.data(), ipc.edge_Num * sizeof(uint2), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(ipc._surfVerts,
                              tetMesh.surfVerts.data(),
                              ipc.surf_vertexNum * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

    ipc.initBVH(d_tetMesh.BoundaryType, d_tetMesh.point_id_to_body_id);

    if(preconditioner_type != 0)
    {
        int neighborListSize = tetMesh.getVertNeighbors();
        ipc.pcg_data.MP.initPreconditioner_Neighbor(ipc.vertexNum - tetMesh.abd_vertexOffset,
                                                    tetMesh.abd_vertexOffset,
                                                    neighborListSize,
                                                    ipc._collisonPairs,
                                                    tetMesh.part_offset * BANKSIZE);

        ipc.pcg_data.MP.neighborListSize = neighborListSize;

        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborListInit,
                                  tetMesh.neighborList.data(),
                                  neighborListSize * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart,
                                  tetMesh.neighborStart.data(),
                                  (ipc.vertexNum - tetMesh.abd_vertexOffset) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborNumInit,
                                  tetMesh.neighborNum.data(),
                                  (ipc.vertexNum - tetMesh.abd_vertexOffset) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_partId_map_real,
                                  tetMesh.partId_map_real.data(),
                                  tetMesh.part_offset * BANKSIZE * sizeof(int),
                                  cudaMemcpyHostToDevice));

        CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_real_map_partId,
                                  tetMesh.real_map_partId.data(),
                                  tetMesh.real_map_partId.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));

        ipc.pcg_data.MP.initPreconditioner_Matrix();
    }

    CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.rest_vertexes,
                              d_tetMesh.o_vertexes,
                              ipc.vertexNum * sizeof(double3),
                              cudaMemcpyDeviceToDevice));

    ipc.buildBVH();
    ipc.init(tetMesh.meanMass, tetMesh.meanVolum, tetMesh.minConer, tetMesh.maxConer);
    ipc.buildCP();

    ipc._moveDir          = ipc.pcg_data.dx;
    ipc.animation_subRate = 1.0 / motion_rate;
    ipc.computeXTilta(d_tetMesh, 1);

    ipc.create_LinearSystem(d_tetMesh);

    fs::create_directories(output_dir);

    for(int frame = 0; frame < frames; ++frame)
    {
        ipc.IPC_Solver(d_tetMesh);
        CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(),
                                  ipc._vertexes,
                                  ipc.vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToHost));
        write_obj(frame_obj_path(output_dir, frame), tetMesh);
    }

    return 0;
}
