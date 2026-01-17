//
// main.cu
// StiffGIPC headless CLI
//

#include <GIPC.cuh>
#include <argparse/argparse.hpp>
#include <gipc/gipc.h>
#include <gipc/utils/json.h>
#include <gipc/utils/timer.h>
#include <gipc/statistics.h>
#include <metis_sort.h>

#include "cuda_tools/cuda_tools.h"
#include "femEnergy.cuh"
#include "fem_parameters.h"
#include "gpu_eigen_libs.cuh"
#include "load_mesh.h"

#include <Eigen/Geometry>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

extern "C"
{
#include <mmio.h>
}

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
        fs::path ext = entry.path().extension();
        if(entry.is_regular_file() && (ext == ".msh" || ext == ".mesh"))
        {
            msh_files.push_back(entry.path());
        }
    }
    std::sort(msh_files.begin(), msh_files.end());

    if(msh_files.empty())
    {
        throw std::runtime_error("No gmsh files found in: " + input_dir.string());
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

TransformParams read_transform(const gipc::Json& j)
{
    const auto r = j.at("rotation").get<std::vector<double>>();
    if(r.size() != 3)
    {
        throw std::runtime_error("Expected transform.rotation vec3");
    }

    const double scale = j.at("scale").get<double>();

    const auto t = j.at("translation").get<std::vector<double>>();
    if(t.size() != 3)
    {
        throw std::runtime_error("Expected transform.translation vec3");
    }

    // Convert rotation from degrees to radians
    constexpr double deg2rad = M_PI / 180.0;

    TransformParams transform;
    transform.rotation = Eigen::Vector3d{r[0] * deg2rad, r[1] * deg2rad, r[2] * deg2rad};
    transform.scale       = scale;
    transform.translation = Eigen::Vector3d{t[0], t[1], t[2]};
    return transform;
}

struct BoxSelection
{
    double3 min;
    double3 max;
};

struct AnimationConstraint
{
    int                       object = -1;  // objects[object]
    std::vector<BoxSelection> boxes;
    double3                   rot_origin   = {0.0, 0.0, 0.0};
    double3                   rot_axis     = {1.0, 0.0, 0.0};
    double                    rot_velocity = 0.0;  // radians/sec
};

struct SoftTargetRotation
{
    double3 origin;
    double3 axis_unit;
    double  angular_velocity;  // radians/sec
};

double3 vec3_sub(const double3& a, const double3& b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

double3 vec3_add(const double3& a, const double3& b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

double3 vec3_scale(const double3& a, double s)
{
    return make_double3(a.x * s, a.y * s, a.z * s);
}

double vec3_dot(const double3& a, const double3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double3 vec3_cross(const double3& a, const double3& b)
{
    return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

double3 vec3_normalize(const double3& a)
{
    const double n2 = vec3_dot(a, a);
    if(n2 <= 0.0)
    {
        throw std::runtime_error("Cannot normalize zero-length axis");
    }
    return vec3_scale(a, 1.0 / std::sqrt(n2));
}

double3 rotate_axis_angle(const double3& p_world,
                          const double3& origin_world,
                          const double3& axis_unit_world,
                          double         theta)
{
    // Rodrigues' rotation formula around axis through origin_world.
    const double3 p = vec3_sub(p_world, origin_world);
    const double  c = std::cos(theta);
    const double  s = std::sin(theta);

    const double3 term0 = vec3_scale(p, c);
    const double3 term1 = vec3_scale(vec3_cross(axis_unit_world, p), s);
    const double3 term2 =
        vec3_scale(axis_unit_world, vec3_dot(axis_unit_world, p) * (1.0 - c));

    return vec3_add(origin_world, vec3_add(vec3_add(term0, term1), term2));
}

bool point_in_box(const double3& p, const BoxSelection& box)
{
    return p.x >= box.min.x && p.x <= box.max.x && p.y >= box.min.y
           && p.y <= box.max.y && p.z >= box.min.z && p.z <= box.max.z;
}

std::vector<AnimationConstraint> read_animation(const gipc::Json& j)
{
    std::vector<AnimationConstraint> cts;
    for(const auto& cj : j)
    {
        AnimationConstraint c;
        c.object = cj.at("object").get<int>();

        for(const auto& bj : cj.at("boxes"))
        {
            BoxSelection box;
            box.min = read_vec3(bj.at("min"));
            box.max = read_vec3(bj.at("max"));
            c.boxes.push_back(std::move(box));
        }

        c.rot_origin   = read_vec3(cj.at("rot_origin"));
        c.rot_axis     = vec3_normalize(read_vec3(cj.at("rot_axis")));
        c.rot_velocity = cj.at("rot_velocity").get<double>();

        cts.push_back(std::move(c));
    }

    return cts;
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
    ipc.gravity           = read_vec3(s.at("gravity"));

    collision_detection_buff_scale =
        s.at("collision_detection_buff_scale").get<double>();
    motion_rate           = s.at("motion_rate").get<double>();
    ipc.IPC_dt            = s.at("ipc_time_step").get<double>();
    ipc.pcg_rel_threshold = s.at("pcg_rel_threshold").get<double>();
    ipc.pcg_abs_threshold = s.at("pcg_abs_threshold").get<double>();
    ipc.pcg_use_preconditioned_norm = s.at("pcg_use_preconditioned_norm").get<bool>();
    ipc.pcg_max_iter     = s.at("pcg_max_iter").get<int>();
    ipc.abs_xdelta_tol   = s.at("abs_xdelta_tol").get<double>();
    ipc.rel_xdelta_tol   = s.at("rel_xdelta_tol").get<double>();
    ipc.relative_dhat    = s.at("IPC_ralative_dHat").get<double>();
    ipc.armijo_c1        = s.at("armijo_c1").get<double>();
    ipc.armijo_beta      = s.at("armijo_beta").get<double>();
    ipc.armijo_alpha_min = s.at("armijo_alpha_min").get<double>();

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

void initFEM(tetrahedra_obj& mesh)
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

        const double density = mesh.tet_densities[i];

        mesh.masses[mesh.tetrahedras[i].x] += vlm * density / 4;
        mesh.masses[mesh.tetrahedras[i].y] += vlm * density / 4;
        mesh.masses[mesh.tetrahedras[i].z] += vlm * density / 4;
        mesh.masses[mesh.tetrahedras[i].w] += vlm * density / 4;

        massSum += vlm * density;
        volumeSum += vlm;
        mesh.DM_inverse.push_back(DM_inverse);
        mesh.volum.push_back(vlm);


        const double poisson_ratio = mesh.tet_poisson_ratios[i];

        double lengthRateLame = mesh.vert_youngth_modules[i] / (2 * (1 + poisson_ratio));
        double volumeRateLame = mesh.vert_youngth_modules[i] * poisson_ratio
                                / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio));
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

namespace
{
struct MmCoordinateMatrixReal
{
    int               M         = 0;
    int               N         = 0;
    int               nz        = 0;
    bool              symmetric = false;
    std::vector<int>  I;  // 0-based
    std::vector<int>  J;  // 0-based
    std::vector<double> V;
};

MmCoordinateMatrixReal read_mm_coordinate_real(const fs::path& path)
{
    FILE* f = std::fopen(path.string().c_str(), "r");
    if(!f)
    {
        throw std::runtime_error("Failed to open MatrixMarket file: " + path.string());
    }

    MM_typecode matcode;
    if(mm_read_banner(f, &matcode) != 0)
    {
        std::fclose(f);
        throw std::runtime_error("Failed to read MatrixMarket banner: " + path.string());
    }

    if(!(mm_is_matrix(matcode) && mm_is_coordinate(matcode) && mm_is_real(matcode)))
    {
        std::string s = mm_typecode_to_str(matcode);
        std::fclose(f);
        throw std::runtime_error("Unsupported MatrixMarket type for A (need coordinate real matrix), got: " + s);
    }

    int M = 0, N = 0, nz = 0;
    if(mm_read_mtx_crd_size(f, &M, &N, &nz) != 0)
    {
        std::fclose(f);
        throw std::runtime_error("Failed to read MatrixMarket coordinate size: " + path.string());
    }

    MmCoordinateMatrixReal A;
    A.M         = M;
    A.N         = N;
    A.nz        = nz;
    A.symmetric = mm_is_symmetric(matcode) != 0;
    A.I.resize(nz);
    A.J.resize(nz);
    A.V.resize(nz);

    if(mm_read_mtx_crd_data(f, M, N, nz, A.I.data(), A.J.data(), A.V.data(), matcode) != 0)
    {
        std::fclose(f);
        throw std::runtime_error("Failed to read MatrixMarket coordinate entries: " + path.string());
    }
    std::fclose(f);

    // Convert to 0-based indices.
    for(int k = 0; k < nz; ++k)
    {
        A.I[k] -= 1;
        A.J[k] -= 1;
    }

    return A;
}

std::vector<gipc::Float> read_mm_array_vector_mx1(const fs::path& path)
{
    FILE* f = std::fopen(path.string().c_str(), "r");
    if(!f)
    {
        throw std::runtime_error("Failed to open MatrixMarket file: " + path.string());
    }

    MM_typecode matcode;
    if(mm_read_banner(f, &matcode) != 0)
    {
        std::fclose(f);
        throw std::runtime_error("Failed to read MatrixMarket banner: " + path.string());
    }

    if(!(mm_is_matrix(matcode) && mm_is_array(matcode) && mm_is_real(matcode)))
    {
        std::string s = mm_typecode_to_str(matcode);
        std::fclose(f);
        throw std::runtime_error("Unsupported MatrixMarket type for b (need array real matrix), got: " + s);
    }

    int M = 0, N = 0;
    if(mm_read_mtx_array_size(f, &M, &N) != 0)
    {
        std::fclose(f);
        throw std::runtime_error("Failed to read MatrixMarket array size: " + path.string());
    }

    if(N != 1)
    {
        std::fclose(f);
        throw std::runtime_error("b must be MatrixMarket array of shape (M x 1): " + path.string());
    }

    std::vector<gipc::Float> b(M);
    for(int i = 0; i < M; ++i)
    {
        double v = 0.0;
        if(std::fscanf(f, "%lg", &v) != 1)
        {
            std::fclose(f);
            throw std::runtime_error("Failed to read b entry " + std::to_string(i) + " from: " + path.string());
        }
        b[i] = static_cast<gipc::Float>(v);
    }

    std::fclose(f);
    return b;
}

struct BlockTriplets
{
    int                        block_rows = 0;
    int                        block_cols = 0;
    std::vector<int>           br;
    std::vector<int>           bc;
    std::vector<Eigen::Matrix3d> bv;
};

static uint64_t pack_block_key(uint32_t r, uint32_t c)
{
    return (uint64_t{r} << 32) | uint64_t{c};
}

static std::pair<uint32_t, uint32_t> unpack_block_key(uint64_t key)
{
    return {static_cast<uint32_t>(key >> 32), static_cast<uint32_t>(key & 0xffffffffu)};
}

BlockTriplets to_upper_triangle_block_triplets(const MmCoordinateMatrixReal& A)
{
    if(A.M != A.N)
    {
        throw std::runtime_error("A must be square");
    }
    if(A.M % 3 != 0)
    {
        throw std::runtime_error("A row count must be divisible by 3 (block size)");
    }

    const int block_n = A.M / 3;

    std::unordered_map<uint64_t, Eigen::Matrix3d> blocks;
    blocks.reserve(static_cast<size_t>(A.nz));

    std::vector<char> diag_present(block_n, 0);

    for(int k = 0; k < A.nz; ++k)
    {
        int r = A.I[k];
        int c = A.J[k];

        if(r < 0 || c < 0 || r >= A.M || c >= A.N)
        {
            throw std::runtime_error("A contains out-of-range indices");
        }

        int bi = r / 3;
        int bj = c / 3;
        int li = r % 3;
        int lj = c % 3;

        // Canonicalize to upper triangle at the *block* level.
        // SpMV expects full 3x3 blocks for (bi <= bj), and handles the symmetric transpose for off-diagonals.
        if(bi > bj)
        {
            if(A.symmetric)
            {
                std::swap(bi, bj);
                std::swap(li, lj);  // transpose within the block
            }
            else
            {
                continue;  // drop lower-triangle blocks to avoid double counting
            }
        }

        auto key = pack_block_key(static_cast<uint32_t>(bi), static_cast<uint32_t>(bj));
        auto [it, inserted] = blocks.try_emplace(key, Eigen::Matrix3d::Zero());
        it->second(li, lj) += A.V[k];
        if(A.symmetric && bi == bj && li != lj)
        {
            // Symmetric MatrixMarket files typically store only one triangle of the scalar matrix.
            // Ensure diagonal 3x3 blocks are fully populated.
            it->second(lj, li) += A.V[k];
        }
        if(bi == bj)
            diag_present[bi] = 1;
    }

    for(int i = 0; i < block_n; ++i)
    {
        if(!diag_present[i])
        {
            throw std::runtime_error("A is missing diagonal 3x3 block at block index " + std::to_string(i));
        }
    }

    std::vector<uint64_t> keys;
    keys.reserve(blocks.size());
    for(const auto& kv : blocks)
    {
        keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end(), [](uint64_t a, uint64_t b) {
        auto [ar, ac] = unpack_block_key(a);
        auto [br, bc] = unpack_block_key(b);
        return (ar < br) || (ar == br && ac < bc);
    });

    BlockTriplets out;
    out.block_rows = block_n;
    out.block_cols = block_n;
    out.br.reserve(keys.size());
    out.bc.reserve(keys.size());
    out.bv.reserve(keys.size());
    for(uint64_t key : keys)
    {
        auto [r, c] = unpack_block_key(key);
        out.br.push_back(static_cast<int>(r));
        out.bc.push_back(static_cast<int>(c));
        out.bv.push_back(blocks.at(key));
    }
    return out;
}

void upload_triplets_to_device(GIPCTripletMatrix& dst, const BlockTriplets& src)
{
    if(dst.block_rows() != src.block_rows || dst.block_cols() != src.block_cols)
    {
        throw std::runtime_error("Loaded A block dimensions do not match configured system size");
    }

    const size_t nnz = src.br.size();
    dst.resize_triplets(nnz);
    if(nnz > 0)
    {
        CUDA_SAFE_CALL(cudaMemcpy(dst.block_row_indices(),
                                  src.br.data(),
                                  sizeof(int) * nnz,
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(dst.block_col_indices(),
                                  src.bc.data(),
                                  sizeof(int) * nnz,
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(dst.block_values(),
                                  src.bv.data(),
                                  sizeof(Eigen::Matrix3d) * nnz,
                                  cudaMemcpyHostToDevice));
    }

    dst.h_unique_key_number = static_cast<int>(nnz);
}
}  // namespace

int main(int argc, char** argv)
{
    argparse::ArgumentParser program("gipc");
    program.add_argument("-j", "--json");
    program.add_argument("--sort").help(
        "offline metis sort mode: --sort <input_dir> --output <output_dir>");
    program.add_argument("--A").help("Solve provided linear system: path to MatrixMarket A (coordinate, real)");
    program.add_argument("--b").help("Solve provided linear system: path to MatrixMarket b (array, real, Mx1)");
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

    const bool write_obj_frames =
        scene.at("simulation").at("write_obj_frames").get<bool>();

    const auto export_linear_system_frames =
        scene.at("simulation").at("export_linear_system_frames").get<std::vector<int>>();

    // This call stores a reference to d_tetMesh.
    ipc.build_gipc_system(d_tetMesh);

    const auto animation = read_animation(scene.at("animation"));
    std::vector<std::pair<int, int>> object_vertex_ranges;
    object_vertex_ranges.reserve(scene.at("objects").size());

    double3 non_obstacle_min = make_double3(std::numeric_limits<double>::infinity(),
                                            std::numeric_limits<double>::infinity(),
                                            std::numeric_limits<double>::infinity());
    double3 non_obstacle_max =
        make_double3(-std::numeric_limits<double>::infinity(),
                     -std::numeric_limits<double>::infinity(),
                     -std::numeric_limits<double>::infinity());

    const auto& objects = scene.at("objects");
    for(const auto& obj : objects)
    {
        const auto  mesh_path     = obj.at("mesh_msh").get<std::string>();
        const auto  part_file     = obj.at("part_file").get<std::string>();
        const bool  is_obstacle   = obj.at("is_obstacle").get<bool>();
        const auto  young_modulus = obj.at("young_modulus").get<double>();
        const auto  density       = obj.at("density").get<double>();
        const auto  poisson_ratio = obj.at("poisson_ratio").get<double>();
        const auto  transform     = read_transform(obj.at("transform"));
        const auto  init_vel      = read_vec3(obj.at("initial_velocity"));
        const auto& pin_boxes     = obj.at("pin_boxes");

        // This is for ABD system only and has no effect on FEM object.
        const BodyBoundaryType boundary = BodyBoundaryType::Free;

        const int v_begin = tetMesh.vertexes.size();

        tetMesh.load_tetrahedraMesh(
            mesh_path, transform, young_modulus, density, poisson_ratio, gipc::BodyType::FEM, boundary);

        if(!is_obstacle)
        {
            non_obstacle_min.x = std::min(non_obstacle_min.x, tetMesh.minTConer.x);
            non_obstacle_min.y = std::min(non_obstacle_min.y, tetMesh.minTConer.y);
            non_obstacle_min.z = std::min(non_obstacle_min.z, tetMesh.minTConer.z);
            non_obstacle_max.x = std::max(non_obstacle_max.x, tetMesh.maxTConer.x);
            non_obstacle_max.y = std::max(non_obstacle_max.y, tetMesh.maxTConer.y);
            non_obstacle_max.z = std::max(non_obstacle_max.z, tetMesh.maxTConer.z);
        }

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

        const int v_end = tetMesh.vertexes.size();
        object_vertex_ranges.emplace_back(v_begin, v_end);

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

    // Build animated soft constraints using box selection.
    std::vector<SoftTargetRotation> soft_target_rotations;  // rotation animation for selected vertices.
    std::vector<double3> host_target_vertices;  // index of selected vertices.
    tetMesh.targetIndex.clear();
    tetMesh.softNum = 0;
    if(!animation.empty())
    {
        std::vector<char> has_soft(tetMesh.vertexes.size(), 0);

        for(const auto& c : animation)
        {
            if(c.object < 0 || c.object >= object_vertex_ranges.size())
            {
                throw std::runtime_error("animation.constraints[].object out of range");
            }

            const auto [v_begin, v_end] = object_vertex_ranges[c.object];
            assert(v_end >= v_begin);

            std::vector<char> selected(v_end - v_begin, 0);
            for(const auto& box : c.boxes)
            {
                for(int vi = v_begin; vi < v_end; ++vi)
                {
                    if(!point_in_box(tetMesh.vertexes[vi], box))
                    {
                        continue;
                    }

                    if(has_soft[vi])
                    {
                        throw std::runtime_error("A vertex is selected by multiple animation constraints");
                    }
                    has_soft[vi] = 1;
                    tetMesh.targetIndex.push_back(vi);
                    SoftTargetRotation tr;
                    tr.origin           = c.rot_origin;
                    tr.axis_unit        = c.rot_axis;
                    tr.angular_velocity = c.rot_velocity;
                    soft_target_rotations.push_back(std::move(tr));
                    host_target_vertices.push_back(tetMesh.vertexes[vi]);
                }
            }
        }
    }

    tetMesh.softNum = tetMesh.targetIndex.size();

    if(non_obstacle_min.x != std::numeric_limits<double>::infinity())
    {
        const double dx = non_obstacle_max.x - non_obstacle_min.x;
        const double dy = non_obstacle_max.y - non_obstacle_min.y;
        const double dz = non_obstacle_max.z - non_obstacle_min.z;
        ipc.scene_diag  = sqrt(dx * dx + dy * dy + dz * dz);
    }
    else
    {
        ipc.scene_diag = 0.0;
    }

    // MAS preconditioner prep.
    if(preconditioner_type != 0)
    {
        setMAS_partition(tetMesh);
    }

    tetMesh.getSurface();
    initFEM(tetMesh);

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

    if(tetMesh.softNum > 0)
    {
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetIndex,
                                  tetMesh.targetIndex.data(),
                                  tetMesh.softNum * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetVert,
                                  host_target_vertices.data(),
                                  tetMesh.softNum * sizeof(double3),
                                  cudaMemcpyHostToDevice));
    }

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

    // Parse ground section
    const auto& ground = scene.at("ground");
    ipc.useGround      = ground.at("enabled").get<bool>();
    if(ipc.useGround)
    {
        const auto ground_normal = read_vec3(ground.at("normal"));
        const auto ground_offset = ground.at("offset").get<double>();
        CUDA_SAFE_CALL(cudaMemcpy(ipc._groundNormal, &ground_normal, sizeof(double3), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ipc._groundOffset, &ground_offset, sizeof(double), cudaMemcpyHostToDevice));
    }

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

    const bool has_A = program.is_used("--A");
    const bool has_b = program.is_used("--b");
    if(has_A || has_b)
    {
        if(!(has_A && has_b))
        {
            std::cerr << "When solving a provided linear system, both --A and --b are required.\n";
            return 1;
        }

        const fs::path A_path = program.get<std::string>("--A");
        const fs::path b_path = program.get<std::string>("--b");

        const auto A = read_mm_coordinate_real(A_path);
        if(A.M != A.N)
        {
            std::cerr << "A must be square.\n";
            return 1;
        }

        const auto b = read_mm_array_vector_mx1(b_path);

        const int expected_M = ipc.gipc_global_triplet.block_rows() * 3;
        if(A.M != expected_M)
        {
            std::cerr << "Loaded A has M=" << A.M << " but configured system expects M=" << expected_M << "\n";
            return 1;
        }
        if(static_cast<int>(b.size()) != expected_M)
        {
            std::cerr << "Loaded b has size=" << b.size() << " but expected M=" << expected_M << "\n";
            return 1;
        }

        const auto blocks = to_upper_triangle_block_triplets(A);
        upload_triplets_to_device(ipc.gipc_global_triplet, blocks);

        const auto iter = ipc.m_global_linear_system->solve_loaded_system(b);
        std::cout << "Loaded linear system solved. Iterations: " << iter << std::endl;
        return 0;
    }

    fs::create_directories(output_dir);

    // Write initial (pre-simulation) state as frame 0.
    if(write_obj_frames)
    {
        CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(),
                                  ipc._vertexes,
                                  ipc.vertexNum * sizeof(double3),
                                  cudaMemcpyDeviceToHost));
        write_obj(frame_obj_path(output_dir, 0), tetMesh);
    }

    for(int frame = 0; frame < frames; ++frame)
    {
        if(ipc.m_global_linear_system)
        {
            ipc.m_global_linear_system->disable_matrix_market_export();
            if(std::find(export_linear_system_frames.begin(),
                         export_linear_system_frames.end(),
                         frame)
               != export_linear_system_frames.end())
            {
                ipc.m_global_linear_system->enable_matrix_market_export(frame,
                                                                         output_dir);
            }
        }

        // apply rotation animation to selected vertices.
        if(ipc.softNum > 0)
        {
            CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(),
                                      ipc._vertexes,
                                      ipc.vertexNum * sizeof(double3),
                                      cudaMemcpyDeviceToHost));

            for(size_t i = 0; i < ipc.softNum; ++i)
            {
                const uint32_t v_id = tetMesh.targetIndex[i];
                const auto&    r    = soft_target_rotations[i];

                host_target_vertices[i] =
                    rotate_axis_angle(tetMesh.vertexes[v_id],
                                      r.origin,
                                      r.axis_unit,
                                      r.angular_velocity * ipc.IPC_dt);
            }

            CUDA_SAFE_CALL(cudaMemcpy(d_tetMesh.targetVert,
                                      host_target_vertices.data(),
                                      ipc.softNum * sizeof(double3),
                                      cudaMemcpyHostToDevice));
        }

        ipc.IPC_Solver(d_tetMesh);

        auto& stats = gipc::Statistics::instance();
        stats.at_current_frame()["timer"] =
            gipc::GlobalTimer::current()->report_merged_as_json();
        gipc::GlobalTimer::current()->print_merged_timings();
        gipc::GlobalTimer::current()->clear();
        stats.write_to_file((fs::path(output_dir) / "stats.json").string());
        stats.frame(stats.frame() + 1);

        if(write_obj_frames)
        {
            CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(),
                                      ipc._vertexes,
                                      ipc.vertexNum * sizeof(double3),
                                      cudaMemcpyDeviceToHost));
            write_obj(frame_obj_path(output_dir, frame + 1), tetMesh);
        }
    }

    return 0;
}
