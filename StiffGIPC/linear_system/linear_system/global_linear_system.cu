#include <linear_system/linear_system/global_linear_system.h>
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/i_preconditioner.h>
#include <gipc/utils/timer.h>
#include <cuda_tools/cuda_tools.h>

#include <filesystem>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

extern "C"
{
#include <mmio.h>
}

namespace gipc
{
void GlobalLinearSystem::clear_matrix_market_export_request()
{
    m_mm_export.pending = false;
    m_mm_export.frame   = -1;
    m_mm_export.output_dir.clear();
}

void GlobalLinearSystem::request_matrix_market_export(int frame, std::string output_dir)
{
    m_mm_export.pending    = true;
    m_mm_export.frame      = frame;
    m_mm_export.output_dir = std::move(output_dir);
}

static std::string format_frame_basename(int frame)
{
    std::ostringstream oss;
    oss << "frame_" << std::setw(5) << std::setfill('0') << frame;
    return oss.str();
}

void GlobalLinearSystem::export_matrix_market_files(int frame, const std::string& output_dir)
{
    if(!gipc_global_triplet)
    {
        throw std::runtime_error("GlobalLinearSystem: gipc_global_triplet is null");
    }

    namespace fs = std::filesystem;
    const fs::path base_dir = fs::path(output_dir) / "linear_system";
    fs::create_directories(base_dir);

    const fs::path A_path = base_dir / (format_frame_basename(frame) + "_A.mtx");
    const fs::path b_path = base_dir / (format_frame_basename(frame) + "_b.mtx");

    const int block_rows = gipc_global_triplet->block_rows();
    const int block_cols = gipc_global_triplet->block_cols();
    const int M          = block_rows * BlockSize;
    const int N          = block_cols * BlockSize;

    // Ensure all device work that produced A/b is visible to host copies.
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Copy b
    std::vector<Float> b_host;
    m_b.copy_to(b_host);
    if(static_cast<int>(b_host.size()) != M)
    {
        throw std::runtime_error("GlobalLinearSystem: b size mismatch with matrix rows");
    }

    // Copy unique block triplets for A
    const int block_nnz = gipc_global_triplet->h_unique_key_number;
    std::vector<int>           br(block_nnz);
    std::vector<int>           bc(block_nnz);
    std::vector<Eigen::Matrix3d> bv(block_nnz);

    if(block_nnz > 0)
    {
        CUDA_SAFE_CALL(cudaMemcpy(br.data(),
                                  gipc_global_triplet->block_row_indices(),
                                  sizeof(int) * block_nnz,
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(bc.data(),
                                  gipc_global_triplet->block_col_indices(),
                                  sizeof(int) * block_nnz,
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(bv.data(),
                                  gipc_global_triplet->block_values(),
                                  sizeof(Eigen::Matrix3d) * block_nnz,
                                  cudaMemcpyDeviceToHost));
    }

    // Write b as MatrixMarket array (M x 1)
    {
        FILE* f = std::fopen(b_path.string().c_str(), "w");
        if(!f)
        {
            throw std::runtime_error("Failed to open b.mtx for write: " + b_path.string());
        }

        MM_typecode matcode;
        mm_initialize_typecode(&matcode);
        mm_set_matrix(&matcode);
        mm_set_array(&matcode);
        mm_set_real(&matcode);
        mm_set_general(&matcode);

        mm_write_banner(f, matcode);
        mm_write_mtx_array_size(f, M, 1);
        if(std::ferror(f))
        {
            std::fclose(f);
            throw std::runtime_error("Failed to write b.mtx header: " + b_path.string());
        }

        for(int i = 0; i < M; ++i)
        {
            std::fprintf(f, "%.17g\n", static_cast<double>(b_host[i]));
        }
        std::fclose(f);
    }

    // Write A as MatrixMarket coordinate (general), expanding symmetric blocks to full matrix.
    {
        int64_t       nnz_full = 0;
        for(int k = 0; k < block_nnz; ++k)
        {
            if(br[k] == bc[k])
            {
                nnz_full += 9;
            }
            else
            {
                nnz_full += 18;
            }
        }

        if(nnz_full > std::numeric_limits<int>::max())
        {
            throw std::runtime_error("A.mtx too large (nnz exceeds int)");
        }

        FILE* f = std::fopen(A_path.string().c_str(), "w");
        if(!f)
        {
            throw std::runtime_error("Failed to open A.mtx for write: " + A_path.string());
        }

        MM_typecode matcode;
        mm_initialize_typecode(&matcode);
        mm_set_matrix(&matcode);
        mm_set_coordinate(&matcode);
        mm_set_real(&matcode);
        mm_set_general(&matcode);

        mm_write_banner(f, matcode);
        mm_write_mtx_crd_size(f, M, N, static_cast<int>(nnz_full));
        if(std::ferror(f))
        {
            std::fclose(f);
            throw std::runtime_error("Failed to write A.mtx header: " + A_path.string());
        }

        for(int k = 0; k < block_nnz; ++k)
        {
            const int bi = br[k];
            const int bj = bc[k];
            const int r0 = bi * BlockSize;
            const int c0 = bj * BlockSize;
            const auto& B = bv[k];

            for(int i = 0; i < BlockSize; ++i)
            {
                for(int j = 0; j < BlockSize; ++j)
                {
                    const int row = r0 + i + 1;
                    const int col = c0 + j + 1;
                    std::fprintf(f, "%d %d %.17g\n", row, col, B(i, j));
                }
            }

            if(bi != bj)
            {
                const int rt0 = bj * BlockSize;
                const int ct0 = bi * BlockSize;
                for(int i = 0; i < BlockSize; ++i)
                {
                    for(int j = 0; j < BlockSize; ++j)
                    {
                        const int row = rt0 + i + 1;
                        const int col = ct0 + j + 1;
                        std::fprintf(f, "%d %d %.17g\n", row, col, B(j, i));
                    }
                }
            }
        }

        std::fclose(f);
    }

    std::cout << "Exported linear system (MatrixMarket):\n"
              << "  A: " << A_path.string() << "\n"
              << "  b: " << b_path.string() << std::endl;
}

bool GlobalLinearSystem::build_linear_system()
{
    auto hessian_provider_count  = m_subsystems.size();
    auto gradient_provider_count = m_inner_subsystems.size();

    // right hand side can only be provided by both LinearSubsystem
    m_rhs_count_per_subsystem.resize(gradient_provider_count);
    m_rhs_offset_per_subsystem.resize(gradient_provider_count);

    for(auto& subsystem : m_subsystems)
        subsystem->report_subsystem_info();

    for(auto& gp : m_inner_subsystems)
    {
        auto i                       = gp->gid();
        m_rhs_count_per_subsystem[i] = gp->right_hand_side_dof();
    }

    std::exclusive_scan(m_rhs_count_per_subsystem.begin(),
                        m_rhs_count_per_subsystem.end(),
                        m_rhs_offset_per_subsystem.begin(),
                        0);

    for(auto& gp : m_inner_subsystems)
    {
        auto i = gp->gid();
        gp->dof_offset(m_rhs_offset_per_subsystem[i]);
    }

    auto total_rhs_count =
        m_rhs_offset_per_subsystem.back() + m_rhs_count_per_subsystem.back();


    if(gipc_global_triplet->global_triplet_offset == 0 || total_rhs_count == 0)
    {
        std::cout << "The global linear system is empty, skip *assembling, *solving and *solution distributing phase."
                  << std::endl;
        return false;
    }


    m_b.resize(total_rhs_count);
    m_x.resize(total_rhs_count);

    auto rhs_view = m_b.view();

    for(auto& subsystem : m_subsystems)
        subsystem->do_assemble(rhs_view);

    int start_preconditioner_id = 0;
    if(m_local_preconditioners.size() && m_local_preconditioners[0]->preconditioner_id == 0)
    {
        m_local_preconditioners[0]->assemble();
        start_preconditioner_id++;
    }
    convert_new();

    if(m_global_preconditioner)
        m_global_preconditioner->do_assemble(*gipc_global_triplet);

    for(int i = start_preconditioner_id; i < m_local_preconditioners.size(); i++)
    {
        m_local_preconditioners[i]->assemble();
    }

    return true;
}

void GlobalLinearSystem::distribute_solution()
{
    auto x_view = std::as_const(m_x).view();

    for(auto& subsystem : m_inner_subsystems)
        subsystem->do_retrieve_solution(x_view);

    muda::wait_device();
}

DiagonalSubsystem& GlobalLinearSystem::_create_subsystem(U<DiagonalSubsystem>&& subsystem)
{
    auto ptr = subsystem.get();
    ptr->gid(m_inner_subsystems.size());
    m_inner_subsystems.push_back(ptr);  // push to gradient providers

    ptr->hid(m_subsystems.size());
    ptr->system(*this);
    m_subsystems.emplace_back(std::move(subsystem));  // push to hessian providers

    return *ptr;
}


IterativeSolver& GlobalLinearSystem::_create_solver(U<IterativeSolver>&& solver)
{
    m_solver = std::move(solver);
    m_solver->system(*this);
    return *m_solver;
}

GlobalLinearSystem::~GlobalLinearSystem() {}

LocalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<LocalPreconditioner>&& preconditioner)
{
    preconditioner->system(*this);
    return *m_local_preconditioners.emplace_back(std::move(preconditioner));
}

GlobalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<GlobalPreconditioner>&& preconditioner)
{
    MUDA_ASSERT(m_global_preconditioner == nullptr, "Global preconditioner already exists.");
    preconditioner->system(*this);
    m_global_preconditioner = std::move(preconditioner);
    return *m_global_preconditioner;
}

gipc::SizeT GlobalLinearSystem::solve_linear_system()
{
    bool success = build_linear_system();
    if(!success)
    {
        // Clear any pending export request to avoid exporting stale frame labels later.
        clear_matrix_market_export_request();
        return 0;
    }

    if(m_mm_export.pending)
    {
        export_matrix_market_files(m_mm_export.frame, m_mm_export.output_dir);
        clear_matrix_market_export_request();
    }
    MUDA_ASSERT(m_solver, "Solver is null, call create_solver() to setup a solver.");
    auto iter = m_solver->solve(m_x, m_b);
    distribute_solution();
    return iter;
}

Json GlobalLinearSystem::as_json() const
{
    Json j;
    j["solver"]     = typeid(*m_solver).name();
    j["subsystems"] = Json::array();
    for(auto& s : m_subsystems)
    {
        j["subsystems"].push_back(s->as_json());
    }
    j["preconditioners"] = Json::array();
    for(auto& p : m_local_preconditioners)
    {
        j["preconditioners"].push_back(p->as_json());
    }
    return j;
}

void GlobalLinearSystem::apply_preconditioner(muda::DenseVectorView<Float>  z,
                                              muda::CDenseVectorView<Float> r)
{
    // first apply global preconditioner
    if(m_global_preconditioner)
        m_global_preconditioner->do_apply(r, z);
    else  // if no global preconditioner, use identity
        z.buffer_view().copy_from(r.buffer_view());

    // then apply local preconditioners
    // it's user's choice to rewrite or reuse the global preconditioner
    for(auto& p : m_local_preconditioners)
        p->do_apply(r, z);
}


void GlobalLinearSystem::convert_new()
{
    m_converter.convert(*gipc_global_triplet,
                        0,
                        gipc_global_triplet->global_triplet_offset,
                        gipc_global_triplet->global_triplet_offset);
    //#ifndef SymGH
    //    m_converter.ge2sym(*gipc_global_triplet);
    //#endif
}


void GlobalLinearSystem::spmv(Float                         a,
                              muda::CDenseVectorView<Float> x,
                              Float                         b,
                              muda::DenseVectorView<Float>  y)
{

    m_spmv.warp_reduce_sym_spmv(a,
                                gipc_global_triplet->block_values(),
                                gipc_global_triplet->block_row_indices(),
                                gipc_global_triplet->block_col_indices(),
                                gipc_global_triplet->h_unique_key_number,
                                x,
                                b,
                                y);
}
}  // namespace gipc
