#include <linear_system/solver/pcg_solver.h>
#include <gipc/utils/timer.h>
#include <gipc/statistics.h>
#include <cuda_tools/cuda_tools.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>


__global__ void PCG_vdv_Reduction(double* squeue, const double* a, const double* b, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= numbers)
        return;

    double temp = a[idx] * b[idx];

    int warpTid = threadIdx.x % 32;
    int warpId  = (threadIdx.x >> 5);
    //double nextTp;
    int warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        temp = tep[threadIdx.x];
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}


__global__ void add_reduction(double* mem, int numbers)
{
    int                      idof = blockIdx.x * blockDim.x;
    int                      idx  = threadIdx.x + idof;
    extern __shared__ double tep[];
    if(idx >= numbers)
        return;
    double temp    = mem[idx];
    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        temp = tep[threadIdx.x];
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        mem[blockIdx.x] = temp;
    }
}


__global__ void update_vector_dx_r(
    double* dx, double* r, const double* c, const double* q, double alpha, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    dx[idx] = dx[idx] + alpha * c[idx];
    r[idx]  = r[idx] - alpha * q[idx];
}

__global__ void update_vector_c(double* c, const double* s, double beta, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    c[idx] = s[idx] + beta * c[idx];
}


double My_PCG_General_v_v_Reduction_Algorithm(double* temp, double* A, double* B, int vertexNum)
{

    int numbers = vertexNum;
    if(numbers < 1)
        return 0;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    PCG_vdv_Reduction<<<blockNum, threadNum, sharedMsize>>>(temp, A, B, numbers);


    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        add_reduction<<<blockNum, threadNum, sharedMsize>>>(temp, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    double result;
    cudaMemcpy(&result, temp, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

namespace gipc
{
PCGSolver::PCGSolver(const PCGSolverConfig& cfg)
    : m_config(cfg)
{
}
SizeT PCGSolver::solve(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b)
{
    Timer timer{"pcg"};

    x.buffer_view().fill(0);
    z.resize(b.size());
    p.resize(b.size());
    r.resize(b.size());
    //temp.resize(b.size());
    Ap.resize(b.size());
    auto iter = pcg(x, b, m_config.max_iter == 0 ? b.size() : m_config.max_iter);

    return iter;
}


SizeT PCGSolver::pcg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter)
{
    SizeT k = 0;

    r.buffer_view().copy_from(b.buffer_view());

    Float alpha, beta, rz, rz0;

    {
        //Timer timer{"preconditioner"};
        apply_preconditioner(z, r);
    }

    {
        //Timer timer{"dot"};
        rz = My_PCG_General_v_v_Reduction_Algorithm(p.buffer_view().data(),
                                                    r.buffer_view().data(),
                                                    z.buffer_view().data(),
                                                    z.size());
    }

    p   = z;
    rz0 = rz;

    Float rr0 = 0;
    if(!m_config.use_preconditioned_residual_norm)
    {
        rr0 = My_PCG_General_v_v_Reduction_Algorithm(Ap.buffer_view().data(),
                                                     r.buffer_view().data(),
                                                     r.buffer_view().data(),
                                                     r.size());
    }

    {
        const Float init_precond_residual = std::sqrt(rz0);
        const Float init_residual =
            m_config.use_preconditioned_residual_norm ? init_precond_residual : std::sqrt(rr0);
        std::cout << std::fixed << std::setprecision(20)
                  << "[PCG] iter=0 residual=" << init_residual
                  << " precond_residual=" << init_precond_residual << " time_ms=0\n";
    }

    bool  converged     = false;
    Float last_residual  = m_config.use_preconditioned_residual_norm ? std::sqrt(rz0) : std::sqrt(rr0);
    Float last_precond   = std::sqrt(rz0);

    for(k = 1; k < max_iter; ++k)
    {
        const auto iter_start = std::chrono::high_resolution_clock::now();

        {
            //Timer timer{"spmv"};
            // Ap = A * p
            spmv(p.cview(), Ap.view());
        }

        {
            //Timer timer{"dot"};

            Float dot_res =
                My_PCG_General_v_v_Reduction_Algorithm(z.buffer_view().data(),
                                                       p.buffer_view().data(),
                                                       Ap.buffer_view().data(),
                                                       z.size());

            alpha = rz / dot_res;
        }

        {
            //Timer timer{"axpby"};
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_dx_r,
                                     x.buffer_view().data(),
                                     r.buffer_view().data(),
                                     (const double*)p.buffer_view().data(),
                                     (const double*)Ap.buffer_view().data(),
                                     alpha,
                                     (int)z.size());
        }

        // Recompute the true residual periodically to avoid drift from recursive updates.
        // This matches the convergence-check granularity (k % 10 == 0).
        if(k % 10 == 0)
        {
            r.buffer_view().copy_from(b.buffer_view());  // r = b
            spmv(-1.0, x.as_const(), 1.0, r);            // r = b - A*x
        }

        {
            //Timer timer{"preconditioner"};
            apply_preconditioner(z, r);
        }

        Float rz_new = 0;
        {
            //Timer timer{"dot"};
            rz_new = My_PCG_General_v_v_Reduction_Algorithm(Ap.buffer_view().data(),
                                                            r.buffer_view().data(),
                                                            z.buffer_view().data(),
                                                            z.size());
        }

        const Float rel2      = m_config.rel_tol * m_config.rel_tol;
        const Float abs2      = m_config.abs_tol * m_config.abs_tol;

        Float rr = 0;
        if(m_config.use_preconditioned_residual_norm)
        {
            if(rz_new <= rel2 * rz0 || rz_new <= abs2)
            {
                converged = true;
            }
        }
        else
        {
            rr = My_PCG_General_v_v_Reduction_Algorithm(Ap.buffer_view().data(),
                                                        r.buffer_view().data(),
                                                        r.buffer_view().data(),
                                                        r.size());
            if(rr <= rel2 * rr0 || rr <= abs2)
            {
                converged = true;
            }
        }

        last_precond  = std::sqrt(rz_new);
        last_residual = m_config.use_preconditioned_residual_norm ? last_precond : std::sqrt(rr);

        if(converged)
        {
            const auto iter_end = std::chrono::high_resolution_clock::now();
            const auto dt = std::chrono::duration<double, std::milli>(iter_end - iter_start);
            std::cout << "[PCG] iter=" << k << " residual=" << last_residual
                      << " precond_residual=" << last_precond
                      << " time_ms=" << dt.count() << "\n";
            break;
        }

        beta = rz_new / rz;

        {
            //Timer timer{"axpby"};
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_c,
                                     p.buffer_view().data(),
                                     (const double*)z.buffer_view().data(),
                                     beta,
                                     (int)z.size());
        }

        rz = rz_new;

        const auto iter_end = std::chrono::high_resolution_clock::now();
        const auto dt = std::chrono::duration<double, std::milli>(iter_end - iter_start);
        std::cout << "[PCG] iter=" << k << " residual=" << last_residual
                  << " precond_residual=" << last_precond
                  << " time_ms=" << dt.count() << "\n";
    }

    if(converged)
    {
        std::cout << "PCG: converged iter " << k << " residual=" << last_residual << std::endl;
    }
    else
    {
        std::cout << "PCG: stopped iter " << k << " (max_iter=" << max_iter
                  << ") residual=" << last_residual << std::endl;
    }
    return k;
}

}  // namespace gipc
