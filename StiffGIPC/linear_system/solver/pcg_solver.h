#pragma once
#include <linear_system/linear_system/i_linear_system_solver.h>

namespace gipc
{
class PCGSolverConfig
{
  public:
    /**
     * \brief the maximum number of iterations will be:
     *  dof * max_iter_ratio
     */
    Float max_iter_ratio  = 0.3;
    Float rel_tol = 1e-2;
    Float abs_tol = 0.0;
    bool  use_preconditioned_residual_norm = true;
    bool  use_bsr         = true;
};

class PCGSolver : public IterativeSolver
{
    using DeviceDenseVector = muda::DeviceDenseVector<Float>;

  public:
    PCGSolver(const PCGSolverConfig& cfg);
    virtual ~PCGSolver() = default;

    void config(const PCGSolverConfig& config) { this->m_config = config; }
    const auto& config() const { return this->m_config; }

  private:

    DeviceDenseVector z;   // preconditioned residual
    DeviceDenseVector r;   // residual
    DeviceDenseVector p;   // search direction
    DeviceDenseVector Ap;  // A*p
    PCGSolverConfig   m_config;

  protected:
    SizeT solve(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b) override;

  private:
    SizeT pcg(muda::DenseVectorView<Float> x, muda::CDenseVectorView<Float> b, SizeT max_iter);
};
}  // namespace gipc
