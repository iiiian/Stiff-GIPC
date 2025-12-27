● Here's what each timer measures:

  Linear System Solving (Top Level)

  | Timer                       | Location                  | What it Measures                                                                                         |
  |-----------------------------|---------------------------|----------------------------------------------------------------------------------------------------------|
  | solve_linear_system         | GIPC.cu:10459             | Total time to solve the linear system (calls solve_linear_system())                                      |
  | pcg                         | pcg_solver.cu:160         | Preconditioned Conjugate Gradient iterations for solving Ax=b                                            |
  | convert3x3                  | converter.cu:28           | Converts 3x3 block triplets - radix sorts indices and performs warp reduction to merge duplicate entries |
  | precomputing Preconditioner | diag_preconditioner.cu:55 | Diagonal preconditioner setup                                                                            |

  Gradient & Hessian Computation (Top Level)

  | Timer                         | Location      | What it Measures                                       |
  |-------------------------------|---------------|--------------------------------------------------------|
  | cal_gradient_hessian          | GIPC.cu:10017 | Parent timer for all gradient/Hessian computation      |
  | cal_kinetic_gradient          | GIPC.cu:10028 | Inertia term: gradient of kinetic energy M(x - x̃)      |
  | cal_barrier_gradient_hessian  | GIPC.cu:10036 | IPC barrier function for collision avoidance           |
  | cal_friction_gradient_hessian | GIPC.cu:10051 | Friction forces between contacting primitives          |
  | cal_fem_gradient_hessian      | GIPC.cu:10123 | FEM elastic energy + bending energy gradients/Hessians |

  ABD (Affine Body Dynamics) System

  | Timer                              | Location                                     | What it Measures                                                             |
  |------------------------------------|----------------------------------------------|------------------------------------------------------------------------------|
  | setup_abd_system_gradient_hessian  | GIPC.cu:10075                                | Parent timer for ABD gradient/Hessian setup                                  |
  | _cal_abd_body_gradient_and_hessian | setup_abd_system_gradient_and_hessian.cu:203 | Per-body kinetic + shape energy (orthogonality constraint) for affine bodies |
  | _cal_abd_system_barrier_gradient   | setup_abd_system_gradient_and_hessian.cu:352 | Transforms vertex barrier gradients to ABD DOFs via Jacobian J^T * g         |
  | _setup_abd_system_hessian          | setup_abd_system_gradient_and_hessian.cu:427 | Assembles ABD contact Hessian blocks (calls convert3x3 internally)           |
