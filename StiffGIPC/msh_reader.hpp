#pragma once

#include <Eigen/Dense>
#include <string>

namespace gipc::io
{
class MshReader
{
  public:
    // Loads a 3D tetrahedral mesh from a Gmsh .msh file.
    // - vertices: #V x 3
    // - tets:     #T x 4 (0-based vertex indices)
    // Throws on any error.
    static void load_tet_mesh(const std::string& path,
                              Eigen::MatrixXd&   vertices,
                              Eigen::MatrixXi&   tets);
};
}  // namespace gipc::io
