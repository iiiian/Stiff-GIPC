#pragma once

#include <Eigen/Dense>
#include <string>

namespace gipc::io
{
// Loads a 3D tetrahedral mesh from a Gmsh .msh file.
// - vertices: #V x 3
// - tets:     #T x 4 (0-based vertex indices)
// Throws on any error.
void load_tet_mesh(const std::string& path, Eigen::MatrixXd& vertices, Eigen::MatrixXi& tets);

// Saves a 3D tetrahedral mesh as a Gmsh 4.1 ASCII .msh file.
// - vertices: #V x 3
// - tets:     #T x 4 (0-based vertex indices)
// Throws on any error.
void save_tet_mesh(const std::string&     path,
                   const Eigen::MatrixXd& vertices,
                   const Eigen::MatrixXi& tets);
}  // namespace gipc::io
