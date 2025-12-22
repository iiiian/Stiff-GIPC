#include "msh_io.hpp"

#include <mshio/mshio.h>

#include <cassert>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace gipc::io
{
void load_tet_mesh(const std::string& path, Eigen::MatrixXd& vertices, Eigen::MatrixXi& tets)
{
    mshio::MshSpec spec;
    // will throw if parsing fails.
    spec = mshio::load_msh(path);

    const auto& nodes = spec.nodes;
    if(nodes.num_nodes == 0)
    {
        throw std::runtime_error("Empty gmsh file (no verts)");
    }
    const auto& elements = spec.elements;
    if(elements.num_elements == 0)
    {
        throw std::runtime_error("Empty gmsh file (no elements)");
    }

    for(const auto& n : nodes.entity_blocks)
    {
        if(n.entity_dim != 3)
        {
            throw std::runtime_error("Only 3D vertices are supported, got dim="
                                     + std::to_string(n.entity_dim));
        }
    }
    for(const auto& e : elements.entity_blocks)
    {
        if(e.element_type != 4)
        {
            throw std::runtime_error("Only tet meshes are supported, got type="
                                     + std::to_string(e.element_type));
        }
    }

    vertices.resize(static_cast<int>(nodes.num_nodes), 3);
    int node_idx = 0;  // re-index node from 0 ... num_nodes-1
    std::vector<int> node_tag_map(nodes.max_node_tag + 1, -1);  // map gmsh tag to idx
    for(const auto& n : nodes.entity_blocks)
    {
        for(int i = 0; i < static_cast<int>(n.num_nodes_in_block); ++i)
        {
            assert(node_idx < static_cast<int>(nodes.num_nodes));
            assert(n.tags[i] <= nodes.max_node_tag);
            node_tag_map[n.tags[i]] = node_idx;
            vertices(node_idx, 0)   = n.data[3 * i];
            vertices(node_idx, 1)   = n.data[3 * i + 1];
            vertices(node_idx, 2)   = n.data[3 * i + 2];

            ++node_idx;
        }
    }

    int tet_idx = 0;
    tets.resize(static_cast<int>(elements.num_elements), 4);
    for(const auto& e : elements.entity_blocks)
    {
        for(int i = 0; i < static_cast<int>(e.num_elements_in_block); ++i)
        {
            assert(tet_idx < static_cast<int>(elements.num_elements));

            // element.data layout:
            // | element tag | node0 tag | node1 tag | node2 tag | node3 tag |
            int tag0 = static_cast<int>(e.data[5 * i + 1]);
            assert(tag0 <= static_cast<int>(nodes.max_node_tag));
            assert(node_tag_map[tag0] != -1);
            int tag1 = static_cast<int>(e.data[5 * i + 2]);
            assert(tag1 <= static_cast<int>(nodes.max_node_tag));
            assert(node_tag_map[tag1] != -1);
            int tag2 = static_cast<int>(e.data[5 * i + 3]);
            assert(tag2 <= static_cast<int>(nodes.max_node_tag));
            assert(node_tag_map[tag2] != -1);
            int tag3 = static_cast<int>(e.data[5 * i + 4]);
            assert(tag3 <= static_cast<int>(nodes.max_node_tag));
            assert(node_tag_map[tag3] != -1);

            tets(tet_idx, 0) = node_tag_map[tag0];
            tets(tet_idx, 1) = node_tag_map[tag1];
            tets(tet_idx, 2) = node_tag_map[tag2];
            tets(tet_idx, 3) = node_tag_map[tag3];

            ++tet_idx;
        }
    }
}

void save_tet_mesh(const std::string&     path,
                   const Eigen::MatrixXd& vertices,
                   const Eigen::MatrixXi& tets)
{
    if(vertices.cols() != 3)
    {
        throw std::runtime_error("save_tet_mesh expects vertices with 3 columns");
    }
    if(tets.cols() != 4)
    {
        throw std::runtime_error("save_tet_mesh expects tets with 4 columns");
    }
    if(vertices.rows() <= 0 || tets.rows() <= 0)
    {
        throw std::runtime_error("save_tet_mesh expects non-empty vertices and tets");
    }

    std::filesystem::path out_path(path);
    if(out_path.has_parent_path())
    {
        std::filesystem::create_directories(out_path.parent_path());
    }

    mshio::MshSpec spec;

    auto& nodes             = spec.nodes;
    nodes.num_entity_blocks = 1;
    nodes.num_nodes         = static_cast<size_t>(vertices.rows());
    nodes.min_node_tag      = 1;
    nodes.max_node_tag      = static_cast<size_t>(vertices.rows());
    nodes.entity_blocks.resize(1);
    {
        auto& block              = nodes.entity_blocks[0];
        block.entity_dim         = 3;
        block.entity_tag         = 1;
        block.parametric         = 0;
        block.num_nodes_in_block = static_cast<size_t>(vertices.rows());
        block.tags.resize(static_cast<size_t>(vertices.rows()));
        block.data.resize(static_cast<size_t>(vertices.rows()) * 3);
        for(int i = 0; i < vertices.rows(); ++i)
        {
            block.tags[static_cast<size_t>(i)] = static_cast<size_t>(i + 1);
            block.data[static_cast<size_t>(3 * i + 0)] = vertices(i, 0);
            block.data[static_cast<size_t>(3 * i + 1)] = vertices(i, 1);
            block.data[static_cast<size_t>(3 * i + 2)] = vertices(i, 2);
        }
    }

    auto& elements             = spec.elements;
    elements.num_entity_blocks = 1;
    elements.num_elements      = static_cast<size_t>(tets.rows());
    elements.min_element_tag   = 1;
    elements.max_element_tag   = static_cast<size_t>(tets.rows());
    elements.entity_blocks.resize(1);
    {
        auto& block                 = elements.entity_blocks[0];
        block.entity_dim            = 3;
        block.entity_tag            = 1;
        block.element_type          = 4;
        block.num_elements_in_block = static_cast<size_t>(tets.rows());
        block.data.resize(static_cast<size_t>(tets.rows()) * 5);
        for(int i = 0; i < tets.rows(); ++i)
        {
            block.data[static_cast<size_t>(5 * i + 0)] = static_cast<size_t>(i + 1);
            block.data[static_cast<size_t>(5 * i + 1)] =
                static_cast<size_t>(tets(i, 0) + 1);
            block.data[static_cast<size_t>(5 * i + 2)] =
                static_cast<size_t>(tets(i, 1) + 1);
            block.data[static_cast<size_t>(5 * i + 3)] =
                static_cast<size_t>(tets(i, 2) + 1);
            block.data[static_cast<size_t>(5 * i + 4)] =
                static_cast<size_t>(tets(i, 3) + 1);
        }
    }

    mshio::save_msh(path, spec);
}
}  // namespace gipc::io
