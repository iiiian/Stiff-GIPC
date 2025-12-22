#include "msh_reader.hpp"

#include <mshio/mshio.h>

#include <filesystem>
#include <stdexcept>
#include <vector>
#include <cassert>

namespace gipc::io
{
void MshReader::load_tet_mesh(const std::string& path,
                              Eigen::MatrixXd&   vertices,
                              Eigen::MatrixXi&   tets)
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

    vertices.resize(nodes.num_nodes, 3);
    int node_idx = 0;  // re-index node from 0 ... num_nodes-1
    std::vector<int> node_tag_map(nodes.max_node_tag + 1, -1);  // map gmsh tag to idx
    for(const auto& n : nodes.entity_blocks)
    {
        for(int i = 0; i < n.num_nodes_in_block; ++i)
        {
            assert(node_idx < nodes.num_nodes);
            assert(n.tags[i] <= nodes.max_node_tag);
            node_tag_map[n.tags[i]] = node_idx;
            vertices(node_idx, 0)   = n.data[3 * i];
            vertices(node_idx, 1)   = n.data[3 * i + 1];
            vertices(node_idx, 2)   = n.data[3 * i + 2];

            ++node_idx;
        }
    }

    int tet_idx = 0;
    tets.resize(elements.num_elements, 4);
    for(const auto& e : elements.entity_blocks)
    {
        for(int i = 0; i < e.num_elements_in_block; ++i)
        {
            assert(tet_idx < elements.num_elements);

            // element.data layout:
            // | element tag | node0 tag | node1 tag | node2 tag | node3 tag |
            int tag0 = e.data[5 * i + 1];
            assert(tag0 <= nodes.max_node_tag);
            assert(node_tag_to_idx[tag0] != -1);
            int tag1 = e.data[5 * i + 2];
            assert(tag1 <= nodes.max_node_tag);
            assert(node_tag_to_idx[tag1] != -1);
            int tag2 = e.data[5 * i + 3];
            assert(tag2 <= nodes.max_node_tag);
            assert(node_tag_to_idx[tag2] != -1);
            int tag3 = e.data[5 * i + 4];
            assert(tag3 <= nodes.max_node_tag);
            assert(node_tag_to_idx[tag3] != -1);

            tets(tet_idx, 0) = node_tag_map[tag0];
            tets(tet_idx, 1) = node_tag_map[tag1];
            tets(tet_idx, 2) = node_tag_map[tag2];
            tets(tet_idx, 3) = node_tag_map[tag3];

            ++tet_idx;
        }
    }
}
}  // namespace gipc::io
