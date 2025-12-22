cmake_minimum_required(VERSION 3.18)

add_subdirectory(MshIO)

add_library(msh_helper STATIC)
target_sources(msh_helper PRIVATE msh_io.cpp)
target_include_directories(msh_helper PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(msh_helper PUBLIC mshio::mshio Eigen3::Eigen)
