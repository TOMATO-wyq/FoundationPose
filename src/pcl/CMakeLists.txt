cmake_minimum_required(VERSION 3.18)
project(box4d_pcl LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_path(PYBIND11_INCLUDE_DIR pybind11/pybind11.h REQUIRED)
find_package(PCL 1.10 REQUIRED COMPONENTS common features segmentation sample_consensus)
Python3_add_library(_pcl_backend MODULE WITH_SOABI pcl_backend.cpp)
target_include_directories(_pcl_backend PRIVATE ${PYBIND11_INCLUDE_DIR} ${PCL_INCLUDE_DIRS})
target_link_directories(_pcl_backend PRIVATE ${PCL_LIBRARY_DIRS})
target_link_libraries(_pcl_backend PRIVATE ${PCL_LIBRARIES})
target_compile_options(_pcl_backend PRIVATE ${PCL_DEFINITIONS})
# Instantiate PCL templates with the same Eigen/compiler ABI as this extension.
# Avoid mixing inline template state with precompiled feature implementations.
target_compile_definitions(_pcl_backend PRIVATE PCL_NO_PRECOMPILE)
set_target_properties(_pcl_backend PROPERTIES
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../box4d")
