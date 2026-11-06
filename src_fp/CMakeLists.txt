cmake_minimum_required(VERSION 3.10)
project(zed_svo_export LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

find_package(ZED REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)

add_executable(zed_svo_export zed_svo_export.cpp)
target_include_directories(zed_svo_export PRIVATE
    ${ZED_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_directories(zed_svo_export PRIVATE
    ${ZED_LIBRARY_DIR}
    ${CUDA_LIBRARY_DIRS}
)
target_link_libraries(zed_svo_export PRIVATE
    ${ZED_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${OpenCV_LIBRARIES}
)
