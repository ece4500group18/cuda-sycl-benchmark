cmake_minimum_required(VERSION 3.18)
project(gridSyncCGReduction LANGUAGES CXX CUDA)

add_executable(app main.cu)
target_include_directories(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(app PROPERTIES CUDA_ARCHITECTURES "70;80;90")
set_target_properties(app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Note: the Makefile in this directory is the primary build path used by
# the benchmark scripts. This CMakeLists.txt is provided for completeness
# / portable standalone builds. CUDA_SEPARABLE_COMPILATION (-rdc=true) is
# required by cudaLaunchCooperativeKernel.
