cmake_minimum_required(VERSION 3.18)
project(cudaStream LANGUAGES CXX CUDA)
add_executable(app main.cu)
set_target_properties(app PROPERTIES CUDA_ARCHITECTURES "70;80;90")
# Note: the pilot tooling builds via the nvcc command in metadata.json. This
# CMakeLists is provided for completeness / portable standalone builds.
