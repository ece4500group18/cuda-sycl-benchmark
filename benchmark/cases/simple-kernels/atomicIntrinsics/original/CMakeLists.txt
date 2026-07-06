cmake_minimum_required(VERSION 3.18)
project(atomicIntrinsics LANGUAGES CXX CUDA)

add_executable(app main.cu)
set_target_properties(app PROPERTIES CUDA_ARCHITECTURES "70;80;90")

# The benchmark scripts build with the nvcc command in metadata.json
# (original/build/app). This CMakeLists.txt is provided for portable
# standalone builds.
