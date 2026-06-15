cmake_minimum_required(VERSION 3.18)
project(mmcvThreeInterpolate LANGUAGES CXX CUDA)
add_executable(app main.cu)
set_target_properties(app PROPERTIES CUDA_ARCHITECTURES "70;80;90")
