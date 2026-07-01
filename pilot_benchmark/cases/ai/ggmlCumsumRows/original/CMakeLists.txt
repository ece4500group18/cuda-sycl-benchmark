cmake_minimum_required(VERSION 3.18)
project(stage1_real_project_batch2 CUDA)
add_executable(app main.cu)
set_target_properties(app PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
