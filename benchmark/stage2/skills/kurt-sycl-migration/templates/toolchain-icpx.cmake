# icpx toolchain file for SYCL builds targeting Intel GPUs.
# Use with: cmake -S sycl -B sycl/build -DCMAKE_TOOLCHAIN_FILE=toolchain-icpx.cmake
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_FLAGS_INIT "-fsycl -O3")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-fsycl")

# For ahead-of-time compilation, use the token for target.aot_device from .sycl/config.json
# (`bmg-g21` for Xe2 Arc Pro B60; `bmg-g31` for Xe2 Arc Pro B70; `cri` for Xe3P CRI), e.g.:
# set(CMAKE_CXX_FLAGS_INIT "-fsycl -O3 -fsycl-targets=spir64_gen -Xs \"-device bmg-g31\"")
# set(CMAKE_EXE_LINKER_FLAGS_INIT "-fsycl -fsycl-targets=spir64_gen -Xs \"-device bmg-g31\"")
