# Occupancy-tuned launch square

The same elementwise-square kernel is launched with a hand-picked (deliberately too-small) 32-thread block and, separately, with a block/grid size suggested by cudaOccupancyMaxPotentialBlockSize for maximal theoretical occupancy on the current GPU. Every index is squared by exactly one thread either way, so both launches produce identical output; this isolates the occupancy-introspection API (no direct SYCL analog).

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleOccupancy/simpleOccupancy.cu

Snapshot: benchmark/collection/simple-kernels/sources/sk-10

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: square reproduced verbatim from NVIDIA/cuda-samples simpleOccupancy.cu. Host driver (both manual and cudaOccupancyMaxPotentialBlockSize-driven launches), gen_array, CMakeLists, and Python oracle are new. Snapshot sk-10.
