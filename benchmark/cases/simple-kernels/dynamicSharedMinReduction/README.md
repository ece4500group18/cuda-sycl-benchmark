# Dynamic shared-memory min reduction

Each block dynamically allocates 2*blockDim floats of shared memory (sized at launch via extern __shared__), loads two elements per thread, and runs a tree-halving loop (d = blockDim..1) keeping the smaller value -- a min reduction. A naive one-thread linear scan is included as a trivially-correct counterpart. min is order-free and every input value is an exact float, so all blocks produce the same exact minimum.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/clock/clock.cu

Snapshot: benchmark/collection/simple-kernels/sources/sk-07

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: timedReduction adapted from NVIDIA/cuda-samples clock.cu (clock_t timer parameter and clock() calls removed; reduction body unmodified). naiveMinReduction counterpart, host driver, gen_input, CMakeLists, and Python oracle are new. Snapshot sk-07.
