# Cooperative-groups thread-block rank sum

cooperative_groups::this_thread_block() partitions the block, and sumReduction runs a shared-memory tree reduction using the group's .sync()/.thread_rank()/.size() API in place of raw __syncthreads()+threadIdx arithmetic. Each block reduces its threads' ranks 0..blockDim-1, whose sum is the closed-form triangular number (blockDim-1)*blockDim/2 = 32640, identical for every block.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleCooperativeGroups/simpleCooperativeGroups.cu

Snapshot: benchmark/collection/simple-kernels/sources/sk-05

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: sumReduction reproduced verbatim from NVIDIA/cuda-samples simpleCooperativeGroups.cu. cgkernel is adapted to write g_odata[blockIdx.x] instead of printf and drops the tiled_partition<16> sub-reduction; host driver, CMakeLists, and Python oracle are new. Snapshot sk-05.
