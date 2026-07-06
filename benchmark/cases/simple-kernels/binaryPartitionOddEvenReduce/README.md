# Binary-partition odd/even reduce

For each element a thread splits its 32-thread warp tile into odd/even sub-groups via cg::binary_partition, each sub-group does one cg::reduce and the rank-0 thread issues a single atomicAdd -- cutting atomic traffic under branch divergence. A naive per-thread-atomic kernel is included for contrast. All three quantities are order-independent integer accumulations, so the grouped CG result matches a plain CPU count-and-sum exactly.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/binaryPartitionCG/binaryPartitionCG.cu

Snapshot: benchmark/collection/simple-kernels/sources/sk-03

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: oddEvenCountAndSumCG reproduced verbatim from NVIDIA/cuda-samples binaryPartitionCG.cu. The naive counterpart kernel, host driver, CMakeLists, and Python oracle are new; the deterministic gen() input replaces the sample's rand()%50. Snapshot sk-03.
