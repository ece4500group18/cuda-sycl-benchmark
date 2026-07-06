# Bitonic vs odd-even merge sort networks

Both a bitonic sort and Batcher's odd-even merge sort load a block's worth of (key,value) pairs into shared memory and apply a network of Comparator conditional-swaps, wired up by thread-index bit arithmetic and separated by cg::sync barriers. The two O(log^2 N) networks have different shapes but sort the same total order, so they converge on the identical permutation. Keys are pairwise distinct, so a position-by-position comparison against a CPU sort-by-key is an exact oracle.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/tree/master/cpp/2_Concepts_and_Techniques/sortingNetworks

Snapshot: benchmark/collection/simple-kernels/sources/sk-04

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: Comparator, bitonicSortShared, and oddEvenMergeSortShared reproduced verbatim from NVIDIA/cuda-samples sortingNetworks (bitonicSort.cu, oddEvenMergeSort.cu, sortingNetworks_common.cuh). Host driver, gen_key/gen_val, CMakeLists, and Python oracle are new. Snapshot sk-04.
