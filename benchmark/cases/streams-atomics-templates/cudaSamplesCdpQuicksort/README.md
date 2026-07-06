# Quicksort via CUDA Dynamic Parallelism

Recursive quicksort where the kernel launches itself on sub-ranges through device-side streams, with a selection-sort base case.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/cdpSimpleQuicksort/cdpSimpleQuicksort.cu

Upstream commit: b7c5481c

License: BSD-3-Clause + CUDA EULA note

Extraction fidelity: extracted

Extraction notes: selection_sort, cdp_simple_quicksort and run_qsort verbatim; deterministic hash input replaces srand. Needs -rdc=true -lcudadevrt.
