# Jaccard edge weights over CSR (nvGRAPH)

Per-edge Jaccard weights: row volumes, sorted-neighbor-list intersections via binary search + atomicAdd, weight map.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/jaccard-cuda/main.cu

Upstream commit: 01f58fc5

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: fill/jaccard_row_sum/jaccard_is/jaccard_jw + warp prefix sum verbatim, unweighted float instantiation (integer-valued atomics stay order-independent); upstream launch geometry. Snapshot graph-07.
