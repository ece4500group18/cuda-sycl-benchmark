# Rodinia LUD blocked LU decomposition

Blocked in-place LU factorization: diagonal, perimeter and internal kernels swept along the matrix diagonal.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/lud-cuda/lud_kernels.cu

Upstream commit: 01f58fc5

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: lud_diagonal/lud_perimeter/lud_internal verbatim (BLOCK_SIZE=16); upstream per-offset launch loop; diagonally dominant deterministic input; residual (L*U vs A) oracle. From the HeCBench raw clone (not yet a collection snapshot).
