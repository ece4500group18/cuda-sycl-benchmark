# Symmetric tridiagonal eigenvalues by parallel bisection

Eigenvalues of a symmetric tridiagonal matrix via Gershgorin bounding, Sturm counting and iterative interval bisection/subdivision.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/eigenvalue-cuda/main.cu

Upstream commit: 01f58fc5

License: AMD BSD-style

Extraction fidelity: extracted

Extraction notes: Both kernels, the device Sturm-sequence counter and host helpers (isComplete, computeGerschgorinInterval) verbatim, with upstream's double-buffered convergence loop; deterministic hash diagonals. From the HeCBench raw clone. Oracle compares converged interval midpoints against numpy eigvalsh (order-independent).
