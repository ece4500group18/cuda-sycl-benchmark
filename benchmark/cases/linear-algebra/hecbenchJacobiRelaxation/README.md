# Jacobi relaxation with shared-memory halos and warp-shuffle reduction

Jacobi iteration for the 2D Laplace system: shared-memory halo tiles, warp __shfl_down_sync error reduction, atomicAdd accumulation.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/jacobi-cuda/main.cu

Upstream commit: 01f58fc5

License: Apache-2.0

Extraction fidelity: extracted

Extraction notes: jacobi_step kernel + initialize_data verbatim (N 2048->512); fixed 50 iterations replace the convergence loop; the atomicAdd error reduction stays exercised every step. From the HeCBench raw clone.
