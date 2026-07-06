# cuThomasBatch batched tridiagonal solver

Thomas algorithm over 1024 interleaved tridiagonal systems of size 64, one system per thread.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/thomas-cuda/cuThomasBatch.cu

Upstream commit: 01f58fc5

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: cuThomasBatch kernel verbatim (BSC cuThomasBatch, double precision, interleaved batched layout, one system per thread); deterministic diagonally-dominant systems. From the HeCBench raw clone.
