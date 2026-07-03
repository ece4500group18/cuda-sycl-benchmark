# ECL-MIS maximal independent set

Maximal independent set via prioritized selection with a lock-free convergence kernel over CSR.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/mis-cuda/main.cu

Upstream commit: 01f58fc5

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: init+findmins kernels and device hash verbatim; lock-free spinning kernel with volatile status bytes; deterministic CSR graph harness. Snapshot graph-05.
