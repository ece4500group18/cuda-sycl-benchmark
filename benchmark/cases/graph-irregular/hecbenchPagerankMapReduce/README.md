# PageRank map+reduce power iteration

PageRank power iteration: scatter outbound rank then gather with damping over a link matrix.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/page-rank-cuda/main.cu

Upstream commit: 01f58fc5

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: map/reduce kernels verbatim; deterministic hash link matrix replaces rand(); fixed 5 iterations. Snapshot graph-03.
