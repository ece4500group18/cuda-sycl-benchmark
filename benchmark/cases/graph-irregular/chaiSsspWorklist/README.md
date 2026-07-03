# Chai SSSP with double-buffered work queues

Single-source shortest paths from the Chai heterogeneous suite (via
HeCBench): the frontier drains through per-block shared-memory local queues
with overflow spill, edges relax via atomicMax over negated costs, and the
new frontier concatenates into a double-buffered global queue — the
dataset's only worklist-pattern case.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/sssp-cuda/main.cu

Upstream commit: 01f58fc5 (snapshot: benchmark/collection/graph-irregular/sources/graph-02)

License: University of Cordoba / University of Illinois (Chai, BSD-style)

Extraction fidelity: extracted

Extraction notes: SSSP_gpu kernel and common.h constants/types are upstream
code verbatim. The host driver keeps upstream's structure (single-threaded
first iteration on the host, queue swap by iteration parity, per-iteration
scalar uploads) but drops the heterogeneous CPU-thread path and file-based
input: GPU-only over a deterministic directed CSR graph. Verification is a
CPU Dijkstra; the kernel's converged costs are execution-order independent.
