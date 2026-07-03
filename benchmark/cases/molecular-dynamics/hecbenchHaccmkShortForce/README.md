# HACC microkernel short-range force

Short-range gravitational force evaluation from the HACC cosmology
microkernel: O(n1*n2) pair loop with a 5th-order polynomial long-range
correction, rsqrt-style force law, and branchless mass gating.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/haccmk-cuda/haccmk.cu

Upstream commit: 01f58fc5 (snapshot: benchmark/collection/molecular-dynamics/sources/md-04)

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: haccmk_kernel verbatim. New host harness: deterministic
hash-derived positions/masses/velocities, single kernel launch, text output.
