# SHOC MD Lennard-Jones force (neighbor list)

Lennard-Jones pairwise force over an explicit per-atom neighbor list with
uncoalesced gather reads — the classic MD force pattern from the SHOC
benchmark suite.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/md-cuda/main.cu

Upstream commit: 01f58fc5 (snapshot: benchmark/collection/molecular-dynamics/sources/md-03)

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: The md() kernel is upstream device code verbatim
(single-precision specialization). New host harness: deterministic
jittered-lattice positions (bounds the LJ force, per upstream's warning
about near-coincident atoms), hash-derived neighbor list, text output.
