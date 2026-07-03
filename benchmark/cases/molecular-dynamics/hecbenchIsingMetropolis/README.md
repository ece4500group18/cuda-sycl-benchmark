# 2D Ising checkerboard Metropolis update

Monte-Carlo spin dynamics on a 2D Ising lattice: black/white checkerboard
decomposition stored as two half-lattices, periodic stencil neighbor sums,
and a Metropolis accept/reject with expf — from NVIDIA's ising-gpu via
HeCBench. Exercises template kernels (update_lattice<is_black>), signed
char lattices, and long long indexing.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/ising-cuda/main.cu

Upstream commit: 01f58fc5 (snapshot: benchmark/collection/molecular-dynamics/sources/md-05)

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: init_spins and update_lattice<is_black> kernels verbatim.
Upstream feeds curand uniforms; the harness feeds deterministic
host-generated uniforms (one per site per color update) so the run is
reproducible without cuRAND. Verification tolerates a tiny site-mismatch
fraction because a borderline expf rounding difference may legitimately
flip a rare spin.
