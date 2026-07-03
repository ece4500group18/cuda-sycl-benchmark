# Particle diffusion random walk (motionsim)

2D random-walk diffusion of particles with per-particle occupancy counting
over a uniform grid — Intel's motionsim sample via HeCBench. Exercises a
per-thread iteration loop, truncf/floorf grid indexing, size_t counters,
and host-pregenerated randomness.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/particle-diffusion-cuda/motionsim.cu

Upstream commit: 01f58fc5 (snapshot: benchmark/collection/molecular-dynamics/sources/md-08)

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: Simulation kernel verbatim. Upstream already pre-generates
randoms on the host; the harness derives them from a deterministic hash
(scale 100 like upstream's rand()%scale) and keeps upstream's (10,10)
initial positions. Verified data has >1e-4 margin to the cell-radius
threshold, so results are bit-exact (tolerance 0).
