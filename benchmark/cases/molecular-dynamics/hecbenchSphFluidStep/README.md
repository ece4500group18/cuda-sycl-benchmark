# SPH fluid simulation step (four-kernel pipeline)

One time step of a smoothed-particle-hydrodynamics fluid: B-spline kernel
density/pressure summation, pairwise pressure + viscosity + surface-tension
forces, boundary-particle repulsion, and leapfrog integration — a real
multi-kernel simulation pipeline in double precision, from HeCBench sph-cuda.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/sph-cuda/fluid.cu

Upstream commit: 01f58fc5 (snapshot: benchmark/collection/molecular-dynamics/sources/md-06)

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: All device functions and the four kernels
(updatePressures, updateAccelerationsFP, updateAccelerationsBP,
updatePositions) are upstream code verbatim, including the idiosyncratic
`xi = (1-x/h)?x<h:0.0;` expression in boundaryGamma (replicated exactly in
the CPU oracle). New host harness: deterministic jittered-lattice fluid
block over a boundary plane, fixed parameters, two pipeline steps.
Upstream ships no verifier (snapshot notes) — the CPU-reference oracle
here is the designed one.
