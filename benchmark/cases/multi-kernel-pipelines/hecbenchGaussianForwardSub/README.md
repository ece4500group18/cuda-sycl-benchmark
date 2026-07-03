# Rodinia Gaussian elimination (fan1/fan2 per-column pipeline)

Forward substitution via a per-pivot-column two-kernel pipeline (multiplier column + submatrix update), then host back-substitution.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/gaussian-cuda/gaussianElim.cu

Upstream commit: 01f58fc5

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: fan1/fan2 kernels + ForwardSub launch loop + BackSub verbatim; deterministic diagonally-dominant system (scheme has no pivoting). From the HeCBench raw clone, registered as mkp-03.
