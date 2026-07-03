# Case: conjugateGradient (hpc)

| field | value |
| --- | --- |
| category | hpc |
| operation | Conjugate gradient (1D Laplacian) |
| correctness | residual_norm / rel_residual (tol 0.001) |
| CUDA features | __global__, reduction, iterative |
| libraries | - |

Notes: SPD 1D Laplacian; verify ||Ax-b||/||b||. sizes=[N,iters].

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
