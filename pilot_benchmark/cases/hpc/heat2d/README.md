# Case: heat2d (hpc)

| field | value |
| --- | --- |
| category | hpc |
| operation | 2D heat equation (explicit) |
| correctness | cpu_reference / max_rel_error (tol 0.001) |
| CUDA features | __global__, iterative, 2D_indexing |
| libraries | - |

Notes: Explicit FTCS, K steps. sizes=[ny,nx,steps].

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
