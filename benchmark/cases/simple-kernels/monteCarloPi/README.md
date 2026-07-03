# Case: monteCarloPi (hpc)

| field | value |
| --- | --- |
| category | hpc |
| operation | Monte Carlo estimate of pi |
| correctness | analytic_reference / abs_error_to_pi (tol 0.02) |
| CUDA features | __global__, reduction, atomicAdd |
| libraries | - |

Notes: Hash-based samples in unit square; |est - pi|. sizes=[samples].

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
