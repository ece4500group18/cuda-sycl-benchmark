# Case: finiteDiff1d (hpc)

| field | value |
| --- | --- |
| category | hpc |
| operation | Central finite-difference derivative |
| correctness | cpu_reference / max_rel_error (tol 0.0001) |
| CUDA features | __global__, finite_difference |
| libraries | - |

Notes: -

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
