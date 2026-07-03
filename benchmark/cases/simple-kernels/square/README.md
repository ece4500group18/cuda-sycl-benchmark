# Case: square (easy)

| field | value |
| --- | --- |
| category | easy |
| operation | Square elements (out = a*a) |
| correctness | cpu_reference / max_abs_error (tol 1e-05) |
| CUDA features | __global__, cudaMalloc, cudaMemcpy, cudaFree |
| libraries | - |

Notes: -

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
