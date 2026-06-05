# Case: dotProduct (medium)

| field | value |
| --- | --- |
| category | medium |
| operation | Dot product via reduction |
| correctness | cpu_reference / max_rel_error (tol 0.001) |
| CUDA features | __shared__, __syncthreads, reduction |
| libraries | - |

Notes: -

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
