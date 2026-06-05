# Case: histogram (medium)

| field | value |
| --- | --- |
| category | medium |
| operation | 256-bin histogram (atomics) |
| correctness | cpu_reference / exact (tol 0.0) |
| CUDA features | __shared__, atomicAdd, cudaMemcpy |
| libraries | - |

Notes: Integer bin counts; exact match. sizes=[n]; 256 bins.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
