# Case: thrustSort (library_api)

| field | value |
| --- | --- |
| category | library_api |
| operation | Thrust sort |
| correctness | cpu_reference / exact (tol 0.0) |
| CUDA features | thrust |
| libraries | thrust |

Notes: thrust::sort ascending.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
