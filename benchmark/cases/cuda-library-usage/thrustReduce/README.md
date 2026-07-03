# Case: thrustReduce (library_api)

| field | value |
| --- | --- |
| category | library_api |
| operation | Thrust reduce (sum) |
| correctness | cpu_reference / max_rel_error (tol 0.001) |
| CUDA features | thrust |
| libraries | thrust |

Notes: thrust::reduce sum.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
