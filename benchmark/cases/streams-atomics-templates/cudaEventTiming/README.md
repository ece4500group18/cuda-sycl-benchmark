# Case: cudaEventTiming (library_api)

| field | value |
| --- | --- |
| category | library_api |
| operation | cudaEvent timing around a kernel |
| correctness | cpu_reference / max_abs_error (tol 1e-05) |
| CUDA features | cudaEvent |
| libraries | - |

Notes: Events time a vector add; correctness = the add result.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
