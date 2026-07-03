# Case: cudaStream (library_api)

| field | value |
| --- | --- |
| category | library_api |
| operation | Multi-stream vector add |
| correctness | cpu_reference / max_abs_error (tol 1e-05) |
| CUDA features | streams, cudaMemcpyAsync |
| libraries | - |

Notes: Work split across 4 streams.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
