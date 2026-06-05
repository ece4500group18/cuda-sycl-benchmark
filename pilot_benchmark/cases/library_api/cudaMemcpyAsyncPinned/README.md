# Case: cudaMemcpyAsyncPinned (library_api)

| field | value |
| --- | --- |
| category | library_api |
| operation | Pinned async copy + scale |
| correctness | cpu_reference / max_abs_error (tol 1e-05) |
| CUDA features | pinned_memory, cudaMemcpyAsync, streams |
| libraries | - |

Notes: cudaHostAlloc pinned + async H2D/D2H; out = 3*a.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
