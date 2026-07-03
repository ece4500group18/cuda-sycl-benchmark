# Case: cublasAxpy (library_api)

| field | value |
| --- | --- |
| category | library_api |
| operation | cuBLAS SAXPY |
| correctness | cpu_reference / max_abs_error (tol 0.0001) |
| CUDA features | cublas |
| libraries | cublas |

Notes: cublasSaxpy; y = alpha*x + y.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
