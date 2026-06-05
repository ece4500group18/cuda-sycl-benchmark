# Case: cublasGemm (library_api)

| field | value |
| --- | --- |
| category | library_api |
| operation | cuBLAS SGEMM |
| correctness | cpu_reference / max_rel_error (tol 0.001) |
| CUDA features | cublas, streams |
| libraries | cublas |

Notes: cublasSgemm; column-major handling. sizes=[N].

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
