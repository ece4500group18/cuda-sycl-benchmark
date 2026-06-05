# Case: gemm (ai)

| field | value |
| --- | --- |
| category | ai |
| operation | GEMM with bias (C = A*B + bias) |
| correctness | cpu_reference / max_rel_error (tol 0.001) |
| CUDA features | __global__, tiling |
| libraries | - |

Notes: Square GEMM + per-column bias. sizes=[N].

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
