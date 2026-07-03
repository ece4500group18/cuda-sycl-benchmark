# Case: gridStride (easy)

| field | value |
| --- | --- |
| category | easy |
| operation | Grid-stride loop vector add |
| correctness | cpu_reference / max_abs_error (tol 1e-05) |
| CUDA features | __global__, grid_stride_loop, cudaMalloc, cudaMemcpy, cudaFree |
| libraries | - |

Notes: Classic grid-stride loop pattern.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
