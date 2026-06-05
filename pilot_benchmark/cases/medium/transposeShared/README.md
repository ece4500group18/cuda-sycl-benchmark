# Case: transposeShared (medium)

| field | value |
| --- | --- |
| category | medium |
| operation | Tiled transpose with shared memory |
| correctness | cpu_reference / exact (tol 0.0) |
| CUDA features | __shared__, __syncthreads, tiling, coalescing |
| libraries | - |

Notes: Shared-memory tiled transpose. sizes=[rows,cols].

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
