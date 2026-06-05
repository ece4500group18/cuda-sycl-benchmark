# Case: conv1dShared (medium)

| field | value |
| --- | --- |
| category | medium |
| operation | 1D convolution with shared-memory halo |
| correctness | cpu_reference / max_rel_error (tol 0.0001) |
| CUDA features | __shared__, __syncthreads, halo, convolution |
| libraries | - |

Notes: Radius-3 fixed-weight stencil with shared-memory halo.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
