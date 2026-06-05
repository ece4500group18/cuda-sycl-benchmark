# Case: nbodyTiled (medium)

| field | value |
| --- | --- |
| category | medium |
| operation | Tiled N-body acceleration |
| correctness | cpu_reference / max_rel_error (tol 0.002) |
| CUDA features | __shared__, __syncthreads, tiling |
| libraries | - |

Notes: Per-body acceleration with softening; sizes=[N]; output 3*N.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
