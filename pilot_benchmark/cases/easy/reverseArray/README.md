# Case: reverseArray (easy)

| field | value |
| --- | --- |
| category | easy |
| operation | Reverse array (out[i] = in[n-1-i]) |
| correctness | cpu_reference / exact (tol 0.0) |
| CUDA features | __global__, cudaMalloc, cudaMemcpy, cudaFree |
| libraries | - |

Notes: Pure data movement.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
