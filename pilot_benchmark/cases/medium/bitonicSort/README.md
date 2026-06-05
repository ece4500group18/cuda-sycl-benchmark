# Case: bitonicSort (medium)

| field | value |
| --- | --- |
| category | medium |
| operation | Bitonic sort (single block) |
| correctness | cpu_reference / exact (tol 0.0) |
| CUDA features | __shared__, __syncthreads, sorting |
| libraries | - |

Notes: Ascending bitonic sort over one block of 1024.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
