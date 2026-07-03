# Case: scanBlock (medium)

| field | value |
| --- | --- |
| category | medium |
| operation | Inclusive prefix sum (single block) |
| correctness | cpu_reference / max_rel_error (tol 0.0001) |
| CUDA features | __shared__, __syncthreads, scan |
| libraries | - |

Notes: Hillis-Steele inclusive scan over one block.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
