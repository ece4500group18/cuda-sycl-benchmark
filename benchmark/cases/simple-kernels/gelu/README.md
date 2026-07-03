# Case: gelu (ai)

| field | value |
| --- | --- |
| category | ai |
| operation | GELU activation (tanh approx) |
| correctness | cpu_reference / max_abs_error (tol 0.0001) |
| CUDA features | __global__, elementwise |
| libraries | - |

Notes: tanh approximation of GELU.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
