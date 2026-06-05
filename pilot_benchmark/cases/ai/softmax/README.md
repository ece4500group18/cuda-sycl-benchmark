# Case: softmax (ai)

| field | value |
| --- | --- |
| category | ai |
| operation | Row-wise softmax |
| correctness | cpu_reference / max_abs_error (tol 1e-05) |
| CUDA features | __global__, __shared__, reduction |
| libraries | - |

Notes: Numerically stable softmax per row. sizes=[rows,cols].

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
