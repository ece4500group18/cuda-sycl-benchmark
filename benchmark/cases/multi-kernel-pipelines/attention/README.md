# Case: attention (ai)

| field | value |
| --- | --- |
| category | ai |
| operation | Scaled dot-product attention |
| correctness | cpu_reference / max_rel_error (tol 0.002) |
| CUDA features | __global__, __shared__, reduction |
| libraries | - |

Notes: Single head softmax(QK^T/sqrt(d))V. sizes=[seq,dim].

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
