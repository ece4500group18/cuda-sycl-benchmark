# Case: rmsnorm (ai)

| field | value |
| --- | --- |
| category | ai |
| operation | RMS normalization |
| correctness | cpu_reference / max_abs_error (tol 0.0001) |
| CUDA features | __global__, __shared__, reduction |
| libraries | - |

Notes: RMSNorm with gamma=1,eps=1e-6. sizes=[rows,cols].

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
