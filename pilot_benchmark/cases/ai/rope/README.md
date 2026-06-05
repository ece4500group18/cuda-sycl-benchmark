# Case: rope (ai)

| field | value |
| --- | --- |
| category | ai |
| operation | Rotary position embedding |
| correctness | cpu_reference / max_abs_error (tol 0.0001) |
| CUDA features | __global__, elementwise |
| libraries | - |

Notes: Apply RoPE to a [seq,dim] tensor. sizes=[seq,dim].

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
