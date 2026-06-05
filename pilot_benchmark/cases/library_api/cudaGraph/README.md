# Case: cudaGraph (library_api)

| field | value |
| --- | --- |
| category | library_api |
| operation | CUDA graph capturing a kernel |
| correctness | cpu_reference / max_abs_error (tol 1e-05) |
| CUDA features | cudaGraph, streams |
| libraries | - |

Notes: Stream-captured graph runs a scale kernel; out = 2*a + 1.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
