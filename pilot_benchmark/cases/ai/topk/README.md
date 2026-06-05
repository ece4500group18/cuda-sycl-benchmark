# Case: topk (ai)

| field | value |
| --- | --- |
| category | ai |
| operation | Top-k per row |
| correctness | cpu_reference / exact (tol 0.0) |
| CUDA features | __global__, selection |
| libraries | - |

Notes: Top-8 values per row, sorted desc. sizes=[rows,cols]; k=8.

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
