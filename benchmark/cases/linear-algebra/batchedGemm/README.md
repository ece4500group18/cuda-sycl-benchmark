# Case: batchedGemm (ai)

| field | value |
| --- | --- |
| category | ai |
| operation | Batched matrix multiply |
| correctness | cpu_reference / max_rel_error (tol 0.001) |
| CUDA features | __global__, batched |
| libraries | - |

Notes: B batches of NxN GEMM. sizes=[batch,N].

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
