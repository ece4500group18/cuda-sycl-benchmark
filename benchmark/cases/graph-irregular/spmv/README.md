# Case: spmv (hpc)

| field | value |
| --- | --- |
| category | hpc |
| operation | Sparse matrix-vector (CSR) |
| correctness | cpu_reference / max_rel_error (tol 0.0001) |
| CUDA features | __global__, sparse, csr |
| libraries | - |

Notes: Tridiagonal CSR matrix; y = A*x. sizes=[N].

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
