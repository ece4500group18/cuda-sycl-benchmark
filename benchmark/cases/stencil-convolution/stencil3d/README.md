# Case: stencil3d (hpc)

| field | value |
| --- | --- |
| category | hpc |
| operation | 7-point 3D stencil |
| correctness | cpu_reference / max_rel_error (tol 0.001) |
| CUDA features | __global__, stencil, 3D_indexing |
| libraries | - |

Notes: sizes=[nz,ny,nx].

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
