# Case: bfs (hpc)

| field | value |
| --- | --- |
| category | hpc |
| operation | Breadth-first search distances |
| correctness | cpu_reference / exact (tol 0.0) |
| CUDA features | __global__, graph, atomicAdd |
| libraries | - |

Notes: Distances from node 0 on a deterministic graph. sizes=[N].

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
