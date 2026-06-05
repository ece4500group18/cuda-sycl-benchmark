# Case: curandUniform (library_api)

| field | value |
| --- | --- |
| category | library_api |
| operation | cuRAND uniform generation |
| correctness | statistical / mean_and_range (tol 0.01) |
| CUDA features | curand |
| libraries | curand |

Notes: Verify mean approx 0.5 and all values in [0,1).

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
