# Case: vectorAdd (easy)

| field | value |
| --- | --- |
| category | easy |
| operation | `C = A + B`, n = 100000 floats |
| CUDA features | `__global__`, cudaMalloc/Memcpy/Free, thread indexing |
| correctness | CPU reference, `max_abs_error < 1e-5` |

## Pipeline
1. `tools/build_cuda.py` builds `original/main.cu` with nvcc.
2. `tools/run_case.py --variant cuda` runs it, writing `output/cuda_output.txt`.
3. `tools/verify_case.py --variant cuda` runs `tests/verify.py`, which
   regenerates A, B in numpy and checks `C`.
4. `tools/run_syclomatic.py` migrates the source into `syclomatic/`.
5. `tools/build_sycl.py` / `run_case.py --variant sycl` / `verify_case.py`
   repeat the cycle for the migrated SYCL code (`output/sycl_output.txt`).

All steps degrade to `skipped_*` when a toolchain or device is unavailable.
