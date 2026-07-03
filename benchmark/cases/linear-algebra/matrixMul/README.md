# Case: matrixMul (easy)

| field | value |
| --- | --- |
| category | easy |
| operation |`C = A*B`, N=128 |
| correctness | CPU reference, `max_rel_error < 1e-3`|

## Pipeline
1. `tools/build_cuda.py` builds `original/main.cu` with nvcc.
2. `tools/run_case.py --variant cuda` runs it, writing `output/cuda_output.txt`.
3. `tools/verify_case.py --variant cuda` runs `tests/verify.py`, which
   regenerates the inputs in numpy and checks the result.
4. `tools/run_syclomatic.py` migrates the source into `syclomatic/`.
5. `tools/build_sycl.py` / `run_case.py --variant sycl` / `verify_case.py`
   repeat the cycle for the migrated SYCL code (`output/sycl_output.txt`).

All steps degrade to `skipped_*` when a toolchain or device is unavailable.
