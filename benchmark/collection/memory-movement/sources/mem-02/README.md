# Case: bankConflictReduction (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | per-block shared-memory sum reduction, sequential vs. interleaved addressing |
| size | n = 1,024,000 floats, ThreadsPerBlock = 256, blocks = 4000 |
| correctness | CPU reference (`reference.h`), tolerance 1e-3 (float accumulation) |

## Source

The two `__global__` kernels in `main.cu`
(`sum_cudakernel` and `sum_cudakernel_bc`) are reproduced **verbatim**
from:

- Project: **CUDAMicroBench**
- File: `BankRedux/sum_cudakernel.cu`
- Repository: https://github.com/passlab/CUDAMicroBench
- Copyright (c) 2021, University of North Carolina at Charlotte and
  Lawrence Livermore National Security, LLC.
- License: BSD-3-Clause (see `LICENSE` in this directory)
- Associated paper: Yi, Xinyao; Stokes, David; Yan, Yonghong; Liao,
  Chunhua. "CUDAMicroBench: Microbenchmarks to Assist CUDA Performance
  Programming." 2021 IEEE IPDPSW, pp. 397-406.
  doi:10.1109/IPDPSW52791.2021.00068

A third kernel in the original file (`sum_warmingup`, identical to
`sum_cudakernel`, used only to warm up the GPU before timing) was
omitted as it adds nothing new. Everything else in this directory
(host driver, `reference.h`, `Makefile`, `CMakeLists.txt`, this
`README.md`) is new code written for this repository, replacing the
original timing-focused driver (`BankRedux/sum_cuda.c`) with a
deterministic, single-file CUDA-vs-SYCL comparison harness.

## What this case demonstrates (methods used)

1. **Shared-memory bank conflicts (memory layout).** Both kernels load
   one element per thread into `__shared__ float cache[256]` and reduce
   it to a single value with a tree of additions guarded by
   `__syncthreads()`:
   - `sum_cudakernel` (sequential addressing): at step `i`
     (`i = blockDim.x/2, .../4, ..., 1`), thread `cacheIndex < i`
     computes `cache[cacheIndex] += cache[cacheIndex + i]`. Active
     threads have **consecutive** `cacheIndex` values, so they touch
     consecutive shared-memory banks -> no bank conflicts.
   - `sum_cudakernel_bc` (interleaved addressing): at step `i`
     (`i = 1, 2, 4, ...`), thread `cacheIndex` computes
     `index = 2*i*cacheIndex` and accesses `cache[index]`/`cache[index+i]`.
     For `i >= 2`, multiple active threads map to shared-memory
     addresses separated by a multiple of the 32-bank width, so several
     threads in a warp hit the **same bank** -> bank conflicts.

   Both kernels read the same input layout and produce the same
   per-block sum; only the *shared-memory access pattern during the
   reduction* differs -- a direct illustration of how memory layout
   inside a single `__shared__` array affects performance.

2. **Tree-based parallel reduction in shared memory** with
   `__syncthreads()` barriers between steps -- a simple but
   non-trivial synchronization pattern.

3. **Per-block partial results**: each block writes one output value
   (`result[blockIdx.x]`), a common pattern for the first stage of a
   two-stage (block-then-host or block-then-second-kernel) reduction.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `x[i] = ((i % 17) - 8) * 0.25f`, `i` in `[0, 1024000)`
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the 4000
  per-block partial sums from `sum_cudakernel` (sequential addressing),
  one `%.9g` float per line, plus a `PASS`/`FAIL` line on stdout
  comparing both kernels' outputs against `reference_block_sum()` in
  `reference.h` (tolerance 1e-3 due to float summation order).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct, build the
result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and compare the two output files within the
same 1e-3 tolerance.
