# Case: asyncCopySingleStage (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | square matrix multiply `C = A*B`, synchronous shared-memory load vs. single-stage `__pipeline_memcpy_async` load |
| size | n = 256 (BLOCK_SIZE = 16, grid = 16x16 blocks of 16x16 threads) |
| correctness | CPU reference (`reference.h`), `max_abs_error < 5e-2` expected (single precision) |

## Source

The two `__global__` kernel templates in `main.cu` (`MatrixMulNaive` and
`MatrixMulAsyncCopySingleStage`, instantiated here for `BLOCK_SIZE=16`,
`float`) are reproduced **verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu`
- Also redistributed unchanged as **CUDAMicroBench**
  `GSOverlap/globalToShmemAsyncCopy.cu`
  (https://github.com/passlab/CUDAMicroBench)
- Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

`USE_CPP_API` is kept at its upstream default of `0` (as `#define`d in
the original file), so both kernels compile to exactly the code path
the original file ships by default -- the `nvcuda::experimental`
C++-wrapper branch (`#if USE_CPP_API`) is present only as dead code,
unchanged, matching upstream.

The original sample's five other kernels (`MatrixMulAsyncCopyMultiStageLargeChunk`,
`MatrixMulAsyncCopyLargeChunk`, `MatrixMulAsyncCopyLargeChunkAWBarrier`,
`MatrixMulAsyncCopyMultiStage`, `MatrixMulNaiveLargeChunk`) and its
`main()` (command-line parsing, `helper_cuda`/`helper_functions`
dependency, multi-kernel timing loop) were omitted -- this directory
keeps only the minimal single-stage-vs-synchronous pair needed to
isolate the async-copy technique. Everything else here (host driver,
`reference.h`, `Makefile`, `CMakeLists.txt`, this `README.md`) is new
code written for this repository, following the same minimal
single-file style used by the other `benchmark/memory` cases.

## What this case demonstrates (methods used)

1. **Asynchronous global-to-shared memory copy (`memory movement`).**
   Both kernels tile `A`/`B` into `BLOCK_SIZE x BLOCK_SIZE` shared-memory
   arrays and accumulate `Csub` over the `k`-dimension identically, but
   load each tile differently:
   - `MatrixMulNaive`: `As[threadIdx.y][threadIdx.x] = A[...]` -- a plain
     assignment, i.e. a synchronous global-memory read into a register
     followed by a shared-memory store, one element per thread.
   - `MatrixMulAsyncCopySingleStage`: `__pipeline_memcpy_async(&As[...],
     A_ptr, sizeof(float))` followed by `__pipeline_commit()` +
     `__pipeline_wait_prior(0)` -- the copy engine moves the element
     from global to shared memory directly, without occupying a
     register or a load/store instruction pair on the issuing thread.

   Both kernels read/write the same arrays in the same layout and
   compute the exact same `C = A*B` with the exact same per-tile
   accumulation order; only *the mechanism used to move a tile from
   global to shared memory* differs -- this is the canonical
   "memory movement" case for the async-copy hardware path introduced
   with compute capability 8.0 (it still compiles and runs correctly on
   older architectures via the pipeline API's software fallback, just
   without the hardware acceleration).

2. **Templated tile size** (`template <int BLOCK_SIZE>`), instantiated
   once for `BLOCK_SIZE=16` -- the same 2D tiled-matmul indexing pattern
   used by `tiledMatmulShmem`, but expressed generically.

3. **`__syncthreads()` barriers** used twice per tile iteration (after
   the tile load -- sync or async -- and after the partial-product
   accumulation), identical in both kernels.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `A[i] = ((i % 17) - 8) * 0.25f`, `B[i] = ((i % 23) - 11) * 0.5f`,
    `i` in `[0, n*n)`, `n = 256`
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the
  256x256 row-major `C = A*B` matrix computed by
  `MatrixMulAsyncCopySingleStage`, one `%.9g` float per line (65536
  lines), plus a `PASS`/`FAIL` line on stdout comparing both kernels'
  outputs against `reference_matmul()` in `reference.h`.

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

Requires a CUDA 11+ toolkit (for `<cuda_pipeline.h>`); the async-copy
hardware path itself requires compute capability 8.0+ (Ampere or
later), though the code also builds and runs correctly on sm_70/sm_75
via the pipeline API's software fallback.

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`__pipeline_memcpy_async`/`__pipeline_commit`/`__pipeline_wait_prior`
typically migrate to `sycl::ext::oneapi::experimental` group/pipeline
async-copy APIs, or may require manual porting -- this is one of the
more interesting migration targets in this benchmark set), build the
result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files within the
same tolerance.
