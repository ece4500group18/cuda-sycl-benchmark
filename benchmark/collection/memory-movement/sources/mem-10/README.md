# Case: memAlign (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | DAXPY, `y[i] += a*x[i]`, aligned vs. misaligned global memory access |
| size | n = 1,048,576 doubles (~8 MB per array) |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected |

## Source

The two `__global__` kernels in `main.cu`
(`axpy_cudakernel_1perThread` and `axpy_cudakernel_1perThread_misaligned`)
are reproduced **verbatim** from:

- Project: **CUDAMicroBench**
- File: `MemAlign/axpy_cudakernel.cu`
- Repository: https://github.com/passlab/CUDAMicroBench
- Copyright (c) 2021, University of North Carolina at Charlotte and
  Lawrence Livermore National Security, LLC.
- License: BSD-3-Clause (see `LICENSE` in this directory)
- Associated paper: Yi, Xinyao; Stokes, David; Yan, Yonghong; Liao,
  Chunhua. "CUDAMicroBench: Microbenchmarks to Assist CUDA Performance
  Programming." 2021 IEEE International Parallel and Distributed
  Processing Symposium Workshops (IPDPSW), pp. 397-406.
  doi:10.1109/IPDPSW52791.2021.00068

Everything else in this directory (the host driver in `main.cu`,
`reference.h`, `Makefile`, `CMakeLists.txt`, this `README.md`) is
new code written for this repository, following the same minimal
single-file style used by the other `pilot_benchmark` cases. The
original CUDAMicroBench driver (`MemAlign/axpy_cuda.c`) depends on
`drand48`/timing harness code that is not needed for a deterministic
CUDA-vs-SYCL output comparison, so it was replaced with a small
self-contained `main()` using the same deterministic input formulas
(`gen_x`/`gen_y`) used elsewhere in this repo.

## What this case demonstrates (methods used)

1. **Global memory alignment / coalescing.** Both kernels perform the
   exact same arithmetic (`y[i] += a*x[i]`) over the same index range,
   but with a one-element (8-byte) offset between them:
   - `axpy_cudakernel_1perThread`: thread `i` (block-global) accesses
     `x[i]`/`y[i]` directly, for `i` in `[1, n)`. Within a warp, the 32
     threads access 32 consecutive doubles starting at a 256-byte
     boundary (warp-aligned).
   - `axpy_cudakernel_1perThread_misaligned`: thread `i` accesses
     `x[i+1]`/`y[i+1]`. The same 32-thread warp now accesses 32
     consecutive doubles starting one element past the aligned
     boundary, so the access spans two memory segments instead of one.

   This is a textbook **memory-layout / memory-movement** case: the
   data layout and computation are identical, only the *alignment of
   the access pattern relative to the memory transaction boundary*
   changes. On real hardware this typically shows up as extra memory
   transactions per warp for the misaligned kernel.

2. **One-thread-per-element grid-stride-free indexing**
   (`blockDim.x * blockIdx.x + threadIdx.x`), a simple but standard
   CUDA indexing pattern.

3. **Host/device memory management**: `cudaMalloc`, `cudaMemcpy`
   (H2D/D2H), `cudaFree`, plus `cudaGetLastError` /
   `cudaDeviceSynchronize` error checking via the repo's `CHECK` macro
   convention.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `x[i] = ((i % 17) - 8) * 0.25`
  - `y[i] = ((i % 23) - 11) * 0.5`
  - `a = 123.456`, `n = 1048576`
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the final
  `y` array (after the *aligned* kernel) written as one `%.17g` double
  per line, plus a `PASS`/`FAIL` line on stdout from comparing against
  `reference_axpy()` in `reference.h`.

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct, build the
result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (or compare
with the same tolerance used by `reference_axpy`).
