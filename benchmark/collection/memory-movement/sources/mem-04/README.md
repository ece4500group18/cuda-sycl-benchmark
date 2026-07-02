# Case: coalescedAxpyDistribution (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | DAXPY, `y[i] += a*x[i]`, cyclic (grid-stride) vs. block/chunked thread-to-element distribution |
| size | n = 1,048,576 doubles, launch = 1024 blocks x 256 threads (262,144 total threads) |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected |

## Source

The two `__global__` kernels in `main.cu`
(`axpy_cudakernel_block` and `axpy_cudakernel_cyclic`) are reproduced
**verbatim** from:

- Project: **CUDAMicroBench**
- File: `CoMem_AXPY/axpy_cudakernel.cu`
- Repository: https://github.com/passlab/CUDAMicroBench
- Copyright (c) 2021, University of North Carolina at Charlotte and
  Lawrence Livermore National Security, LLC.
- License: BSD-3-Clause (see `LICENSE` in this directory)
- Associated paper: Yi, Xinyao; Stokes, David; Yan, Yonghong; Liao,
  Chunhua. "CUDAMicroBench: Microbenchmarks to Assist CUDA Performance
  Programming." 2021 IEEE IPDPSW, pp. 397-406.
  doi:10.1109/IPDPSW52791.2021.00068

Two other kernels in the original file (`axpy_cudakernel_warmingup`,
`axpy_cudakernel_1perThread`, both a plain one-thread-per-element AXPY
already covered by `memAlign` in this repo) were omitted. Everything
else in this directory (host driver, `reference.h`, `Makefile`,
`CMakeLists.txt`, this `README.md`) is new code written for this
repository, replacing the original driver's `axpy_cuda()` (which just
launches all four kernels back-to-back with no verification) with a
deterministic, single-file CUDA-vs-SYCL comparison harness.

## What this case demonstrates (methods used)

1. **Thread-to-data mapping and coalescing (memory movement / memory
   layout).** Both kernels visit every index in `[0, n)` exactly once
   and apply the identical update, but assign indices to threads
   differently:
   - `axpy_cudakernel_cyclic`: thread `tid` handles
     `tid, tid + total_threads, tid + 2*total_threads, ...` (a
     grid-stride loop). At each step, the 32 threads of a warp have
     **consecutive** `tid` values and therefore touch 32 **consecutive**
     array elements -- one coalesced memory transaction per warp per
     step.
   - `axpy_cudakernel_block`: thread `tid` handles one **contiguous
     chunk** of `n / total_threads` elements starting at
     `tid * block_size`. At each step, the 32 threads of a warp touch
     32 elements that are `block_size` apart -- a strided access
     pattern spread across up to 32 different memory segments per warp
     instead of one.

   Both kernels touch the same data and produce the exact same final
   `y` array; only *which global-memory addresses a warp's threads
   access together, at the same instant* differs -- this is the
   canonical illustration of why thread-to-data mapping (not just
   per-thread arithmetic) determines whether accesses coalesce.

2. **Grid-stride loop pattern** (`for (i = tid; i < n; i += total_threads)`),
   one of the most common idioms in real-world CUDA code for
   decoupling problem size from launch configuration.

3. **Host/device memory management**: `cudaMalloc`, `cudaMemcpy`
   (H2D/D2H), `cudaFree`, plus `cudaGetLastError` /
   `cudaDeviceSynchronize` error checking via the repo's `CHECK` macro
   convention.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `x[i] = ((i % 17) - 8) * 0.25`
  - `y[i] = ((i % 23) - 11) * 0.5`
  - `a = 123.456`, `n = 1048576` (a multiple of the 262,144 total
    threads launched, so `axpy_cudakernel_block`'s `n / total_threads`
    divides evenly, matching the original kernel's "dividable" comment)
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the final
  `y` array after `axpy_cudakernel_cyclic`, one `%.17g` double per
  line, plus a `PASS`/`FAIL` line on stdout comparing both kernels'
  outputs against `reference_axpy()` in `reference.h` (exact match
  expected, since each `y[i]` update is an independent multiply-add
  regardless of which thread performs it or in what order).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct, build the
result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
