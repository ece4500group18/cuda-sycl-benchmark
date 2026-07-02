# Case: warpDivergence (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | parity-dependent linear combination `z[i] = a*x[i] + b*y[i]`, branchy vs. branch-free coefficient selection |
| size | n = 1,024,000 floats |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected (exact) |

## Source

The two `__global__` kernels in `main.cu` (`warpDivergence` and
`noWarpDivergence`) are reproduced **verbatim** from:

- Project: **CUDAMicroBench**
- File: `WarpDivRedux/warpDivergenceTest_cudakernel.cu`
- Repository: https://github.com/passlab/CUDAMicroBench
- Copyright (c) 2021, University of North Carolina at Charlotte and
  Lawrence Livermore National Security, LLC.
- License: BSD-3-Clause (see `LICENSE` in this directory)
- Associated paper: Yi, Xinyao; Stokes, David; Yan, Yonghong; Liao,
  Chunhua. "CUDAMicroBench: Microbenchmarks to Assist CUDA Performance
  Programming." 2021 IEEE IPDPSW, pp. 397-406.
  doi:10.1109/IPDPSW52791.2021.00068

A third kernel in the original file (`warmingup`, used only to warm up
the GPU before timing, functionally identical to `warpDivergence`) was
omitted. Everything else in this directory (host driver,
`reference.h`, `Makefile`, `CMakeLists.txt`, this `README.md`) is new
code written for this repository, replacing the original
timing-focused driver (`WarpDivRedux/warpDivergenceTest_cuda.c`) with
a deterministic, single-file CUDA-vs-SYCL comparison harness.

## What this case demonstrates (methods used)

1. **Warp divergence vs. branch-free (predication-style) code.** Both
   kernels compute the same per-element result
   (`z[i] = 2*x[i]+3*y[i]` for even `i`, `z[i] = 3*x[i]+2*y[i]` for odd
   `i`), but select the coefficients differently:
   - `warpDivergence`: an `if (tid % 2 != 0) { a=3; b=2; }` branch.
     Within every warp, even- and odd-indexed lanes take *different*
     control-flow paths, so the warp must execute both the if- and
     else-bodies (serialized) -- the textbook definition of warp
     divergence.
   - `noWarpDivergence`: computes `even = (tid % 2 == 0)` as an integer
     0/1 and blends the two coefficient sets arithmetically
     (`a = even*2 + (1-even)*3`), with **no data-dependent branch** --
     every lane in the warp executes identical instructions.

   This is a "simple but not trivial" kernel: the arithmetic itself is
   a one-line AXPY-like combination, but the *technique* of rewriting
   a per-thread conditional as branch-free arithmetic to avoid SIMT
   divergence is a non-trivial, widely-used GPU optimization pattern.

2. **One-thread-per-element indexing**
   (`blockIdx.x * blockDim.x + threadIdx.x`), the standard simple CUDA
   indexing pattern, applied to three flat float arrays.

3. **Host/device memory management**: `cudaMalloc`, `cudaMemcpy`
   (H2D/D2H), `cudaFree`, plus `cudaGetLastError` /
   `cudaDeviceSynchronize` error checking via the repo's `CHECK` macro
   convention.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `x[i] = ((i % 17) - 8) * 0.25f`
  - `y[i] = ((i % 23) - 11) * 0.5f`
  - `i` in `[0, 1024000)`
- **Output**: `argv[1]` (default `output/cuda_output.txt`), `z[]`
  computed by the branch-free kernel (`noWarpDivergence`), one `%.9g`
  float per line, plus a `PASS`/`FAIL` line on stdout comparing both
  kernels' outputs against `reference_z()` in `reference.h` (exact
  match expected, since both kernels do the same float arithmetic in
  the same order as the reference).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct, build the
result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
