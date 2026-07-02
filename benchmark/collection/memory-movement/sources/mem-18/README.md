# Case: tiledMatmulShmem (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | square matrix multiply `C = A*B`, global-memory-only tiling vs. shared-memory tiling |
| size | n = 128 (BLOCK_SIZE = 16, grid = 8x8 blocks of 16x16 threads) |
| correctness | CPU reference (`reference.h`), `max_abs_error < 1e-9` expected (double precision, exact) |

## Source

The two `__global__` kernels in `main.cu` (`global_block` and
`shared_block`) are reproduced **verbatim** from:

- Project: **CUDAMicroBench**
- File: `Shmem/mm_kernel.cu`
- Repository: https://github.com/passlab/CUDAMicroBench
- Copyright (c) 2021, University of North Carolina at Charlotte and
  Lawrence Livermore National Security, LLC.
- License: BSD-3-Clause (see `LICENSE` in this directory)
- Associated paper: Yi, Xinyao; Stokes, David; Yan, Yonghong; Liao,
  Chunhua. "CUDAMicroBench: Microbenchmarks to Assist CUDA Performance
  Programming." 2021 IEEE IPDPSW, pp. 397-406.
  doi:10.1109/IPDPSW52791.2021.00068

A third kernel in the original file (`global_element`, a naive
one-thread-per-output-element kernel with no tiling at all) was
omitted since `global_block` already covers the "global-memory-only"
baseline in a directly comparable (tile-loop) form. Everything else in
this directory (host driver, `reference.h`, `Makefile`,
`CMakeLists.txt`, this `README.md`) is new code written for this
repository, replacing the original OpenMP/CUDA dual-mode driver
(`Shmem/mm_omp_cuda.c`) with a deterministic, single-file
CUDA-vs-SYCL comparison harness.

## What this case demonstrates (methods used)

1. **Shared-memory tiling (memory movement / memory layout).** Both
   kernels divide the n x n matrices into `BLOCK_SIZE x BLOCK_SIZE`
   (16x16) tiles and accumulate `Csub` over the `k`-dimension tile by
   tile:
   - `global_block`: for each tile, every thread directly reads
     `BLOCK_SIZE` elements of `A` and `B` from **global memory** in
     the inner `k`-loop. Each element of a given A/B tile is read
     redundantly by every thread in the block (16x re-reads of the
     same global-memory tile per block).
   - `shared_block`: for each tile, the block first cooperatively
     copies its A-tile and B-tile into `__shared__ REAL As[16][16]`
     and `Bs[16][16]` (one element per thread, `__syncthreads()` to
     ensure the copy completes), then every thread computes its
     partial dot product by reading **shared memory** instead. Each
     global-memory element of A/B is read exactly once per block.

   Both kernels read/write the same global arrays in the same
   row-major layout and produce the same `C = A*B`; only *where the
   working tile lives* (global vs. on-chip shared memory) differs --
   this is the canonical "memory movement and memory layout"
   optimization (data reuse via an explicit on-chip copy).

2. **2D thread/block indexing** (`blockIdx`/`threadIdx` in `x`/`y`,
   `dim3` grid/block) over a tiled iteration space -- a simple but
   non-trivial 2D parallel pattern.

3. **`__syncthreads()` barriers** used twice per tile iteration (after
   the shared-memory load, and after the partial-product accumulation)
   to coordinate a cooperative load/compute pipeline within a block.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `A[i] = ((i % 17) - 8) * 0.25`, `B[i] = ((i % 23) - 11) * 0.5`,
    `i` in `[0, n*n)`, `n = 128`
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the
  128x128 row-major `C = A*B` matrix computed by `shared_block`, one
  `%.17g` double per line (16384 lines), plus a `PASS`/`FAIL` line on
  stdout comparing both kernels' outputs against `reference_matmul()`
  in `reference.h`.

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`__shared__` 2D arrays inside the kernel typically migrate to
`sycl::local_accessor` or `group_local_memory`), build the result, run
it with the same `argv[1]` convention (e.g. `output/sycl_output.txt`),
and diff the two output files (exact match expected in double
precision).
