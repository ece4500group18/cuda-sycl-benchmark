# Case: pyramidTiledPathfinderDP (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout -- temporal tiling) |
| operation | Rodinia "pathfinder" shortest-path DP recurrence, batched pyramid tiling (4 rows/launch) vs. one-row-per-launch, same kernel |
| size | rows = 64, cols = 512, BLOCK_SIZE = 256; batched = 16 kernel launches, singlestep = 63 kernel launches |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected (integer MIN/ADD recurrence) |

## Source

The `__global__` kernel in `main.cu` (`dynproc_kernel`, together with the
`BLOCK_SIZE`/`HALO` constants and `IN_RANGE`/`MIN` helper macros it
depends on) is reproduced **verbatim** from:

- Project: **Rodinia** benchmark suite (gpu-rodinia mirror)
- File: `cuda/pathfinder/pathfinder.cu`
- Repository: https://github.com/yuhc/gpu-rodinia/blob/master/cuda/pathfinder/pathfinder.cu
- Copyright (c) 2008-2011 University of Virginia. All rights reserved.
- License: BSD-style, permissive (see `LICENSE` in this directory for
  the full text; GitHub's automatic license detector tags the upstream
  repository "NOASSERTION" because the wording isn't a byte-exact match
  for the cataloged SPDX `BSD-3-Clause` template, but it grants the same
  three permissive conditions plus the standard "AS IS" disclaimer).

Everything else in the upstream file -- `main()`, `run()`, `init()`
(which seeded `wall[][]` with `rand() % 10` under a fixed `srand(9)`,
not reproducible across libc versions), the `calc_path()` host loop,
`TIMING`-gated `gettimeofday` instrumentation, and command-line
argument parsing -- was omitted. This directory's `main()` is new code
that reimplements the same ping-pong buffer / kernel-launch loop
structure as upstream's `calc_path()` (twice, once per pyramid height),
replacing `rand()` with a deterministic index formula and upstream's
plain `printf`-only output with a deterministic CUDA-vs-CPU-reference
comparison harness. `reference.h` (deterministic input generator plus
the CPU DP recurrence) is new code written for this repository.
Everything else here (`Makefile`, `CMakeLists.txt`, this `README.md`)
is likewise new.

## What this case demonstrates (methods used)

1. **Pyramid / temporal tiling as a memory-traffic-reduction technique
   (memory movement).** `dynproc_kernel` takes an `iteration` parameter:
   it loads one row-slice of the current DP state into `__shared__`
   memory *once* per launch, then advances the recurrence `iteration`
   times entirely out of shared memory (re-synchronizing only between
   rows with `__syncthreads()`), before writing just the final row back
   to global memory. To keep the shared-memory recurrence consistent at
   a block's edges after `iteration` steps, each block additionally
   loads a `border = iteration * HALO` halo of extra columns from its
   neighbors on each side -- the classic "pyramid" shape (the valid
   output region shrinks by one column per side per step).
   - **Batched path**: `iteration = 4` -> 16 kernel launches for the 63
     row-updates, each amortizing one halo exchange (and one round trip
     through `gpuWall`/`gpuResult` global memory) over 4 rows of DP
     work.
   - **Singlestep path**: `iteration = 1` -> 63 kernel launches, one row
     (and one halo exchange) per launch -- the straightforward,
     un-tiled recurrence.
   Both paths call the identical `dynproc_kernel`, read the identical
   `wall` cost array, and produce the identical final `result[cols]`
   array; only the number of DP time-steps batched behind each
   halo-exchange / kernel launch differs. This isolates *temporal*
   tiling (reusing shared memory across several sequential time-steps
   of the same spatial region) as a distinct memory-movement technique
   from the *spatial* shared-memory tiling in this repo's
   `tiledMatmulShmem` case (where a shared-memory tile always
   corresponds to one fixed output region computed once, not several
   time-steps of a recurrence).

2. **Halo/ghost-zone exchange sizing.** `border = iteration * HALO`
   ties the amount of redundantly-loaded boundary data directly to how
   many time-steps a launch batches -- a concrete illustration of the
   tiling-depth-vs-redundant-halo-traffic trade-off pyramid tiling makes
   (deeper pyramids launch fewer kernels and touch global memory fewer
   times per row-update, at the cost of a wider halo, i.e. more
   redundantly-loaded/recomputed boundary columns per block, and a
   smaller valid output region per launch).

3. **Exact correctness regardless of batching.** The recurrence
   `result[t][j] = wall[t][j] + min(result[t-1][j-1], result[t-1][j],
   result[t-1][j+1])` (edge-clamped at the array boundaries) is pure
   integer MIN/ADD arithmetic with no floating-point accumulation-order
   sensitivity. Since pyramid tiling changes only *when* (which kernel
   launch) each row is computed, not *what* is computed or in what
   row-to-row order, the batched and singlestep GPU paths -- and the
   sequential CPU reference in `reference.h` -- are all expected to
   agree bit-for-bit.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`): a `rows x cols` row-major array
  `data[i*cols+j] = (i*31 + j*17) % 10`, `rows = 64`, `cols = 512`. Row
  0 is the initial DP source row (copied to the device as the ping-pong
  buffer's initial state, matching upstream's `gpuResult[0]`); rows
  `1..rows-1` are the per-step wall costs (`gpuWall`, matching
  upstream's `data + cols` offset).
- **Output**: `argv[1]` (default `output/cuda_output.txt`), 512 lines,
  the final DP `result[cols]` array (one `%d` int per line) from the
  batched (pyramid height 4) path, plus a `PASS`/`FAIL` line on stdout
  comparing both the batched and singlestep paths against
  `reference_pathfinder()` in `reference.h`, and against each other
  (exact integer match expected in all three comparisons).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (the
`__shared__` arrays become `sycl::local_accessor`s / local memory, and
`__syncthreads()` becomes `item.barrier(sycl::access::fence_space::local_space)`),
build the result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
