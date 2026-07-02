# Case: gridSyncCGReduction (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | sum reduction of n doubles: one cooperative-launch kernel using `cg::this_grid()`/`grid.sync()` vs. two ordinary kernel launches computing the same partial-sum tree |
| size | n = 262,144 (`1<<18`) doubles; launch = up to 64 blocks x 128 threads (clamped to what the device can run as one co-resident cooperative grid) |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected for **both** GPU paths |

## Source

The device function `reduceBlock` and the `__global__` kernel
`reduceSinglePassMultiBlockCG` in `main.cu` are reproduced from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/2_Concepts_and_Techniques/reductionMultiBlockCG/reductionMultiBlockCG.cu`
- URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/reductionMultiBlockCG/reductionMultiBlockCG.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

`reduceBlock` is reproduced **completely unmodified**.
`reduceSinglePassMultiBlockCG` is reproduced with exactly one mechanical
type change: upstream declares `const float *g_idata, float *g_odata` (with
an internal `double` shared-memory accumulator); here both are `double*`
instead, so a block's partial sum is no longer truncated back to `float`
precision when written to global memory. That single change is what makes
an exact-match correctness oracle possible (see "What this case
demonstrates" below) -- no other line of the algorithm, and none of its
control flow, is touched.

The original file's `main()`/`runTest()`/`benchmarkReduce()` (command-line
parsing, `helper_cuda`/`helper_functions` dependency, `rand()`-based input,
`StopWatchInterface` timing loop over 100 iterations, Kahan-summation CPU
reference with a `1e-8 * size` tolerance) are not needed for a
deterministic, single-file CUDA-vs-SYCL output comparison and are omitted.
This directory's `main()` replaces them with a fixed, deterministic launch
of both a cooperative and a non-cooperative reduction path and an exact
CPU oracle. The two new kernels `reduceBlockPartial` and
`reduceCombinePartials` (see next section) are new code written for this
repository -- they do not exist upstream -- built specifically to give the
cooperative kernel a same-arithmetic, non-cooperative counterpart to
compare against.

## What this case demonstrates (methods used)

1. **Grid-wide barrier via a single cooperative-launch kernel
   (`cg::this_grid()` + `cg::sync(grid)`, launched with
   `cudaLaunchCooperativeKernel`).** `reduceSinglePassMultiBlockCG` does the
   entire reduction -- per-block partial sums *and* the final combination
   of all blocks' partial sums into one scalar -- inside a single kernel
   launch. This is only legal because a cooperative launch guarantees every
   block in the grid is resident on the device at the same time, so
   `cg::sync(grid)` (a barrier across *all* blocks, not just one) can
   safely separate "every block computes its partial sum" from "block 0
   combines all the partial sums."

2. **The same computation split across two ordinary kernel launches, with
   no cooperative launch and no grid-wide barrier at all.**
   `reduceBlockPartial` is exactly `reduceSinglePassMultiBlockCG`'s first
   half (same grid-stride assignment of input elements to threads, same
   unmodified `reduceBlock` warp-shuffle-tree-then-leader-loop reduction,
   same per-block partial written to `g_odata[blockIdx.x]`), launched as an
   ordinary kernel. `reduceCombinePartials` is exactly
   `reduceSinglePassMultiBlockCG`'s second half (the same sequential
   `for (block = 1; block < numBlocks; ++block) g_odata[0] += g_odata[block];`
   loop), launched as its own, separate kernel afterward. Ordinary
   kernel-launch/stream ordering (each kernel only starts once the
   previous one on the same stream has finished) stands in for the
   in-kernel grid-wide barrier -- this is the "two or more kernel calls"
   alternative that the upstream file's own doc comment explicitly
   contrasts itself with ("as opposed to two or more kernel calls as shown
   in the 'reduction' CUDA Sample").

   Both paths visit every input element exactly once via the identical
   grid-stride assignment, reduce each block with the identical
   `reduceBlock` helper, and combine the resulting per-block partial sums
   with the identical sequential loop -- so the *only* difference between
   the two code paths is **whether the two phases are joined by an
   in-kernel `grid.sync()` inside one cooperative launch, or by ordinary
   kernel-launch ordering across two separate launches.** This isolates
   the grid-sync cooperative-launch technique as the sole thing being
   compared.

3. **Why `max_abs_error == 0` is a valid oracle even though the two paths
   (and, at a finer grain, the warp-shuffle tree inside `reduceBlock`)
   combine values in different groupings/orders.** The input
   `input[i] = ((i % 23) - 11) * 0.5` is always an exact multiple of 0.5
   with magnitude <= 5.5. For n = 262,144 such values, every partial sum
   that either reduction can ever produce, in any order or grouping, has
   magnitude well under 2^52 (in the "value x2" integer sense), so it is
   always exactly representable in `double`. Since IEEE-754 addition is
   exact whenever the true mathematical result is itself representable, no
   addition step performed by either reduction (or by a plain sequential
   CPU sum) ever rounds -- so the warp-shuffle tree, the sequential
   per-block leader loop, the in-kernel grid-sync combine, the two-launch
   combine, and a trivial CPU `for` loop are all guaranteed to compute the
   bit-identical final double. See `reference.h` for the same argument
   written next to the code.

## Input / Output

- **Input** (generated deterministically on the host, see `reference.h`):
  `input[i] = ((i % 23) - 11) * 0.5`, `i` in `[0, n)`, `n = 262144`.
- **Launch configuration**: `threadsPerBlock = 128`; `numBlocks` is
  computed at runtime from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
  x the device's SM count, clamped to at most 64 blocks (so the cooperative
  grid is guaranteed launchable as one co-resident grid, and modest in
  size). Both GPU paths use the same `numBlocks`/`threadsPerBlock`. The
  exact-match argument above holds for *any* valid `numBlocks`/
  `threadsPerBlock`, so this clamp only affects performance, never
  correctness.
- **Output**: `argv[1]` (default `output/cuda_output.txt`), one line: the
  scalar sum computed by the cooperative single-pass kernel
  (`%.17g` format), plus a `PASS`/`FAIL` line on stdout comparing *both*
  the cooperative kernel's result and the two-launch kernels' result
  against `reference_reduce_sum()` in `reference.h` (exact match expected
  for both).

## Build & run

```bash
make run            # nvcc build (-rdc=true, required by cudaLaunchCooperativeKernel), writes output/cuda_output.txt
```

Requires a GPU that reports `cudaDeviceProp::cooperativeLaunch == true`
(true for essentially all discrete GPUs sm_70/sm_80/sm_90 and newer); the
program checks this at startup and exits early with a message if the
device does not support cooperative kernel launch. Single-GPU only -- no
multi-GPU cooperative launch is used or required.

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cudaLaunchCooperativeKernel` + `cg::this_grid()`/`grid.sync()` typically
migrate to a `sycl::nd_range` kernel using the root group's
`ext::oneapi::experimental::group_barrier`/root-group synchronization APIs,
or may require a manual restructure -- this is one of the more interesting
migration targets in this benchmark set, since SYCL's grid-wide
synchronization support is still evolving across backends), build the
result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
