# Case: shflScanWarpPrefixSum (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | warp-shuffle inclusive prefix sum (`shfl_scan_test`), launched at blockDim.x=32 (single warp) vs. blockDim.x=256 (8 warps, shared-memory cross-warp broadcast) |
| size | n = 262,144 ints; configuration A: 8192 blocks x 32 threads; configuration B: 1024 blocks x 256 threads |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected (segmented inclusive scan) |

## Source

The `__global__` kernel in `main.cu` (`shfl_scan_test`) is reproduced
**verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

**Deviation from the original task brief, and why:** this case was
originally scoped around two kernels, `shfl_scan_test` and
`shfl2_scan_test`. After fetching `shfl_scan.cu` and its companion
`shfl_integral_image.cuh` in full from the URL above (and cross-checking
every tagged release of the file back to CUDA 5.5), the upstream file
contains exactly **one** scan kernel, `shfl_scan_test` -- there is no
`shfl2_scan_test` anywhere in the sample's history. Rather than invent a
kernel that does not exist upstream, this case uses the one real kernel
that does, instantiated at two different block sizes. This is not an
artificial substitute: `shfl_scan_test`'s own body already implements a
generic two-stage algorithm (intra-warp shuffle scan, then an optional
shared-memory-assisted cross-warp broadcast that only activates when
`blockDim.x / warpSize > 1`), so launching it with `blockDim.x = 32`
(1 warp) vs. `blockDim.x = 256` (8 warps) exercises exactly the
"single-warp-only" vs. "multi-warp hybrid with shared-memory broadcast"
contrast the task was after, using only verbatim, real, upstream code.

The original sample's `uniform_add` kernel (used to propagate a
block's carry into a *second* launch over multiple kernel calls, to
scan arrays far larger than one block) and its `main()` /
`shuffle_simple_test()` / `shuffle_integral_image_test()` driver
(command-line device selection, `helper_cuda`/`helper_functions`
dependency, `cudaMallocHost` pinned buffers, CUDA event timing, the
separate 1920x1080 integral-image demo) were omitted -- this directory
keeps only the single scan kernel needed to isolate the warp-shuffle
technique, launched with `partial_sums = NULL` so each block's segment
is scanned independently (no cross-block carry, so no second kernel
launch or `uniform_add` is needed). Everything else here (host driver,
`reference.h`, `Makefile`, `CMakeLists.txt`, this `README.md`) is new
code written for this repository, replacing the original driver's
timing/verification loop with a deterministic, single-file
CUDA-vs-SYCL comparison harness.

## What this case demonstrates (methods used)

1. **Register-to-register warp-shuffle scan (`__shfl_up_sync`).**
   `shfl_scan_test` computes an inclusive prefix sum within a warp
   purely through register-to-register communication: in a
   `log2(width)`-step loop, each lane reads the value held `i` lanes
   below it (`__shfl_up_sync(mask, value, i, width)`) and adds it in,
   with no shared memory, no global memory round-trip, and no
   `__syncthreads()` -- the fastest possible intra-warp communication
   primitive on the GPU.

2. **Shared-memory-assisted cross-warp broadcast, isolated by launch
   configuration.** The same kernel also contains a second stage: each
   warp's total is written to shared memory, warp 0 scans those warp
   totals (again via `__shfl_up_sync`), and every thread then adds its
   warp's prefix (`blockSum`) read back from shared memory. Whether
   this second stage does anything observable depends entirely on how
   many warps are in the block:
   - At **blockDim.x = 32** (1 warp/block), `blockDim.x / warpSize == 1`,
     so `warp_id` is always `0` and `blockSum` is always `0` for every
     thread -- stage 2 is a functional no-op, and the kernel reduces to
     a pure single-warp, shared-memory-free shuffle scan of 32 elements.
   - At **blockDim.x = 256** (8 warps/block), stage 2 is fully
     exercised: 8 warp totals are broadcast through `sums[]` in shared
     memory and uniformly added, producing a correct 256-element scan.

   Both configurations are guaranteed to produce the same result as an
   ordinary running sum over their respective segment, because integer
   addition is associative and commutative: the CPU reference
   (`reference_segmented_scan` in `reference.h`) computes each
   segment's inclusive sum with a plain left-to-right loop, and no
   matter how the GPU kernel groups its partial sums (intra-warp first,
   then inter-warp), the total for each prefix is identical -- an
   **exact** match (`max_abs_error == 0`) is expected, not a tolerance.

3. **Segmented (per-block-independent) scan via `partial_sums =
   NULL`.** Both launches skip the cross-block carry-propagation path
   entirely (`partial_sums` defaults to `NULL` in the kernel signature,
   and no second kernel or `uniform_add` call follows), so each block
   computes a self-contained inclusive scan of its own contiguous
   segment of the input -- avoiding the need to reproduce the original
   sample's second-level scan-of-partial-sums kernel launch while still
   exercising the exact same warp-shuffle and shared-memory-broadcast
   code paths inside `shfl_scan_test` itself.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`): `in[i] = (i % 9) + 1` for `i` in `[0, n)`, `n =
  262144` (a common multiple of both block sizes used, 32 and 256).
  The same input array is copied to two separate device buffers before
  each launch.
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the
  262,144-element inclusive-scan result from the 256-wide (multi-warp)
  configuration, one `%d` int per line, plus a `PASS`/`FAIL` line on
  stdout comparing *both* configurations' outputs against
  `reference_segmented_scan()` with the matching segment length (32 or
  256) -- exact match expected for both.

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`__shfl_up_sync` typically migrates to
`sycl::permute_group_by_xor`/`sycl::shift_group_left`-style
sub-group shuffle calls, e.g. `sg.shuffle_up()`, and `extern __shared__
int sums[]` becomes a `sycl::local_accessor<int, 1>`; the fixed shuffle
`width` argument constrains the sub-group logic used and is worth
checking carefully against the target sub-group size), build the
result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
