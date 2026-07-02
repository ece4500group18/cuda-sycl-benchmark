# Case: sharedScanVsShuffleScan (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | 256-element unsigned-int exclusive prefix sum (scan): shared-memory-buffered Hillis-Steele scan vs. warp-shuffle register-to-register scan |
| size | n = 256 uints, single thread block (`scanExclusiveShared`: 64 threads x uint4; `scanExclusiveShuffle`: 256 threads x uint) |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected (exact) |

## Source

The device helper functions `scan1Inclusive`, `scan1Exclusive`,
`scan4Inclusive`, `scan4Exclusive`, and the `__global__` kernel
`scanExclusiveShared` in `main.cu` are reproduced with **unmodified
bodies** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/2_Concepts_and_Techniques/scan/scan.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/scan/scan.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

Only the `THREADBLOCK_SIZE` launch-configuration constant was changed,
from the upstream default of 256 (1024 scalar elements per block, one
`uint4` per thread) down to 64, so that a single block processes
exactly 256 scalar elements (`4 * THREADBLOCK_SIZE`) -- the modest,
easy-to-verify problem size used throughout this case. The upstream
file's other two kernels (`scanExclusiveShared2`, `uniformUpdate`) and
all of its host-side "interface" functions (`initScan`, `closeScan`,
`scanExclusiveShort`, `scanExclusiveLarge`, `factorRadix2`, `iDivUp`)
exist only to batch many single-block scans into an arbitrarily large
multi-level scan (via `scan_common.h` and the `cuda-samples` shared
`helper_cuda.h`); they were omitted because this case only needs one
single-block invocation of `scanExclusiveShared` and pulling them in
would require vendoring `scan_common.h` and the `helper_cuda`/
`helper_functions` headers for no benefit to the comparison being made
here. `scanExclusiveShuffle` (the warp-shuffle scan compared against
it) does **not** appear anywhere in the upstream file -- it is new code
written for this repository, implementing the identical exclusive-scan
mathematics via `__shfl_up_sync` instead of a shared-memory buffer.
Everything else in this directory (host driver, `reference.h`,
`Makefile`, `CMakeLists.txt`, this `README.md`) is likewise new code
written for this repository, replacing the original file's
batching-oriented interface with a deterministic, single-file
CUDA-vs-SYCL comparison harness.

## What this case demonstrates (methods used)

1. **Shared-memory-buffered intra-block scan (`memory movement`).**
   `scanExclusiveShared` computes each thread's 4-element `uint4`
   level-0 partial sum in registers, then runs a Hillis-Steele
   inclusive scan (`scan1Inclusive`) over the 64 per-thread partial
   sums entirely through a padded `__shared__ uint s_Data[]` buffer:
   at every one of `log2(64) = 6` doubling steps, every thread reads
   two values out of shared memory, adds them, and writes the result
   back to shared memory, bracketed by two block-wide barriers
   (`cg::sync(cta)`, i.e. `__syncthreads()`) per step so the
   read-before-overwrite hazard across threads is respected.

2. **Warp-shuffle register-to-register scan (`memory movement`).**
   `scanExclusiveShuffle` computes the same 256-element exclusive scan
   without ever staging the *data* through shared memory: each of the
   8 warps runs its own 5-step Hillis-Steele scan using
   `__shfl_up_sync` to read a value directly out of another lane's
   register (no shared-memory round trip, no barrier needed within a
   warp -- shuffles are automatically synchronizing within the warp).
   Shared memory is touched only to combine the 8 warps' totals into
   8 per-warp offsets (`s_WarpTotal`/`s_WarpOffset`, done by a single
   thread) -- an unavoidable few-word exchange once a scan spans more
   than one warp, in sharp contrast to `scanExclusiveShared`'s full
   256-element shared-memory buffer that is read and written at every
   doubling step.

   Both kernels touch the same 256 input elements and produce the
   exact same exclusive prefix sum; only *how a partial sum moves from
   one thread to another* (buffered through shared memory vs.
   register-to-register via warp shuffle) differs -- the canonical
   memory-movement contrast this case isolates.

3. **Order-independent unsigned accumulation.** `in[i]` is always in
   `[1, 7]`, so the running sum over 256 elements never comes close to
   `2^32`. Unsigned-integer addition is commutative and associative
   and cannot overflow here, so *any* reduction-tree shape -- the
   64-wide Hillis-Steele shared-memory tree, the 32-wide per-warp
   shuffle tree combined with a sequential 8-way warp-offset step, or
   a plain sequential CPU loop -- must produce the bit-identical
   result. This is what makes an **exact** (`max_abs_error == 0`)
   correctness oracle valid here, unlike floating-point reductions
   where accumulation order can change the result.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`): `in[i] = (i % 7) + 1` for `i` in `[0, 256)`.
- **Output**: `argv[1]` (default `output/cuda_output.txt`), 256 lines,
  the exclusive prefix sum computed by `scanExclusiveShuffle`, one
  `%u` unsigned int per line, plus a `PASS`/`FAIL` line on stdout
  comparing **both** kernels' outputs against
  `reference_exclusive_scan()` in `reference.h` (exact match expected).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cg::thread_block`/`cg::sync` typically migrate to a SYCL `nd_item`'s
`barrier()`, and `__shfl_up_sync` typically migrates to
`sycl::sub_group`'s `shift_group_right`/`sycl::shift_group_right`
free function on a subgroup), build the result, run it with the same
`argv[1]` convention (e.g. `output/sycl_output.txt`), and diff the two
output files (exact match expected).
