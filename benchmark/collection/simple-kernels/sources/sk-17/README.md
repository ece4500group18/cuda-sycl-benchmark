# Case: warpAggregatedAtomicCompaction (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | stream compaction (`dst[] = { src[i] : src[i] > 0 }`), warp-aggregated `atomicAdd` (`cg::coalesced_threads()` + `shfl` broadcast) vs. naive per-thread `atomicAdd` |
| size | n = 1,048,576 ints (`1 << 20`), launch = 2048 blocks x 512 threads |
| correctness | CPU reference (`reference.h`); exact **set-equality** (sorted-array) match, not positional -- see below |

## Source

The `atomicAggInc` `__device__` helper and the `filter_arr` `__global__`
kernel in `main.cu` are reproduced **verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/3_CUDA_Features/warpAggregatedAtomicsCG/warpAggregatedAtomicsCG.cu`
- URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/warpAggregatedAtomicsCG/warpAggregatedAtomicsCG.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

The original file's other kernels (`atomicAggIncMulti`/`mapToBuckets`,
`atomicAggMaxMulti`/`calculateMaxInEachBuckets`, and their
`mapIndicesToBuckets`/`calculateMaxInBuckets` host-side drivers), which
require compute capability 7.0+ (`__CUDA_ARCH__ >= 700`) for the
`cg::labeled_partition`/match-based multi-bucket variant, were omitted --
this case keeps only the single-bucket `atomicAggInc`/`filter_arr` pair,
which is the minimal, architecture-portable illustration of the
warp-aggregated-atomic technique. The original `main()` (`rand()`-based
input, `helper_cuda`/`helper_functions` dependency, `findCudaDevice`,
`checkCudaErrors`) is not needed for a deterministic CUDA-vs-SYCL output
comparison.

`filter_arr_naive` is **new code** written for this repository: the
identical stream-compaction loop, but with a plain per-thread
`atomicAdd(nres, 1)` instead of the warp-aggregated helper, added
specifically to give the warp-aggregated technique something to be
compared against (the upstream file only ships the aggregated version).
Everything else in this directory (host driver, `reference.h`,
`Makefile`, `CMakeLists.txt`, this `README.md`) is new code written for
this repository.

## What this case demonstrates (methods used)

1. **Warp-aggregated atomics via Cooperative Groups.** `filter_arr`'s
   `atomicAggInc` helper calls `cg::coalesced_threads()` to find the
   subset of the warp's lanes that are still active and executing this
   instruction together; lane 0 of that group (`active.thread_rank() ==
   0`) performs a **single** `atomicAdd(counter, active.size())` for the
   whole group, and `active.shfl(res, 0)` broadcasts the resulting base
   offset to every other active lane, which adds its own
   `thread_rank()` to land on a distinct output slot. This issues **at
   most one** atomic read-modify-write per active warp per loop
   iteration, instead of one per thread -- up to a 32x reduction in
   atomic traffic on the shared counter when most/all of a warp's lanes
   keep an element.

2. **Naive per-thread atomics as the baseline.** `filter_arr_naive` does
   the same predicate-and-append (`if (src[i] > 0) dst[atomicAdd(nres,
   1)] = src[i];`) with zero warp-level coordination: every lane that
   keeps an element issues its own `atomicAdd`, contending directly with
   every other lane (in the same warp and across the whole grid) that
   also kept an element in the same instant. This is "simple but not
   trivial": the compaction predicate itself is a one-line comparison,
   but the difference between the two kernels is entirely in *how many
   atomic instructions actually execute* to produce the *same* result --
   a technique, not a different algorithm.

3. **A documented set-equality (not positional) correctness oracle.**
   Both kernels visit every index in `[0, n)` exactly once and are
   guaranteed to keep exactly the same multiset of values (every
   `src[i] > 0`), but *which slot in `dst[]` a given kept value lands
   at* depends on unspecified block/warp scheduling order and differs
   between runs and between the two kernels -- there is no guaranteed
   positional correspondence. The oracle in `main.cu` therefore sorts
   both GPU outputs (`std::sort`) and a CPU-filtered-and-sorted
   reference (`reference_filter_sorted` in `reference.h`) and compares
   them elementwise (exact integer equality, since these are compaction
   counts/values with no floating-point accumulation involved) -- a
   multiset-equality check, explicitly not an exact positional match.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`): `src[i] = (i % 13) - 6`, for `i` in `[0, n)`,
  `n = 1048576` -- 13-periodic, values in `[-6, 6]`; exactly 6 of every
  13 values (`1..6`) are `> 0`.
- **Output**: `argv[1]` (default `output/cuda_output.txt`): first line =
  the compacted count produced by `filter_arr` (`*nres`), followed by
  that many lines, one int per line, `filter_arr`'s compacted `dst[]`
  array **sorted ascending**. Plus a `PASS`/`FAIL` line on stdout: `PASS`
  iff both kernels' compacted counts match the CPU reference count *and*
  both kernels' sorted output arrays are elementwise equal to the CPU
  reference's sorted filtered array.

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cg::coalesced_threads()`/`active.shfl()`/`active.thread_rank()`
typically migrate to `sycl::ext::oneapi::experimental::group_ballot` /
`sycl::group_broadcast` / non-uniform sub-group query APIs over a
`sycl::sub_group` -- one of the more interesting migration targets in
this benchmark set -- while the naive `atomicAdd` migrates to a
`sycl::atomic_ref::fetch_add`), build the result, run it with the same
`argv[1]` convention (e.g. `output/sycl_output.txt`), and compare the two
output files using the same sort-then-compare (set-equality) procedure
described above, not a raw line-by-line diff.
