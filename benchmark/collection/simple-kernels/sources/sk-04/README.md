# Case: bitonicVsOddEvenMergeSort (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | in-shared-memory bitonic sorting network (`bitonicSortShared`) vs. Batcher's odd-even merge sorting network (`oddEvenMergeSortShared`), over the same 1024 (key, value) pairs |
| size | arrayLength = 1024 (== `SHARED_SIZE_LIMIT`); both launches = 1 block x 512 threads (single thread block only) |
| correctness | CPU reference (`reference.h`, `std::sort` by key), exact position-by-position array match expected for both GPU kernels and between them |

## Source

The device function `Comparator` and both `__global__` kernels in
`main.cu` (`bitonicSortShared`, `oddEvenMergeSortShared`) are
reproduced **verbatim** from:

- Project: **NVIDIA/cuda-samples**
- Directory: `cpp/2_Concepts_and_Techniques/sortingNetworks/`
- URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/sortingNetworks/
- Files: `bitonicSort.cu` (`bitonicSortShared`), `oddEvenMergeSort.cu`
  (`oddEvenMergeSortShared`), `sortingNetworks_common.cuh`
  (`Comparator`)
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

The `typedef unsigned int uint;` and `#define SHARED_SIZE_LIMIT 1024U`
definitions used to make the kernels compile standalone are adapted
(values only) from the same project's
`sortingNetworks_common.h`/`sortingNetworks_common.cuh`.

Everything else in the original sample was omitted:
`bitonicSortShared1`/`bitonicMergeGlobal`/`bitonicMergeShared` and
`oddEvenMergeGlobal` (the multi-block, "large array" bitonic/odd-even
merge stages that stitch together many `SHARED_SIZE_LIMIT`-sized
blocks into one long sorted batch), the `bitonicSort()`/
`oddEvenMergeSort()` host-side interface functions that drive those
multi-stage launches, and `main.cpp`/`sortingNetworks_validate.cpp`
(the original driver, which sorts a batch of many arrays with
`rand()`-generated keys, times it with `helper_timer`, and validates
via histogram-comparison against the input rather than a fixed
reference order). Those all exist to extend a single block's network
to arbitrarily large arrays and batches; they are a separate technique
(multi-block merge orchestration) layered on top of the two
shared-memory networks themselves, so they fall outside this case's
isolated theme of comparing the two *network topologies* directly.
`bitonicSortShared` and `oddEvenMergeSortShared` alone are each a
complete, self-contained sorting network for one block's worth of data
(here, exactly `arrayLength = SHARED_SIZE_LIMIT = 1024`, so a single
kernel launch with one thread block sorts the whole input for each
kernel).

`reference.h`, the host driver in `main.cu`, `Makefile`,
`CMakeLists.txt`, and this `README.md` are new code written for this
repository, replacing the original sample's `rand()`-seeded,
multi-batch, timed driver with a deterministic, single-file
CUDA-vs-SYCL comparison harness that isolates just the two
shared-memory network kernels against a shared CPU oracle.

## What this case demonstrates (methods used)

1. **Two different sorting-network topologies computing the same
   total order.** Both kernels load `SHARED_SIZE_LIMIT` (key, value)
   pairs into shared memory and repeatedly call the shared
   `Comparator` on pairs of slots selected by pure thread-index bit
   arithmetic, separated by `cg::sync(cta)` barriers -- but they wire
   up *which* slots get compared, and in what sequence, completely
   differently:
   - `bitonicSortShared` builds bitonic sequences of doubling size via
     a recursive bitonic-merge butterfly (`pos = 2*tid - (tid &
     (stride - 1))`, direction flipped by `threadIdx.x & (size / 2)`),
     finishing with one full bitonic merge pass in the fixed `dir`.
   - `oddEvenMergeSortShared` builds Batcher's odd-even merge networks
     of doubling size, where each stage does one unconditional
     comparator pass at `stride = size/2` followed by a sequence of
     *conditionally executed* comparator passes (`if (offset >=
     stride)`) at halving strides -- a structurally different circuit
     with different data dependencies and fewer total comparator
     invocations for large arrays than the bitonic network.

   Both are complete O(log^2 N)-stage comparator networks that
   provably sort *any* input of `SHARED_SIZE_LIMIT` elements. Because
   all 1024 keys used here are pairwise distinct (see below), the
   sorted permutation of any correct sort is unique, so the two
   structurally different networks are guaranteed to converge on the
   exact same, bit-identical output array -- a "same-result-via-
   different-algorithm" comparison, not a memory-movement one. This is
   "simple but not trivial": each comparator call is a one-line
   conditional swap, but recognizing why two differently-shaped
   compare-and-swap circuits are both guaranteed to sort correctly,
   and hence must agree exactly, is the substantive part.

2. **Thread-index bit arithmetic replacing data-dependent control
   flow.** In both kernels, every comparator partner index and
   direction bit is computed from `threadIdx.x` and the current
   `size`/`stride` stage constants alone (`&`, `^`, shifts) -- no
   branching on the keys themselves (aside from `oddEvenMergeShared`'s
   `if (offset >= stride)` gate, which is again purely index-derived).

3. **Intra-block barrier synchronization** via
   `cooperative_groups::sync(cta)` between every comparator stage, so
   that all 512 threads observe every prior stage's shared-memory
   writes before the next stage reads them.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `key[i] = (i * 40503u) % 65536u` for `i` in `[0, 1024)` (`uint32_t`
    arithmetic). Since 40503 is odd, `gcd(40503, 65536) == 1`
    (`65536 == 2^16`), so `i -> (i * 40503u) % 65536u` is a bijection
    on `Z/65536Z` -- all 1024 generated keys are pairwise distinct.
  - `val[i] = i` (the original index).
  - `arrayLength = 1024` (`= SHARED_SIZE_LIMIT`), `dir = 1` (ascending,
    fixed); launch = 1 block x 512 threads for each kernel (single
    thread block only, no multi-batch host-side merge).
- **Output**: `argv[1]` (default `output/cuda_output.txt`), 1024 lines
  of `"key val"` pairs (space-separated), `bitonicSortShared`'s sorted
  result, plus a `PASS`/`FAIL` line on stdout: `PASS` iff
  `bitonicSortShared`'s output, `oddEvenMergeSortShared`'s output, and
  a CPU `std::sort` by key (`reference_sort()` in `reference.h`) are
  all three exactly, bit-identically equal (`max_abs_error == 0`,
  trivially, since all values are integers with no floating-point
  accumulation involved).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (both
kernels' `__shared__` arrays become `sycl::local_accessor`s or
`group_local_memory`, and `cg::sync(cta)` becomes
`item.barrier(sycl::access::fence_space::local_space)`), build the
result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
