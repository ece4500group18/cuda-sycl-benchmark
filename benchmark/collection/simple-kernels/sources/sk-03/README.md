# Case: binaryPartitionOddEvenReduce (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | `cg::binary_partition()` warp split into odd/even sub-groups + `cg::reduce()` per sub-group vs. naive per-thread `atomicAdd` of odd/even counts and sums |
| size | arrSize = 65,536 ints, launch = 128 blocks x 256 threads (32,768 threads, grid-stride loop) |
| correctness | CPU reference (`reference.h`), `exact_ok` (both kernels), exact integer match expected |

## Source

The `__global__` kernel `oddEvenCountAndSumCG` in `main.cu` is
reproduced **verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/3_CUDA_Features/binaryPartitionCG/binaryPartitionCG.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/binaryPartitionCG/binaryPartitionCG.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

What was kept verbatim: the entire kernel body -- the
`cg::this_thread_block()` / `cg::this_grid()` / `cg::tiled_partition<32>`
setup, the grid-stride loop, the `cg::binary_partition(tile32, elem & 1)`
call splitting each warp tile into an odd sub-group and an even sub-group,
the per-sub-group `cg::reduce(subTile, elem, cg::plus<int>())`, the
rank-0-only `atomicAdd` calls, and the closing `cg::sync(tile32)` -- is
byte-identical to upstream.

What was omitted: the original file's `initOddEvenArr` (`rand() % 50`,
replaced by a deterministic generator in `reference.h`), its `main()`
(command-line device selection via `findCudaDevice`/`helper_cuda.h`,
`cudaOccupancyMaxPotentialBlockSize`-derived launch configuration, pinned
host buffers via `cudaMallocHost`, `printf`-only reporting with no exit
code), and its dependency on `helper_cuda.h` (not needed for a
deterministic, dependency-free comparison harness).

What is new in this directory: `oddEvenCountAndSumNaive` (a second
`__global__` kernel, not present upstream) computes the exact same three
quantities the upstream kernel does -- count of odd elements, sum of odd
elements, sum of even elements -- with a plain per-thread `atomicAdd` for
each element it touches, and no partitioning or local reduction at all.
It exists purely as the "naive" counterpart needed to isolate what
`binary_partition` + `reduce` actually buys: fewer, larger atomic
operations under warp divergence, without changing the result. The host
driver, `reference.h`, `Makefile`, `CMakeLists.txt`, and this `README.md`
are new code written for this repository, replacing the original's
device-query/timing-free driver with a deterministic, single-file
CUDA-vs-SYCL comparison harness that launches both kernels back-to-back
and diffs their outputs against a shared CPU reference.

## What this case demonstrates (methods used)

1. **`cg::binary_partition()`: splitting a divergent warp into two
   sub-groups (simple but not trivial).** Every thread in a 32-thread
   `tile32` checks whether its own element is odd or even, then calls
   `cg::binary_partition(tile32, elem & 1)`. Threads that pass the same
   predicate value end up in the same new, smaller cooperative-groups
   thread block tile (`subTile`) -- one sub-group for the odd threads of
   that warp, one for the even threads -- whatever their relative sizes
   happen to be for that particular 32 elements. This is the "simple but
   not trivial" idiom this case isolates: the API call itself is a
   one-liner, but reasoning correctly about *what group a given thread
   ends up in*, and that every thread in a sub-group is guaranteed to
   still be convergent for a subsequent `cg::reduce()`/`.sync()` on that
   sub-group, is the substantive part.

2. **`cg::reduce()` per sub-group, one atomic per sub-group instead of
   one per thread.** Each sub-group (odd or even) performs a single
   `cg::reduce(subTile, elem, cg::plus<int>())` -- an intra-group
   tree/shuffle reduction that combines every member's `elem` into one
   value, entirely in registers/shuffles, no shared memory or global
   traffic. Only the sub-group's rank-0 thread then issues global
   atomics: one `atomicAdd(numOfOdds, subTile.size())` (odd sub-group
   only) and one `atomicAdd(&sumOfOddAndEvens[...], groupSum)`. A fully
   diverged warp (16 odd, 16 even) therefore performs at most **2**
   atomic RMW operations total, instead of the 32 that a naive
   per-thread scheme (`oddEvenCountAndSumNaive`) performs on the same 32
   elements -- the whole point of partitioning a divergent warp before
   reducing, rather than letting every thread hit the same global
   counters independently.

3. **Byte-identical result guaranteed by commutative/associative integer
   addition.** Both kernels visit every index in `[0, size)` exactly
   once (grid-stride loop) and combine each element into exactly one of
   three plain `int` accumulators (odd count, odd sum, even sum) via
   addition. Integer addition is commutative and associative regardless
   of how many terms are pre-combined locally (via `cg::reduce`) before
   the atomic add, or in what order across warps/blocks/loop iterations
   the atomics land -- so `oddEvenCountAndSumCG` and
   `oddEvenCountAndSumNaive` are guaranteed to produce **exactly** the
   same three final totals as each other and as a straightforward CPU
   odd/even count-and-sum loop (`max_abs_error == 0` / exact integer
   equality, no floating point involved anywhere in this case).

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`): `arrSize = 65536` ints,
  `gen(i) = (i * 7 + 3) % 50` (replacing the upstream sample's own
  `rand() % 50`); launch = 128 blocks x 256 threads (256 is a multiple
  of the warp size 32, so `tile32 = cg::tiled_partition<32>(cta)`
  divides each block evenly into 8 full warp tiles).
- **Output**: `argv[1]` (default `output/cuda_output.txt`), 3 lines:
  number of odd elements, sum of odd elements, sum of even elements, as
  computed by `oddEvenCountAndSumCG`, plus a `PASS`/`FAIL` line on
  stdout comparing **both** kernels' three outputs against
  `reference_odd_even_count_and_sum()` in `reference.h` (exact integer
  match expected for both).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cooperative_groups::tiled_partition<32>`/`binary_partition`/`reduce`
have no direct one-to-one SYCL equivalent -- `sycl::sub_group` supports
shuffle-based reductions over a whole sub-group but not an on-the-fly
predicate-based binary split the way `cg::binary_partition` does, so
this is one of the more interesting manual-porting targets in this
benchmark set, e.g. via `sycl::group_ballot`/mask-based partitioning
plus manual shuffle reduction, or `sycl::ext::oneapi::experimental`
extensions where available), build the result, run it with the same
`argv[1]` convention (e.g. `output/sycl_output.txt`), and diff the two
output files (exact match expected).
