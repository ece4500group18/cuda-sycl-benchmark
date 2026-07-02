# Case: threadFenceSinglePassReduction (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | single-precision sum reduction of N floats: single-kernel-launch `__threadfence()` + atomic-ticket "last block finishes" vs. classic two-kernel-launch tree reduction |
| size | N = 131,072 floats (= 2 x 256 x 256, a power of 2); level 1 / Phase 1 = 256 blocks x 256 threads; two-launch level 2 = 1 block x 128 threads |
| correctness | CPU reference (`reference.h`), one exact-tree-replica reference per variant, `max_abs_error == 0` expected for both |

## Source

The device functions/kernels in `main.cu` (`reduceBlock`, `reduceBlocks`,
`reduceMultiPass`, `reduceSinglePass`, `retirementCount`,
`setRetirementCount`) are reproduced **verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/2_Concepts_and_Techniques/threadFenceReduction/threadFenceReduction_kernel.cuh`
- Repository: https://github.com/NVIDIA/cuda-samples
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

The original sample's host driver (`threadFenceReduction.cu`) was **not**
reused: it depends on `helper_cuda.h`/`helper_functions.h` (command-line
parsing, `StopWatchInterface` timing), reduces a `rand()`-filled array, and
picks thread/block counts at runtime via a `getNumBlocksAndThreads()`
heuristic aimed at performance shmoo sweeps, none of which is needed for a
deterministic single-file CUDA-vs-SYCL comparison. This directory's
`main()` is new code that fixes N and the launch geometry so the two
techniques being compared operate over an identical level-1 partitioning
of the input (see below), and `reference.h` is a new, from-scratch CPU
re-implementation of the exact reduction trees the two kernels perform
(the original sample only ever checks its GPU result with `abs(gpu -
cpu) < 1e-8 * size` against a Kahan-summed CPU reference -- a tolerance
check appropriate for a `rand()`-filled array of arbitrary size, but
looser than the exact match this repository prefers when an exact
per-kernel tree replica is feasible). The giant `switch (threads) { case
512: ... case 1: ... }` launch-dispatch wrapper functions
(`reduce()`/`reduceSinglePass()` in the original `.cuh`, needed there to
support any power-of-2 thread count from a command line flag) were
omitted; this directory instead directly instantiates the two kernel
templates it needs (`<256, true>` and `<128, true>`).

## What this case demonstrates (methods used)

1. **`__threadfence()` + global atomic ticket for cross-block
   synchronization within a single kernel launch.** `reduceSinglePass`
   has every block reduce its own slice of the input to one partial sum
   (Phase 1, via `reduceBlocks`/`reduceBlock` -- ordinary shared-memory
   tree reduction), then calls `__threadfence()` (blocks the calling
   thread until *all* of its prior global-memory writes, i.e. this
   block's partial-sum write, are visible to every other block in the
   grid -- a stronger guarantee than `__syncthreads()`, which only
   orders memory within one block) before thread 0 calls
   `atomicInc(&retirementCount, gridDim.x)`. Because
   `atomicInc`/`atomicAdd` on a single global counter is strictly
   serialized by hardware, exactly one block's ticket equals
   `gridDim.x - 1`; that block alone is provably the last to finish
   Phase 1 (every other block must have already taken a strictly
   smaller ticket, and thanks to the fence, its partial sum is already
   visible in global memory). That single "last" block then sums all
   `gridDim.x` partial results and writes the final answer -- **all
   without ending the kernel and launching a second one.**

2. **Classic two-kernel-launch tree reduction as the control-flow
   counterpart.** `reduceMultiPass` (the exact same
   `reduceBlocks`/`reduceBlock` machinery as Phase 1 above, just without
   any Phase 2) is launched twice: once with 256 blocks over the
   original 131,072-element input (producing 256 partial sums), then
   again with a single 128-thread block over those 256 partial sums
   (producing the final answer). Here, "did the previous level finish"
   is guaranteed by the CUDA runtime's implicit ordering of kernel
   launches on the default stream, not by a fence/atomic pair -- the
   two techniques solve the identical inter-block-visibility problem
   (how does a later stage know the earlier stage's writes are done and
   visible?) via two structurally different mechanisms, one
   intra-kernel and one inter-kernel.

3. **Why an exact match is possible despite reduction being
   order-sensitive.** Float addition is not associative, so a plain
   linear-order CPU sum is not guaranteed to equal *either* GPU
   kernel's result bit-for-bit, and the two GPU kernels are not
   required to (and in general do not) produce identical results to
   *each other*, since `reduceSinglePass`'s Phase 2 combines 256
   partial sums with a 256-wide tree while the two-launch path's
   second kernel combines the same 256 values with a 128-wide tree
   (two elements folded together per thread before entering the
   tree). `reference.h` instead provides **two** single-threaded CPU
   functions, `reference_single_pass()` and `reference_two_launch()`,
   each a straight-line re-implementation -- in `float`, matching the
   GPU's precision, not `double` -- of that specific kernel's exact
   sequence of additions (including the warp-level stride-halving tree
   inside `reduceBlock`, simulated round-by-round with an explicit
   snapshot of the shared array so that every lane's read in a round
   sees the pre-round state, exactly as real SIMT hardware guarantees).
   Comparing each GPU output only to its own matching reference
   therefore gives `max_abs_error == 0` for both variants, without
   relying on any tolerance.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`): `input[i] = ((i % 29) - 14) * 0.25`, `i` in `[0,
  131072)`.
- Launch geometry: level 1 / Phase 1 = 256 blocks x 256 threads for
  both variants (`reduceSinglePass<256,true>` and
  `reduceMultiPass<256,true>`); the two-launch variant's second kernel
  launch is `reduceMultiPass<128,true>` with 1 block x 128 threads over
  the 256 level-1 partial sums (in place).
- **Output**: `argv[1]` (default `output/cuda_output.txt`), two lines:
  the single-pass kernel's final sum, then the two-kernel-launch
  reduction's final sum (each `%.9g`, sufficient to round-trip a
  float32 exactly), plus a `PASS`/`FAIL` line on stdout comparing both
  against `reference_single_pass()`/`reference_two_launch()` in
  `reference.h` (exact match expected).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`__threadfence()` migrates to `sycl::atomic_fence(sycl::memory_order::
seq_cst, sycl::memory_scope::device)`, and the global `atomicInc`
ticket counter typically migrates to a `sycl::atomic_ref` on a
device-visible `unsigned int` with `fetch_add`/wraparound handled
manually since SYCL has no direct `atomicInc`-with-wrap equivalent;
`cg::thread_block`/`cg::thread_block_tile<32>` map to
`sycl::nd_item::barrier()` and `sycl::sub_group` operations), build the
result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
