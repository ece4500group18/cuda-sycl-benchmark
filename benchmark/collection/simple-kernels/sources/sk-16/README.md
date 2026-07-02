# Case: voteAnyAll (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | per-warp `any(pred)` / `all(pred)` via `__any_sync`/`__all_sync` |
| size | 128 threads (4 warps), single block |
| correctness | CPU reference (`reference.h`), exact boolean-truth-value match expected |

## Source

The two `__global__` kernels in `main.cu` (`VoteAnyKernel1` and
`VoteAllKernel2`) are reproduced **verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/0_Introduction/simpleVoteIntrinsics/simpleVote_kernel.cuh`
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

A third kernel in the original file (`VoteAnyKernel3`, a directed test
that additionally checks cross-warp/half-warp boundary behavior using
a `bool*` output array and its own launch configuration) was omitted
to keep this case a direct two-kernel comparison, matching the style
of the other cases in this directory. The deterministic input pattern
(`gen_vote_pattern` in `reference.h`) is adapted from the original
sample's `genVoteTestPattern`. Everything else here (host driver,
`reference.h`, `Makefile`, `CMakeLists.txt`, this `README.md`) is new
code written for this repository, replacing the original driver's
`helper_cuda`/`helper_functions` dependency and per-group
`checkErrors1`/`checkErrors2` printf-based verification with a
deterministic, single-file CUDA-vs-SYCL comparison harness.

## What this case demonstrates (methods used)

1. **Warp vote intrinsics (`__any_sync`, `__all_sync`).** Both kernels
   evaluate a per-lane predicate (`input[tx] != 0`) and reduce it across
   the 32 lanes of a warp with a single instruction:
   - `VoteAnyKernel1`: `__any_sync(mask, input[tx])` -- true if *any*
     lane in the warp has a nonzero input.
   - `VoteAllKernel2`: `__all_sync(mask, input[tx])` -- true if *all*
     lanes in the warp have a nonzero input.

   This is a "simple but not trivial" kernel: the reduction itself
   ("did any/all of 32 booleans come out true") is one line of serial
   logic, but performing it as a *single hardware instruction* across a
   warp -- with no shared memory, no `__syncthreads()`, and no loop --
   is the non-trivial, widely-used technique it isolates (the same
   family as `warpShuffleReduction`'s `__shfl_down_sync`, but for a
   boolean reduction instead of a sum).

2. **Four-warp deterministic test pattern.** The 128-thread (4-warp)
   input is split into groups so every combination of "any" and "all"
   truth values is exercised: warp 0 is all-false, warps 1 and 2 are
   mixed (odd-only / even-only nonzero, so `Any=true, All=false` for
   both), and warp 3 is all-true.

3. **`threadIdx.x`-only indexing** (no `blockIdx`) over a single block
   -- both kernels are meant to be launched with exactly one block, so
   each warp's vote is self-contained within that launch.

## Input / Output

- **Input** (generated deterministically on the host, see
  `gen_vote_pattern` in `reference.h`):
  - warp 0 (lanes 0-31): all `0`
  - warp 1 (lanes 32-63): odd lane index nonzero, even lane `0`
  - warp 2 (lanes 64-95): even lane index nonzero, odd lane `0`
  - warp 3 (lanes 96-127): all `0xffffffff`
- **Output**: `argv[1]` (default `output/cuda_output.txt`), 128 lines,
  one `0`/`1` per line -- whether `VoteAnyKernel1`'s result for that
  lane is nonzero -- plus a `PASS`/`FAIL` line on stdout comparing
  both kernels' per-lane truth values (`result != 0`) against
  `reference_vote()` in `reference.h` (exact boolean match expected;
  `__any_sync`/`__all_sync` are guaranteed by the CUDA Programming
  Guide to return zero exactly when the predicate is false and a
  nonzero value when it is true, so the truth value, not the raw
  integer, is the invariant being checked).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct. Note that
`__any_sync`/`__all_sync` typically migrate to
`sycl::any_of_group`/`sycl::all_of_group` (or
`sycl::ext::oneapi::any_of_group`/`all_of_group` on older SYCL
versions) over a `sycl::sub_group` -- this is one of the more
interesting migration targets in this benchmark set. Build the
result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
