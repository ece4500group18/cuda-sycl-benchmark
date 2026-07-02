# Case: atomicIntrinsics (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | 11 global-memory atomic read-modify-write intrinsics, raced on by every thread |
| size | numThreads = 256, numBlocks = 64 (16,384 threads total) |
| correctness | CPU reference (`reference.h`); 9/11 slots exact match, 2/11 slots range-checked (see below) |

## Source

The `__global__` kernel in `main.cu` (`testKernel`) is reproduced
**verbatim**, and the correctness formulas in `reference.h` are ported
directly, from:

- Project: **NVIDIA/cuda-samples**
- Files: `cpp/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics_kernel.cuh`,
  `cpp/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics_cpu.cpp`
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

One formula was adapted for host-side portability: the original
`computeGold`'s OR check does `val |= (1 << i)` for `i` up to
`len - 1 = 16383`, which relies on GPU/x86 hardware masking the shift
amount to 5 bits (`i mod 32`) -- technically undefined behavior for a
32-bit shift count `>= 32` in portable C++, even though it is exactly
what both the CUDA `atomicOr` instruction and an x86 `shl` instruction
actually do. `reference.h` makes this masking explicit
(`1 << (i & 31)`) so the reference is well-defined C++ while computing
the identical value. The original driver (`helper_cuda`/
`helper_functions` dependency, `StopWatchInterface` timing,
`cudaMallocHost` pinned staging buffer, `cudaStreamNonBlocking`) is not
needed for a deterministic CUDA-vs-SYCL output comparison; this
directory's `main()` reimplements the same 11-slot setup and launch
configuration with plain synchronous `cudaMemcpy`. Everything here
other than the kernel and reference formulas listed above (host
driver, `Makefile`, `CMakeLists.txt`, this `README.md`) is new code
written for this repository.

## What this case demonstrates (methods used)

1. **Global-memory atomic read-modify-write intrinsics.** All 16,384
   threads call the same 11 atomics (`atomicAdd`, `atomicSub`,
   `atomicExch`, `atomicMax`, `atomicMin`, `atomicInc`, `atomicDec`,
   `atomicCAS`, `atomicAnd`, `atomicOr`, `atomicXor`) on the same 11
   output slots, with zero coordination between threads beyond the
   atomics themselves. This is "simple but not trivial": each atomic
   call is a one-line primitive, but working out the guaranteed final
   value of a slot that 16,384 threads raced on -- and recognizing
   *which* operations even have one -- is the substantive content.

2. **Order-independent vs. order-dependent atomics.** Nine of the
   eleven slots are associative or otherwise combine identically
   regardless of which thread executes when (`add`/`sub`: net effect
   is `+-10` per thread; `max`/`min`: converge to the global
   max/min thread id; `inc`/`dec`: deterministic modulo counters;
   `and`/`or`/`xor`: bitwise ops over all thread ids, all
   commutative and associative) -- these are checked for an **exact**
   match against `reference_check_exact()`. The remaining two
   (`atomicExch`, `atomicCAS`) have **no** single correct final value:
   whichever thread happens to execute last "wins," and that depends
   on the scheduler. Both are only checked for being **some** valid
   thread id in `[0, len)` via `reference_check_range()`, exactly as
   the original NVIDIA sample's own `computeGold` does -- an
   instructive, real-world example of a correctness oracle that must
   itself account for which parts of a parallel computation are
   genuinely nondeterministic.

## Input / Output

- **Input**: 11 output ints initialized to `0`, except slot 8 (AND)
  and slot 10 (XOR) initialized to `0xff` (so those two tests exercise
  a non-trivial starting value); launch = 64 blocks x 256 threads.
- **Output**: `argv[1]` (default `output/cuda_output.txt`), 11 lines,
  the final value of each output slot, plus a `PASS`/`FAIL` line on
  stdout: `PASS` iff all 9 exact-match slots equal
  `reference_check_exact()`'s formulas *and* both range-checked slots
  (`exch`, `cas`) hold a value in `[0, 16384)`.

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`atomicAdd`/`atomicSub`/etc. on raw pointers typically migrate to
`sycl::atomic_ref` member functions -- `fetch_add`, `fetch_sub`,
`exchange`, `fetch_max`, `fetch_min`, `compare_exchange_strong`,
`fetch_and`, `fetch_or`, `fetch_xor`; SYCL has no direct equivalent of
CUDA's modulo-wrapping `atomicInc`/`atomicDec`, so those typically need
a manual `compare_exchange` loop), build the result, run it with the
same `argv[1]` convention (e.g. `output/sycl_output.txt`), and apply
the same exact/range-check split when comparing the two output files.
