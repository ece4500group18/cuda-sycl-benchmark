# Case: systemScopeAtomicAdd (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | 9 atomic read-modify-write ops, run with `*_system` (host-device-coherent) scope vs. ordinary device scope, on a shared 10-int array |
| size | numThreads = 256, numBlocks = 64, LOOP_NUM = 50; 32,768 total logical contributors (`len = 2 * numBlocks * numThreads`) |
| correctness | CPU reference (`reference.h`); 8/10 slots exact match (system path, device path, and system-vs-device cross-check all identical), 2/10 slots range-checked |

## Source

The `__global__` kernel `atomicKernel` and the host function
`atomicKernel_CPU` in `main.cu` are reproduced **verbatim** (aside from
formatting), and the correctness formulas in `reference.h` are ported
directly, from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/0_Introduction/systemWideAtomics/systemWideAtomics.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/systemWideAtomics/systemWideAtomics.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

`atomicKernel` uses nine `*_system` atomic intrinsics
(`atomicAdd_system`, `atomicExch_system`, `atomicMax_system`,
`atomicMin_system`, `atomicInc_system`, `atomicDec_system`,
`atomicCAS_system`, `atomicAnd_system`, `atomicOr_system`,
`atomicXor_system`) on `cudaMallocManaged` memory; `atomicKernel_CPU`
performs the analogous operations on the host via GCC/Clang `__sync_*`
builtins, covering the other half of the same logical index range on
the *same* managed array -- this host+device coherence is the entire
point of the original sample.

Omitted from the original file: the `main()` command-line/device-query
scaffolding (`findCudaDevice`, `helper_cuda` dependency,
`cudaDevAttrComputeMode`/`cudaDevAttrConcurrentManagedAccess`/
`pageableMemoryAccess` branching used to decide whether an explicit
sync is needed before the host touches the array). This directory
always synchronizes before the host mirror runs, so the result is
deterministic on every device this repo targets (sm_70/80/90), instead
of depending on a platform-specific concurrent-access attribute.
Everything else here (host driver, the new `atomicKernel_device`
kernel described below, `reference.h`, `Makefile`, `CMakeLists.txt`,
this `README.md`) is new code written for this repository.

**New, adapted kernel:** `atomicKernel_device` in `main.cu` is *not*
from the upstream sample -- it is a hand-written twin of `atomicKernel`
(same nine operations, same order, same `LOOP_NUM` repeat count) using
ordinary **device-scope** atomics (no `_system` suffix), launched as a
single GPU kernel over the *entire* logical index range `[0, len)`
that the system path splits between the GPU kernel and the host mirror
function. This twin exists specifically to give this case something to
compare `*_system` atomics *against* (the upstream sample itself has no
device-scope counterpart to check its own result relative to).

## What this case demonstrates (methods used)

1. **Atomic memory scope (`*_system` vs. device scope).** CUDA atomics
   come in three scopes: block (`_block`), device (the default, e.g.
   `atomicAdd`), and system (`_system`, e.g. `atomicAdd_system`).
   `_system` atomics are guaranteed coherent with concurrent accesses
   from the *host CPU* (or other GPUs) to the same page-migratable
   allocation; ordinary device-scope atomics only guarantee ordering
   and atomicity among threads running on the *same* GPU. This case
   isolates that scope/coherence axis directly: the "system" path
   (`atomicKernel` + `atomicKernel_CPU`) has the GPU and the CPU
   *both* mutate the same `cudaMallocManaged` array via `*_system`
   atomics, while the "device" path (`atomicKernel_device`) does the
   identical battery of operations, over the identical total index
   range, entirely on the GPU with ordinary atomics. Because all nine
   operations are either associative/commutative (add, max, min, inc,
   dec, and, or, xor) or -- for exch/cas -- only ever guaranteed to
   land on *some* contributor's index, scope cannot change *which*
   value is legal, only *whether* concurrently-racing host and device
   writes are guaranteed to be mutually visible. `main.cu` checks this
   explicitly: the eight order-independent slots must match the exact
   reference **and** the system array must equal the device array on
   those same eight slots, byte-for-byte.

2. **Distinct from `atomicIntrinsics` (this repo's other atomics
   case).** `atomicIntrinsics` already covers "which RMW ops have one
   guaranteed final value vs. which are inherently order-dependent"
   using eleven ordinary, single-GPU, device-scope atomics -- it never
   leaves device scope. This case does **not** re-cover that ground;
   it holds the RMW-op-classification question constant (reusing the
   same exact/range split, now over nine of those same eleven ops) and
   varies only the *scope*/coherence-domain argument, which
   `atomicIntrinsics` does not touch at all.

3. **Unified/managed memory (`cudaMallocManaged`).** The system path's
   array lives in a single allocation accessible, coherently, from both
   host and device code without an explicit `cudaMemcpy` -- the
   mechanism that makes `*_system` atomics meaningful in the first
   place. Requires compute capability 6.0+ (Pascal or later); this
   repo's `Makefile`/`CMakeLists.txt` only target sm_70/80/90, so no
   runtime capability check is performed.

## Input / Output

- **Input**: a 10-int array, initialized to `0` except slot 7 (AND)
  and slot 9 (XOR) set to `0xff` (matches upstream, makes those two
  tests non-trivial). `numThreads = 256`, `numBlocks = 64`,
  `LOOP_NUM = 50`; `len = 2 * numBlocks * numThreads = 32768` is the
  total number of logical contributors in both paths (the system path
  splits this as 16,384 GPU threads + a host loop covering the other
  16,384 indices; the device path launches `2*numBlocks = 128` blocks
  of 256 threads directly covering all 32,768 indices in one kernel).
- **Output**: `argv[1]` (default `output/cuda_output.txt`), 10 lines,
  the final value of each slot of the **system** array (GPU
  `atomicKernel` + host `atomicKernel_CPU`), plus a `PASS`/`FAIL` line
  on stdout. `PASS` iff: the system array's 8 exact-match slots (add,
  max, min, inc, dec, and, or, xor) equal `reference_check_exact()`'s
  closed-form/loop formulas, the device array's same 8 slots also
  match, the system and device arrays agree on all 8 of those slots
  with each other, and both arrays' 2 range-checked slots (exch, cas)
  hold a value in `[0, len)` via `reference_check_range()`.

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

Requires a CUDA toolkit with Unified Memory / `*_system` atomics
support (compute capability 6.0+) and a GCC/Clang-compatible host
compiler for `atomicKernel_CPU`'s `__sync_*` builtins (as in the
original sample).

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cudaMallocManaged` typically migrates to `sycl::malloc_shared`, and
`*_system` atomics typically migrate to `sycl::atomic_ref` with
`sycl::memory_scope::system`, vs. plain device-scope atomics migrating
to `sycl::memory_scope::device`; SYCL has no direct equivalent of
CUDA's modulo-wrapping `atomicInc`/`atomicDec`, so those typically need
a manual `compare_exchange` loop, same as `atomicIntrinsics`), build
the result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
