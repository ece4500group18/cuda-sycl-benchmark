---
name: cuda-to-sycl-migration
description: Translate standalone CUDA C++ benchmark cases into functionally equivalent SYCL 2020 programs for Intel GPUs. Use when reading main.cu, producing main.sycl.cpp, compiling with icpx -fsycl, running through Level Zero, and repairing compile, runtime, or numeric-correctness failures without access to hidden verifier data.
---

# Migrate CUDA to SYCL

## Workflow

1. Read `TASK.md`, `main.cu`, and the original build file completely.
2. Identify every CUDA dependency: runtime API, kernels, launches, streams,
   synchronization, atomics, texture/cooperative features, and libraries.
3. Choose a direct SYCL 2020 implementation. Preserve data generation, seeds,
   sizes, launch-dependent semantics, numeric types, and output order.
4. Write one self-contained `main.sycl.cpp` that accepts the output path as
   `argv[1]`.
5. Run `./sycl_build.sh`. Repair all compiler errors and warnings that indicate
   incorrect address spaces, captures, dimensions, or unsupported extensions.
6. Run `./sycl_run.sh`. Repair asynchronous exceptions, ordering errors,
   out-of-bounds access, and invalid work-group assumptions.
7. Re-read the CUDA source and compare control flow, indexing, synchronization,
   initialization, reductions, and output formatting before declaring success.

Do not invent verifier tolerances, read outside the current sandbox, embed
captured output, or replace the GPU computation with constants.

## Required implementation rules

- Create the queue with an asynchronous exception handler and wait with
  `wait_and_throw()` at dependency and result boundaries.
- Let `ONEAPI_DEVICE_SELECTOR` select the Intel Level Zero GPU. Do not select a
  CPU or silently fall back to a host implementation.
- Prefer SYCL 2020 USM for direct CUDA-runtime translations. Pair allocations
  with `sycl::free`, preserve copy directions and byte counts, and express
  ordering with events or an explicitly in-order queue.
- Map CUDA block/thread indexing deliberately. Check global/local range order
  for every dimension; do not assume CUDA's x/y/z order matches SYCL's printed
  dimension order.
- Use `sycl::local_accessor` for shared memory. Preserve every conditional
  barrier rule; all work-items in a group must encounter compatible barriers.
- Use `sycl::atomic_ref` with the narrowest correct memory order and scope.
- Guard padded global ranges exactly as the CUDA kernel guards excess threads.
- Preserve deterministic host-side initialization and exact whitespace-separated
  output order and precision.

Read [references/patterns.md](references/patterns.md) when the case uses shared
memory, reductions, atomics, streams, CUDA libraries, or advanced CUDA features.
Read [references/sources.md](references/sources.md) only when a specification or
portability decision needs confirmation.

## Completion gate

- `main.sycl.cpp` exists and contains the complete solution.
- `./sycl_build.sh` succeeds with `icpx -fsycl`.
- `./sycl_run.sh` succeeds on the selected Intel GPU and writes the requested file.
- No CUDA headers, launch syntax, runtime calls, or device-only CUDA qualifiers remain.
- The implementation does not depend on hidden files or network access.
