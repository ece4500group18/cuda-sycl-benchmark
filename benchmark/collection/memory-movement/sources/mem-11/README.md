# Case: pinnedAsyncIncrement (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | per-element increment `g_data[i] += inc_value`, pinned host memory + `cudaMemcpyAsync` on a stream vs. pageable host memory + synchronous `cudaMemcpy` |
| size | n = 1,048,576 (1 << 20) ints, launch = 2048 blocks x 512 threads |
| correctness | CPU reference (`reference.h`), exact match expected (mismatch count == 0) |

## Source

The `__global__` kernel in `main.cu` (`increment_kernel`) is reproduced
**verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/0_Introduction/asyncAPI/asyncAPI.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/asyncAPI/asyncAPI.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

The upstream sample's real purpose is demonstrating GPU-side timing via
`cudaEvent_t` and CPU/GPU overlap via a CPU busy-wait loop
(`while (cudaEventQuery(stop) == cudaErrorNotReady) counter++;`), all
wrapped around a single `cudaMallocHost` pinned buffer and one
H2D-async -> kernel -> D2H-async pipeline issued to the default stream.
None of that timing/overlap machinery affects correctness, so it (plus
the `helper_cuda`/`helper_functions` dependency, `cudaProfilerStart/Stop`,
and `StopWatchInterface`) was omitted. Everything else in this directory
(host driver, `reference.h`, `Makefile`, `CMakeLists.txt`, this
`README.md`) is new code written for this repository: a deterministic
harness that runs the *same* `increment_kernel` twice, once through a
pinned-memory + `cudaMemcpyAsync`-on-an-explicit-stream path and once
through a plain pageable-`malloc` + synchronous-`cudaMemcpy` path, so the
two memory-movement mechanisms can be diffed against a single CPU
reference instead of a wall-clock timer.

## What this case demonstrates (methods used)

1. **Pinned (page-locked) host memory vs. pageable host memory (memory
   movement).** `cudaMallocHost` allocates host memory that the OS
   cannot page out or move, so the GPU's DMA engine can copy directly to
   and from it. Plain `malloc` returns ordinary pageable memory, which
   the CUDA driver must first stage through an internal pinned bounce
   buffer before any DMA transfer can occur -- an extra host-side copy
   that pinned memory skips entirely.

2. **Asynchronous (`cudaMemcpyAsync` on a stream) vs. synchronous
   (`cudaMemcpy`) transfers.** `cudaMemcpyAsync` on a non-default,
   explicitly created `cudaStream_t` returns to the CPU immediately once
   the copy is enqueued; the H2D copy, the `increment_kernel` launch, and
   the D2H copy are all enqueued back-to-back on that one stream (so CUDA
   still serializes them relative to each other, exactly as the upstream
   sample's "all to stream 0" comment describes) and only
   `cudaStreamSynchronize` blocks the CPU. `cudaMemcpy` with no stream
   argument is always synchronous and blocks the calling CPU thread until
   the transfer completes.

3. **Same kernel, same data, two host data-paths.** Both paths launch the
   identical, unmodified `increment_kernel` over identically-initialized
   input (`g_data[i] = i`) with the same launch configuration and the
   same `inc_value`. Since `g_data[idx] = g_data[idx] + inc_value` is an
   independent update of a single index touched by exactly one thread,
   the result cannot depend on which memory-allocation kind or
   copy/launch mechanism was used to get the data to and from the
   device -- an exact byte-for-byte match against the CPU reference
   (`i + inc_value` for every `i`) is the correctness invariant, and any
   mismatch would indicate an actual data-race/synchronization bug in
   the async pipeline (e.g. reading back `a_pinned` before the D2H copy
   or the kernel has actually completed), not a numerical-tolerance
   issue.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`): `g_data[i] = i`, for `i` in `[0, n)`, `n = 1048576`;
  `inc_value = 17` (fixed); launch = 2048 blocks x 512 threads (matching
  the upstream sample's 512-thread block width).
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the final
  `g_data` array from the pinned+async path, one int per line
  (1,048,576 lines), plus a `PASS`/`FAIL` line on stdout comparing both
  paths' outputs against `reference_increment()` in `reference.h` (exact
  match expected; `PASS` iff both paths report zero mismatches).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cudaMallocHost` typically migrates to `sycl::malloc_host`,
`cudaStreamCreate`/`cudaMemcpyAsync`/`cudaStreamSynchronize` to a
`sycl::queue` with in-order execution or explicit event dependencies,
and plain `cudaMemcpy` to a blocking `queue.memcpy(...).wait()`), build
the result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
