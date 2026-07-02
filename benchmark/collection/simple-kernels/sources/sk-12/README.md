# Case: streamOrderedAllocVectorAdd (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | elementwise `c[i] = a[i] + b[i]` via `vectorAddGPU`, classic `cudaMalloc`/`cudaFree` vs. stream-ordered `cudaMallocAsync`/`cudaFreeAsync` allocation |
| size | nelem = 1,048,576 (= 1 << 20) floats, launch = 4096 blocks x 256 threads |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected |

## Source

The `__global__` kernel in `main.cu` (`vectorAddGPU`) is reproduced
**verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/2_Concepts_and_Techniques/streamOrderedAllocation/streamOrderedAllocation.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/streamOrderedAllocation/streamOrderedAllocation.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

The original sample's two driver functions
(`basicStreamOrderedAllocation()` and `streamOrderedAllocationPostSync()`,
the latter looping `MAX_ITER=20` times and tuning a memory pool's release
threshold via `cudaMemPoolSetAttribute` purely to demonstrate steady-state
pool reuse under timing) were not reproduced -- they exist upstream to
illustrate *repeated* allocate/free cycling and its interaction with a
pool's release threshold, which is a performance-timing concern, not a
correctness one, and needs `helper_cuda`/`helper_functions` (command-line
parsing, `checkCudaErrors`, `findCudaDevice`) that this repo's cases avoid.
Everything else in this directory (host driver, `reference.h`, `Makefile`,
`CMakeLists.txt`, this `README.md`) is new code written for this
repository: a single deterministic pass that runs the *same* kernel once
through a plain `cudaMalloc`/`cudaFree` path and once through a
`cudaMallocAsync`/`cudaFreeAsync` stream-ordered path (mirroring the shape
of upstream's `basicStreamOrderedAllocation()`, minus its own
internal norm-ratio check, replaced by this repo's CPU-reference
comparison), instead of the original's `rand()`-seeded, un-checked-against-
a-CPU-oracle driver.

## What this case demonstrates (methods used)

1. **Stream-ordered memory-pool allocation vs. the classic CUDA
   allocator (simple but not trivial).** Both code paths in `main.cu`
   launch the identical `vectorAddGPU` kernel over the identical input
   data; the only difference is *how the three device buffers `a`, `b`,
   `c` are obtained and released*:
   - **classic**: `cudaMalloc()` before the launch (a synchronous call
     into the CUDA driver's allocator) and `cudaFree()` after, on the
     default stream.
   - **streamOrdered**: `cudaMallocAsync()`/`cudaFreeAsync()` issued
     *inside* a non-blocking stream, so the allocation and free
     themselves become stream-ordered operations backed by a CUDA
     memory pool (`cudaMemPool_t`) rather than synchronous driver calls
     -- the same pattern as upstream's `basicStreamOrderedAllocation()`:
     allocate `d_a`/`d_b`/`d_c`, copy inputs in, launch the kernel, free
     `d_a`/`d_b` (already consumed), copy the result out, free `d_c`,
     all enqueued on one stream ahead of a single
     `cudaStreamSynchronize()`.

   The kernel itself (`vectorAddGPU`) is deliberately trivial -- a
   single guarded addition per thread -- so that the only variable
   under test is genuinely the *host-side allocation-strategy control
   path*, not any kernel-internal memory-access pattern. This is a
   "simple but not trivial" case in the sense that the arithmetic is
   one line, but recognizing that a memory pool's allocations are
   still ordered with respect to the rest of a stream's work (so no
   extra synchronization is required around them, unlike the
   synchronous allocator) is the substantive, non-obvious part.

2. **Guaranteed identical result regardless of allocation strategy.**
   `c[idx] = a[idx] + b[idx]` is an independent, per-element operation
   with no cross-thread communication and no floating-point
   accumulation of any kind; which allocator produced the backing
   memory for `a`, `b`, `c`, or which stream (default vs. explicit
   non-blocking) the kernel launch was enqueued on, has zero effect on
   the value computed for any given `idx`. Both paths are therefore
   held to an **exact** match (`max_abs_error == 0`) against the CPU
   reference, not a tolerance.

3. **Non-blocking stream + async H2D/D2H copies** (`cudaStreamCreateWithFlags(...,
   cudaStreamNonBlocking)`, `cudaMemcpyAsync`, `cudaFreeAsync` issued
   before the corresponding data has even finished copying out, relying
   on stream ordering rather than manual synchronization to keep `d_c`
   alive until its `cudaMemcpyAsync` completes) -- the idiomatic shape
   of a stream-ordered allocation/free sequence.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `a[i] = (i % 23) - 11` (range `[-11, 11]`)
  - `b[i] = (i % 19) - 9` (range `[-9, 9]`)
  - `nelem = 1048576`, launch = `ceil(nelem / 256)` blocks x 256 threads
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the
  1,048,576-element `c = a + b` array computed by the stream-ordered
  (`cudaMallocAsync`/`cudaFreeAsync`) path, one `%.9g` float per line,
  plus a `PASS`/`FAIL` line on stdout comparing *both* paths' outputs
  against `reference_vector_add()` in `reference.h` (exact match
  expected, since both inputs are small integers and their sum is
  exactly representable in `float`).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

Requires a CUDA 11.2+ toolkit and a matching runtime/driver for
`cudaMallocAsync`/`cudaFreeAsync`/`cudaMemPool` support (this is a hard
build/run prerequisite for this case, not merely a performance
consideration).

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cudaMallocAsync`/`cudaFreeAsync` typically migrate to
`sycl::ext::oneapi::experimental::async_malloc`/`async_free`, or, absent
that extension on a given SYCL backend, to a plain USM
`sycl::malloc_device`/`sycl::free` inside the same queue-ordered
submission; `cudaMalloc`/`cudaFree` migrate to `sycl::malloc_device`/
`sycl::free` directly), build the result, run it with the same
`argv[1]` convention (e.g. `output/sycl_output.txt`), and diff the two
output files (exact match expected).
