# Case: occupancyTunedLaunch (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | elementwise square (`array[i] *= array[i]`), launched with a `cudaOccupancyMaxPotentialBlockSize`-suggested configuration vs. a naive fixed block size |
| size | arrayCount = 1,048,576 (`1 << 20`) `uint32_t`s; manual launch = 32 threads/block, automatic launch = driver-suggested block size |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected for both launch configurations |

## Source

The `__global__` kernel in `main.cu` (`square`) is reproduced
**verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/0_Introduction/simpleOccupancy/simpleOccupancy.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleOccupancy/simpleOccupancy.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

The original sample's `reportPotentialOccupancy()` (an informational
`cudaOccupancyMaxActiveBlocksPerMultiprocessor()`-based wrapper that
prints an occupancy percentage but does not affect correctness),
`cudaEvent_t`-based timing, `helper_cuda.h` dependency, and its
`main()`/`test()` driver (command-line-free but `std::cout`-heavy,
with no machine-checkable pass/fail beyond a `std::cout` message) were
omitted. This directory's `main()` reimplements the same "manual vs.
automatic launch configuration" comparison as a deterministic,
single-file CUDA-vs-SYCL harness: it launches `square` once with a
fixed, hand-picked block size and once with the block size (and
minimum grid size) suggested by `cudaOccupancyMaxPotentialBlockSize`,
then verifies both results against a CPU reference. Everything other
than the `square` kernel listed above (host driver, `reference.h`,
`Makefile`, `CMakeLists.txt`, this `README.md`) is new code written for
this repository.

## What this case demonstrates (methods used)

1. **Occupancy-driven automatic launch configuration
   (`cudaOccupancyMaxPotentialBlockSize`).** Instead of hand-picking a
   block size, the CUDA runtime is queried at run time for a block
   size (and the minimum grid size needed to saturate the device at
   that block size) that maximizes theoretical occupancy for the
   *actual* kernel (`square`) on the *actual* GPU present -- a form of
   runtime introspection (querying register usage, shared-memory
   usage, and per-SM resource limits behind the scenes) that a plain
   kernel launch does not otherwise expose. This is "simple but not
   trivial": the kernel itself is a one-line multiply, but reasoning
   about *why* a fixed, too-small block size (here 32 threads/block --
   the original sample's own deliberately suboptimal
   `manualBlockSize`) under-occupies the GPU relative to a
   driver-suggested configuration is the substantive, CUDA-specific
   content.

2. **Launch-configuration independence of the actual computation.**
   `square` addresses itself purely via
   `idx = threadIdx.x + blockIdx.x * blockDim.x` plus a bounds check
   (`if (idx < arrayCount)`), so it is correct for *any* grid/block
   shape that collectively covers `[0, arrayCount)` -- the manual and
   automatic launches use entirely different block sizes, grid sizes,
   and total thread counts, yet touch the exact same indices with the
   exact same per-element arithmetic. Because each `array[i] *=
   array[i]` is an independent integer operation with no
   cross-thread communication or reduction, the result is
   bit-for-bit identical (`max_abs_error == 0`) regardless of which
   launch configuration was used -- isolating occupancy tuning as a
   pure *performance* knob with zero effect on correctness.

3. **CUDA-runtime API with no direct SYCL analog.**
   `cudaOccupancyMaxPotentialBlockSize` (and the underlying
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` it's built on) is a
   CUDA-runtime-specific occupancy calculator; SYCL has no standard,
   portable equivalent that queries a device for a kernel's suggested
   work-group size from its resource usage. This makes the case a
   deliberately imperfect 1:1 migration target -- see "Build & run"
   below.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`): `array[i] = i % 1000`, for `i` in
  `[0, 1048576)`.
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the
  1,048,576-element `array` after the automatic
  (occupancy-tuned) launch's `square`, one `uint32_t` per line, plus a
  `PASS`/`FAIL` line on stdout comparing *both* the manual-launch and
  automatic-launch results against `reference_square()` in
  `reference.h` (exact match expected for both, since each element is
  an independent, order-independent integer multiply).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct: the
`square` kernel itself migrates directly (a `sycl::nd_item`-indexed
elementwise multiply), but `cudaOccupancyMaxPotentialBlockSize` has no
direct SYCL/DPC++ equivalent -- SYCLomatic typically leaves an
occupancy-calculator call either commented out or replaced with a
fixed/vendor-suggested work-group size (e.g. a multiple of the
sub-group size reported by `device::get_info<info::device::...>()`,
or simply a constant such as 256). Document whichever fallback is used
on the SYCL side as an intentionally imperfect migration of this API,
run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected, since the choice of work-group/block size cannot change the
per-element result here).
