# Case: separableConvHaloTiling (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | Separable 2D convolution (row pass then column pass, 9-tap / radius-4 kernel), shared-memory halo-tile kernel vs. naive kernel that re-reads halo pixels directly from global memory every time |
| size | 256 x 256 image (65,536 pixels), KERNEL_RADIUS = 4 (KERNEL_LENGTH = 9) |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected for both GPU paths |

## Source

The two `__global__` kernels in `main.cu` (`convolutionRowsKernel` and
`convolutionColumnsKernel`) are reproduced **verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

What was kept verbatim: both kernel bodies, including their
`cg::thread_block` / `cg::sync` synchronization (the sample's
`cooperative_groups`-based spelling of `__syncthreads()`), their
`ROWS_BLOCKDIM_*` / `COLUMNS_BLOCKDIM_*` / `*_RESULT_STEPS` /
`*_HALO_STEPS` tiling constants, and their shared-memory tile layout and
indexing exactly as upstream.

What was omitted and why: the upstream file's `extern "C"` host launcher
wrappers (`setConvolutionKernel`, `convolutionRowsGPU`,
`convolutionColumnsGPU`), its `#include <helper_cuda.h>` and
`getLastCudaError`/`checkCudaErrors` calls, and its companion
`convolutionSeparable_common.h` / `convolutionSeparable_gold.cpp` /
`main.cpp` (which use a random image, `helper_image` PPM I/O, and a
`StopWatchInterface`-based timing harness) are not needed for a
deterministic, dependency-free CUDA-vs-SYCL output comparison. This
directory's `main()` replaces that driver with its own deterministic
setup, launch, and this repo's `CHECK()` error-checking convention.
`KERNEL_RADIUS` is fixed here at 4 (9-tap) rather than upstream's
default of 8 (17-tap); the kernel bodies are radius-agnostic (the radius
is only ever a compile-time constant baked into `#pragma unroll` loop
bounds), and radius 4 keeps every one of the upstream launch-configuration
asserts (`ROWS_BLOCKDIM_X*ROWS_HALO_STEPS >= KERNEL_RADIUS`,
`COLUMNS_BLOCKDIM_Y*COLUMNS_HALO_STEPS >= KERNEL_RADIUS`) satisfied with a
smaller, faster benchmark.

What is new code, and why: `convolutionRowsNaiveKernel` and
`convolutionColumnsNaiveKernel` in `main.cu` (one thread per output pixel,
reading every input pixel directly from global memory with no
shared-memory staging) do not exist upstream -- they were written for
this repository to give the shared-memory tiled kernel a same-math,
same-order counterpart that skips the shared-memory halo tile entirely,
which is exactly the "memory movement" contrast this case is built to
isolate. `reference.h`'s row/column convolution loops port the algorithm
shape of the upstream sample's own CPU gold reference
(`convolutionSeparable_gold.cpp`: `sum = 0; for (k = -R..R) { d = x + k;
if (d in bounds) sum += Src[d] * Kernel[R-k]; }`), and `Makefile`,
`CMakeLists.txt`, and this `README.md` are all new for this repository.

## What this case demonstrates (methods used)

1. **Shared-memory halo-tile reuse (memory movement).** Each thread
   block in `convolutionRowsKernel` / `convolutionColumnsKernel` loads a
   tile of the source image -- its "main data" plus a one-block-wide halo
   on each side -- into `__shared__ s_Data` exactly once (three small
   `#pragma unroll` loops: main, left/upper halo, right/lower halo, with
   zero substituted for any position that falls outside the image), syncs
   once, and then every output pixel the block produces reads its 9-tap
   neighborhood back out of that shared-memory tile. Each shared-memory
   element loaded is read by up to `KERNEL_LENGTH = 9` different output
   computations without ever touching global memory again.

2. **Naive direct-global-memory re-reads, as the contrasting technique.**
   `convolutionRowsNaiveKernel` / `convolutionColumnsNaiveKernel` assign
   one thread to one output pixel and read each of that pixel's 9
   neighbors straight from global memory, with the same
   "out-of-image -> 0" boundary rule and the same left-to-right
   accumulation order (`for j = -KERNEL_RADIUS..KERNEL_RADIUS: sum +=
   kernel[KERNEL_RADIUS-j] * pixel`) as the tiled kernel's inner loop.
   Every interior pixel is consequently re-fetched from global memory up
   to 9 times -- once per neighboring output pixel that needs it --
   instead of the tiled kernel's single shared-memory load. Both kernel
   pairs read/write the same arrays and compute the identical weighted
   sum in the identical order; the only difference is *how many times,
   and through which memory space,* each source pixel is moved off
   global memory -- the isolated variable this case is testing.

3. **Two-pass separable convolution.** A 2D 9x9 convolution is
   decomposed into a 1D horizontal pass (`convolutionRows*`) followed by
   a 1D vertical pass (`convolutionColumns*`) on the row pass's output,
   reducing the per-pixel work from `O(K^2)` to `O(2K)` taps -- the
   standard separable-filter technique, applied identically by both the
   tiled and the naive kernel pair so the comparison isolates only the
   halo-tiling question above.

4. **Why the results are guaranteed to match exactly.** The image values
   (`((x+y) % 13) - 6`, small integers) and kernel weights
   (`((i % 5) - 2) * 0.25`, multiples of a power-of-two fraction) are
   deliberately dyadic: every partial product and partial sum that occurs
   anywhere in either convolution pass is exactly representable in
   IEEE-754 single precision, with zero rounding error at any step. With
   no rounding error to accumulate, fused-multiply-add vs.
   separate-multiply-then-add, and CPU float/double arithmetic vs. GPU
   float arithmetic, all collapse to the identical mathematical value --
   so the tiled kernel, the naive kernel, and the `reference.h` CPU
   reference (which uses the same left-to-right accumulation order and
   the same boundary rule) all agree with `max_abs_error == 0`, a genuine
   exact match rather than a tolerance-masked one.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `image[y*imageW + x] = ((x + y) % 13) - 6` for `imageW = imageH = 256`
  - `kernel[i] = ((i % 5) - 2) * 0.25` for `i` in `[0, 9)` (`KERNEL_RADIUS
    = 4`, `KERNEL_LENGTH = 9`), copied to `__constant__ c_Kernel` via
    `cudaMemcpyToSymbol`
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the
  shared-memory tiled kernel's final image (row pass then column pass),
  one `%.9g` float per line in row-major order (65,536 lines), plus a
  `PASS`/`FAIL` line on stdout: `PASS` iff both the tiled kernel's output
  and the naive kernel's output match `reference.h`'s CPU reference with
  `max_abs_error == 0`.

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cg::thread_block` / `cg::sync` typically migrate to `sycl::group` /
`item.barrier(sycl::access::fence_space::local_space)`, `__constant__
c_Kernel` typically migrates to a `sycl::constant_buffer` or a
device-side global in constant address space, and the two-dimensional
`__shared__` tile arrays migrate to `sycl::local_accessor`s sized the
same way), build the result, run it with the same `argv[1]` convention
(e.g. `output/sycl_output.txt`), and diff the two output files (exact
match expected).
