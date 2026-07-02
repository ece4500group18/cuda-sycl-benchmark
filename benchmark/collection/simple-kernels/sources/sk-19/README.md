# Case: warpShuffleReduction (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | per-block sum reduction: shared-memory tree (`reduce2`) vs. warp-shuffle-accelerated (`reduce4`) |
| size | n = 1,048,576 (2^20) floats, 256 threads/block |
| correctness | CPU reference (`reference.h`), tolerance 1e-2 (float accumulation order differs) |

## Source

The kernels and helper templates in `main.cu` (`SharedMemory<T>`,
`reduce2`, `warpReduceSum`, `reduce4`) are reproduced **verbatim**
(with the generic templates instantiated for `float` /
`blockSize=256`) from:

- Project: **NVIDIA/cuda-samples**,
  `Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu`
- Also redistributed unchanged as **CUDAMicroBench**
  `Shuffle/cuda_shuffle/reduction_kernel.cu`
  (https://github.com/passlab/CUDAMicroBench)
- Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)
- Reference: "Faster Parallel Reductions on Kepler", NVIDIA Developer
  Blog, https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/

The original `reduction.cpp` host driver depends on NVIDIA's shared
`helper_cuda.h` / `helper_functions.h` (command-line parsing, error-
checking macros, `--shmoo` timing harness) which are not needed for a
deterministic CUDA-vs-SYCL output comparison. Everything in this
directory other than the kernels/templates listed above (host driver
in `main.cu`, `reference.h`, `Makefile`, `CMakeLists.txt`, this
`README.md`) is new code written for this repository, following the
same minimal single-file style used by the other cases.

## What this case demonstrates (methods used)

1. **Warp-shuffle intrinsics (`__shfl_down_sync`).** `reduce4`'s
   `warpReduceSum<T>` reduces 32 values held in 32 different threads'
   registers down to lane 0 using `__shfl_down_sync`, with **no shared
   memory and no `__syncthreads()`** for the final 32-element step --
   values are exchanged directly between a warp's register files.

2. **Shared-memory tree reduction with `__syncthreads()`.** `reduce2`
   reduces `blockDim.x` shared-memory elements to 1 with
   `for (s = blockDim.x/2; s > 0; s >>= 1)`, each step guarded by
   `cg::sync(cta)` (`__syncthreads()`).

3. **"First add during load" (Brent's-theorem-style work reduction).**
   `reduce4` uses half as many threads as `reduce2` for the same input
   size: each thread reads *two* elements
   (`g_idata[i]` and `g_idata[i+blockSize]`) and adds them before the
   shared-memory phase even begins, halving the shared-memory traffic
   compared to `reduce2`.

4. **Cooperative Groups** (`cg::thread_block`, `cg::tiled_partition<32>`)
   used as a modern, structured replacement for raw
   `__syncthreads()`/`__shfl_down_sync` calls.

5. **Generic templated kernels** (`template <class T>`,
   `template <class T, unsigned int blockSize>`) with a
   compile-time-specialized shared-memory helper (`SharedMemory<T>`,
   specialized for `double` to satisfy alignment requirements) --
   illustrates how a "simple" reduction kernel becomes genuinely
   non-trivial once made generic and high-performance.

   This case isolates exactly *how values move between threads* during
   a reduction (shared memory + syncthreads vs. warp-register shuffle)
   while computing the same per-block sums -- a simple operation
   (`+=`) implemented with a non-trivial, widely-used data-movement
   technique.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `x[i] = ((i % 17) - 8) * 0.25f`, `i` in `[0, 1048576)`
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the 2048
  per-block partial sums from `reduce4`, one `%.9g` float per line,
  plus a `PASS`/`FAIL` line on stdout comparing both kernels' outputs
  against `reference_block_sum()` in `reference.h` (tolerance 1e-2 due
  to differing float-summation order between `reduce2`, `reduce4`, and
  the reference's left-to-right sum).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct. Note that
`cooperative_groups` and `__shfl_down_sync` typically migrate to
`sycl::group`/`sycl::sub_group` and `sycl::sub_group::shuffle_down`
respectively -- this is one of the more interesting migration targets
in this benchmark set. Build the result, run it with the same
`argv[1]` convention (e.g. `output/sycl_output.txt`), and compare the
two output files within the same 1e-2 tolerance.
