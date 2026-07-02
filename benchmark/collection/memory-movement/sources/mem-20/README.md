# Case: zeroCopyMappedVectorAdd (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | elementwise vector add `c[i] = a[i] + b[i]`, zero-copy mapped pinned host memory vs. classic `cudaMalloc` + explicit H2D/D2H copies |
| size | n = 1,048,576 floats (1 << 20), launch = ceil(n/256) blocks x 256 threads |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected (exact) |

## Source

The `__global__` kernel in `main.cu` (`vectorAddGPU`) is reproduced
**verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/0_Introduction/simpleZeroCopy/simpleZeroCopy.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleZeroCopy/simpleZeroCopy.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

The original sample's `main()` (`helper_cuda`/`helper_functions`
dependency, `--device=`/`--use_generic_memory`/`--help` command-line
flags, the `malloc()` + `cudaHostRegister()` "generic pinned memory"
alternative path, and an accumulated-norm-ratio (`errorNorm/refNorm <
1e-6`) pass/fail check driven by `rand()`-seeded input) is not needed
for a deterministic CUDA-vs-SYCL output comparison; this directory's
`main()` reimplements the same `cudaHostAlloc(cudaHostAllocMapped)` +
`cudaHostGetDevicePointer` zero-copy setup with deterministic
index-formula inputs, always targets device 0, and always uses the
`cudaHostAlloc` (not the generic/`cudaHostRegister`) path. It also adds
a second, classic `cudaMalloc` + explicit `cudaMemcpy` path launching
the *same* kernel over the *same* input, so the two results can be
diffed against each other and against a CPU reference. Everything here
other than the kernel listed above (host driver, `reference.h`,
`Makefile`, `CMakeLists.txt`, this `README.md`) is new code written for
this repository.

## What this case demonstrates (methods used)

1. **Zero-copy mapped pinned host memory (`memory movement`).** `a`,
   `b`, `c` are allocated with `cudaHostAlloc(..., cudaHostAllocMapped)`
   -- pinned host memory that is simultaneously mapped into the GPU's
   address space. `cudaHostGetDevicePointer()` produces GPU-side
   pointers (`d_a`, `d_b`, `d_c`) that alias that *same* physical host
   memory, so `vectorAddGPU` can read/write it directly over
   PCIe/NVLink, on demand, with **no `cudaMemcpy` call anywhere** in
   that path -- the host array `c` already holds the final result the
   instant `cudaDeviceSynchronize()` returns.

2. **Classic discrete-device-memory staging, for contrast.** The same
   kernel is launched again over separate `cudaMalloc`'d device
   buffers, with the same input explicitly staged in via
   `cudaMemcpy(..., cudaMemcpyHostToDevice)` beforehand and the result
   staged back out via `cudaMemcpy(..., cudaMemcpyDeviceToHost)`
   afterwards. Both paths run the identical kernel over the identical
   input, so the only thing that differs is *how the operand bytes get
   from host DRAM to the GPU and back* -- on-demand mapped access vs.
   an explicit bulk copy into and out of dedicated device memory. Since
   `c[i] = a[i] + b[i]` is a fully independent per-element computation
   with no cross-element accumulation, both paths must produce
   bit-identical output.

3. **Capability assertion instead of runtime fallback.** Following the
   upstream sample, this case checks `deviceProp.canMapHostMemory` via
   `cudaGetDeviceProperties()` at startup and treats a `false` result
   as a hard failure (matching the `CHECK()` convention) rather than
   silently degrading to a different memory path -- every CUDA device
   this benchmark's `ARCH` list targets (sm_70/80/90) supports mapped
   host memory, so requiring it up front keeps the comparison
   unambiguous.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `a[i] = (i % 23) - 11`
  - `b[i] = (i % 19) - 9`
  - `i` in `[0, n)`, `n = 1048576` (`1 << 20`)
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the final
  `c` array from the zero-copy path, one `%.9g` float per line
  (1,048,576 lines), plus a `PASS`/`FAIL` line on stdout comparing
  *both* the zero-copy result and the classic result against
  `reference_vector_add()` in `reference.h` (exact match expected,
  since each `c[i]` is an independent add regardless of which memory
  path supplied its operands).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cudaHostAlloc(..., cudaHostAllocMapped)` +
`cudaHostGetDevicePointer` typically migrate to a USM host allocation
via `sycl::malloc_host`/`sycl::aligned_alloc_host`, which SYCL USM
already exposes to device kernels without a separate "get device
pointer" step -- an instructive divergence point worth checking in the
migrated code), build the result, run it with the same `argv[1]`
convention (e.g. `output/sycl_output.txt`), and diff the two output
files (exact match expected).
