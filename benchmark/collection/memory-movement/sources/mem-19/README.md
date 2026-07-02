# Case: unifiedMemoryAccess (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | strided gather `y[j] = a*x[j*stride]`, discrete device memory vs. CUDA Unified/managed memory |
| size | n = 1,048,576 floats input, stride = 16 (65,536 outputs) |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected (exact) |

## Source

The `__global__` kernel in `main.cu`
(`LowAccessDensityTest_cudakernel`) is reproduced **verbatim** from:

- Project: **CUDAMicroBench**
- File: `UniMem/LowAccessDensityTest_cuda.cu`
- Repository: https://github.com/passlab/CUDAMicroBench
- Copyright (c) 2021, University of North Carolina at Charlotte and
  Lawrence Livermore National Security, LLC.
- License: BSD-3-Clause (see `LICENSE` in this directory)
- Associated paper: Yi, Xinyao; Stokes, David; Yan, Yonghong; Liao,
  Chunhua. "CUDAMicroBench: Microbenchmarks to Assist CUDA Performance
  Programming." 2021 IEEE IPDPSW, pp. 397-406.
  doi:10.1109/IPDPSW52791.2021.00068

The original file's two host-side wrapper functions
(`LowAccessDensityTest_cuda_discrete_memory`,
`LowAccessDensityTest_cuda_unified_memory`) and its `.c` driver
(`drand48`-seeded random input, a 100-iteration timing loop, `libm`
`ftime`-based timers) are not needed for a deterministic
CUDA-vs-SYCL output comparison; this directory's `main()` reimplements
the same discrete-vs-managed contrast with deterministic inputs and a
single run of each path. Everything here other than the kernel listed
above (host driver, `reference.h`, `Makefile`, `CMakeLists.txt`, this
`README.md`) is new code written for this repository.

## What this case demonstrates (methods used)

1. **Explicit copy vs. Unified/managed memory (memory movement).** The
   *same* kernel is launched against two differently-provisioned copies
   of the input array:
   - **discrete**: `cudaMalloc` a device buffer, then one explicit,
     synchronous `cudaMemcpy(..., cudaMemcpyHostToDevice)` moves the
     entire array before the kernel runs.
   - **managed**: `cudaMallocManaged` a Unified Memory buffer; the host
     writes into it with a plain `memcpy` (no CUDA API call at all).
     The kernel then reads it directly -- any page not yet resident on
     the device is faulted in and migrated by the CUDA runtime on first
     touch, rather than being bulk-copied ahead of time.

   Because the kernel and the input values are identical in both
   cases, the two runs are guaranteed to produce byte-identical
   output; what differs is *how the input array's pages actually get
   from host to device memory* -- an explicit bulk copy chosen by the
   programmer vs. on-demand page migration chosen by the CUDA runtime
   -- the canonical "memory movement" contrast at the host/device
   boundary (as opposed to the other cases in this directory, which
   contrast movement *within* device memory).

2. **Strided ("low access density") global read.** Each output thread
   `i` reads only `x[i*stride]`, one element out of every `stride` --
   a sparse gather pattern (as opposed to `memAlign`'s dense,
   1-element-offset access), included here mainly as the payload that
   makes the discrete-vs-managed contrast meaningful for a single,
   non-repeated kernel launch.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `x[i] = ((i % 17) - 8) * 0.25f`, `i` in `[0, 1048576)`
  - `a = 123.456f`, `stride = 16`
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the 65,536
  outputs from the *managed*-memory run, one `%.9g` float per line,
  plus a `PASS`/`FAIL` line on stdout comparing both the discrete and
  the managed run's outputs against `reference_strided_axpy()` in
  `reference.h` (exact match expected: a single multiply per output
  element, independent of provisioning strategy).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cudaMallocManaged` typically migrates to SYCL Unified Shared Memory,
`sycl::malloc_shared`), build the result, run it with the same
`argv[1]` convention (e.g. `output/sycl_output.txt`), and diff the two
output files (exact match expected).
