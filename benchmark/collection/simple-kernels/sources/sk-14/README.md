# Case: templatedSharedMemIdiom (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | C++ template `__global__` kernel (`testKernel<T>`) using the `SharedMemory<T>` template-specialization idiom to route each element through a dynamically-sized shared-memory array, then multiply it by the runtime thread count |
| size | N = 256 threads / 1 block, run for T = int and T = float |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected for both instantiations |

## Source

The `SharedMemory<T>` template (and all of its type specializations) and
the `testKernel<T>` `__global__` function in `main.cu` are reproduced
**verbatim**, and `reference.h`'s `reference_transform<T>()` is ported
directly, from:

- Project: **NVIDIA/cuda-samples**
- Files: `cpp/0_Introduction/simpleTemplates/sharedmem.cuh`,
  `cpp/0_Introduction/simpleTemplates/simpleTemplates.cu`
- Repository: https://github.com/NVIDIA/cuda-samples
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

What was **kept verbatim**: the entire `sharedmem.cuh` header (the
unspecialized `SharedMemory<T>` primary template, which deliberately
fails to compile if instantiated via an undefined `error()` call, plus
all eleven POD-type full specializations `int`, `unsigned int`, `char`,
`unsigned char`, `short`, `unsigned short`, `long`, `unsigned long`,
`bool`, `float`, `double`); and `testKernel<T>` itself, unmodified.

What was **omitted, and why**: the original `simpleTemplates.cu`'s
`main()`/`runTest<T>()` driver (which depends on `helper_cuda.h`,
`helper_functions.h`, `StopWatchInterface` timing, `findCudaDevice()`,
and cutil-style `ArrayComparator<T>`/`ArrayFileWriter<T>` class
specializations wrapping `compareData()`/`sdkWriteFile()` -- none of
which are available outside the full cuda-samples tree); `main.cu`
here has a new, self-contained driver instead. The two other host-side
class templates (`ArrayComparator<T>`, `ArrayFileWriter<T>`) are pure
sample-harness plumbing with no computational content and were dropped
entirely.

What is **new** in this directory: `main.cu`'s `run_case<T>()` /
`main()` host driver (allocation, `cudaMemcpy`, launch, error checking
via this repo's `CHECK()` convention, and the single-file
deterministic-input / CPU-reference / `argv[1]`-output convention used
throughout this repo), `reference.h`'s `gen_idata<T>()` input
generator, `Makefile`, `CMakeLists.txt`, and this `README.md`.

## What this case demonstrates (methods used)

1. **The `SharedMemory<T>` template-specialization idiom for
   dynamically-sized templated shared memory.** CUDA does not allow
   `extern __shared__ T sdata[];` inside a function template -- nvcc
   rejects a templated `extern __shared__` declaration, since multiple
   instantiations would otherwise collide on one external symbol name
   with different element types. The idiom sidesteps this: an
   unspecialized `SharedMemory<T>::getPointer()` that intentionally
   won't compile if ever instantiated, and one full specialization per
   concrete type, each declaring its *own*, uniquely-named,
   non-templated `extern __shared__` array (`s_int`, `s_float`, ...)
   and returning a pointer to it. Because only one dynamic shared-memory
   allocation is ever live per kernel launch, every specialization
   safely refers to the same underlying shared-memory bank at runtime
   -- the compile-time dispatch (which specialization gets selected for
   a given `T`) is exactly what makes a single templated kernel body
   usable with any of eleven different element types without ever
   writing `extern __shared__ T`.

2. **Per-thread, non-reducing shared-memory round trip.**
   `testKernel<T>` copies `g_idata[tid]` into shared memory
   (`__syncthreads()`), multiplies that shared-memory slot in place by
   `num_threads` (a value known only at kernel-launch time, i.e. not a
   compile-time constant), syncs again, then writes it back to
   `g_odata[tid]`. Every output element is produced entirely by its own
   thread; no thread ever reads another thread's shared-memory slot.
   This isolates the "get a templated, dynamically-sized shared-memory
   pointer and use it as a private scratch buffer" idiom from any
   actual cross-thread communication or reduction, so the two GPU
   implementations (CUDA vs. migrated SYCL) can be checked
   **byte-for-bit exactly**: `out[i] = in[i] * num_threads` for
   `in[i] = i`, `num_threads = 256` (an exact power of two), has no
   accumulation-order sensitivity for either T = int (max product
   `255 * 256 = 65280`, far under `INT_MAX`) or T = float (multiplying
   by an exact power of two never rounds in IEEE-754 binary32, given
   the small magnitudes here).

3. **A single template kernel body instantiated for two different
   element types** (`testKernel<int>`, `testKernel<float>`) from one
   source definition -- the "simple but not trivial" content here is
   recognizing *why* the shared-memory declaration needs the
   specialization trick at all (rather than, say, writing two
   hand-duplicated non-template kernels), which is exactly the kind of
   idiom a CUDA-to-SYCL migration tool must translate correctly (SYCL's
   `sycl::local_accessor<T>` / dynamic local memory extension is
   natively templated and needs no such workaround).

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`): `g_idata[i] = (T)i` for `i` in `[0, N)`, `N = 256`,
  for both `T = int` and `T = float`; launch = 1 block x 256 threads,
  dynamic shared memory = `256 * sizeof(T)` bytes.
- **Output**: `argv[1]` (default `output/cuda_output.txt`), 512 lines
  total -- the first 256 lines are `testKernel<int>`'s `g_odata[]` (one
  int per line), the next 256 lines are `testKernel<float>`'s
  `g_odata[]` (one float per line, `%.9g`) -- plus a `PASS`/`FAIL` line
  on stdout comparing both instantiations' output against
  `reference_transform<T>()` in `reference.h` (exact match expected for
  both types, since each output element is an independent
  multiply-by-a-power-of-two with no cross-thread accumulation).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (the
`SharedMemory<T>` specializations and their `extern __shared__`
declarations typically migrate to a single templated
`sycl::local_accessor<T, 1>` parameter or an `external_dynamic_shared`
memory helper -- worth inspecting closely, since this is one of the
idioms migration tooling has to specifically recognize rather than
translate line-by-line), build the result, run it with the same
`argv[1]` convention (e.g. `output/sycl_output.txt`), and diff the two
output files (exact match expected).
