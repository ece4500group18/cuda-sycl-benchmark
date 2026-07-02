# Case: dynamicSharedMinReduction (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | dynamically-sized `extern __shared__` tree-halving min-reduction per block (`timedReduction`) vs. a naive single-thread linear-scan min over the same block's input (`naiveMinReduction`) |
| size | NUM_BLOCKS = 8, NUM_THREADS = 64 (each block reduces 2*64 = 128 floats) |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected |

## Source

The `timedReduction` `__global__` kernel in `main.cu` is adapted from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/0_Introduction/clock/clock.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/clock/clock.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

Only the `clock_t *timer` kernel parameter and the two `clock()` calls
that recorded a per-block start/end timestamp (plus the trailing
`__syncthreads()` that existed solely to make the second timestamp
include all reduction work) were removed -- that instrumentation
measures wall-clock duration, not a value that participates in the
reduction, and is irrelevant to a deterministic CUDA-vs-SYCL output
comparison. The dynamic-shared-memory declaration
(`extern __shared__ float shared[]`, sized at launch via the kernel's
third `<<<>>>` argument), the two-elements-per-thread load, and the
`for (d = blockDim.x; d > 0; d /= 2)` tree-halving comparison loop are
otherwise unmodified from the upstream file. The original sample's
`main()` (which used `findCudaDevice`/`checkCudaErrors` from
`helper_cuda.h`/`helper_functions.h`, printed an average elapsed-clocks
number, and never checked the reduction result against a reference) is
replaced by this directory's `main()`, plus a new `naiveMinReduction`
kernel (a single-thread-per-block linear scan) and `reference.h`, to
form a deterministic, single-file CUDA-vs-SYCL comparison harness with
an actual correctness check.

## What this case demonstrates (methods used)

1. **Dynamically-sized shared memory (`extern __shared__`).** Unlike a
   compile-time-sized `__shared__ float shared[256]`, the array's size
   is only known at kernel-launch time and is passed as the third
   `<<<grid, block, sharedMemBytes>>>` launch argument
   (`sizeof(float) * 2 * NUM_THREADS` here). This is the idiomatic CUDA
   pattern whenever the amount of shared memory a kernel needs depends
   on the launch configuration (e.g. `blockDim.x`) rather than being
   fixed at compile time.

2. **Tree-halving reduction control flow.** Each thread first loads two
   elements into shared memory (`shared[tid]` and
   `shared[tid + blockDim.x]`), then a loop halves a stride `d` from
   `blockDim.x` down to `1`; at each step, only the first `d` threads
   compare `shared[tid]` against `shared[tid + d]` and keep the smaller
   value, guarded by `__syncthreads()` so every thread sees the
   previous step's results before reading. After `log2(blockDim.x) + 1`
   steps, `shared[0]` holds the minimum of all `2 * blockDim.x` loaded
   values. This is "simple but not trivial": the operation is just a
   `<` comparison, but correctly reasoning about the halving stride,
   the synchronization placement, and why exactly `2 * blockDim.x`
   elements collapse to one value is the substantive content.

3. **Why both kernels are guaranteed to agree exactly.** Both kernels
   compute the minimum of the *same* `2 * NUM_THREADS = 128` input
   floats -- every block reads from `input[0 .. 127]` with no
   per-block offset, exactly as in the upstream sample (which does not
   index `input` by `blockIdx.x` either), so all `NUM_BLOCKS` blocks
   are expected to produce the identical minimum. `min` is commutative
   and associative, and every candidate value here is an exact `float`
   (a small integer times `0.5f`, representable without rounding), so
   the minimum of a fixed set of floats does not depend on the order
   or grouping in which candidates are compared: a tree-halving
   reduction (`timedReduction`), a plain sequential scan
   (`naiveMinReduction`), and the CPU reference's own sequential scan
   all must land on the exact same `float` bit pattern. There is no
   floating-point accumulation-order sensitivity here (unlike a sum
   reduction), so this case uses an exact-match oracle
   (`max_abs_error == 0`), not a tolerance.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`): `input[i] = ((i % 37) - 18) * 0.5f` for
  `i in [0, 128)` -- values cycle through `{-9.0f, -8.5f, ..., +8.5f,
  +9.0f}` every 37 indices. Launch: `timedReduction<<<8, 64,
  sizeof(float)*128>>>`; `naiveMinReduction<<<8, 1>>>` (one active
  thread per block, since the naive scan is intentionally serial).
- **Output**: `argv[1]` (default `output/cuda_output.txt`), 8 lines
  (`%.9g` float per line), the per-block minimum computed by
  `timedReduction`, plus a `PASS`/`FAIL` line on stdout comparing both
  kernels' per-block outputs against `reference_min()` in
  `reference.h` (exact match expected, since every block reduces the
  same 128-element input and `min` is order-independent).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`extern __shared__` dynamic shared memory typically migrates to a
`sycl::local_accessor` or a `sycl::group_local_memory`-backed pointer
sized from the kernel launch's local range, and `__syncthreads()`
becomes `item.barrier(sycl::access::fence_space::local_space)`), build
the result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
