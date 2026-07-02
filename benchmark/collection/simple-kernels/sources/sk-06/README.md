# Case: deviceAssertGuard (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | device-side `assert(gtid < N)` as a compile-in correctness guard, vs. a kernel that records the same predicate as a 0/1 flag |
| size | N = 1000, Nblocks = 8, Nthreads = 128 (1024 threads total, 24 of which violate the predicate) |
| correctness | split oracle -- `reference.h` exact match (`max_abs_error == 0`) on the 1024-entry flag array, plus an expected-`cudaErrorAssert` check on the asserting kernel |

## Source

The `__global__` kernel `testKernel` in `main.cu` is reproduced
**verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/0_Introduction/simpleAssert/simpleAssert.cu`
- URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleAssert/simpleAssert.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

The original file's `main()`/`runTest()` (command-line parsing via
`findCudaDevice`, `helper_cuda`/`helper_functions` dependency, an
OS-detection block that skips the test on macOS, and a driver that only
ever launches `testKernel` once with a fixed `Nblocks=2, Nthreads=32,
N=60`) was omitted; this directory replaces it with a deterministic,
single-file CUDA-vs-SYCL comparison harness that scales the launch up to
1024 threads (`Nblocks=8, Nthreads=128`) against `N=1000` and adds a
second kernel, `testKernelFlag`, that is **new code written for this
repository** (not present upstream). `testKernelFlag` evaluates the
exact same `gtid < N` predicate as `testKernel` but writes the 0/1
result to an output array instead of asserting on it, because a
device-side assert has no numeric "answer" for a harness to diff --
see "What this case demonstrates" below for why both halves are needed.
Everything else here (`Makefile`, `CMakeLists.txt`, `reference.h`, this
`README.md`, and all host driver code in `main.cu`) is original to this
repository.

## What this case demonstrates (methods used)

1. **Device-side `assert()` as a compile-in correctness guard.**
   `testKernel(N)` computes `gtid = blockIdx.x*blockDim.x + threadIdx.x`
   and calls `assert(gtid < N)` -- a non-arithmetic, GPU-only
   control-flow/debugging technique with no numeric result at all. It is
   "simple but not trivial": the one-line predicate is trivial, but
   understanding *what actually happens* when it's violated -- an
   asynchronous, per-thread trap that surfaces only on the next
   synchronizing CUDA call, as `cudaErrorAssert`, rather than a normal
   return value or a host-side crash -- is the substantive,
   easy-to-get-wrong part of using device asserts in real code.

2. **Deliberately over-launching to guarantee a deterministic assert
   trip.** `Nblocks*Nthreads = 1024` is launched against `N = 1000`, so
   the 24 threads with `gtid` in `[1000, 1024)` are *known in advance* to
   violate the predicate -- this is not a hypothetical bug being
   demonstrated, it's a controlled, reproducible trigger of the assert
   path so the harness can verify the guard actually fires (and fires
   only for the expected threads) rather than merely verifying it stays
   silent on well-formed input.

3. **A split oracle for a technique with no numeric output.** Because
   `assert()` either passes silently or kills the kernel invocation with
   an error code -- it never *computes* anything a CPU reference could
   diff -- this case pairs `testKernel` with `testKernelFlag`, a second
   kernel (new code, not from upstream) that evaluates the identical
   `gtid < N` predicate but writes `(gtid < N) ? 1 : 0` into an output
   array. Each of the two paths is verified independently:
   - `testKernelFlag`'s 1024-entry flag array is diffed **bit-for-bit**
     (`max_abs_error == 0`) against `reference_predicate_array()` in
     `reference.h` -- every entry is an independent, order-free
     predicate evaluation, so an exact match is guaranteed regardless of
     thread/block scheduling.
   - `testKernel` is launched separately and is expected to make
     `cudaDeviceSynchronize()` return exactly `cudaErrorAssert` (not
     `cudaSuccess`, not some other error) -- confirming the guard trips
     precisely when, and only when, the known-bad subset of threads runs.

   The overall `PASS`/`FAIL` verdict requires **both** halves to hold.
   This documents, in one file, how to give a hard numeric
   pass/fail oracle to a GPU technique whose native behavior is a crash
   rather than a computed value.

4. **Sticky device-side error state after an assert.** Once a device
   assert trips, the CUDA context reports the sticky `cudaErrorAssert`
   on the *next* call, and further CUDA API calls remain affected until
   `cudaGetLastError()` clears it. `main.cu` performs and fully drains
   all of `testKernelFlag`'s work (launch, sync, copy back, free) before
   ever launching the asserting `testKernel`, and clears the sticky
   error immediately afterward -- a real-world gotcha when mixing
   assert-guarded kernels with other GPU work in the same process.

## Input / Output

- **Input**: no host-generated arrays -- both kernels are driven purely
  by the launch configuration and the scalar `N = 1000`
  (`Nblocks = 8`, `Nthreads = 128`, 1024 threads total).
- **Output**: `argv[1]` (default `output/cuda_output.txt`), 1024 lines,
  the 0/1 predicate flag from `testKernelFlag` for every absolute thread
  id `gtid` in `[0, 1024)` (`1` for `gtid < 1000`, `0` for
  `gtid >= 1000`). A `PASS`/`FAIL` line is also printed on stdout,
  `PASS` iff the flag array matches `reference_predicate_array()`
  exactly **and** `testKernel`'s launch makes
  `cudaDeviceSynchronize()` return `cudaErrorAssert` as expected. Expect
  to see device-side "Assertion failed" diagnostic lines on stderr
  between the `-- Begin/End expected device assert output --` markers --
  that is normal, intentional output from the 24 threads that violate
  the predicate, not a harness failure.

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note: device
`assert()` typically migrates directly, since SYCL/DPC++ supports
standard `assert()` in device code the same way; there is no direct
SYCL equivalent of `cudaErrorAssert`, so the migrated harness will
likely need to detect the assert either via an exception/abort from the
queue's async-error handler or via a raw `cl::sycl::exception` at
`wait()`/synchronization time -- adjust the "expected failure" check
accordingly), build the result, run it with the same `argv[1]`
convention (e.g. `output/sycl_output.txt`), and diff the two flag-array
output files (exact match expected) while separately confirming the
migrated build's asserting path still fails exactly as expected.
