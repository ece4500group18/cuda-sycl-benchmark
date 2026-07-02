// deviceAssertGuard: device-side assert(gtid < N) as a compile-in
// correctness guard, contrasted against a kernel that instead writes a
// 0/1 pass/fail flag per thread.
//
// The __global__ kernel below (`testKernel`) is reproduced, unmodified,
// from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/simpleAssert/simpleAssert.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// `testKernel(N)` computes gtid = blockIdx.x*blockDim.x + threadIdx.x and
// calls `assert(gtid < N)`. It is "simple but not trivial": there is no
// arithmetic result to inspect at all -- the kernel's entire "output" is
// whether the predicate holds for every thread, checked device-side by
// the assert machinery itself (a non-arithmetic, GPU-only control-flow /
// debugging technique).
//
// Nblocks*Nthreads is intentionally launched *larger* than N (1024
// threads vs. N=1000), so a known, deterministic subset of gtid values
// (gtid in [1000, 1024), 24 threads) violate `gtid < N` and must trip
// the device assert. That failure surfaces asynchronously as
// cudaErrorAssert, returned by the next synchronizing CUDA call
// (cudaDeviceSynchronize()/cudaGetLastError()) after the kernel runs --
// there is no crash on the host, just an error code.
//
// Because a triggered device assert has no numeric "answer" to diff
// against a CPU reference, this case pairs `testKernel` with a second,
// new (not from upstream) kernel, `testKernelFlag`, that evaluates the
// *exact same* predicate but instead of asserting writes
// `(gtid < N) ? 1 : 0` into an output array -- giving the harness a
// bit-exact, CPU-checkable numeric oracle for one half of the guard,
// while the other half (that the assert actually fires exactly when
// expected, and only then) is checked separately via the expected CUDA
// error code. See README.md ("split oracle") for the full rationale.
//
// Deterministic setup (replicated by reference.h):
//   N = 1000, Nblocks = 8, Nthreads = 128 (1024 threads total)
//   -> gtid in [0, 1000) satisfies the predicate; gtid in [1000, 1024)
//      (24 threads) does not.
//
// Output: argv[1] (default output/cuda_output.txt), 1024 lines, one
// 0/1 predicate flag per thread, from testKernelFlag.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "reference.h"

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                          \
      return 2;                                                             \
    }                                                                        \
  } while (0)

// --- begin: verbatim from NVIDIA/cuda-samples simpleAssert.cu ---

__global__ void testKernel(int N)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    assert(gtid < N);
}

// --- end: verbatim from NVIDIA/cuda-samples ---

// --- begin: new code for this repository (not from upstream) ---
//
// Evaluates the exact same predicate as testKernel above (gtid < N), but
// instead of asserting, records the 0/1 result per thread so the harness
// has a numeric value to diff against a CPU reference, rather than
// having to rely solely on the presence/absence of a device-side crash.
__global__ void testKernelFlag(int *g_flag, int N)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    g_flag[gtid] = (gtid < N) ? 1 : 0;
}
// --- end: new code for this repository ---

int main(int argc, char **argv) {
  const int N = 1000;
  const int Nblocks = 8;
  const int Nthreads = 128;
  const int total = Nblocks * Nthreads;  // 1024, intentionally > N
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  // --- Part 1: numeric oracle via testKernelFlag ---
  // Run and fully drain this part (including freeing device memory)
  // *before* triggering the device assert in Part 2 below: a triggered
  // device-side assert leaves the CUDA context in a state where further
  // CUDA API calls report the sticky error until cudaGetLastError() is
  // called, so all "real" device work is done first.
  int *hFlag = (int *)malloc((size_t)total * sizeof(int));
  int *dFlag;
  CHECK(cudaMalloc(&dFlag, (size_t)total * sizeof(int)));

  testKernelFlag<<<Nblocks, Nthreads>>>(dFlag, N);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(hFlag, dFlag, (size_t)total * sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(dFlag);

  int *href = (int *)malloc((size_t)total * sizeof(int));
  reference_predicate_array(href, total, N);

  int flag_exact_ok = 1;
  for (int i = 0; i < total; ++i) {
    if (hFlag[i] != href[i]) { flag_exact_ok = 0; break; }
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < total; ++i) fprintf(f, "%d\n", hFlag[i]);
  fclose(f);

  // --- Part 2: device-side assert guard (testKernel, verbatim upstream) ---
  // A known, deterministic subset of gtid values (N..total-1, 24
  // threads) violate `gtid < N` and must trip the device assert. Expect
  // some device-side "Assertion failed" diagnostic lines on stderr from
  // the CUDA runtime itself -- that is expected output, not a harness
  // failure. The failure surfaces asynchronously as cudaErrorAssert from
  // cudaDeviceSynchronize(); we then clear the sticky device-side error
  // state via cudaGetLastError() (a documented requirement after a
  // device-side assert) before doing anything else.
  printf("\n-- Begin expected device assert output --\n\n");
  testKernel<<<Nblocks, Nthreads>>>(N);
  cudaError_t sync_err = cudaDeviceSynchronize();
  printf("\n-- End expected device assert output --\n\n");

  int assert_ok = (sync_err == cudaErrorAssert);
  const char *sync_err_name = cudaGetErrorString(sync_err);
  cudaGetLastError();  // clear sticky error state left by the assert

  int expected_fail_count = total - N;  // 24

  printf("deviceAssertGuard done: N=%d, total_threads=%d, expected_fail_count=%d, "
         "flag_exact_ok=%d, assert_triggered_as_expected=%d "
         "(cudaDeviceSynchronize -> %s) -> %s\n",
         N, total, expected_fail_count, flag_exact_ok, assert_ok, sync_err_name, out);
  printf("%s\n", (flag_exact_ok && assert_ok) ? "PASS" : "FAIL");

  free(hFlag);
  free(href);
  return 0;
}
