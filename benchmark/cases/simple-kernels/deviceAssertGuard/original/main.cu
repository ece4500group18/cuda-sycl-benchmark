// deviceAssertGuard: device-side assert(gtid < N) as a compile-in correctness
// guard, paired with a kernel that instead writes a 0/1 pass/fail flag per
// thread (so the harness has a numeric oracle).
//
// The __global__ kernel testKernel below is reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/simpleAssert/simpleAssert.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// testKernelFlag is new code: it evaluates the same predicate (gtid < N) but
// records (gtid < N) ? 1 : 0 per thread, giving a bit-exact CPU-checkable
// oracle. The device assert half is exercised too (Part 2): the launch is
// intentionally larger than N (1024 threads vs N=1000), so gtid in [1000, 1024)
// trip the assert, which surfaces as cudaErrorAssert -- expected, not a crash.
//
// Deterministic setup: N = 1000, Nblocks = 8, Nthreads = 128 (1024 threads).
//
// Output: argv[1] (default output/cuda_output.txt), 1024 lines, one 0/1
// predicate flag per thread from testKernelFlag. Checked by tests/verify.py.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                           \
      return 2;                                                              \
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
// Evaluates the same predicate as testKernel (gtid < N), but records the 0/1
// result per thread so the harness has a numeric value to diff against a CPU
// reference.
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
  const char *out = (argc > 1) ? argv[1] : "output/cuda_output.txt";

  // --- Part 1: numeric oracle via testKernelFlag (all real device work is
  // done and drained before the assert in Part 2 poisons the context) ---
  int *hFlag = (int *)malloc((size_t)total * sizeof(int));
  int *dFlag;
  CHECK(cudaMalloc(&dFlag, (size_t)total * sizeof(int)));

  testKernelFlag<<<Nblocks, Nthreads>>>(dFlag, N);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(hFlag, dFlag, (size_t)total * sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(dFlag);

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < total; ++i) fprintf(f, "%d\n", hFlag[i]);
  fclose(f);

  // --- Part 2: device-side assert guard (testKernel, verbatim upstream) ---
  // gtid in [N, total) trip the assert; expect device-side "Assertion failed"
  // diagnostics on stderr -- that is expected output. The failure surfaces as
  // cudaErrorAssert from cudaDeviceSynchronize(); clear the sticky error via
  // cudaGetLastError() (documented requirement) before returning.
  printf("\n-- Begin expected device assert output --\n\n");
  testKernel<<<Nblocks, Nthreads>>>(N);
  cudaError_t sync_err = cudaDeviceSynchronize();
  printf("\n-- End expected device assert output --\n\n");
  (void)sync_err;
  cudaGetLastError();  // clear sticky error state left by the assert

  free(hFlag);
  return 0;
}
