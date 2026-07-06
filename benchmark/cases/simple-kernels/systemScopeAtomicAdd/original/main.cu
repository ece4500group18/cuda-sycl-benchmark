// systemScopeAtomicAdd: system-scope (*_system), host-device-coherent atomics
// vs. ordinary device-scope atomics, computing the same closed-form aggregate
// over a 10-element int array.
//
// The __global__ kernel atomicKernel and the host function atomicKernel_CPU
// below are reproduced, unmodified (aside from formatting), from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/systemWideAtomics/systemWideAtomics.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// atomicKernel_device is a new, adapted device-scope twin (same 9 operations,
// same order, same LOOP_NUM, ordinary atomics, whole [0,len) range in one
// launch). Requires cudaMallocManaged + *_system atomics (compute capability
// 6.0+; this repo targets sm_70/80/90).
//
// Deterministic setup: numThreads=256, numBlocks=64, LOOP_NUM=50,
// len = 2*numBlocks*numThreads = 32768; array all 0 except slot 7 (AND) and
// slot 9 (XOR) = 0xff.
//
// Output: argv[1] (default output/cuda_output.txt), 10 lines, the final integer
// value of each slot of the *system* array. Checked by tests/verify.py: eight
// order-independent slots matched exactly; atomicExch (slot 1) and atomicCAS
// (slot 6) range-checked to a valid contributor index in [0, len).

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

#define LOOP_NUM 50

#ifndef min
#define min(a, b) (a) < (b) ? (a) : (b)
#endif
#ifndef max
#define max(a, b) (a) > (b) ? (a) : (b)
#endif

// --- begin: verbatim from NVIDIA/cuda-samples systemWideAtomics.cu ---

__global__ void atomicKernel(int *atom_arr)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < LOOP_NUM; i++) {
        // Atomic addition
        atomicAdd_system(&atom_arr[0], 10);

        // Atomic exchange
        atomicExch_system(&atom_arr[1], tid);

        // Atomic maximum
        atomicMax_system(&atom_arr[2], tid);

        // Atomic minimum
        atomicMin_system(&atom_arr[3], tid);

        // Atomic increment (modulo 17+1)
        atomicInc_system((unsigned int *)&atom_arr[4], 17);

        // Atomic decrement
        atomicDec_system((unsigned int *)&atom_arr[5], 137);

        // Atomic compare-and-swap
        atomicCAS_system(&atom_arr[6], tid - 1, tid);

        // Bitwise atomic instructions

        // Atomic AND
        atomicAnd_system(&atom_arr[7], 2 * tid + 7);

        // Atomic OR
        atomicOr_system(&atom_arr[8], 1 << tid);

        // Atomic XOR
        atomicXor_system(&atom_arr[9], tid);
    }
}

void atomicKernel_CPU(int *atom_arr, int no_of_threads)
{
    for (int i = no_of_threads; i < 2 * no_of_threads; i++) {
        for (int j = 0; j < LOOP_NUM; j++) {
            // Atomic addition
            __sync_fetch_and_add(&atom_arr[0], 10);

            // Atomic exchange
            __sync_lock_test_and_set(&atom_arr[1], i);

            // Atomic maximum
            int old, expected;
            do {
                expected = atom_arr[2];
                old      = __sync_val_compare_and_swap(&atom_arr[2], expected, max(expected, i));
            } while (old != expected);

            // Atomic minimum
            do {
                expected = atom_arr[3];
                old      = __sync_val_compare_and_swap(&atom_arr[3], expected, min(expected, i));
            } while (old != expected);

            // Atomic increment (modulo 17+1)
            int limit = 17;
            do {
                expected = atom_arr[4];
                old      = __sync_val_compare_and_swap(&atom_arr[4], expected, (expected >= limit) ? 0 : expected + 1);
            } while (old != expected);

            // Atomic decrement
            limit = 137;
            do {
                expected = atom_arr[5];
                old      = __sync_val_compare_and_swap(
                    &atom_arr[5], expected, ((expected == 0) || (expected > limit)) ? limit : expected - 1);
            } while (old != expected);

            // Atomic compare-and-swap
            __sync_val_compare_and_swap(&atom_arr[6], i - 1, i);

            // Bitwise atomic instructions

            // Atomic AND
            __sync_fetch_and_and(&atom_arr[7], 2 * i + 7);

            // Atomic OR
            __sync_fetch_and_or(&atom_arr[8], 1 << i);

            // Atomic XOR
            __sync_fetch_and_xor(&atom_arr[9], i);
        }
    }
}

// --- end: verbatim from NVIDIA/cuda-samples ---

// --- begin: new, adapted from atomicKernel above (device-scope, no host
// participation) -- same 9 ops / same order / same LOOP_NUM, ordinary
// device-scope atomics, tid over the entire [0, 2*numBlocks*numThreads). ---

__global__ void atomicKernel_device(int *atom_arr)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < LOOP_NUM; i++) {
        atomicAdd(&atom_arr[0], 10);
        atomicExch(&atom_arr[1], tid);
        atomicMax(&atom_arr[2], tid);
        atomicMin(&atom_arr[3], tid);
        atomicInc((unsigned int *)&atom_arr[4], 17);
        atomicDec((unsigned int *)&atom_arr[5], 137);
        atomicCAS(&atom_arr[6], tid - 1, tid);
        atomicAnd(&atom_arr[7], 2 * tid + 7);
        atomicOr(&atom_arr[8], 1 << tid);
        atomicXor(&atom_arr[9], tid);
    }
}

// --- end: new, adapted ---

int main(int argc, char **argv) {
  const unsigned int numThreads = 256;
  const unsigned int numBlocks  = 64;
  const unsigned int numData    = 10;
  const size_t memSize = sizeof(int) * numData;
  const char *out = (argc > 1) ? argv[1] : "output/cuda_output.txt";

  const unsigned int numContribGPU = numBlocks * numThreads;      // 16384

  // --- "system" path: cudaMallocManaged + atomicKernel (*_system) on the GPU,
  // then atomicKernel_CPU on the host, both touching the same coherent array. ---
  int *sysArr;
  CHECK(cudaMallocManaged(&sysArr, memSize));
  for (unsigned int i = 0; i < numData; ++i) sysArr[i] = 0;
  sysArr[7] = sysArr[9] = 0xff;  // make AND/XOR tests non-trivial

  atomicKernel<<<numBlocks, numThreads>>>(sysArr);
  CHECK(cudaGetLastError());
  // Synchronize before the host touches managed memory, so this case runs
  // deterministically regardless of cudaDevAttrConcurrentManagedAccess.
  CHECK(cudaDeviceSynchronize());

  atomicKernel_CPU(sysArr, (int)numContribGPU);

  // --- "device" path: plain cudaMalloc + atomicKernel_device (ordinary
  // device-scope atomics), single launch over the full [0,len) range. ---
  int hDevInit[numData];
  for (unsigned int i = 0; i < numData; ++i) hDevInit[i] = 0;
  hDevInit[7] = hDevInit[9] = 0xff;

  int *dDevArr;
  CHECK(cudaMalloc(&dDevArr, memSize));
  CHECK(cudaMemcpy(dDevArr, hDevInit, memSize, cudaMemcpyHostToDevice));

  atomicKernel_device<<<2 * numBlocks, numThreads>>>(dDevArr);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (unsigned int i = 0; i < numData; ++i) fprintf(f, "%d\n", sysArr[i]);
  fclose(f);

  cudaFree(dDevArr);
  cudaFree(sysArr);
  return 0;
}
