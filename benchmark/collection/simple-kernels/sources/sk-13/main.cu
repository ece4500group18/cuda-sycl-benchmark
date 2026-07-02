// systemScopeAtomicAdd: system-scope (*_system), host-device-coherent
// atomics vs. ordinary device-scope atomics, computing the same
// closed-form aggregate over a tiny (10-element) int array.
//
// The __global__ kernel `atomicKernel` and the host function
// `atomicKernel_CPU` below are reproduced, unmodified (aside from
// formatting), from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/systemWideAtomics/systemWideAtomics.cu
//   https://github.com/NVIDIA/cuda-samples
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// `atomicKernel` uses nine *_system atomic intrinsics
// (atomicAdd_system, atomicExch_system, atomicMax_system,
// atomicMin_system, atomicInc_system, atomicDec_system,
// atomicCAS_system, atomicAnd_system, atomicOr_system,
// atomicXor_system) on `cudaMallocManaged` memory. Unlike the plain
// (device-scope) atomics used by this repo's `atomicIntrinsics` case,
// *_system atomics are guaranteed coherent with concurrent CPU
// accesses to the same managed allocation -- that is the entire point
// of this sample: `atomicKernel` (GPU) and `atomicKernel_CPU` (host,
// using __sync_* GCC/Clang builtins) both update the *same* array,
// each covering half of a shared logical index range, and the combined
// result must equal a single closed-form/range oracle exactly as if
// one single scope had done all the work.
//
// `atomicKernel_device` is a *new*, adapted (not verbatim) twin of
// `atomicKernel`: the same nine operations, in the same order, with
// the same LOOP_NUM repeat count, but using ordinary *device-scope*
// atomics (no _system suffix) in a single GPU kernel launched over the
// same total logical index range [0, 2*numBlocks*numThreads) that the
// system path splits between GPU and host. Because all nine operations
// are either associative/commutative (add, max, min, inc, dec, and,
// or, xor) or -- for exch/cas -- only ever guaranteed to land on *some*
// contributor's index, *scope* (system vs. device) cannot change the
// final value; it only changes whether concurrent host and device
// accesses to the same address are guaranteed to observe each other's
// writes. This isolates the atomic memory-scope/coherence-domain axis
// specifically -- distinct from this repo's `atomicIntrinsics` case,
// which already covers the general "which RMW ops have one guaranteed
// outcome" question but never leaves single-GPU, device-scope atomics.
//
// Deterministic setup (replicated by reference.h, following the
// original sample's own verify() exactly, generalized to a `len`
// parameter):
//   numThreads = 256, numBlocks = 64, LOOP_NUM = 50 (all as upstream)
//   len = 2 * numBlocks * numThreads = 32768 total logical contributors
//     (system path: 16384 from the GPU kernel + 16384 from the host
//     mirror function; device path: 16384*2 = 32768 GPU threads in one
//     launch)
//   initial array values: all 0, except slot 7 (AND) and slot 9 (XOR)
//   initialized to 0xff (matches upstream).
//
// Requires cudaMallocManaged (Unified Memory) and *_system atomics,
// i.e. compute capability 6.0+; this repo targets sm_70/80/90 only, so
// no runtime capability check is performed (see Makefile ARCH).
//
// Output: argv[1] (default output/cuda_output.txt), 10 lines, one
// final integer value per output slot of the *system* array.

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
// participation) -- same 9 ops / same order / same LOOP_NUM, but every
// atomic below is the ordinary device-scope intrinsic instead of its
// *_system counterpart, and `tid` ranges over the *entire* logical
// index range [0, 2*numBlocks*numThreads) in a single kernel launch
// (2x the blocks of atomicKernel above), instead of being split
// between a GPU kernel and a host mirror function. ---

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
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  const unsigned int numContribGPU = numBlocks * numThreads;      // 16384
  const int len = (int)(2 * numContribGPU);                       // 32768

  // --- "system" path: cudaMallocManaged + atomicKernel (*_system) on the
  // GPU, followed by atomicKernel_CPU on the host, both touching the
  // same coherent allocation. ---
  int *sysArr;
  CHECK(cudaMallocManaged(&sysArr, memSize));
  for (unsigned int i = 0; i < numData; ++i) sysArr[i] = 0;
  sysArr[7] = sysArr[9] = 0xff;  // make AND/XOR tests non-trivial

  atomicKernel<<<numBlocks, numThreads>>>(sysArr);
  CHECK(cudaGetLastError());
  // Always synchronize before the host touches managed memory here, so
  // this case runs deterministically regardless of whether the device
  // reports cudaDevAttrConcurrentManagedAccess (the original sample
  // skips this sync on platforms that do report it).
  CHECK(cudaDeviceSynchronize());

  atomicKernel_CPU(sysArr, (int)numContribGPU);

  // --- "device" path: plain cudaMalloc + atomicKernel_device (ordinary,
  // device-scope atomics only), single launch covering the full [0,len)
  // range that the system path split across GPU + host. ---
  int hDevInit[numData];
  for (unsigned int i = 0; i < numData; ++i) hDevInit[i] = 0;
  hDevInit[7] = hDevInit[9] = 0xff;

  int *dDevArr;
  CHECK(cudaMalloc(&dDevArr, memSize));
  CHECK(cudaMemcpy(dDevArr, hDevInit, memSize, cudaMemcpyHostToDevice));

  atomicKernel_device<<<2 * numBlocks, numThreads>>>(dDevArr);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  int hDevArr[numData];
  CHECK(cudaMemcpy(hDevArr, dDevArr, memSize, cudaMemcpyDeviceToHost));

  // CPU reference check (mirrors the original sample's own verify(),
  // generalized to `len`/LOOP_NUM; see reference.h).
  int exact_ok_sys = reference_check_exact(sysArr, len, LOOP_NUM);
  int range_ok_sys = reference_check_range(sysArr, len);
  int exact_ok_dev = reference_check_exact(hDevArr, len, LOOP_NUM);
  int range_ok_dev = reference_check_range(hDevArr, len);

  // Scope (system vs. device) must not change the outcome of the eight
  // order-independent ops: the two arrays must agree exactly on those
  // slots even though one path split contributions across host+device
  // and the other kept everything on the GPU.
  int same_exact_slots = 1;
  const int exact_slots[8] = {0, 2, 3, 4, 5, 7, 8, 9};
  for (int k = 0; k < 8; ++k) {
    int s = exact_slots[k];
    if (sysArr[s] != hDevArr[s]) same_exact_slots = 0;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (unsigned int i = 0; i < numData; ++i) fprintf(f, "%d\n", sysArr[i]);
  fclose(f);

  int pass = exact_ok_sys && range_ok_sys && exact_ok_dev && range_ok_dev &&
             same_exact_slots;

  printf("systemScopeAtomicAdd done: len=%d, exact_ok_sys=%d, range_ok_sys=%d, "
         "exact_ok_dev=%d, range_ok_dev=%d, same_exact_slots=%d -> %s\n",
         len, exact_ok_sys, range_ok_sys, exact_ok_dev, range_ok_dev,
         same_exact_slots, out);
  printf("%s\n", pass ? "PASS" : "FAIL");

  cudaFree(dDevArr);
  cudaFree(sysArr);
  return 0;
}
