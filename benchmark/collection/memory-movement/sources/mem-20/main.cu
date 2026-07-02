// zeroCopyMappedVectorAdd: elementwise vector add, zero-copy pinned host
// memory addressed directly by the GPU vs. classic discrete-device-memory
// staging via explicit H2D/D2H copies.
//
// The __global__ kernel below is reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/simpleZeroCopy/simpleZeroCopy.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// The *same* kernel (`vectorAddGPU`) is launched twice, over the same
// deterministic input, via two different memory-movement mechanisms:
//
// "zero-copy": `a`, `b`, `c` are allocated with `cudaHostAlloc(...,
//   cudaHostAllocMapped)` -- ordinary pinned host memory that is also
//   mapped into the GPU's address space. `cudaHostGetDevicePointer()`
//   hands the kernel GPU-side pointers (`d_a`, `d_b`, `d_c`) that alias
//   the *same physical host memory* -- every load/store the kernel issues
//   travels over PCIe/NVLink on demand. There is no `cudaMemcpy` at all:
//   the host arrays `a`/`b` are the input the kernel reads, and `c` is
//   already the final result the instant `cudaDeviceSynchronize()`
//   returns.
//
// "classic": `a`, `b`, `c` are ordinary pageable host arrays; `d_a`, `d_b`,
//   `d_c` are separate buffers obtained from `cudaMalloc` in the GPU's own
//   device memory. Inputs are staged with explicit `cudaMemcpy(...,
//   cudaMemcpyHostToDevice)` before the kernel runs, and the result is
//   staged back with an explicit `cudaMemcpy(...,
//   cudaMemcpyDeviceToHost)` afterwards.
//
// Both paths run the exact same kernel over the exact same input, so the
// two `c` arrays must match bit-for-bit -- only *how the kernel's operand
// bytes get from host DRAM to the GPU and back* differs (mapped/zero-copy
// on-demand PCIe access vs. an explicit bulk copy into dedicated device
// memory), the canonical "memory movement" contrast for zero-copy mapped
// memory.
//
// Deterministic inputs (replicated by reference.h):
//   a[i] = (i % 23) - 11
//   b[i] = (i % 19) - 9
// n = 1048576 (1 << 20)
//
// Following the upstream sample, this file checks
// `deviceProp.canMapHostMemory` via `cudaGetDeviceProperties()` at
// startup and *asserts* it is true (via CHECK-style hard failure) rather
// than silently falling back to a non-zero-copy path at runtime -- every
// CUDA device with compute capability >= 1.2 (i.e. every device this
// benchmark's ARCH list targets: sm_70/80/90) supports mapped host
// memory, so this is a fair invariant to assume. The original sample's
// `--device=` and `--use_generic_memory` CLI flags (which merely select
// which GPU to run on, or swap `cudaHostAlloc` for `malloc` +
// `cudaHostRegister`) are dropped; this directory always uses device 0
// and the `cudaHostAlloc` path.
//
// Output: argv[1] (default output/cuda_output.txt), one float per line,
// c[] = a[] + b[] computed by the zero-copy path.

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

// --- begin: verbatim from NVIDIA/cuda-samples simpleZeroCopy.cu ---

/* Add two vectors on the GPU */
__global__ void vectorAddGPU(float *a, float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// --- end: verbatim from NVIDIA/cuda-samples ---

int main(int argc, char **argv) {
  const int n = 1048576;  // 1 << 20
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)n * sizeof(float);
  const int idev = 0;

  CHECK(cudaSetDevice(idev));

  // Verify the device supports mapped host memory, and enable it, exactly
  // as the upstream sample does -- but assert rather than fall back.
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, idev));
  if (!deviceProp.canMapHostMemory) {
    fprintf(stderr,
            "device %d does not support mapping CPU host memory "
            "(cudaDeviceProp.canMapHostMemory == false); this benchmark "
            "requires it\n",
            idev);
    return 2;
  }
  CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

  // --- zero-copy path: cudaHostAlloc(cudaHostAllocMapped) + ---
  // --- cudaHostGetDevicePointer, no cudaMemcpy at all.       ---
  float *a, *b, *c;  // pinned, mapped host memory
  CHECK(cudaHostAlloc((void **)&a, bytes, cudaHostAllocMapped));
  CHECK(cudaHostAlloc((void **)&b, bytes, cudaHostAllocMapped));
  CHECK(cudaHostAlloc((void **)&c, bytes, cudaHostAllocMapped));

  for (int i = 0; i < n; ++i) {
    a[i] = gen_a(i);
    b[i] = gen_b(i);
  }

  float *d_a, *d_b, *d_c;  // device-space aliases of the same host memory
  CHECK(cudaHostGetDevicePointer((void **)&d_a, (void *)a, 0));
  CHECK(cudaHostGetDevicePointer((void **)&d_b, (void *)b, 0));
  CHECK(cudaHostGetDevicePointer((void **)&d_c, (void *)c, 0));

  dim3 block(256);
  dim3 grid((unsigned int)((n + block.x - 1) / block.x));
  vectorAddGPU<<<grid, block>>>(d_a, d_b, d_c, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  // `c` already holds the result -- no cudaMemcpy needed.

  // --- classic path: cudaMalloc device buffers + explicit H2D/D2H copies. ---
  float *ha = (float *)malloc(bytes);
  float *hb = (float *)malloc(bytes);
  float *hc = (float *)malloc(bytes);
  for (int i = 0; i < n; ++i) {
    ha[i] = gen_a(i);
    hb[i] = gen_b(i);
  }

  float *da, *db, *dc;
  CHECK(cudaMalloc(&da, bytes));
  CHECK(cudaMalloc(&db, bytes));
  CHECK(cudaMalloc(&dc, bytes));

  CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));

  vectorAddGPU<<<grid, block>>>(da, db, dc, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));

  // CPU reference check.
  float *href = (float *)malloc(bytes);
  reference_vector_add(href, n);

  double max_abs_err_zerocopy = 0.0, max_abs_err_classic = 0.0;
  for (int i = 0; i < n; ++i) {
    double d1 = (double)c[i] - (double)href[i];
    double d2 = (double)hc[i] - (double)href[i];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_zerocopy) max_abs_err_zerocopy = d1;
    if (d2 > max_abs_err_classic) max_abs_err_classic = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", c[i]);
  fclose(f);

  printf("zeroCopyMappedVectorAdd done: n=%d, "
         "max_abs_error(zero-copy vs ref)=%.3e, max_abs_error(classic vs ref)=%.3e -> %s\n",
         n, max_abs_err_zerocopy, max_abs_err_classic, out);
  printf("%s\n", (max_abs_err_zerocopy == 0.0 && max_abs_err_classic == 0.0) ? "PASS" : "FAIL");

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  free(ha);
  free(hb);
  free(hc);
  free(href);
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);
  return 0;
}
