// pinnedAsyncIncrement: cudaMallocHost pinned staging + cudaMemcpyAsync
// H2D/kernel/D2H on a stream, vs. plain pageable malloc + synchronous
// cudaMemcpy, for the exact same per-element increment kernel.
//
// The __global__ kernel below is reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/asyncAPI/asyncAPI.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Both host-side paths below launch the identical `increment_kernel` over
// the identical data and produce the identical result; only *how the data
// gets to and from the device* differs:
//
// "pinned + async": the host array is allocated with `cudaMallocHost`
//   (page-locked / pinned memory) and moved with `cudaMemcpyAsync` on a
//   non-default stream -- the DMA engine can copy directly out of/into
//   pinned pages without an intermediate driver-side bounce buffer, and
//   the call returns to the CPU immediately (true asynchrony), matching
//   the upstream sample's H2D -> kernel -> D2H pipeline issued "all to
//   one stream".
//
// "pageable + sync": the host array is allocated with plain `malloc`
//   (pageable memory) and moved with the synchronous `cudaMemcpy` -- the
//   driver must stage each transfer through an internal pinned bounce
//   buffer before the DMA engine can touch it, and the call blocks the
//   CPU until the copy completes.
//
// Both paths touch every element exactly once with the exact same
// `g_data[idx] += inc_value` update, so their outputs must be bit-for-bit
// identical -- only the memory-allocation kind (pinned vs. pageable) and
// the copy/launch mechanism (async-on-a-stream vs. synchronous-on-the-
// default-stream) differ. This isolates the pinned/async memory-movement
// path itself, not the (trivial) per-element arithmetic.
//
// Dropped from the upstream sample: `cudaEvent_t` GPU timing, the
// `StopWatchInterface` CPU timer, the `cudaProfilerStart/Stop` calls, and
// the CPU busy-wait loop (`while (cudaEventQuery(stop) == cudaErrorNotReady)
// counter++;`) used only to demonstrate CPU/GPU overlap -- none of that
// affects correctness, and this benchmark isolates the memory-movement
// technique, not timing.
//
// Deterministic input (replicated by reference.h):
//   g_data[i] = i, for i in [0, n)
//   inc_value = 17 (fixed)
// n = 1 << 20 = 1,048,576 ints (launch = 2048 blocks x 512 threads, the
// same 512-thread block width as the upstream sample, scaled to n).
//
// Output: argv[1] (default output/output.txt), one int per line, g_data[]
// after the pinned+async path's increment_kernel call.

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// --- begin: verbatim from NVIDIA/cuda-samples asyncAPI.cu ---

__global__ void increment_kernel(int *g_data, int inc_value)
{
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}

// --- end: verbatim from NVIDIA/cuda-samples ---

int main(int argc, char **argv) {
  const int n = 1 << 20;  // 1,048,576 ints
  const int inc_value = 17;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)n * sizeof(int);

  dim3 threads(512, 1);
  dim3 blocks((unsigned)(n / threads.x), 1);

  // --- path 1: pinned host memory + cudaMemcpyAsync on a stream ---
  int *a_pinned = nullptr;
  CHECK(cudaMallocHost((void **)&a_pinned, bytes));
  for (int i = 0; i < n; ++i) a_pinned[i] = gen_data(i);

  int *d_a_pinned = nullptr;
  CHECK(cudaMalloc((void **)&d_a_pinned, bytes));

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  CHECK(cudaMemcpyAsync(d_a_pinned, a_pinned, bytes, cudaMemcpyHostToDevice, stream));
  increment_kernel<<<blocks, threads, 0, stream>>>(d_a_pinned, inc_value);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpyAsync(a_pinned, d_a_pinned, bytes, cudaMemcpyDeviceToHost, stream));
  CHECK(cudaStreamSynchronize(stream));

  // --- path 2: pageable host memory + synchronous cudaMemcpy ---
  int *a_pageable = (int *)malloc(bytes);
  for (int i = 0; i < n; ++i) a_pageable[i] = gen_data(i);

  int *d_a_pageable = nullptr;
  CHECK(cudaMalloc((void **)&d_a_pageable, bytes));

  CHECK(cudaMemcpy(d_a_pageable, a_pageable, bytes, cudaMemcpyHostToDevice));
  increment_kernel<<<blocks, threads>>>(d_a_pageable, inc_value);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(a_pageable, d_a_pageable, bytes, cudaMemcpyDeviceToHost));

  // CPU reference check (exact: g_data[i] must equal i + inc_value).
  int diff_pinned = 0, diff_pageable = 0;
  for (int i = 0; i < n; ++i) {
    int ref = reference_increment(i, inc_value);
    if (a_pinned[i] != ref) diff_pinned++;
    if (a_pageable[i] != ref) diff_pageable++;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < n; ++i) fprintf(f, "%d\n", a_pinned[i]);
  fclose(f);

  printf("pinnedAsyncIncrement done: n=%d, inc_value=%d, "
         "mismatches(pinned+async)=%d, mismatches(pageable+sync)=%d -> %s\n",
         n, inc_value, diff_pinned, diff_pageable, out);
  printf("%s\n", (diff_pinned == 0 && diff_pageable == 0) ? "PASS" : "FAIL");

  CHECK(cudaStreamDestroy(stream));
  cudaFree(d_a_pinned);
  cudaFree(d_a_pageable);
  cudaFreeHost(a_pinned);
  free(a_pageable);
  return 0;
}
