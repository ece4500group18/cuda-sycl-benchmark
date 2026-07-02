// streamOrderedAllocVectorAdd: classic cudaMalloc/cudaFree vs. stream-ordered
// cudaMallocAsync/cudaFreeAsync memory-pool allocation, both driving the
// identical trivial vectorAddGPU kernel.
//
// The __global__ kernel below is reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/2_Concepts_and_Techniques/streamOrderedAllocation/streamOrderedAllocation.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// "classic": device buffers are obtained with plain cudaMalloc() before the
//   kernel launch and released with cudaFree() after, on the default stream.
//
// "streamOrdered": device buffers are obtained with cudaMallocAsync()
//   *inside* a non-blocking stream, i.e. the allocation itself becomes an
//   operation ordered within that stream's sequence (backed by a CUDA
//   memory pool rather than a synchronous driver call), and released with
//   cudaFreeAsync() -- mirroring the original sample's
//   basicStreamOrderedAllocation(): allocate d_a/d_b/d_c, copy H2D, launch
//   vectorAddGPU, free d_a/d_b, copy D2H, free d_c, all enqueued on the same
//   stream before a single cudaStreamSynchronize().
//
// Both paths feed vectorAddGPU the exact same input values and the exact
// same per-element, order-independent arithmetic (c[idx] = a[idx] +
// b[idx]); only *how and when the device memory backing a/b/c is obtained
// and released* differs -- this isolates the host-side allocation-strategy
// control path (synchronous driver allocator vs. stream-ordered memory
// pool) rather than any kernel-internal memory-access pattern.
//
// Requires a CUDA runtime/driver of 11.2 or newer for cudaMallocAsync /
// cudaFreeAsync / cudaMemPool support (build prerequisite; see README.md).
//
// Deterministic inputs (replicated by reference.h):
//   a[i] = (i % 23) - 11
//   b[i] = (i % 19) - 9
// nelem = 1048576 (= 1 << 20)
//
// Output: argv[1] (default output/cuda_output.txt), one float per line,
// c[] computed by the stream-ordered (cudaMallocAsync/cudaFreeAsync) path.

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

// --- begin: verbatim from NVIDIA/cuda-samples streamOrderedAllocation.cu ---

/* Add two vectors on the GPU */
__global__ void vectorAddGPU(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// --- end: verbatim from NVIDIA/cuda-samples ---

int main(int argc, char **argv) {
  const int nelem = 1 << 20;  // 1,048,576
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)nelem * sizeof(float);

  float *ha = (float *)malloc(bytes);
  float *hb = (float *)malloc(bytes);
  float *hc_classic = (float *)malloc(bytes);
  float *hc_stream = (float *)malloc(bytes);

  for (int i = 0; i < nelem; ++i) {
    ha[i] = gen_a(i);
    hb[i] = gen_b(i);
  }

  dim3 block(256);
  dim3 grid((unsigned int)((nelem + block.x - 1) / block.x));

  // --- classic path: plain cudaMalloc/cudaFree on the default stream ---
  {
    float *da, *db, *dc;
    CHECK(cudaMalloc(&da, bytes));
    CHECK(cudaMalloc(&db, bytes));
    CHECK(cudaMalloc(&dc, bytes));

    CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));

    vectorAddGPU<<<grid, block>>>(da, db, dc, nelem);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(hc_classic, dc, bytes, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(da));
    CHECK(cudaFree(db));
    CHECK(cudaFree(dc));
  }

  // --- stream-ordered path: cudaMallocAsync/cudaFreeAsync on a non-blocking
  //     stream, mirroring the original sample's basicStreamOrderedAllocation() ---
  {
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    float *da, *db, *dc;
    CHECK(cudaMallocAsync(&da, bytes, stream));
    CHECK(cudaMallocAsync(&db, bytes, stream));
    CHECK(cudaMallocAsync(&dc, bytes, stream));

    CHECK(cudaMemcpyAsync(da, ha, bytes, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(db, hb, bytes, cudaMemcpyHostToDevice, stream));

    vectorAddGPU<<<grid, block, 0, stream>>>(da, db, dc, nelem);
    CHECK(cudaGetLastError());

    CHECK(cudaFreeAsync(da, stream));
    CHECK(cudaFreeAsync(db, stream));
    CHECK(cudaMemcpyAsync(hc_stream, dc, bytes, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaFreeAsync(dc, stream));
    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaStreamDestroy(stream));
  }

  // CPU reference check.
  float *href = (float *)malloc(bytes);
  reference_vector_add(ha, hb, href, nelem);

  double max_abs_err_classic = 0.0, max_abs_err_stream = 0.0;
  for (int i = 0; i < nelem; ++i) {
    double d1 = (double)hc_classic[i] - (double)href[i];
    double d2 = (double)hc_stream[i] - (double)href[i];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_classic) max_abs_err_classic = d1;
    if (d2 > max_abs_err_stream) max_abs_err_stream = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < nelem; ++i) fprintf(f, "%.9g\n", hc_stream[i]);
  fclose(f);

  printf("streamOrderedAllocVectorAdd done: nelem=%d, "
         "max_abs_error(classic vs ref)=%.3e, max_abs_error(streamOrdered vs ref)=%.3e -> %s\n",
         nelem, max_abs_err_classic, max_abs_err_stream, out);
  printf("%s\n", (max_abs_err_classic == 0.0 && max_abs_err_stream == 0.0) ? "PASS" : "FAIL");

  free(ha);
  free(hb);
  free(hc_classic);
  free(hc_stream);
  free(href);
  return 0;
}
