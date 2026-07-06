// streamOrderedAllocVectorAdd: classic cudaMalloc/cudaFree vs. stream-ordered
// cudaMallocAsync/cudaFreeAsync memory-pool allocation, both driving the
// identical trivial vectorAddGPU kernel.
//
// The __global__ kernel vectorAddGPU below is reproduced, unmodified, from:
//   NVIDIA/cuda-samples,
//   cpp/2_Concepts_and_Techniques/streamOrderedAllocation/streamOrderedAllocation.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Deterministic inputs (reproduced by tests/verify.py):
//   a[i] = (i % 23) - 11,  b[i] = (i % 19) - 9,  nelem = 1<<20 = 1048576.
// Requires CUDA 11.2+ for cudaMallocAsync/cudaFreeAsync/memory pools.
//
// Output: argv[1] (default output/cuda_output.txt), one float per line, c[]
// computed by the stream-ordered (cudaMallocAsync/cudaFreeAsync) path.

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

// Deterministic input generators (new host code; mirrored in tests/verify.py).
static inline float gen_a(long i) { return (float)((i % 23) - 11); }
static inline float gen_b(long i) { return (float)((i % 19) - 9); }

int main(int argc, char **argv) {
  const int nelem = 1 << 20;  // 1,048,576
  const char *out = (argc > 1) ? argv[1] : "output/cuda_output.txt";
  const size_t bytes = (size_t)nelem * sizeof(float);

  float *ha = (float *)malloc(bytes);
  float *hb = (float *)malloc(bytes);
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

    CHECK(cudaFree(da));
    CHECK(cudaFree(db));
    CHECK(cudaFree(dc));
  }

  // --- stream-ordered path: cudaMallocAsync/cudaFreeAsync on a non-blocking
  //     stream, mirroring the sample's basicStreamOrderedAllocation() ---
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

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < nelem; ++i) fprintf(f, "%.9g\n", hc_stream[i]);
  fclose(f);

  free(ha);
  free(hb);
  free(hc_stream);
  return 0;
}
