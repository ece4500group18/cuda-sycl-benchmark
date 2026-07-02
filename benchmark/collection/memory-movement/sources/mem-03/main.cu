// chunkedStreamPipelineIncrement: one large buffer split into STREAM_COUNT
// chunks, each with its own pinned staging buffers and its own CUDA stream
// so that chunk c's H2D copy, chunk c's kernel, and chunk c's D2H copy can
// overlap with the neighboring chunks' copies/kernels on other streams --
// vs. a single monolithic stream that copies/processes/copies back the
// entire buffer in one shot.
//
// The __global__ kernel below is reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/simpleMultiCopy/simpleMultiCopy.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Both paths below launch the identical `incKernel` (with inner_reps fixed
// at 1) over every index in [0, N) exactly once and write the identical
// per-element result; only *how the buffer is divided up and moved* differs:
//
// "chunked / pipelined": the N-element buffer is split into STREAM_COUNT=4
//   equal chunks. Each chunk gets its own pinned (`cudaHostAlloc`) input/
//   output staging buffers, its own device input/output buffers, and its
//   own `cudaStream_t`. For every chunk c, an H2D `cudaMemcpyAsync`, an
//   `incKernel` launch, and a D2H `cudaMemcpyAsync` are all enqueued on
//   `stream[c]` (exactly the enqueue order used by upstream's
//   `processWithStreams`), so a GPU with independent copy engines can
//   overlap chunk c's copy with chunk (c-1)'s or (c+1)'s kernel/copy.
//
// "single / monolithic": the same N-element buffer, also pinned, is copied
//   H2D in one `cudaMemcpyAsync`, processed by one `incKernel` launch over
//   the full N, and copied D2H in one `cudaMemcpyAsync`, all on a single
//   stream -- there is nothing else in flight to overlap with.
//
// Both paths use pinned host memory and asynchronous copies throughout, so
// the only variable being isolated is the chunk count / stream count (1 vs
// 4), not pinned-vs-pageable memory (that contrast is covered by the
// separate `pinnedAsyncIncrement` case in this repo). Every index is
// touched by exactly one chunk in the chunked path and by the single pass
// in the monolithic path, with the exact same per-element arithmetic, so
// the two paths' outputs must be bit-for-bit identical.
//
// Dropped from the upstream sample: `cudaEvent_t`-based timing
// (`memcpy_h2d_time`, `memcpy_d2h_time`, `kernel_time`, `cycleDone`
// events), the `cudaDeviceGetAttribute(cudaDevAttrGpuOverlap, ...)` /
// `deviceProp.asyncEngineCount` overlap-capability queries, the
// `nreps`-repetition averaging loop, the GFLOPS-based device auto-select
// (`gpuGetMaxGflopsDeviceId`) and `helper_cuda`/`helper_functions`
// dependency, and the `SIMULATE_IO` host-side memcpy option -- none of
// that affects correctness, and this benchmark isolates the chunked/
// pipelined memory-movement technique itself, not timing or device
// selection.
//
// Deterministic input (replicated by reference.h):
//   in[i] = i, for i in [0, N)
//   inner_reps = 1 (fixed; upstream defaults to 5 repeated increments per
//   element, which would just add 5 instead of 1 -- fixing it at 1 keeps
//   the arithmetic itself trivial so the case isolates memory movement)
// N = 1 << 22 = 4,194,304 ints, STREAM_COUNT = 4 (chunk_size = 1,048,576,
// evenly divisible by both STREAM_COUNT and the 512-thread block width).
//
// Output: argv[1] (default output/output.txt), one int per line, the
// chunked/pipelined path's output buffer (out[i] = in[i] + 1 for all i).

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

// --- begin: verbatim from NVIDIA/cuda-samples simpleMultiCopy.cu ---

__global__ void incKernel(int *g_out, int *g_in, int N, int inner_reps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        for (int i = 0; i < inner_reps; ++i) {
            g_out[idx] = g_in[idx] + 1;
        }
    }
}

// --- end: verbatim from NVIDIA/cuda-samples ---

#define STREAM_COUNT 4

int main(int argc, char **argv) {
  const long N = 1L << 22;             // 4,194,304 ints total
  const int inner_reps = 1;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  const long chunk_size = N / STREAM_COUNT;  // 1,048,576, evenly divisible
  const size_t chunk_bytes = (size_t)chunk_size * sizeof(int);
  const size_t total_bytes = (size_t)N * sizeof(int);

  const dim3 block(512);
  const dim3 grid_chunk((unsigned)(chunk_size / block.x));
  const dim3 grid_mono((unsigned)(N / block.x));

  // ---------------- path 1: chunked / pipelined across 4 streams ----------------

  int *h_in_chunk[STREAM_COUNT];
  int *h_out_chunk[STREAM_COUNT];
  int *d_in_chunk[STREAM_COUNT];
  int *d_out_chunk[STREAM_COUNT];
  cudaStream_t stream[STREAM_COUNT];

  for (int c = 0; c < STREAM_COUNT; ++c) {
    CHECK(cudaHostAlloc(&h_in_chunk[c], chunk_bytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc(&h_out_chunk[c], chunk_bytes, cudaHostAllocDefault));
    CHECK(cudaMalloc(&d_in_chunk[c], chunk_bytes));
    CHECK(cudaMalloc(&d_out_chunk[c], chunk_bytes));
    CHECK(cudaStreamCreate(&stream[c]));

    for (long j = 0; j < chunk_size; ++j) {
      h_in_chunk[c][j] = gen_in((long)c * chunk_size + j);
    }
  }

  // Enqueue H2D copy, kernel, and D2H copy for every chunk on that chunk's
  // own stream (same enqueue order as upstream's processWithStreams), so
  // independent copy engines can overlap different chunks' transfers with
  // each other's kernel execution.
  for (int c = 0; c < STREAM_COUNT; ++c) {
    CHECK(cudaMemcpyAsync(d_in_chunk[c], h_in_chunk[c], chunk_bytes,
                           cudaMemcpyHostToDevice, stream[c]));
    incKernel<<<grid_chunk, block, 0, stream[c]>>>(d_out_chunk[c], d_in_chunk[c],
                                                    (int)chunk_size, inner_reps);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpyAsync(h_out_chunk[c], d_out_chunk[c], chunk_bytes,
                           cudaMemcpyDeviceToHost, stream[c]));
  }

  for (int c = 0; c < STREAM_COUNT; ++c) {
    CHECK(cudaStreamSynchronize(stream[c]));
  }

  int *h_out_pipelined = (int *)malloc(total_bytes);
  for (int c = 0; c < STREAM_COUNT; ++c) {
    for (long j = 0; j < chunk_size; ++j) {
      h_out_pipelined[(long)c * chunk_size + j] = h_out_chunk[c][j];
    }
  }

  // ---------------- path 2: single / monolithic stream ----------------

  int *h_in_mono, *h_out_mono;
  CHECK(cudaHostAlloc(&h_in_mono, total_bytes, cudaHostAllocDefault));
  CHECK(cudaHostAlloc(&h_out_mono, total_bytes, cudaHostAllocDefault));
  for (long i = 0; i < N; ++i) h_in_mono[i] = gen_in(i);

  int *d_in_mono, *d_out_mono;
  CHECK(cudaMalloc(&d_in_mono, total_bytes));
  CHECK(cudaMalloc(&d_out_mono, total_bytes));

  cudaStream_t mono_stream;
  CHECK(cudaStreamCreate(&mono_stream));

  CHECK(cudaMemcpyAsync(d_in_mono, h_in_mono, total_bytes,
                         cudaMemcpyHostToDevice, mono_stream));
  incKernel<<<grid_mono, block, 0, mono_stream>>>(d_out_mono, d_in_mono, (int)N,
                                                   inner_reps);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpyAsync(h_out_mono, d_out_mono, total_bytes,
                         cudaMemcpyDeviceToHost, mono_stream));
  CHECK(cudaStreamSynchronize(mono_stream));

  // CPU reference check.
  int *h_in_all = (int *)malloc(total_bytes);
  int *href = (int *)malloc(total_bytes);
  for (long i = 0; i < N; ++i) h_in_all[i] = gen_in(i);
  reference_inc(h_in_all, href, N);

  long mismatches_pipelined = 0, mismatches_mono = 0;
  for (long i = 0; i < N; ++i) {
    if (h_out_pipelined[i] != href[i]) mismatches_pipelined++;
    if (h_out_mono[i] != href[i]) mismatches_mono++;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (long i = 0; i < N; ++i) fprintf(f, "%d\n", h_out_pipelined[i]);
  fclose(f);

  printf("chunkedStreamPipelineIncrement done: N=%ld, STREAM_COUNT=%d, "
         "chunk_size=%ld, mismatches(pipelined)=%ld, mismatches(mono)=%ld -> %s\n",
         N, STREAM_COUNT, chunk_size, mismatches_pipelined, mismatches_mono, out);
  printf("%s\n", (mismatches_pipelined == 0 && mismatches_mono == 0) ? "PASS" : "FAIL");

  for (int c = 0; c < STREAM_COUNT; ++c) {
    cudaFree(d_in_chunk[c]);
    cudaFree(d_out_chunk[c]);
    cudaFreeHost(h_in_chunk[c]);
    cudaFreeHost(h_out_chunk[c]);
    cudaStreamDestroy(stream[c]);
  }
  cudaFree(d_in_mono);
  cudaFree(d_out_mono);
  cudaFreeHost(h_in_mono);
  cudaFreeHost(h_out_mono);
  cudaStreamDestroy(mono_stream);
  free(h_out_pipelined);
  free(h_in_all);
  free(href);
  return 0;
}
