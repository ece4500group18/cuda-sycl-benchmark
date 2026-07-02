// histogramSharedPrivatization: per-block privatized shared-memory 64-bin
// histogram (with a bank-conflict-avoiding thread-index permutation),
// reduced across partial histograms with a second kernel, vs. a naive
// single-pass kernel that does one atomicAdd straight into global memory
// per input byte.
//
// The two __global__ kernels `histogram64Kernel` and
// `mergeHistogram64Kernel` (plus their `addByte`/`addWord` device helpers)
// below are reproduced, unmodified, from:
//   NVIDIA/cuda-samples,
//   cpp/2_Concepts_and_Techniques/histogram/histogram64.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// "histogram64Kernel": each thread block owns a private HISTOGRAM64_BIN_COUNT
//   (=64) x HISTOGRAM64_THREADBLOCK_SIZE (=64)-byte region of __shared__
//   memory -- one 64-bin sub-histogram per thread -- and every thread
//   accumulates counts for its grid-strided slice of the input entirely in
//   shared memory (`addByte`/`addWord`), using a bit-permuted thread index
//   (`threadPos`) so that the SHARED_MEMORY_BANKS=16 threads of a
//   half-warp land on 16 consecutive shared-memory banks instead of
//   colliding on one. Only once all input has been consumed does each
//   block reduce its 64 per-thread sub-histograms into one 64-bin partial
//   histogram and write *that* (one global-memory write per bin per
//   block) to `d_PartialHistograms`.
// "mergeHistogram64Kernel": one thread block per output bin; each block
//   sums the same bin across all of the per-block partial histograms
//   (a small parallel tree reduction in shared memory) into the final
//   64-bin `d_Histogram`.
// "naiveHistogramKernel" (new code, not from upstream): every thread reads
//   one input byte at a time and calls `atomicAdd(&d_Histogram[bin], 1)`
//   directly on the single global-memory output histogram -- no shared
//   memory, no privatization, every one of the 1,048,576 byte updates is
//   a global atomic contending with every other thread that happens to
//   land on the same bin.
//
// Both GPU paths compute the exact same 64-bin histogram over the exact
// same input bytes; only *where* the read-modify-write count updates
// happen (private per-thread shared memory, merged in two stages, vs.
// directly in global memory) differs. Since histogram counts are integers
// under addition -- commutative and associative -- the total count in
// each bin is independent of update order, thread scheduling, or block
// interleaving, so both paths, and a single sequential CPU pass, must
// agree exactly.
//
// Deterministic input (replicated by reference.h):
//   data[i] = (i * 31 + 7) % 256, for i in [0, byteCount), byteCount = 1<<20
//   (a multiple of sizeof(uint4)=16, as histogram64Kernel expects its
//   input packed as uint4 words). 64 bins; bin b of byte value v is
//   b = (v >> 2) & 0x3F (the top 6 bits of the byte).
//
// Output: argv[1] (default output/cuda_output.txt), 64 lines, the final
// bin counts from the privatized-shared-memory + merge path.

#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "reference.h"

namespace cg = cooperative_groups;

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                          \
      return 2;                                                             \
    }                                                                        \
  } while (0)

// --- begin: verbatim from NVIDIA/cuda-samples histogram64.cu (plus the
// constants/macros/typedefs it relies on from the companion
// histogram_common.h in the same upstream directory) ---

typedef unsigned int  uint;
typedef unsigned char uchar;

#define HISTOGRAM64_BIN_COUNT 64
#define SHARED_MEMORY_BANKS   16
#define HISTOGRAM64_THREADBLOCK_SIZE (4 * SHARED_MEMORY_BANKS)
#define UMUL(a, b)    ((a) * (b))
#define UMAD(a, b, c) (UMUL((a), (b)) + (c))

// Data type used for input data fetches
typedef uint4 data_t;

////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute gridDim.x partial histograms
////////////////////////////////////////////////////////////////////////////////
// Count a byte into shared-memory storage
inline __device__ void addByte(uchar *s_ThreadBase, uint data)
{
    s_ThreadBase[UMUL(data, HISTOGRAM64_THREADBLOCK_SIZE)]++;
}

// Count four bytes of a word
inline __device__ void addWord(uchar *s_ThreadBase, uint data)
{
    // Only higher 6 bits of each byte matter, as this is a 64-bin histogram
    addByte(s_ThreadBase, (data >> 2) & 0x3FU);
    addByte(s_ThreadBase, (data >> 10) & 0x3FU);
    addByte(s_ThreadBase, (data >> 18) & 0x3FU);
    addByte(s_ThreadBase, (data >> 26) & 0x3FU);
}

__global__ void histogram64Kernel(uint *d_PartialHistograms, data_t *d_Data, uint dataCount)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Encode thread index in order to avoid bank conflicts in s_Hist[] access:
    // each group of SHARED_MEMORY_BANKS threads accesses consecutive shared
    // memory banks
    // and the same bytes [0..3] within the banks
    // Because of this permutation block size should be a multiple of 4 *
    // SHARED_MEMORY_BANKS
    const uint threadPos = ((threadIdx.x & ~(SHARED_MEMORY_BANKS * 4 - 1)) << 0)
                         | ((threadIdx.x & (SHARED_MEMORY_BANKS - 1)) << 2)
                         | ((threadIdx.x & (SHARED_MEMORY_BANKS * 3)) >> 4);

    // Per-thread histogram storage
    __shared__ uchar s_Hist[HISTOGRAM64_THREADBLOCK_SIZE * HISTOGRAM64_BIN_COUNT];
    uchar           *s_ThreadBase = s_Hist + threadPos;

// Initialize shared memory (writing 32-bit words)
#pragma unroll

    for (uint i = 0; i < (HISTOGRAM64_BIN_COUNT / 4); i++) {
        ((uint *)s_Hist)[threadIdx.x + i * HISTOGRAM64_THREADBLOCK_SIZE] = 0;
    }

    // Read data from global memory and submit to the shared-memory histogram
    // Since histogram counters are byte-sized, every single thread can't do more
    // than 255 submission
    cg::sync(cta);

    for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x)) {
        data_t data = d_Data[pos];
        addWord(s_ThreadBase, data.x);
        addWord(s_ThreadBase, data.y);
        addWord(s_ThreadBase, data.z);
        addWord(s_ThreadBase, data.w);
    }

    // Accumulate per-thread histograms into per-block and write to global memory
    cg::sync(cta);

    if (threadIdx.x < HISTOGRAM64_BIN_COUNT) {
        uchar *s_HistBase = s_Hist + UMUL(threadIdx.x, HISTOGRAM64_THREADBLOCK_SIZE);

        uint sum = 0;
        uint pos = 4 * (threadIdx.x & (SHARED_MEMORY_BANKS - 1));

#pragma unroll

        for (uint i = 0; i < (HISTOGRAM64_THREADBLOCK_SIZE / 4); i++) {
            sum += s_HistBase[pos + 0] + s_HistBase[pos + 1] + s_HistBase[pos + 2] + s_HistBase[pos + 3];
            pos = (pos + 4) & (HISTOGRAM64_THREADBLOCK_SIZE - 1);
        }

        d_PartialHistograms[blockIdx.x * HISTOGRAM64_BIN_COUNT + threadIdx.x] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merge histogram64() output
// Run one threadblock per bin; each threadbock adds up the same bin counter
// from every partial histogram. Reads are uncoalesced, but mergeHistogram64
// takes only a fraction of total processing time
////////////////////////////////////////////////////////////////////////////////
#define MERGE_THREADBLOCK_SIZE 256

__global__ void mergeHistogram64Kernel(uint *d_Histogram, uint *d_PartialHistograms, uint histogramCount)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint  data[MERGE_THREADBLOCK_SIZE];

    uint sum = 0;

    for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE) {
        sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM64_BIN_COUNT];
    }

    data[threadIdx.x] = sum;

    for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        cg::sync(cta);

        if (threadIdx.x < stride) {
            data[threadIdx.x] += data[threadIdx.x + stride];
        }
    }

    if (threadIdx.x == 0) {
        d_Histogram[blockIdx.x] = data[0];
    }
}

// --- end: verbatim from NVIDIA/cuda-samples ---

// naiveHistogramKernel: new code written for this repository (not part of
// the upstream sample). One atomicAdd straight into global memory per
// input byte, no shared memory, no per-block privatization -- the
// baseline this case contrasts histogram64Kernel/mergeHistogram64Kernel
// against.
__global__ void naiveHistogramKernel(uint *d_Histogram, const uchar *d_Data, uint byteCount)
{
    for (uint pos = blockIdx.x * blockDim.x + threadIdx.x; pos < byteCount;
         pos += blockDim.x * gridDim.x) {
        uint bin = ((uint)d_Data[pos] >> 2) & 0x3FU;
        atomicAdd(&d_Histogram[bin], 1);
    }
}

int main(int argc, char **argv) {
  const long byteCount = 1L << 20;  // 1,048,576 bytes; multiple of sizeof(uint4)=16
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  uchar *hData = (uchar *)malloc(byteCount);
  for (long i = 0; i < byteCount; ++i) hData[i] = gen_byte(i);

  uchar *dData;
  CHECK(cudaMalloc(&dData, byteCount));
  CHECK(cudaMemcpy(dData, hData, byteCount, cudaMemcpyHostToDevice));

  // --- privatized shared-memory histogram + merge (matches upstream
  // histogram64() host launch geometry in histogram64.cu) ---
  const uint dataCount = (uint)(byteCount / (long)sizeof(data_t));  // # of uint4 words
  // iDivUp(byteCount, HISTOGRAM64_THREADBLOCK_SIZE * iSnapDown(255, sizeof(data_t)))
  const uint snapped255 = 255U - (255U % (uint)sizeof(data_t));  // = 240
  const uint denom = HISTOGRAM64_THREADBLOCK_SIZE * snapped255;
  const uint histogramCount = (uint)((byteCount % denom != 0) ? (byteCount / denom + 1)
                                                                : (byteCount / denom));

  uint *dPartialHistograms;
  uint *dHistPriv;
  CHECK(cudaMalloc(&dPartialHistograms, (size_t)histogramCount * HISTOGRAM64_BIN_COUNT * sizeof(uint)));
  CHECK(cudaMalloc(&dHistPriv, HISTOGRAM64_BIN_COUNT * sizeof(uint)));

  histogram64Kernel<<<histogramCount, HISTOGRAM64_THREADBLOCK_SIZE>>>(
      dPartialHistograms, (data_t *)dData, dataCount);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  mergeHistogram64Kernel<<<HISTOGRAM64_BIN_COUNT, MERGE_THREADBLOCK_SIZE>>>(
      dHistPriv, dPartialHistograms, histogramCount);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // --- naive single-pass global-atomic histogram over the same bytes ---
  uint *dHistNaive;
  CHECK(cudaMalloc(&dHistNaive, HISTOGRAM64_BIN_COUNT * sizeof(uint)));
  CHECK(cudaMemset(dHistNaive, 0, HISTOGRAM64_BIN_COUNT * sizeof(uint)));

  naiveHistogramKernel<<<256, 256>>>(dHistNaive, dData, (uint)byteCount);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  uint hHistPriv[HISTOGRAM64_BIN_COUNT];
  uint hHistNaive[HISTOGRAM64_BIN_COUNT];
  CHECK(cudaMemcpy(hHistPriv, dHistPriv, sizeof(hHistPriv), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hHistNaive, dHistNaive, sizeof(hHistNaive), cudaMemcpyDeviceToHost));

  // CPU reference check.
  uint href[HISTOGRAM64_BIN_COUNT];
  reference_histogram64(hData, byteCount, href);

  long max_abs_err_priv = 0, max_abs_err_naive = 0;
  for (int b = 0; b < HISTOGRAM64_BIN_COUNT; ++b) {
    long d1 = (long)hHistPriv[b] - (long)href[b];
    long d2 = (long)hHistNaive[b] - (long)href[b];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_priv) max_abs_err_priv = d1;
    if (d2 > max_abs_err_naive) max_abs_err_naive = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int b = 0; b < HISTOGRAM64_BIN_COUNT; ++b) fprintf(f, "%u\n", hHistPriv[b]);
  fclose(f);

  printf("histogramSharedPrivatization done: byteCount=%ld, bins=%d, "
         "max_abs_error(privatized vs ref)=%ld, max_abs_error(naive vs ref)=%ld -> %s\n",
         byteCount, HISTOGRAM64_BIN_COUNT, max_abs_err_priv, max_abs_err_naive, out);
  printf("%s\n", (max_abs_err_priv == 0 && max_abs_err_naive == 0) ? "PASS" : "FAIL");

  cudaFree(dData);
  cudaFree(dPartialHistograms);
  cudaFree(dHistPriv);
  cudaFree(dHistNaive);
  free(hData);
  return 0;
}
