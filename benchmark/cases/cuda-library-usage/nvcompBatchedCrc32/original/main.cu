// nvCOMP batched CRC32: PKZIP CRC32 checksums of data chunks on the GPU.
//
// Extracted from nvCOMP examples/nvcomp_crc32.cu (snapshot lib-06 in
// benchmark/collection/cuda-library-usage/sources; Apache-2.0, NVIDIA).
// The nvCOMP API sequence is upstream's verbatim: heuristic kernel config
// via nvcompBatchedCRC32GetHeuristicConf, then nvcompBatchedCRC32Async over
// per-chunk device pointers/sizes; the reverse() and cpu_crc32() reference
// helpers are upstream code verbatim and validate in-binary like upstream.
// The harness replaces file input + BatchData helpers with deterministic
// hash-generated chunks and plain device arrays, and dumps the checksums.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include "nvcomp/crc32.h"

#define CUDA_CHECK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(2);}}

// ---- upstream reference helpers (verbatim) -----------------------------------
static uint32_t reverse(uint32_t x)
{
  x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
  x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
  x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
  x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));

  return((x >> 16) | (x << 16));
}

static uint32_t cpu_crc32(const nvcompCRC32Spec_t& spec, size_t n, const void *m_)
{
  const unsigned char *m = static_cast<const unsigned char *>(m_);
  uint32_t crc = spec.init;

  while (n--) {
      crc ^= spec.ref_in ? reverse(*m++) : (*m++ << 24);
      for (int i = 0; i < 8; i++) {
          crc = (crc << 1) ^ ((crc & 0x80000000) ? spec.poly : 0);
      }
  }

  if (spec.ref_out) {
      crc = reverse(crc);
  }

  return crc ^ spec.xorout;
}
// ---- end upstream helpers -----------------------------------------------------

static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char* argv[])
{
  const size_t chunk_size = 4096;
  const size_t chunk_count = 8;
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";

  // Deterministic input chunks (replaces upstream file input).
  std::vector<std::vector<unsigned char>> chunks(chunk_count);
  for (size_t c = 0; c < chunk_count; ++c) {
    chunks[c].resize(chunk_size);
    for (size_t i = 0; i < chunk_size; ++i)
      chunks[c][i] = (unsigned char)(h01((unsigned)(c * chunk_size + i), 71) * 256.0f);
  }

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Per-chunk device buffers plus device arrays of pointers and sizes
  // (replaces upstream's BatchData helper).
  std::vector<void*> h_ptrs(chunk_count);
  std::vector<size_t> h_sizes(chunk_count, chunk_size);
  for (size_t c = 0; c < chunk_count; ++c) {
    CUDA_CHECK(cudaMalloc(&h_ptrs[c], chunk_size));
    CUDA_CHECK(cudaMemcpy(h_ptrs[c], chunks[c].data(), chunk_size, cudaMemcpyHostToDevice));
  }
  void **d_ptrs; size_t *d_sizes; uint32_t *d_crc32;
  CUDA_CHECK(cudaMalloc(&d_ptrs, chunk_count * sizeof(void*)));
  CUDA_CHECK(cudaMalloc(&d_sizes, chunk_count * sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&d_crc32, chunk_count * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_ptrs, h_ptrs.data(), chunk_count * sizeof(void*), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sizes, h_sizes.data(), chunk_count * sizeof(size_t), cudaMemcpyHostToDevice));

  // Heuristically determine the optimal kernel configuration.
  nvcompCRC32KernelConf_t kernel_conf{};
  if (nvcompBatchedCRC32GetHeuristicConf(
        nvcompCRC32IgnoredInputChunkBytes,
        chunk_count,
        &kernel_conf,
        chunk_size,
        stream) != nvcompSuccess) {
    fprintf(stderr, "ERROR: nvcompBatchedCRC32GetHeuristicConf() not successful\n");
    return 2;
  }

  nvcompBatchedCRC32Opts_t opts{nvcompCRC32, kernel_conf, {}};

  if (nvcompBatchedCRC32Async(
        d_ptrs,
        d_sizes,
        chunk_count,
        d_crc32,
        opts,
        nvcompCRC32OnlySegment,
        /*device_statuses=*/nullptr,
        stream) != nvcompSuccess) {
    fprintf(stderr, "ERROR: nvcompBatchedCRC32Async() not successful\n");
    return 2;
  }

  std::vector<uint32_t> crc32_values(chunk_count);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(crc32_values.data(), d_crc32, chunk_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  // Validate against the upstream CPU reference, like the original example.
  for (size_t c = 0; c < chunk_count; ++c) {
    uint32_t ref = cpu_crc32(nvcompCRC32, chunk_size, chunks[c].data());
    if (ref != crc32_values[c]) {
      fprintf(stderr, "chunk %zu: GPU %08x != CPU %08x\n", c, crc32_values[c], ref);
      return 1;
    }
  }

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (size_t c = 0; c < chunk_count; ++c) fprintf(f, "%u\n", crc32_values[c]);
  fclose(f);

  for (size_t c = 0; c < chunk_count; ++c) cudaFree(h_ptrs[c]);
  cudaFree(d_ptrs); cudaFree(d_sizes); cudaFree(d_crc32);
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
