// PageRank power iteration: map (scatter outbound rank) + reduce (gather and
// damp) kernel pair over a link matrix.
//
// Extracted from HeCBench src/page-rank-cuda/main.cu (origin: Ostrich/McGill
// benchmark suite, MIT-licensed upstream header preserved in the snapshot).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5.
// The map and reduce kernels below are upstream device code verbatim. The
// host harness replaces upstream's rand()-based random_pages() with a
// deterministic hash-derived link matrix and runs a fixed iteration count.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

#define D_FACTOR (0.85f)
#define BLOCK_SIZE 256

// ---- upstream kernels (verbatim) --------------------------------------------
__global__
void map(const int *__restrict__ pages,
         const float *__restrict__ page_ranks,
               float *__restrict__ maps,
         const unsigned int *__restrict__ noutlinks,
         const int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j;
  if(i < n){
    float outbound_rank = page_ranks[i]/(float)noutlinks[i];
    for(j=0; j<n; ++j){
      maps[(size_t)i*n+j] = pages[(size_t)i*n+j]*outbound_rank;
    }
  }
}

__global__
void reduce(      float *__restrict__ page_ranks,
            const float *__restrict__ maps,
            const int n,
                  float *__restrict__ dif)
{

  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i;
  float new_rank;
  float old_rank;

  if(j<n){
    old_rank = page_ranks[j];
    new_rank = 0.0f;
    for(i=0; i< n; ++i){
      new_rank += maps[(size_t)i*n + j];
    }

    new_rank = ((1.f-D_FACTOR)/n)+(D_FACTOR*new_rank);
    dif[j] = fmaxf(fabsf(new_rank - old_rank), dif[j]);
    page_ranks[j] = new_rank;
  }
}
// ---- end upstream kernels ----------------------------------------------------

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const int n = 1024;
  const int iters = 5;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  int *pages = (int*)calloc((size_t)n * n, sizeof(int));
  unsigned int *noutlinks = (unsigned int*)calloc(n, sizeof(unsigned int));
  float *page_ranks = (float*)malloc(n * sizeof(float));
  float *difs = (float*)malloc(n * sizeof(float));

  // Deterministic link matrix (~1% density) with a guaranteed outlink per
  // page (upstream's random_pages() also enforces >= 1 outlink).
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      int link = (i != j) && (h01((unsigned)(i * n + j), 7) < 0.01f);
      if (j == (i + 1) % n) link = 1;
      pages[(size_t)i * n + j] = link;
      noutlinks[i] += link;
    }
    page_ranks[i] = 1.0f / (float)n;
  }

  int *d_pages; float *d_maps, *d_ranks, *d_difs; unsigned int *d_nout;
  CK(cudaMalloc(&d_pages, (size_t)n * n * sizeof(int)));
  CK(cudaMalloc(&d_maps, (size_t)n * n * sizeof(float)));
  CK(cudaMalloc(&d_ranks, n * sizeof(float)));
  CK(cudaMalloc(&d_difs, n * sizeof(float)));
  CK(cudaMalloc(&d_nout, n * sizeof(unsigned int)));
  CK(cudaMemcpy(d_pages, pages, (size_t)n * n * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_ranks, page_ranks, n * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_nout, noutlinks, n * sizeof(unsigned int), cudaMemcpyHostToDevice));
  CK(cudaMemset(d_difs, 0, n * sizeof(float)));

  size_t block_size = n < BLOCK_SIZE ? n : BLOCK_SIZE;
  size_t num_blocks = (n + block_size - 1) / block_size;
  for (int t = 0; t < iters; ++t) {
    map<<<dim3(num_blocks), dim3(block_size)>>>(d_pages, d_ranks, d_maps, d_nout, n);
    reduce<<<dim3(num_blocks), dim3(block_size)>>>(d_ranks, d_maps, n, d_difs);
  }
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(page_ranks, d_ranks, n * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(difs, d_difs, n * sizeof(float), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", page_ranks[i]);
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", difs[i]);
  fclose(f);

  cudaFree(d_pages); cudaFree(d_maps); cudaFree(d_ranks); cudaFree(d_difs); cudaFree(d_nout);
  free(pages); free(noutlinks); free(page_ranks); free(difs);
  return 0;
}
