// Gaussian elimination forward substitution: fan1 (multiplier column) +
// fan2 (submatrix update) kernel pair launched per pivot column, followed
// by host back-substitution — the Rodinia two-kernel iterative pipeline.
//
// Extracted from HeCBench src/gaussian-cuda/gaussianElim.cu (origin: Rodinia
// gaussian, with Ke Wang's internal input generation).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5 (BSD-3-Clause).
// The fan1/fan2 kernels, the ForwardSub per-column launch loop and the
// BackSub host routine below are upstream code verbatim (upstream block
// geometry). The harness uses a deterministic diagonally-dominant system
// (no pivoting in this scheme) and dumps the solution vector.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(2);}}

#define BLOCK_SIZE_0 256
#define BLOCK_SIZE_1_X 16
#define BLOCK_SIZE_1_Y 16

// ---- upstream kernels (verbatim) --------------------------------------------
__global__ void
fan1 (const float*__restrict__ a,
            float*__restrict__ m,
      const int size, const int t)
{
  int globalId = blockDim.x * blockIdx.x + threadIdx.x;
  if (globalId < size-1-t) {
    m[size * (globalId + t + 1)+t] =
      a[size * (globalId + t + 1) + t] / a[size * t + t];
  }
}

__global__ void
fan2 (float*__restrict__ a,
      float*__restrict__ b,
      const float*__restrict__ m,
      const int size, const int t)
{
  int globalIdy = blockDim.x * blockIdx.x + threadIdx.x;
  int globalIdx = blockDim.y * blockIdx.y + threadIdx.y;
  if (globalIdx < size-1-t && globalIdy < size-t) {
    a[size*(globalIdx+1+t)+(globalIdy+t)] -=
      m[size*(globalIdx+1+t)+t] * a[size*t+(globalIdy+t)];

    if(globalIdy == 0){
      b[globalIdx+1+t] -=
        m[size*(globalIdx+1+t)+(globalIdy+t)] * b[t];
    }
  }
}
// ---- end upstream kernels ----------------------------------------------------

// ---- upstream host routines (verbatim) ---------------------------------------
void ForwardSub(float *a, float *b, float *m, int size, int timing) {

  dim3 blockDim_fan1 (BLOCK_SIZE_0);
  dim3 gridDim_fan1 ((size + BLOCK_SIZE_0 - 1) / BLOCK_SIZE_0);

  dim3 blockDim_fan2 (BLOCK_SIZE_1_Y, BLOCK_SIZE_1_X);
  dim3 gridDim_fan2 ((size + BLOCK_SIZE_1_Y - 1) / BLOCK_SIZE_1_Y,
                     (size + BLOCK_SIZE_1_X - 1) / BLOCK_SIZE_1_X);

  float *d_a, *d_b, *d_m;
  cudaMalloc((void**)&d_a, size*size*sizeof(float));
  cudaMalloc((void**)&d_b, size*sizeof(float));
  cudaMalloc((void**)&d_m, size*size*sizeof(float));

  cudaMemcpy(d_a, a, size*size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, m, size*size*sizeof(float), cudaMemcpyHostToDevice);

  for (int t=0; t<(size-1); t++) {
    fan1<<<gridDim_fan1, blockDim_fan1>>> (d_a, d_m, size, t);
    fan2<<<gridDim_fan2, blockDim_fan2>>> (d_a, d_b, d_m, size, t);
  }

  cudaDeviceSynchronize();
  (void)timing;

  cudaMemcpy(a, d_a, size*size*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(b, d_b, size*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(m, d_m, size*size*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_m);
}

void BackSub(float *a, float *b, float *finalVec, int size)
{
  // solve "bottom up"
  int i,j;
  for(i=0;i<size;i++){
    finalVec[size-i-1]=b[size-i-1];
    for(j=0;j<i;j++)
    {
      finalVec[size-i-1]-=*(a+size*(size-i-1)+(size-j-1)) * finalVec[size-j-1];
    }
    finalVec[size-i-1]=finalVec[size-i-1]/ *(a+size*(size-i-1)+(size-i-1));
  }
}
// ---- end upstream host routines -----------------------------------------------

static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const int size = 64;
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";

  float *a = (float*)malloc((size_t)size * size * sizeof(float));
  float *b = (float*)malloc(size * sizeof(float));
  float *m = (float*)calloc((size_t)size * size, sizeof(float));
  float *finalVec = (float*)malloc(size * sizeof(float));

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      float v = 2.0f * h01((unsigned)(i * size + j), 81) - 1.0f;
      if (i == j) v += (float)size;  // diagonally dominant, no pivoting needed
      a[i * size + j] = v;
    }
    b[i] = 2.0f * h01((unsigned)i, 82) - 1.0f;
  }

  ForwardSub(a, b, m, size, 0);
  BackSub(a, b, finalVec, size);

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (int i = 0; i < size; ++i) fprintf(f, "%.9g\n", finalVec[i]);
  fclose(f);

  free(a); free(b); free(m); free(finalVec);
  return 0;
}
