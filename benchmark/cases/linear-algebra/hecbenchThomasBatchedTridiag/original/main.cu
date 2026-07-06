// cuThomasBatch: batched tridiagonal solve (Thomas algorithm), one system
// per thread over interleaved storage.
//
// Extracted from HeCBench src/thomas-cuda/cuThomasBatch.cu (origin:
// cuThomasBatch, Barcelona Supercomputing Center).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5 (BSD-3-Clause).
// The cuThomasBatch kernel is upstream device code verbatim (double
// precision, interleaved layout: element i of system s lives at
// i*BATCHCOUNT+s). The harness builds deterministic diagonally-dominant
// systems and dumps the solutions.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

// ---- upstream kernel (verbatim) ----------------------------------------------
__global__ void cuThomasBatch(const double *__restrict__ L,
                              const double *__restrict__ D,
                                    double *__restrict__ U,
                                    double *__restrict__ RHS,
                              const int M,
                              const int BATCHCOUNT)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  if(tid < BATCHCOUNT) {

    int first = tid;
    int last  = BATCHCOUNT*(M-1)+tid;

    U[first] /= D[first];
    RHS[first] /= D[first];

    for (int i = first + BATCHCOUNT; i < last; i+=BATCHCOUNT) {
      U[i] /= D[i] - L[i] * U[i-BATCHCOUNT];
      RHS[i] = ( RHS[i] - L[i] * RHS[i-BATCHCOUNT] ) /
        ( D[i] - L[i] * U[i-BATCHCOUNT] );
    }

    RHS[last] = ( RHS[last] - L[last] * RHS[last-BATCHCOUNT] ) /
      ( D[last] - L[last] * U[last-BATCHCOUNT] );

    for (int i = last-BATCHCOUNT; i >= first; i-=BATCHCOUNT) {
      RHS[i] -= U[i] * RHS[i+BATCHCOUNT];
    }
  }
}
// ---- end upstream kernel -------------------------------------------------------

static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const int M = 64, BATCH = 1024;
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t n = (size_t)M * BATCH;

  double *L = (double*)malloc(n * sizeof(double));
  double *D = (double*)malloc(n * sizeof(double));
  double *U = (double*)malloc(n * sizeof(double));
  double *RHS = (double*)malloc(n * sizeof(double));
  // Interleaved layout; diagonally dominant so the solve is stable.
  for (size_t k = 0; k < n; ++k) {
    L[k] = 2.0 * (double)h01((unsigned)k, 101) - 1.0;
    U[k] = 2.0 * (double)h01((unsigned)k, 102) - 1.0;
    D[k] = 4.0 + (double)h01((unsigned)k, 103);
    RHS[k] = 2.0 * (double)h01((unsigned)k, 104) - 1.0;
  }

  double *d_L, *d_D, *d_U, *d_RHS;
  CK(cudaMalloc(&d_L, n * sizeof(double)));
  CK(cudaMalloc(&d_D, n * sizeof(double)));
  CK(cudaMalloc(&d_U, n * sizeof(double)));
  CK(cudaMalloc(&d_RHS, n * sizeof(double)));
  CK(cudaMemcpy(d_L, L, n * sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_D, D, n * sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_U, U, n * sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_RHS, RHS, n * sizeof(double), cudaMemcpyHostToDevice));

  const int BlockSize = 128;
  cuThomasBatch<<<(BATCH / BlockSize) + 1, BlockSize>>>(d_L, d_D, d_U, d_RHS, M, BATCH);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(RHS, d_RHS, n * sizeof(double), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (size_t k = 0; k < n; ++k) fprintf(f, "%.12g\n", RHS[k]);
  fclose(f);

  cudaFree(d_L); cudaFree(d_D); cudaFree(d_U); cudaFree(d_RHS);
  free(L); free(D); free(U); free(RHS);
  return 0;
}
