// spmv: sparse matrix-vector y = A*x where A is the 1D Laplacian in CSR
// (diag 2, off-diagonals -1). N = 100000. x[i] = h01(i, 123).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void spmv(const int *rowptr, const int *colidx, const float *vals,
                     const float *x, float *y, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N) {
    float acc = 0.0f;
    for (int k = rowptr[row]; k < rowptr[row + 1]; ++k)
      acc += vals[k] * x[colidx[k]];
    y[row] = acc;
  }
}

int main(int argc, char **argv) {
  const int N = 100000;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  // Build CSR for the tridiagonal 1D Laplacian on the host.
  int *rowptr = (int *)malloc((size_t)(N + 1) * sizeof(int));
  int nnz = 0;
  for (int i = 0; i < N; ++i) { rowptr[i] = nnz; nnz += 3 - (i == 0) - (i == N - 1); }
  rowptr[N] = nnz;
  int *colidx = (int *)malloc((size_t)nnz * sizeof(int));
  float *vals = (float *)malloc((size_t)nnz * sizeof(float));
  float *hx = (float *)malloc((size_t)N * sizeof(float));
  float *hy = (float *)malloc((size_t)N * sizeof(float));
  int p = 0;
  for (int i = 0; i < N; ++i) {
    if (i > 0)     { colidx[p] = i - 1; vals[p++] = -1.0f; }
    colidx[p] = i; vals[p++] = 2.0f;
    if (i < N - 1) { colidx[p] = i + 1; vals[p++] = -1.0f; }
    hx[i] = h01(i, 123);
  }

  int *drow, *dcol; float *dval, *dx, *dy;
  CK(cudaMalloc(&drow, (size_t)(N + 1) * sizeof(int)));
  CK(cudaMalloc(&dcol, (size_t)nnz * sizeof(int)));
  CK(cudaMalloc(&dval, (size_t)nnz * sizeof(float)));
  CK(cudaMalloc(&dx, (size_t)N * sizeof(float)));
  CK(cudaMalloc(&dy, (size_t)N * sizeof(float)));
  CK(cudaMemcpy(drow, rowptr, (size_t)(N + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcol, colidx, (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dval, vals, (size_t)nnz * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dx, hx, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
  int tpb = 256, blocks = (N + tpb - 1) / tpb;
  spmv<<<blocks, tpb>>>(drow, dcol, dval, dx, dy, N);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy, dy, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < N; ++i) fprintf(f, "%.9g\n", hy[i]); fclose(f);
  printf("spmv done: N=%d nnz=%d -> %s\n", N, nnz, out);
  cudaFree(drow); cudaFree(dcol); cudaFree(dval); cudaFree(dx); cudaFree(dy);
  free(rowptr); free(colidx); free(vals); free(hx); free(hy);
  return 0;
}
