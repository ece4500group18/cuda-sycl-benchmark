// conjugateGradient: solve A x = b with CG, matrix-free.
// A is the diagonally-dominant 1D operator (diag 4, off-diagonals -1), SPD.
// N = 1024, up to K = 200 iterations. b[i] = h01(i, 123). Output: x (N values).
// Multi-kernel pipeline: matvec + dot (shared-memory reduction) + axpy.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void matvec(const float *p, float *Ap, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float v = 4.0f * p[i];
    if (i > 0) v -= p[i - 1];
    if (i < N - 1) v -= p[i + 1];
    Ap[i] = v;
  }
}
__global__ void dotK(const float *a, const float *b, float *res, int N) {
  extern __shared__ float s[];
  int t = threadIdx.x;
  float v = 0.0f;
  for (int i = t; i < N; i += blockDim.x) v += a[i] * b[i];
  s[t] = v; __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) { if (t < st) s[t] += s[t + st]; __syncthreads(); }
  if (t == 0) res[0] = s[0];
}
__global__ void axpy(float *y, float a, const float *x, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < N) y[i] += a * x[i];
}
__global__ void xpby(float *p, const float *r, float beta, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < N) p[i] = r[i] + beta * p[i];
}

static float dot(const float *a, const float *b, float *dres, int N) {
  dotK<<<1, 256, 256 * sizeof(float)>>>(a, b, dres, N);
  float h; cudaMemcpy(&h, dres, sizeof(float), cudaMemcpyDeviceToHost); return h;
}

int main(int argc, char **argv) {
  const int N = 1024, K = 200;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)N * sizeof(float);
  float *hb = (float *)malloc(bytes), *hx = (float *)malloc(bytes);
  for (int i = 0; i < N; ++i) hb[i] = h01(i, 123);

  float *db, *dx, *dr, *dp, *dAp, *dres;
  CK(cudaMalloc(&db, bytes)); CK(cudaMalloc(&dx, bytes)); CK(cudaMalloc(&dr, bytes));
  CK(cudaMalloc(&dp, bytes)); CK(cudaMalloc(&dAp, bytes)); CK(cudaMalloc(&dres, sizeof(float)));
  CK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));
  CK(cudaMemset(dx, 0, bytes));
  CK(cudaMemcpy(dr, hb, bytes, cudaMemcpyHostToDevice));  // r = b - A*0 = b
  CK(cudaMemcpy(dp, hb, bytes, cudaMemcpyHostToDevice));  // p = r
  int tpb = 256, blocks = (N + tpb - 1) / tpb;

  float rsold = dot(dr, dr, dres, N);
  for (int k = 0; k < K; ++k) {
    matvec<<<blocks, tpb>>>(dp, dAp, N);
    float pAp = dot(dp, dAp, dres, N);
    float alpha = rsold / pAp;
    axpy<<<blocks, tpb>>>(dx, alpha, dp, N);
    axpy<<<blocks, tpb>>>(dr, -alpha, dAp, N);
    float rsnew = dot(dr, dr, dres, N);
    if (sqrtf(rsnew) < 1e-7f) { rsold = rsnew; break; }
    xpby<<<blocks, tpb>>>(dp, dr, rsnew / rsold, N);
    rsold = rsnew;
  }
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hx, dx, bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < N; ++i) fprintf(f, "%.9g\n", hx[i]); fclose(f);
  printf("conjugateGradient done: N=%d final_res=%.3e -> %s\n", N, sqrtf(rsold), out);
  cudaFree(db); cudaFree(dx); cudaFree(dr); cudaFree(dp); cudaFree(dAp); cudaFree(dres);
  free(hb); free(hx);
  return 0;
}
