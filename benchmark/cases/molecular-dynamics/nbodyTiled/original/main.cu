// nbodyTiled: per-body gravitational acceleration with shared-memory tiling.
// N = 2048 bodies, unit mass, softening eps2 = 1e-4.
//   pos.x[i]=h01(i,11), pos.y[i]=h01(i,22), pos.z[i]=h01(i,33)
// Output: 3*N floats (ax, ay, az per body).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}
#define BS 256

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void nbody(const float4 *p, float *acc, int N) {
  __shared__ float4 sh[BS];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float4 pi = (i < N) ? p[i] : make_float4(0, 0, 0, 0);
  float ax = 0, ay = 0, az = 0;
  int tiles = (N + BS - 1) / BS;
  for (int tile = 0; tile < tiles; ++tile) {
    int j = tile * BS + threadIdx.x;
    sh[threadIdx.x] = (j < N) ? p[j] : make_float4(0, 0, 0, 0);
    __syncthreads();
    for (int k = 0; k < BS; ++k) {
      int jj = tile * BS + k;
      if (jj >= N) break;
      float dx = sh[k].x - pi.x, dy = sh[k].y - pi.y, dz = sh[k].z - pi.z;
      float dist2 = dx * dx + dy * dy + dz * dz + 1e-4f;
      float inv = rsqrtf(dist2);
      float inv3 = inv * inv * inv;
      float f = sh[k].w * inv3;
      ax += f * dx; ay += f * dy; az += f * dz;
    }
    __syncthreads();
  }
  if (i < N) { acc[3 * i] = ax; acc[3 * i + 1] = ay; acc[3 * i + 2] = az; }
}

int main(int argc, char **argv) {
  const int N = 2048;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  float4 *hp = (float4 *)malloc((size_t)N * sizeof(float4));
  for (int i = 0; i < N; ++i)
    hp[i] = make_float4(h01(i, 11), h01(i, 22), h01(i, 33), 1.0f);
  float *hacc = (float *)malloc((size_t)3 * N * sizeof(float));

  float4 *dp; float *dacc;
  CK(cudaMalloc(&dp, (size_t)N * sizeof(float4)));
  CK(cudaMalloc(&dacc, (size_t)3 * N * sizeof(float)));
  CK(cudaMemcpy(dp, hp, (size_t)N * sizeof(float4), cudaMemcpyHostToDevice));
  nbody<<<(N + BS - 1) / BS, BS>>>(dp, dacc, N);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hacc, dacc, (size_t)3 * N * sizeof(float), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < 3 * N; ++i) fprintf(f, "%.9g\n", hacc[i]); fclose(f);
  printf("nbodyTiled done: N=%d -> %s\n", N, out);
  cudaFree(dp); cudaFree(dacc); free(hp); free(hacc);
  return 0;
}
