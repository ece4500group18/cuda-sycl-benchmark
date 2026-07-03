#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 2; \
} } while (0)

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__host__ __device__ static inline float hs(unsigned i, unsigned s) {
  return 2.0f * h01(i, s) - 1.0f;
}

static void write_vec(const char *path, const float *data, int n) {
  FILE *f = fopen(path, "w");
  if (!f) { fprintf(stderr, "open %s\n", path); exit(2); }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", data[i]);
  fclose(f);
}

__global__ void avg_pool2d(const float *x, float *y, int n, int c, int h, int w) {
  int oh = h / 2, ow = w / 2;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * oh * ow;
  if (idx < total) {
    int tmp = idx;
    int ox = tmp % ow; tmp /= ow;
    int oy = tmp % oh; tmp /= oh;
    int ch = tmp % c; int batch = tmp / c;
    int base = ((batch * c + ch) * h + 2 * oy) * w + 2 * ox;
    y[idx] = 0.25f * (x[base] + x[base + 1] + x[base + w] + x[base + w + 1]);
  }
}

int main(int argc, char **argv) {
  const int n=4, c=3, h=64, w=64, in_n=n*c*h*w, out_n=n*c*(h/2)*(w/2);
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  float *hx=(float*)malloc((size_t)in_n*sizeof(float)), *hy=(float*)malloc((size_t)out_n*sizeof(float));
  for (int i=0;i<in_n;++i) hx[i] = hs(i, 123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)in_n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)out_n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)in_n*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(out_n+tpb-1)/tpb; avg_pool2d<<<grid,tpb>>>(dx,dy,n,c,h,w);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)out_n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, out_n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
