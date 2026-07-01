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

__host__ __device__ static inline float hs(unsigned i, unsigned s) {
  return 2.0f * h01(i, s) - 1.0f;
}

static void write_vec(const char *path, const float *data, int n) {
  FILE *f = fopen(path, "w");
  if (!f) { fprintf(stderr, "open %s\n", path); exit(2); }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", data[i]);
  fclose(f);
}

__global__ void particle_update(const float *x, const float *v, const float *a, float *out, int n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float vn = v[i] + a[i] * dt;
    float xn = x[i] + vn * dt;
    out[2*i] = xn; out[2*i+1] = vn;
  }
}

int main(int argc, char **argv) {
  const int n = 262144; const float dt = 0.01f;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hx=(float*)malloc(bytes), *hv=(float*)malloc(bytes), *ha=(float*)malloc(bytes), *hy=(float*)malloc((size_t)2*n*sizeof(float));
  for (int i=0;i<n;++i) { hx[i]=hs(i,1); hv[i]=0.1f*hs(i,2); ha[i]=0.01f*hs(i,3); }
  float *dx,*dv,*da,*dy; CK(cudaMalloc(&dx,bytes)); CK(cudaMalloc(&dv,bytes)); CK(cudaMalloc(&da,bytes)); CK(cudaMalloc(&dy,(size_t)2*n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv,hv,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(da,ha,bytes,cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  particle_update<<<blocks,tpb>>>(dx,dv,da,dy,n,dt);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)2*n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, 2*n);
  cudaFree(dx); cudaFree(dv); cudaFree(da); cudaFree(dy); free(hx); free(hv); free(ha); free(hy); return 0;
}
