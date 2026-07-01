#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 2; } } while (0)

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

__global__ void image_kernel(const float*x,float*y,int N,int H,int W,int C){int idx=blockIdx.x*blockDim.x+threadIdx.x,total=N*H*W*C;if(idx<total){int c=idx%3; float mean=(c==0?0.485f:(c==1?0.456f:0.406f)); float inv=(c==0?4.3668f:(c==1?4.4643f:4.4444f)); y[idx]=(x[idx]-mean)*inv;}}
int main(int argc,char**argv){const int N=4,H=64,W=64,C=3,total=N*H*W*C;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)total*sizeof(float)),*hy=(float*)malloc((size_t)total*sizeof(float));for(int i=0;i<total;++i)hx[i]=h01(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)total*sizeof(float)));CK(cudaMalloc(&dy,(size_t)total*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)total*sizeof(float),cudaMemcpyHostToDevice));image_kernel<<<(total+255)/256,256>>>(dx,dy,N,H,W,C);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)total*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,total);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
