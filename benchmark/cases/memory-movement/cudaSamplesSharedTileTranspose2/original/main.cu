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

__global__ void tile_t(const float*x,float*y,int H,int W){__shared__ float tile[16][17];int c=blockIdx.x*16+threadIdx.x,r=blockIdx.y*16+threadIdx.y;if(r<H&&c<W)tile[threadIdx.y][threadIdx.x]=x[r*W+c];__syncthreads();int orow=blockIdx.x*16+threadIdx.y,ocol=blockIdx.y*16+threadIdx.x;if(orow<W&&ocol<H)y[orow*H+ocol]=tile[threadIdx.x][threadIdx.y];}
int main(int argc,char**argv){const int H=128,W=128,n=H*W;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));tile_t<<<dim3((W+15)/16,(H+15)/16),dim3(16,16)>>>(dx,dy,H,W);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
