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

__global__ void ell_spmv(const float*val,const float*x,float*y,int rows,int width){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r<rows){float s=0.0f;for(int k=0;k<width;++k){int c=(r*17+k*13)%rows;s+=val[r*width+k]*x[c];}y[r]=s;}}
int main(int argc,char**argv){const int rows=4096,width=8,n=rows*width;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hv=(float*)malloc((size_t)n*sizeof(float)),*hx=(float*)malloc((size_t)rows*sizeof(float)),*hy=(float*)malloc((size_t)rows*sizeof(float));for(int i=0;i<n;++i)hv[i]=0.1f*hs(i,123);for(int i=0;i<rows;++i)hx[i]=hs(i,321);float*dv,*dx,*dy;CK(cudaMalloc(&dv,(size_t)n*sizeof(float)));CK(cudaMalloc(&dx,(size_t)rows*sizeof(float)));CK(cudaMalloc(&dy,(size_t)rows*sizeof(float)));CK(cudaMemcpy(dv,hv,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dx,hx,(size_t)rows*sizeof(float),cudaMemcpyHostToDevice));ell_spmv<<<(rows+255)/256,256>>>(dv,dx,dy,rows,width);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)rows*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,rows);cudaFree(dv);cudaFree(dx);cudaFree(dy);free(hv);free(hx);free(hy);return 0;}
