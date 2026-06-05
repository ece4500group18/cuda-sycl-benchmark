// rope: rotary position embedding on a [seq, dim] tensor. seq=128, dim=64.
//   theta_k = 10000^(-2k/dim), angle = p * theta_k, for pair (2k, 2k+1).
//   out[p,2k]   = x0*cos - x1*sin
//   out[p,2k+1] = x0*sin + x1*cos
//   x[idx] = h01(idx, 123)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void rope(const float *x, float *y, int seq, int dim) {
  int half = dim / 2;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < seq * half) {
    int p = idx / half, k = idx % half;
    float theta = powf(10000.0f, -2.0f * k / dim);
    float ang = p * theta;
    float cs = cosf(ang), sn = sinf(ang);
    const float *xr = x + (size_t)p * dim;
    float x0 = xr[2 * k], x1 = xr[2 * k + 1];
    y[(size_t)p * dim + 2 * k]     = x0 * cs - x1 * sn;
    y[(size_t)p * dim + 2 * k + 1] = x0 * sn + x1 * cs;
  }
}

int main(int argc, char **argv) {
  const int seq = 128, dim = 64, total = seq * dim;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)total * sizeof(float);
  float *hx=(float*)malloc(bytes),*hy=(float*)malloc(bytes);
  for (int i=0;i<total;++i) hx[i]=h01(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,bytes)); CK(cudaMalloc(&dy,bytes));
  CK(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice));
  int tpb=128, work=seq*(dim/2), blocks=(work+tpb-1)/tpb;
  rope<<<blocks,tpb>>>(dx,dy,seq,dim);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,bytes,cudaMemcpyDeviceToHost));
  FILE *f=fopen(out,"w"); if(!f){fprintf(stderr,"open %s\n",out);return 2;}
  for(int i=0;i<total;++i) fprintf(f,"%.9g\n",hy[i]); fclose(f);
  printf("rope done: seq=%d dim=%d -> %s\n", seq, dim, out);
  cudaFree(dx);cudaFree(dy);free(hx);free(hy);
  return 0;
}
