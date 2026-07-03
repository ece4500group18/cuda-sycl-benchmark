// HACC microkernel: short-range gravitational force evaluation.
//
// Extracted from HeCBench src/haccmk-cuda/haccmk.cu (origin: HACC microkernel).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5 (BSD-3-Clause).
// The haccmk_kernel below is upstream device code verbatim, including the
// 5th-order polynomial long-range correction and the branchless mass gating.
// Only the host harness is new: deterministic hash inputs and a text dump.
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

// ---- upstream kernel (verbatim) --------------------------------------------
__global__ void
haccmk_kernel (
    const int n1,  // outer loop count
    const int n2,  // inner loop count
    const float *__restrict__ xx,
    const float *__restrict__ yy,
    const float *__restrict__ zz,
    const float *__restrict__ mass,
          float *__restrict__ vx2,
          float *__restrict__ vy2,
          float *__restrict__ vz2,
    const float fsrmax,
    const float mp_rsm,
    const float fcoeff )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n1) return;

  const float ma0 = 0.269327f;
  const float ma1 = -0.0750978f;
  const float ma2 = 0.0114808f;
  const float ma3 = -0.00109313f;
  const float ma4 = 0.0000605491f;
  const float ma5 = -0.00000147177f;

  float dxc, dyc, dzc, m, r2, f, xi, yi, zi;

  xi = 0.f;
  yi = 0.f;
  zi = 0.f;

  float xxi = xx[i];
  float yyi = yy[i];
  float zzi = zz[i];

  for ( int j = 0; j < n2; j++ ) {
    dxc = xx[j] - xxi;
    dyc = yy[j] - yyi;
    dzc = zz[j] - zzi;

    r2 = dxc * dxc + dyc * dyc + dzc * dzc;

    //if ( r2 < fsrmax ) m = mass[j]; else m = 0.f;
    m = mass[j] * (r2 < fsrmax);

    f = r2 + mp_rsm;
    f = m * (1.f / (f * sqrtf(f)) - (ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5))))));

    xi = xi + f * dxc;
    yi = yi + f * dyc;
    zi = zi + f * dzc;
  }

  vx2[i] += xi * fcoeff;
  vy2[i] += yi * fcoeff;
  vz2[i] += zi * fcoeff;
}
// ---- end upstream kernel ----------------------------------------------------

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}
__host__ __device__ static inline float hs(unsigned i, unsigned s) {
  return 2.0f * h01(i, s) - 1.0f;
}

int main(int argc, char **argv) {
  const int n1 = 2048, n2 = 2048;
  const float fsrmax = 0.5f, mp_rsm = 0.1f, fcoeff = 0.23f;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t b1 = n1 * sizeof(float), b2 = n2 * sizeof(float);

  float *xx = (float*)malloc(b2), *yy = (float*)malloc(b2), *zz = (float*)malloc(b2);
  float *mass = (float*)malloc(b2);
  float *vx2 = (float*)malloc(b1), *vy2 = (float*)malloc(b1), *vz2 = (float*)malloc(b1);
  for (int j = 0; j < n2; ++j) {
    xx[j] = 2.0f * hs(j, 1);
    yy[j] = 2.0f * hs(j, 2);
    zz[j] = 2.0f * hs(j, 3);
    mass[j] = 0.5f + h01(j, 4);
  }
  for (int i = 0; i < n1; ++i) {
    vx2[i] = 0.1f * hs(i, 5);
    vy2[i] = 0.1f * hs(i, 6);
    vz2[i] = 0.1f * hs(i, 7);
  }

  float *d_xx, *d_yy, *d_zz, *d_mass, *d_vx2, *d_vy2, *d_vz2;
  CK(cudaMalloc(&d_xx, b2)); CK(cudaMalloc(&d_yy, b2)); CK(cudaMalloc(&d_zz, b2));
  CK(cudaMalloc(&d_mass, b2));
  CK(cudaMalloc(&d_vx2, b1)); CK(cudaMalloc(&d_vy2, b1)); CK(cudaMalloc(&d_vz2, b1));
  CK(cudaMemcpy(d_xx, xx, b2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_yy, yy, b2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_zz, zz, b2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_mass, mass, b2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_vx2, vx2, b1, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_vy2, vy2, b1, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_vz2, vz2, b1, cudaMemcpyHostToDevice));

  const int block_size = 256;
  dim3 grids((n1 + block_size - 1) / block_size);
  dim3 blocks(block_size);
  haccmk_kernel<<<grids, blocks>>>(n1, n2, d_xx, d_yy, d_zz, d_mass,
                                   d_vx2, d_vy2, d_vz2, fsrmax, mp_rsm, fcoeff);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(vx2, d_vx2, b1, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(vy2, d_vy2, b1, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(vz2, d_vz2, b1, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < n1; ++i)
    fprintf(f, "%.9g\n%.9g\n%.9g\n", vx2[i], vy2[i], vz2[i]);
  fclose(f);

  cudaFree(d_xx); cudaFree(d_yy); cudaFree(d_zz); cudaFree(d_mass);
  cudaFree(d_vx2); cudaFree(d_vy2); cudaFree(d_vz2);
  free(xx); free(yy); free(zz); free(mass); free(vx2); free(vy2); free(vz2);
  return 0;
}
