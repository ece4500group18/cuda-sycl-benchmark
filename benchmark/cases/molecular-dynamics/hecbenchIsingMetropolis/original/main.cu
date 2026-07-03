// 2D Ising model: checkerboard Metropolis spin updates.
//
// Extracted from HeCBench src/ising-cuda/main.cu (origin: NVIDIA ising-gpu).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5 (BSD-3-Clause).
// The init_spins and update_lattice<is_black> kernels below are upstream
// device code verbatim. The upstream feeds curand-generated uniforms; this
// harness feeds deterministic host-generated uniforms instead (same kernel
// contract: one fresh uniform per site per color update), so results are
// reproducible without cuRAND.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

#define TCRIT 2.26918531421f
#define THREADS 128

// ---- upstream kernels (verbatim) --------------------------------------------
// Initialize lattice spins
__global__ void init_spins(signed char* lattice,
                           const float* __restrict__ randvals,
                           const long long nx,
                           const long long ny) {
  const long long  tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
  if (tid >= nx * ny) return;

  float randval = randvals[tid];
  signed char val = (randval < 0.5f) ? -1 : 1;
  lattice[tid] = val;
}

template<bool is_black>
__global__ void update_lattice(signed char* lattice,
                               const signed char* __restrict__ op_lattice,
                               const float* __restrict__ randvals,
                               const float inv_temp,
                               const long long nx,
                               const long long ny) {
  const long long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
  const int i = tid / ny;
  const int j = tid % ny;

  if (i >= nx || j >= ny) return;

  // Set stencil indices with periodicity
  int ipp = (i + 1 < nx) ? i + 1 : 0;
  int inn = (i - 1 >= 0) ? i - 1: nx - 1;
  int jpp = (j + 1 < ny) ? j + 1 : 0;
  int jnn = (j - 1 >= 0) ? j - 1: ny - 1;

  // Select off-column index based on color and row index parity
  int joff;
  if (is_black) {
    joff = (i % 2) ? jpp : jnn;
  } else {
    joff = (i % 2) ? jnn : jpp;
  }

  // Compute sum of nearest neighbor spins
  signed char nn_sum = op_lattice[inn * ny + j] + op_lattice[i * ny + j] + op_lattice[ipp * ny + j] + op_lattice[i * ny + joff];

  // Determine whether to flip spin
  signed char lij = lattice[i * ny + j];
  float acceptance_ratio = expf(-2.0f * inv_temp * nn_sum * lij);
  if (randvals[i*ny + j] < acceptance_ratio) {
    lattice[i * ny + j] = -lij;
  }
}
// ---- end upstream kernels ----------------------------------------------------

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

static void fill_randvals(float *h, float *d, long long n, unsigned seed) {
  for (long long k = 0; k < n; ++k) h[k] = h01((unsigned)k, seed);
  cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
}

int main(int argc, char **argv) {
  const long long nx = 128, ny = 128, ny_half = ny / 2;
  const int nsweeps = 4;
  const float inv_temp = 1.0f / TCRIT;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const long long nhalf = nx * ny_half;

  float *h_rand = (float*)malloc(nhalf * sizeof(float));
  signed char *h_lat = (signed char*)malloc(2 * nhalf);
  float *d_rand; signed char *d_black, *d_white;
  CK(cudaMalloc(&d_rand, nhalf * sizeof(float)));
  CK(cudaMalloc(&d_black, nhalf));
  CK(cudaMalloc(&d_white, nhalf));

  int blocks = (int)((nhalf + THREADS - 1) / THREADS);

  fill_randvals(h_rand, d_rand, nhalf, 11);
  init_spins<<<blocks, THREADS>>>(d_black, d_rand, nx, ny_half);
  fill_randvals(h_rand, d_rand, nhalf, 12);
  init_spins<<<blocks, THREADS>>>(d_white, d_rand, nx, ny_half);
  CK(cudaGetLastError());

  for (int s = 0; s < nsweeps; ++s) {
    fill_randvals(h_rand, d_rand, nhalf, 1000 + 2 * s);
    update_lattice<true><<<blocks, THREADS>>>(d_black, d_white, d_rand, inv_temp, nx, ny_half);
    fill_randvals(h_rand, d_rand, nhalf, 1001 + 2 * s);
    update_lattice<false><<<blocks, THREADS>>>(d_white, d_black, d_rand, inv_temp, nx, ny_half);
  }
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  CK(cudaMemcpy(h_lat, d_black, nhalf, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h_lat + nhalf, d_white, nhalf, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (long long k = 0; k < 2 * nhalf; ++k) fprintf(f, "%d\n", (int)h_lat[k]);
  fclose(f);

  cudaFree(d_rand); cudaFree(d_black); cudaFree(d_white);
  free(h_rand); free(h_lat);
  return 0;
}
