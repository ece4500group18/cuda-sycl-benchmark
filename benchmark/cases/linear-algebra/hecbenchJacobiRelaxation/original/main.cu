// Jacobi relaxation of the 2D Laplace equation with shared-memory halo
// tiles, warp-shuffle error reduction and atomicAdd accumulation.
//
// Extracted from HeCBench src/jacobi-cuda/main.cu (origin: NVIDIA, 2021).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5 (Apache-2.0).
// The jacobi_step kernel and initialize_data are upstream code verbatim
// (N reduced from 2048 to 512). The harness runs a fixed iteration count
// instead of upstream's convergence loop (deterministic output) and dumps
// the final field; the atomicAdd error path stays exercised every step.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <cuda_runtime.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

// A multiple of thread block size
#define N 512

#define IDX(i, j) ((i) + (j) * N)

// ---- upstream code (verbatim) ------------------------------------------------
void initialize_data (float* f) {
  // Set up simple sinusoidal boundary conditions
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {

      if (i == 0 || i == N-1) {
        f[IDX(i,j)] = sinf(j * 2 * M_PI / (N - 1));
      }
      else if (j == 0 || j == N-1) {
        f[IDX(i,j)] = sinf(i * 2 * M_PI / (N - 1));
      }
      else {
        f[IDX(i,j)] = 0.0f;
      }

    }
  }
}

__global__ void jacobi_step (float*__restrict__ f,
                             const float*__restrict__ f_old,
                             float*__restrict__ error) {
  __shared__ float f_old_tile[18][18];

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  // First read in the "interior" data, one value per thread
  // Note the offset by 1, to reserve space for the "left"/"bottom" halo

  f_old_tile[threadIdx.y+1][threadIdx.x+1] = f_old[IDX(i,j)];

  // Now read in the halo data; we'll pick the "closest" thread
  // to each element. When we do this, make sure we don't fall
  // off the end of the global memory array. Note that this
  // code does not fill the corners, as they are not used in
  // this stencil.

  if (threadIdx.x == 0 && i >= 1) {
    f_old_tile[threadIdx.y+1][threadIdx.x+0] = f_old[IDX(i-1,j)];
  }
  if (threadIdx.x == 15 && i <= N-2) {
    f_old_tile[threadIdx.y+1][threadIdx.x+2] = f_old[IDX(i+1,j)];
  }
  if (threadIdx.y == 0 && j >= 1) {
    f_old_tile[threadIdx.y+0][threadIdx.x+1] = f_old[IDX(i,j-1)];
  }
  if (threadIdx.y == 15 && j <= N-2) {
    f_old_tile[threadIdx.y+2][threadIdx.x+1] = f_old[IDX(i,j+1)];
  }

  // Synchronize all threads
  __syncthreads();

  float err = 0.0f;

  if (j >= 1 && j <= N-2) {
    if (i >= 1 && i <= N-2) {
      // Perform the read from shared memory
      f[IDX(i,j)] = 0.25f * (f_old_tile[threadIdx.y+1][threadIdx.x+2] +
                             f_old_tile[threadIdx.y+1][threadIdx.x+0] +
                             f_old_tile[threadIdx.y+2][threadIdx.x+1] +
                             f_old_tile[threadIdx.y+0][threadIdx.x+1]);
      float df = f[IDX(i,j)] - f_old_tile[threadIdx.y+1][threadIdx.x+1];
      err = df * df;
    }
  }

  // Sum over threads in the warp
  // For simplicity, we do this outside the above conditional
  // so that all threads participate
  for (int offset = 8; offset > 0; offset /= 2) {
    err += __shfl_down_sync(0xffffffff, err, offset);
  }

  // If we're thread 0 in the warp, update our value to shared memory
  // Note that we're assuming exactly a 16x16 block and that the warp ID
  // is equivalent to threadIdx.y. For the general case, we would have to
  // write more careful code.
  __shared__ float reduction_array[16];
  if (threadIdx.x == 0) {
    reduction_array[threadIdx.y] = err;
  }

  // Synchronize the block before reading any values from smem
  __syncthreads();

  // Using the first warp in the block, reduce over the partial sums
  // in the shared memory array.
  if (threadIdx.y == 0) {
    err = reduction_array[threadIdx.x];
    for (int offset = 8; offset > 0; offset /= 2) {
      err += __shfl_down_sync(0xffffffff, err, offset);
    }
    if (threadIdx.x == 0) {
      atomicAdd(error, err);
    }
  }
}
// ---- end upstream code ---------------------------------------------------------

int main (int argc, char **argv) {
  const int iters = 50;
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";

  float *d_f, *d_f_old, *d_error;
  float *f = (float*)malloc((size_t)N * N * sizeof(float));
  float *f_old = (float*)malloc((size_t)N * N * sizeof(float));

  initialize_data(f);
  initialize_data(f_old);

  CK(cudaMalloc(&d_f, (size_t)N * N * sizeof(float)));
  CK(cudaMemcpy(d_f, f, (size_t)N * N * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d_f_old, (size_t)N * N * sizeof(float)));
  CK(cudaMemcpy(d_f_old, f_old, (size_t)N * N * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d_error, sizeof(float)));

  dim3 grid(N / 16, N / 16);
  dim3 block(16, 16);

  for (int t = 0; t < iters; ++t) {
    CK(cudaMemset(d_error, 0, 4));
    jacobi_step<<<grid, block>>>(d_f, d_f_old, d_error);
    std::swap(d_f, d_f_old);
  }
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  // After the final swap, d_f_old holds the latest field.
  CK(cudaMemcpy(f, d_f_old, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost));

  FILE *fp = fopen(out_path, "w");
  if (!fp) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (size_t k = 0; k < (size_t)N * N; ++k) fprintf(fp, "%.9g\n", f[k]);
  fclose(fp);

  cudaFree(d_f); cudaFree(d_f_old); cudaFree(d_error);
  free(f); free(f_old);
  return 0;
}
