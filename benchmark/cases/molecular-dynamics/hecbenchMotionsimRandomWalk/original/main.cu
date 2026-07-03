// Particle diffusion by 2D random walk with per-particle grid-cell counting.
//
// Extracted from HeCBench src/particle-diffusion-cuda/motionsim.cu
// (origin: Intel oneAPI motionsim sample).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5 (BSD-3-Clause).
// The Simulation kernel below is upstream device code verbatim. Upstream
// already pre-generates its random numbers on the host; this harness keeps
// that design but derives them from a deterministic hash (scale 100, like
// upstream's rand()%scale), and initializes particles at (10,10) exactly as
// upstream does.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

// ---- upstream kernel (verbatim) --------------------------------------------
__global__
void Simulation(float*__restrict__ a_particleX,
                float*__restrict__ a_particleY,
		const float*__restrict__ a_randomX,
                const float*__restrict__ a_randomY,
		size_t *__restrict__ a_map,
                const size_t n_particles,
                unsigned int nIterations,
                int grid_size,
                float radius)
{
  size_t ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii >= n_particles) return;
  // Start iterations
  // Each iteration:
  //  1. Updates the position of all water molecules
  //  2. Checks if water molecule is inside a cell or not.
  //  3. Updates counter in cells array
  size_t iter = 0;
  float pX = a_particleX[ii];
  float pY = a_particleY[ii];
  size_t map_base = ii * grid_size * grid_size;
  while (iter < nIterations) {
    // Computes random displacement for each molecule
    // This example shows random distances between
    // -0.05 units and 0.05 units in both X and Y directions
    // Moves each water molecule by a random vector

    float randnumX = a_randomX[iter * n_particles + ii];
    float randnumY = a_randomY[iter * n_particles + ii];

    // Transform the scaled random numbers into small displacements
    float displacementX = randnumX / 1000.0f - 0.0495f;
    float displacementY = randnumY / 1000.0f - 0.0495f;

    // Move particles using random displacements
    pX += displacementX;
    pY += displacementY;

    // Compute distances from particle position to grid point
    float dX = pX - truncf(pX);
    float dY = pY - truncf(pY);

    // Compute grid point indices
    int iX = floorf(pX);
    int iY = floorf(pY);

    // Check if particle is still in computation grid
    if ((pX < grid_size) && (pY < grid_size) && (pX >= 0) && (pY >= 0)) {
      // Check if particle is (or remained) inside cell.
      // Increment cell counter in map array if so
      if ((dX * dX + dY * dY <= radius * radius))
        // The map array is organized as (particle, y, x)
        a_map[map_base + iY * grid_size + iX]++;
    }

    iter++;

  }  // Next iteration

  a_particleX[ii] = pX;
  a_particleY[ii] = pY;
}
// ---- end upstream kernel ----------------------------------------------------

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const size_t n_particles = 512;
  const unsigned int nIterations = 32;
  const int grid_size = 16;
  const float radius = 0.5f;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t map_len = n_particles * grid_size * grid_size;
  const size_t rand_len = (size_t)nIterations * n_particles;

  float *pX = (float*)malloc(n_particles * sizeof(float));
  float *pY = (float*)malloc(n_particles * sizeof(float));
  float *rX = (float*)malloc(rand_len * sizeof(float));
  float *rY = (float*)malloc(rand_len * sizeof(float));
  size_t *map = (size_t*)calloc(map_len, sizeof(size_t));

  // Upstream initializes every particle at (10, 10).
  for (size_t i = 0; i < n_particles; ++i) { pX[i] = 10.0f; pY[i] = 10.0f; }
  // Upstream scale = 100: random integers in [0, 100).
  for (size_t k = 0; k < rand_len; ++k) {
    rX[k] = (float)(int)(h01((unsigned)k, 21) * 100.0f);
    rY[k] = (float)(int)(h01((unsigned)k, 22) * 100.0f);
  }

  float *d_pX, *d_pY, *d_rX, *d_rY; size_t *d_map;
  CK(cudaMalloc(&d_pX, n_particles * sizeof(float)));
  CK(cudaMalloc(&d_pY, n_particles * sizeof(float)));
  CK(cudaMalloc(&d_rX, rand_len * sizeof(float)));
  CK(cudaMalloc(&d_rY, rand_len * sizeof(float)));
  CK(cudaMalloc(&d_map, map_len * sizeof(size_t)));
  CK(cudaMemcpy(d_pX, pX, n_particles * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_pY, pY, n_particles * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_rX, rX, rand_len * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_rY, rY, rand_len * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_map, map, map_len * sizeof(size_t), cudaMemcpyHostToDevice));

  const int tpb = 256;
  Simulation<<<((int)n_particles + tpb - 1) / tpb, tpb>>>(
      d_pX, d_pY, d_rX, d_rY, d_map, n_particles, nIterations, grid_size, radius);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(pX, d_pX, n_particles * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(pY, d_pY, n_particles * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(map, d_map, map_len * sizeof(size_t), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (size_t i = 0; i < n_particles; ++i) fprintf(f, "%.9g\n%.9g\n", pX[i], pY[i]);
  for (size_t k = 0; k < map_len; ++k) fprintf(f, "%zu\n", map[k]);
  fclose(f);

  cudaFree(d_pX); cudaFree(d_pY); cudaFree(d_rX); cudaFree(d_rY); cudaFree(d_map);
  free(pX); free(pY); free(rX); free(rY); free(map);
  return 0;
}
