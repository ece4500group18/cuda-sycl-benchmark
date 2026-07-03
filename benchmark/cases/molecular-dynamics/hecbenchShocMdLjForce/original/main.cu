// SHOC MD Lennard-Jones force kernel with an explicit neighbor list.
//
// Extracted from HeCBench src/md-cuda/main.cu (origin: SHOC benchmark suite).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5 (BSD-3-Clause).
// The md() kernel below is the upstream device code verbatim, specialized to
// the single-precision configuration (FPTYPE=float, POSVECTYPE=float4).
// Only the host harness is new: deterministic jittered-lattice positions,
// a deterministic pseudo-random neighbor list, and a text output dump.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

typedef float FPTYPE;
typedef float4 POSVECTYPE;
typedef float4 FORCEVECTYPE;
#define zero make_float4(0.0f, 0.0f, 0.0f, 0.0f)

// ---- upstream kernel (verbatim) --------------------------------------------
__global__ void md (
  const POSVECTYPE* __restrict__ position,
        FORCEVECTYPE* __restrict__ force,
  const int* __restrict__ neighborList,
  const int nAtom,
  const int maxNeighbors,
  const FPTYPE lj1_t,
  const FPTYPE lj2_t,
  const FPTYPE cutsq_t )
{
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nAtom) return;

  POSVECTYPE ipos = position[idx];
  FORCEVECTYPE f = zero;

  int j = 0;
  while (j < maxNeighbors)
  {
    int jidx = neighborList[j*nAtom + idx];

    // Uncoalesced read
    POSVECTYPE jpos = position[jidx];

    // Calculate distance
    FPTYPE delx = ipos.x - jpos.x;
    FPTYPE dely = ipos.y - jpos.y;
    FPTYPE delz = ipos.z - jpos.z;
    FPTYPE r2inv = delx*delx + dely*dely + delz*delz;

    // If distance is less than cutoff, calculate force
    if (r2inv > 0 && r2inv < cutsq_t)
    {
      r2inv = (FPTYPE)1.0 / r2inv;
      FPTYPE r6inv = r2inv * r2inv * r2inv;
      FPTYPE forceC = r2inv * r6inv * (lj1_t * r6inv - lj2_t);

      f.x += delx * forceC;
      f.y += dely * forceC;
      f.z += delz * forceC;
    }
    j++;
  }
  force[idx] = f;
}
// ---- end upstream kernel ----------------------------------------------------

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const int nAtom = 4096;          // 16^3 jittered lattice
  const int maxNeighbors = 32;
  const float spacing = 1.25f;
  const float lj1 = 1.5f, lj2 = 2.0f, cutsq = 6.25f;  // cutoff radius 2.5
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  POSVECTYPE *pos = (POSVECTYPE*)malloc(nAtom * sizeof(POSVECTYPE));
  FORCEVECTYPE *frc = (FORCEVECTYPE*)malloc(nAtom * sizeof(FORCEVECTYPE));
  int *nbr = (int*)malloc((size_t)maxNeighbors * nAtom * sizeof(int));

  // Jittered cubic lattice: guarantees a minimum separation so the LJ force
  // stays bounded (upstream warns about near-coincident random positions).
  for (int i = 0; i < nAtom; ++i) {
    int ix = i % 16, iy = (i / 16) % 16, iz = i / 256;
    pos[i].x = ((float)ix + 0.5f * h01(i, 1)) * spacing;
    pos[i].y = ((float)iy + 0.5f * h01(i, 2)) * spacing;
    pos[i].z = ((float)iz + 0.5f * h01(i, 3)) * spacing;
    pos[i].w = 0.0f;
  }
  // Deterministic pseudo-random neighbor list (column-major like upstream).
  for (int j = 0; j < maxNeighbors; ++j)
    for (int i = 0; i < nAtom; ++i) {
      int jidx = (int)(h01(i, 100 + j) * (float)nAtom);
      if (jidx > nAtom - 1) jidx = nAtom - 1;
      nbr[j*nAtom + i] = jidx;
    }

  POSVECTYPE *d_pos; FORCEVECTYPE *d_frc; int *d_nbr;
  CK(cudaMalloc(&d_pos, nAtom * sizeof(POSVECTYPE)));
  CK(cudaMalloc(&d_frc, nAtom * sizeof(FORCEVECTYPE)));
  CK(cudaMalloc(&d_nbr, (size_t)maxNeighbors * nAtom * sizeof(int)));
  CK(cudaMemcpy(d_pos, pos, nAtom * sizeof(POSVECTYPE), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_nbr, nbr, (size_t)maxNeighbors * nAtom * sizeof(int), cudaMemcpyHostToDevice));

  dim3 grids((nAtom + 255) / 256);
  dim3 block(256);
  md<<<grids, block>>>(d_pos, d_frc, d_nbr, nAtom, maxNeighbors, lj1, lj2, cutsq);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(frc, d_frc, nAtom * sizeof(FORCEVECTYPE), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < nAtom; ++i)
    fprintf(f, "%.9g\n%.9g\n%.9g\n", frc[i].x, frc[i].y, frc[i].z);
  fclose(f);

  cudaFree(d_pos); cudaFree(d_frc); cudaFree(d_nbr);
  free(pos); free(frc); free(nbr);
  return 0;
}
