// DEM sphere-sphere collision over a uniform hashed grid, from the CUDA
// Samples "particles" demo.
//
// Extracted from NVIDIA/cuda-samples cpp/2_Concepts_and_Techniques/particles
// (particles_kernel_impl.cuh). Upstream: @ b7c5481c (BSD-style NVIDIA license).
// calcGridPos, calcGridHash, collideSpheres, collideCell and collideD below
// are upstream device code verbatim (cudaParams kept as the __constant__
// parameter block). Upstream builds the sorted grid with thrust::sort plus a
// reorder kernel; this harness prebuilds the same sorted arrays and
// cellStart/cellEnd deterministically on the host (documented substitution),
// so the case focuses on the collision kernel itself. The float3/float4
// helpers are the needed subset of cuda-samples helper_math.h.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

typedef unsigned int uint;

// ---- helper_math.h subset (cuda-samples) ------------------------------------
inline __host__ __device__ float3 make_float3(float s) { return make_float3(s, s, s); }
inline __host__ __device__ float3 make_float3(float4 a) { return make_float3(a.x, a.y, a.z); }
inline __host__ __device__ float4 make_float4(float3 a, float w) { return make_float4(a.x, a.y, a.z, w); }
inline __host__ __device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __host__ __device__ void operator+=(float3 &a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
inline __host__ __device__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline __host__ __device__ float3 operator*(float a, float3 b) { return make_float3(a * b.x, a * b.y, a * b.z); }
inline __host__ __device__ float3 operator*(float3 a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
inline __host__ __device__ float3 operator/(float3 a, float b) { return make_float3(a.x / b, a.y / b, a.z / b); }
inline __host__ __device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __host__ __device__ float length(float3 v) { return sqrtf(dot(v, v)); }
inline __host__ __device__ int3 operator+(int3 a, int3 b) { return make_int3(a.x + b.x, a.y + b.y, a.z + b.z); }

// ---- simulation parameters (subset of upstream SimParams) --------------------
struct SimParams {
  uint3 gridSize;
  float3 worldOrigin;
  float3 cellSize;
  float spring;
  float damping;
  float shear;
  float attraction;
  float particleRadius;
  float3 colliderPos;
  float colliderRadius;
};

__constant__ SimParams cudaParams;

// ---- upstream device code (verbatim) -----------------------------------------
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floorf((p.x - cudaParams.worldOrigin.x) / cudaParams.cellSize.x);
    gridPos.y = floorf((p.y - cudaParams.worldOrigin.y) / cudaParams.cellSize.y);
    gridPos.z = floorf((p.z - cudaParams.worldOrigin.z) / cudaParams.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (cudaParams.gridSize.x - 1); // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (cudaParams.gridSize.y - 1);
    gridPos.z = gridPos.z & (cudaParams.gridSize.z - 1);
    return __umul24(__umul24(gridPos.z, cudaParams.gridSize.y), cudaParams.gridSize.x)
         + __umul24(gridPos.y, cudaParams.gridSize.x) + gridPos.x;
}

// collide two spheres using DEM method
__device__ float3
collideSpheres(float3 posA, float3 posB, float3 velA, float3 velB, float radiusA, float radiusB, float attraction)
{
    // calculate relative position
    float3 relPos = posB - posA;

    float dist        = length(relPos);
    float collideDist = radiusA + radiusB;

    float3 force = make_float3(0.0f);

    if (dist < collideDist) {
        float3 norm = relPos / dist;

        // relative velocity
        float3 relVel = velB - velA;

        // relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

        // spring force
        force = -cudaParams.spring * (collideDist - dist) * norm;
        // dashpot (damping) force
        force += cudaParams.damping * relVel;
        // tangential shear force
        force += cudaParams.shear * tanVel;
        // attraction
        force += attraction * relPos;
    }

    return force;
}

// collide a particle against all other particles in a given cell
__device__ float3 collideCell(int3    gridPos,
                              uint    index,
                              float3  pos,
                              float3  vel,
                              float4 *oldPos,
                              float4 *oldVel,
                              uint   *cellStart,
                              uint   *cellEnd)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex = cellStart[gridHash];

    float3 force = make_float3(0.0f);

    if (startIndex != 0xffffffff) // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];

        for (uint j = startIndex; j < endIndex; j++) {
            if (j != index) // check not colliding with self
            {
                float3 pos2 = make_float3(oldPos[j]);
                float3 vel2 = make_float3(oldVel[j]);

                // collide two spheres
                force += collideSpheres(
                    pos, pos2, vel, vel2, cudaParams.particleRadius, cudaParams.particleRadius, cudaParams.attraction);
            }
        }
    }

    return force;
}

__global__ void collideD(float4 *newVel,            // output: new velocity
                         float4 *oldPos,            // input: sorted positions
                         float4 *oldVel,            // input: sorted velocities
                         uint   *gridParticleIndex, // input: sorted particle indices
                         uint   *cellStart,
                         uint   *cellEnd,
                         uint    numParticles)
{
    uint index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles)
        return;

    // read particle data from sorted arrays
    float3 pos = make_float3(oldPos[index]);
    float3 vel = make_float3(oldVel[index]);

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    float3 force = make_float3(0.0f);

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                force += collideCell(neighbourPos, index, pos, vel, oldPos, oldVel, cellStart, cellEnd);
            }
        }
    }

    // collide with cursor sphere
    force += collideSpheres(pos,
                            cudaParams.colliderPos,
                            vel,
                            make_float3(0.0f, 0.0f, 0.0f),
                            cudaParams.particleRadius,
                            cudaParams.colliderRadius,
                            0.0f);

    // write new velocity back to original unsorted location
    uint originalIndex    = gridParticleIndex[index];
    newVel[originalIndex] = make_float4(vel + force, 0.0f);
}
// ---- end upstream device code -------------------------------------------------

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

// Host mirror of calcGridPos/calcGridHash (for the deterministic grid build).
static uint host_hash(const SimParams &p, float x, float y, float z) {
  int gx = (int)floorf((x - p.worldOrigin.x) / p.cellSize.x) & (p.gridSize.x - 1);
  int gy = (int)floorf((y - p.worldOrigin.y) / p.cellSize.y) & (p.gridSize.y - 1);
  int gz = (int)floorf((z - p.worldOrigin.z) / p.cellSize.z) & (p.gridSize.z - 1);
  return ((uint)gz * p.gridSize.y * p.gridSize.x) + ((uint)gy * p.gridSize.x) + (uint)gx;
}

int main(int argc, char **argv) {
  const uint n = 2048;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  SimParams p;
  p.gridSize = make_uint3(16, 16, 16);
  p.worldOrigin = make_float3(-1.0f, -1.0f, -1.0f);
  p.particleRadius = 1.0f / 16.0f;                     // cellSize = 2*radius
  p.cellSize = make_float3(0.125f, 0.125f, 0.125f);
  p.spring = 0.5f;
  p.damping = 0.02f;
  p.shear = 0.1f;
  p.attraction = 0.0f;
  p.colliderPos = make_float3(-1.2f, -0.8f, 0.8f);
  p.colliderRadius = 0.2f;

  const uint ncells = p.gridSize.x * p.gridSize.y * p.gridSize.z;
  float4 *pos = (float4*)malloc(n * sizeof(float4));
  float4 *vel = (float4*)malloc(n * sizeof(float4));
  for (uint i = 0; i < n; ++i) {
    pos[i] = make_float4(2.0f * h01(i, 1) - 1.0f, 2.0f * h01(i, 2) - 1.0f,
                         2.0f * h01(i, 3) - 1.0f, 1.0f);
    vel[i] = make_float4(0.02f * (2.0f * h01(i, 4) - 1.0f),
                         0.02f * (2.0f * h01(i, 5) - 1.0f),
                         0.02f * (2.0f * h01(i, 6) - 1.0f), 0.0f);
  }

  // Deterministic host grid build (replaces thrust sort + reorder kernel):
  // counting sort by cell hash, stable in original particle order.
  uint *hashv = (uint*)malloc(n * sizeof(uint));
  uint *count = (uint*)calloc(ncells + 1, sizeof(uint));
  for (uint i = 0; i < n; ++i) {
    hashv[i] = host_hash(p, pos[i].x, pos[i].y, pos[i].z);
    count[hashv[i] + 1]++;
  }
  for (uint c = 0; c < ncells; ++c) count[c + 1] += count[c];
  float4 *sortedPos = (float4*)malloc(n * sizeof(float4));
  float4 *sortedVel = (float4*)malloc(n * sizeof(float4));
  uint *gridIndex = (uint*)malloc(n * sizeof(uint));
  uint *cellStart = (uint*)malloc(ncells * sizeof(uint));
  uint *cellEnd = (uint*)malloc(ncells * sizeof(uint));
  memset(cellStart, 0xff, ncells * sizeof(uint));
  memset(cellEnd, 0, ncells * sizeof(uint));
  uint *cursor = (uint*)malloc((ncells) * sizeof(uint));
  memcpy(cursor, count, ncells * sizeof(uint));
  for (uint i = 0; i < n; ++i) {
    uint slot = cursor[hashv[i]]++;
    sortedPos[slot] = pos[i];
    sortedVel[slot] = vel[i];
    gridIndex[slot] = i;
  }
  for (uint c = 0; c < ncells; ++c) {
    if (count[c + 1] > count[c]) { cellStart[c] = count[c]; cellEnd[c] = count[c + 1]; }
  }

  float4 *d_newVel, *d_oldPos, *d_oldVel; uint *d_gridIndex, *d_cellStart, *d_cellEnd;
  CK(cudaMalloc(&d_newVel, n * sizeof(float4)));
  CK(cudaMalloc(&d_oldPos, n * sizeof(float4)));
  CK(cudaMalloc(&d_oldVel, n * sizeof(float4)));
  CK(cudaMalloc(&d_gridIndex, n * sizeof(uint)));
  CK(cudaMalloc(&d_cellStart, ncells * sizeof(uint)));
  CK(cudaMalloc(&d_cellEnd, ncells * sizeof(uint)));
  CK(cudaMemcpy(d_oldPos, sortedPos, n * sizeof(float4), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_oldVel, sortedVel, n * sizeof(float4), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_gridIndex, gridIndex, n * sizeof(uint), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_cellStart, cellStart, ncells * sizeof(uint), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_cellEnd, cellEnd, ncells * sizeof(uint), cudaMemcpyHostToDevice));
  CK(cudaMemcpyToSymbol(cudaParams, &p, sizeof(SimParams)));

  const uint tpb = 256;
  collideD<<<(n + tpb - 1) / tpb, tpb>>>(d_newVel, d_oldPos, d_oldVel,
                                         d_gridIndex, d_cellStart, d_cellEnd, n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  float4 *newVel = (float4*)malloc(n * sizeof(float4));
  CK(cudaMemcpy(newVel, d_newVel, n * sizeof(float4), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (uint i = 0; i < n; ++i)
    fprintf(f, "%.9g\n%.9g\n%.9g\n", newVel[i].x, newVel[i].y, newVel[i].z);
  fclose(f);

  cudaFree(d_newVel); cudaFree(d_oldPos); cudaFree(d_oldVel);
  cudaFree(d_gridIndex); cudaFree(d_cellStart); cudaFree(d_cellEnd);
  free(pos); free(vel); free(hashv); free(count); free(cursor);
  free(sortedPos); free(sortedVel); free(gridIndex); free(cellStart); free(cellEnd); free(newVel);
  return 0;
}
