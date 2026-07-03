// K-means clustering iteration: feature transpose kernel + per-point nearest
// centroid assignment kernel, alternating with host-side centroid
// recomputation — the classic Rodinia device/host pipeline.
//
// Extracted from HeCBench src/kmeans-cuda/cluster.cu (origin: Rodinia
// kmeans, Northwestern University license preserved upstream).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5.
// The feature_transpose and find_membership kernels below are upstream
// device code verbatim. The host harness fixes the iteration count, seeds
// centroids with the first nclusters points, and accumulates centroid sums
// in double for a reproducible CPU reference.
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

// ---- upstream kernels (verbatim) --------------------------------------------
// copy the feature to a feature swap region
__global__
void feature_transpose (float* feature_swap,
                        const float* feature,
                        const int nfeatures,
                        const int npoints)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < npoints) {
    for(int i = 0; i <  nfeatures; i++)
      feature_swap[i * npoints + tid] = feature[tid * nfeatures + i];
  }
}

__global__
void find_membership (const float*__restrict__ feature,
                      const float*__restrict__ cluster,
                              int*__restrict__ member,
                      const int nclusters,
                      const int nfeatures,
                      const int npoints)
{
  int point_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_id < npoints) {
    int index = 0;
    float min_dist = FLT_MAX;
    for (int i = 0; i < nclusters; i++) {
      float dist = 0;
      float ans  = 0;
      for (int l = 0; l < nfeatures; l++) {
        ans += (feature[l * npoints + point_id] - cluster[i * nfeatures + l]) *
               (feature[l * npoints + point_id] - cluster[i * nfeatures + l]) ;
      }
      dist = ans;
      if (dist < min_dist) {
        min_dist = dist;
        index    = i;
      }
    }
    member[point_id] = index;
  }
}
// ---- end upstream kernels ----------------------------------------------------

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const int npoints = 4096, nfeatures = 8, nclusters = 5, iters = 4;
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";

  float *feature = (float*)malloc((size_t)npoints * nfeatures * sizeof(float));
  for (int p = 0; p < npoints; ++p)
    for (int l = 0; l < nfeatures; ++l)
      feature[p * nfeatures + l] = 2.0f * h01((unsigned)(p * nfeatures + l), 17) - 1.0f;

  float *clusters = (float*)malloc((size_t)nclusters * nfeatures * sizeof(float));
  for (int c = 0; c < nclusters; ++c)
    for (int l = 0; l < nfeatures; ++l)
      clusters[c * nfeatures + l] = feature[c * nfeatures + l];

  int *member = (int*)malloc(npoints * sizeof(int));

  float *d_feature, *d_swap, *d_clusters; int *d_member;
  CK(cudaMalloc(&d_feature, (size_t)npoints * nfeatures * sizeof(float)));
  CK(cudaMalloc(&d_swap, (size_t)npoints * nfeatures * sizeof(float)));
  CK(cudaMalloc(&d_clusters, (size_t)nclusters * nfeatures * sizeof(float)));
  CK(cudaMalloc(&d_member, npoints * sizeof(int)));
  CK(cudaMemcpy(d_feature, feature, (size_t)npoints * nfeatures * sizeof(float), cudaMemcpyHostToDevice));

  const int tpb = 256;
  const int blocks = (npoints + tpb - 1) / tpb;
  feature_transpose<<<blocks, tpb>>>(d_swap, d_feature, nfeatures, npoints);

  double *sums = (double*)malloc((size_t)nclusters * nfeatures * sizeof(double));
  int *counts = (int*)malloc(nclusters * sizeof(int));
  for (int t = 0; t < iters; ++t) {
    CK(cudaMemcpy(d_clusters, clusters, (size_t)nclusters * nfeatures * sizeof(float), cudaMemcpyHostToDevice));
    find_membership<<<blocks, tpb>>>(d_swap, d_clusters, d_member, nclusters, nfeatures, npoints);
    CK(cudaGetLastError());
    CK(cudaMemcpy(member, d_member, npoints * sizeof(int), cudaMemcpyDeviceToHost));

    // Host-side centroid recomputation (double accumulation, point order).
    for (int c = 0; c < nclusters * nfeatures; ++c) sums[c] = 0.0;
    for (int c = 0; c < nclusters; ++c) counts[c] = 0;
    for (int p = 0; p < npoints; ++p) {
      counts[member[p]]++;
      for (int l = 0; l < nfeatures; ++l)
        sums[member[p] * nfeatures + l] += (double)feature[p * nfeatures + l];
    }
    for (int c = 0; c < nclusters; ++c)
      if (counts[c] > 0)
        for (int l = 0; l < nfeatures; ++l)
          clusters[c * nfeatures + l] = (float)(sums[c * nfeatures + l] / counts[c]);
  }
  CK(cudaDeviceSynchronize());

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (int p = 0; p < npoints; ++p) fprintf(f, "%d\n", member[p]);
  for (int c = 0; c < nclusters * nfeatures; ++c) fprintf(f, "%.9g\n", clusters[c]);
  fclose(f);

  cudaFree(d_feature); cudaFree(d_swap); cudaFree(d_clusters); cudaFree(d_member);
  free(feature); free(clusters); free(member); free(sums); free(counts);
  return 0;
}
