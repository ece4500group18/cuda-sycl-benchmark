// bfs: level-synchronous breadth-first search distances from node 0.
// Deterministic directed graph, out-degree 3 per node:
//   neighbors(i) = { (i+1)%N, (2*i+1)%N, (7*i+13)%N }
// N = 20000. Output: N integer distances (-1 if unreachable).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__global__ void bfsStep(const int *rowptr, const int *col, int *dist,
                        int level, int *changed, int N) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u < N && dist[u] == level) {
    for (int k = rowptr[u]; k < rowptr[u + 1]; ++k) {
      int v = col[k];
      if (dist[v] == -1) { dist[v] = level + 1; *changed = 1; }
    }
  }
}

int main(int argc, char **argv) {
  const int N = 20000, deg = 3;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  int *rowptr = (int *)malloc((size_t)(N + 1) * sizeof(int));
  int *col = (int *)malloc((size_t)N * deg * sizeof(int));
  for (int i = 0; i < N; ++i) {
    rowptr[i] = i * deg;
    col[i * deg + 0] = (i + 1) % N;
    col[i * deg + 1] = (2 * i + 1) % N;
    col[i * deg + 2] = (7 * i + 13) % N;
  }
  rowptr[N] = N * deg;

  int *drow, *dcol, *ddist, *dchanged;
  CK(cudaMalloc(&drow, (size_t)(N + 1) * sizeof(int)));
  CK(cudaMalloc(&dcol, (size_t)N * deg * sizeof(int)));
  CK(cudaMalloc(&ddist, (size_t)N * sizeof(int)));
  CK(cudaMalloc(&dchanged, sizeof(int)));
  CK(cudaMemcpy(drow, rowptr, (size_t)(N + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcol, col, (size_t)N * deg * sizeof(int), cudaMemcpyHostToDevice));

  int *hdist = (int *)malloc((size_t)N * sizeof(int));
  for (int i = 0; i < N; ++i) hdist[i] = -1;
  hdist[0] = 0;
  CK(cudaMemcpy(ddist, hdist, (size_t)N * sizeof(int), cudaMemcpyHostToDevice));

  int tpb = 256, blocks = (N + tpb - 1) / tpb;
  for (int level = 0; level < N; ++level) {
    int changed = 0;
    CK(cudaMemcpy(dchanged, &changed, sizeof(int), cudaMemcpyHostToDevice));
    bfsStep<<<blocks, tpb>>>(drow, dcol, ddist, level, dchanged, N);
    CK(cudaGetLastError());
    CK(cudaMemcpy(&changed, dchanged, sizeof(int), cudaMemcpyDeviceToHost));
    if (!changed) break;
  }
  CK(cudaMemcpy(hdist, ddist, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < N; ++i) fprintf(f, "%d\n", hdist[i]); fclose(f);
  printf("bfs done: N=%d -> %s\n", N, out);
  cudaFree(drow); cudaFree(dcol); cudaFree(ddist); cudaFree(dchanged);
  free(rowptr); free(col); free(hdist);
  return 0;
}
