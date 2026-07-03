// ECL-MIS: maximal independent set via prioritized selection.
//
// Extracted from HeCBench src/mis-cuda/main.cu (origin: ECL-MIS, Burtscher et
// al., Texas State University).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5 (BSD-3-Clause).
// The init and findmins kernels (and the device hash) below are upstream
// code verbatim: hash-derived per-node priorities, then a lock-free spinning
// kernel that admits local priority maxima and knocks out their neighbors.
// The fixed point is unique, so results are deterministic. The host harness
// builds a deterministic undirected CSR graph.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

static const int ThreadsPerBlock = 256;

typedef unsigned char stattype;
static const stattype in = 0xfe;
static const stattype out = 0;

// ---- upstream kernels (verbatim) --------------------------------------------
/* main computation kernel */

__global__
void findmins(const int nodes,
    const int* const __restrict nidx,
    const int* const __restrict nlist,
    volatile stattype* const __restrict nstat)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  int missing;
  do {
    missing = 0;
    for (int v = from; v < nodes; v += incr) {
      const stattype nv = nstat[v];
      if (nv & 1) {
        int i = nidx[v];
        while ((i < nidx[v + 1]) && ((nv > nstat[nlist[i]]) || ((nv == nstat[nlist[i]]) && (v > nlist[i])))) {
          i++;
        }
        if (i < nidx[v + 1]) {
          missing = 1;
        } else {
          for (int i = nidx[v]; i < nidx[v + 1]; i++) {
            nstat[nlist[i]] = out;
          }
          nstat[v] = in;
        }
      }
    }
  } while (missing != 0);
}

/* hash function to generate random values */

// source of hash function: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
__device__
unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

/* prioritized-selection initialization kernel */

__global__
void init(const int nodes,
    const int edges,
    const int* const __restrict nidx,
    stattype* const __restrict nstat)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  const float avg = (float)edges / nodes;
  const float scaledavg = ((in / 2) - 1) * avg;

  for (int i = from; i < nodes; i += incr) {
    stattype val = in;
    const int degree = nidx[i + 1] - nidx[i];
    if (degree > 0) {
      float x = degree - (hash(i) * 0.00000000023283064365386962890625f);
      int res = int(scaledavg / (avg + x));
      val = (res + res) | 1;
    }
    nstat[i] = val;
  }
}
// ---- end upstream kernels ----------------------------------------------------

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

// Deterministic undirected graph: adjacency matrix from k hash picks per
// node, symmetrized, emitted as CSR with ascending column indices.
static int build_graph(int nodes, int picks, int **nidx_out, int **nlist_out) {
  unsigned char *adj = (unsigned char*)calloc((size_t)nodes * nodes, 1);
  for (int u = 0; u < nodes; ++u)
    for (int k = 0; k < picks; ++k) {
      int w = (int)(h01((unsigned)u, 300 + k) * (float)nodes);
      if (w > nodes - 1) w = nodes - 1;
      if (w != u) { adj[(size_t)u * nodes + w] = 1; adj[(size_t)w * nodes + u] = 1; }
    }
  int *nidx = (int*)malloc((nodes + 1) * sizeof(int));
  int edges = 0;
  for (int u = 0; u < nodes; ++u)
    for (int w = 0; w < nodes; ++w) edges += adj[(size_t)u * nodes + w];
  int *nlist = (int*)malloc(edges * sizeof(int));
  int pos = 0;
  for (int u = 0; u < nodes; ++u) {
    nidx[u] = pos;
    for (int w = 0; w < nodes; ++w)
      if (adj[(size_t)u * nodes + w]) nlist[pos++] = w;
  }
  nidx[nodes] = pos;
  free(adj);
  *nidx_out = nidx; *nlist_out = nlist;
  return edges;
}

int main(int argc, char **argv) {
  const int nodes = 2000, picks = 3;
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";

  int *nidx, *nlist;
  int edges = build_graph(nodes, picks, &nidx, &nlist);
  stattype *nstat = (stattype*)malloc(nodes);

  int *d_nidx, *d_nlist; stattype *d_nstat;
  CK(cudaMalloc(&d_nidx, (nodes + 1) * sizeof(int)));
  CK(cudaMalloc(&d_nlist, edges * sizeof(int)));
  CK(cudaMalloc(&d_nstat, nodes));
  CK(cudaMemcpy(d_nidx, nidx, (nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_nlist, nlist, edges * sizeof(int), cudaMemcpyHostToDevice));

  const int blocks = 24;  // upstream launch geometry
  init<<<blocks, ThreadsPerBlock>>>(nodes, edges, d_nidx, d_nstat);
  findmins<<<blocks, ThreadsPerBlock>>>(nodes, d_nidx, d_nlist, d_nstat);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(nstat, d_nstat, nodes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (int i = 0; i < nodes; ++i) fprintf(f, "%d\n", (int)nstat[i]);
  fclose(f);

  cudaFree(d_nidx); cudaFree(d_nlist); cudaFree(d_nstat);
  free(nidx); free(nlist); free(nstat);
  return 0;
}
