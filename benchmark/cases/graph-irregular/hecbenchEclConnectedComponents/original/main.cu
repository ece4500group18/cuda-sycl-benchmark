// ECL-CC: connected components via lock-free hooking and pointer jumping —
// a five-kernel pipeline with degree-based work partitioning (thread / warp /
// block granularity worklists).
//
// Extracted from HeCBench src/cc-cuda/main.cu (origin: ECL-CC, Jaiganesh &
// Burtscher, Texas State University).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5 (BSD-3-Clause).
// The five kernels (init, compute1, compute2, compute3, flatten), the
// representative() helper and the device worklist counters below are
// upstream code verbatim. The host harness builds a deterministic undirected
// CSR graph and uses a fixed grid size. Hooking always links the larger
// representative to the smaller, so final labels are the component minima —
// deterministic regardless of the atomics' interleaving.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

static const int ThreadsPerBlock = 256;
static const int warpsize = 32;

static __device__ int topL, posL, topH, posH;

// ---- upstream kernels (verbatim) --------------------------------------------
/* initialize with first smaller neighbor ID */

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void init(const int nodes,
          const int* const __restrict__ nidx,
          const int* const __restrict__ nlist,
                int* const __restrict__ nstat)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    const int beg = nidx[v];
    const int end = nidx[v + 1];
    int m = v;
    int i = beg;
    while ((m == v) && (i < end)) {
      m = min(m, nlist[i]);
      i++;
    }
    nstat[v] = m;
  }

  if (from == 0) {topL = 0; posL = 0; topH = nodes - 1; posH = nodes - 1;}
}

/* intermediate pointer jumping */

static inline __device__ int representative(const int idx, int* const __restrict__ nstat)
{
  int curr = nstat[idx];
  if (curr != idx) {
    int next, prev = idx;
    while (curr > (next = nstat[curr])) {
      nstat[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
}

/* process low-degree vertices at thread granularity and fill worklists */

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute1(const int nodes,
              const int* const __restrict__ nidx,
              const int* const __restrict__ nlist,
                    int* const __restrict__ nstat,
                    int* const __restrict__ wl)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    const int vstat = nstat[v];
    if (v != vstat) {
      const int beg = nidx[v];
      const int end = nidx[v + 1];
      int deg = end - beg;
      if (deg > 16) {
        int idx;
        if (deg <= 352) {
          idx = atomicAdd(&topL, 1);
        } else {
          idx = atomicAdd(&topH, -1);
        }
        wl[idx] = v;
      } else {
        int vstat = representative(v, nstat);
        for (int i = beg; i < end; i++) {
          const int nli = nlist[i];
          if (v > nli) {
            int ostat = representative(nli, nstat);
            bool repeat;
            do {
              repeat = false;
              if (vstat != ostat) {
                int ret;
                if (vstat < ostat) {
                  if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                    ostat = ret;
                    repeat = true;
                  }
                } else {
                  if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                    vstat = ret;
                    repeat = true;
                  }
                }
              }
            } while (repeat);
          }
        }
      }
    }
  }
}

/* process medium-degree vertices at warp granularity */

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute2(const int nodes,
              const int* const __restrict__ nidx,
              const int* const __restrict__ nlist,
                    int* const __restrict__ nstat,
              const int* const __restrict__ wl)
{
  const int lane = threadIdx.x % warpsize;

  int idx;
  if (lane == 0) idx = atomicAdd(&posL, 1);
  idx = __shfl_sync(0xffffffff, idx, 0);
  while (idx < topL) {
    const int v = wl[idx];
    int vstat = representative(v, nstat);
    for (int i = nidx[v] + lane; i < nidx[v + 1]; i += warpsize) {
      const int nli = nlist[i];
      if (v > nli) {
        int ostat = representative(nli, nstat);
        bool repeat;
        do {
          repeat = false;
          if (vstat != ostat) {
            int ret;
            if (vstat < ostat) {
              if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                ostat = ret;
                repeat = true;
              }
            } else {
              if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                vstat = ret;
                repeat = true;
              }
            }
          }
        } while (repeat);
      }
    }
    if (lane == 0) idx = atomicAdd(&posL, 1);
    idx = __shfl_sync(0xffffffff, idx, 0);
  }
}

/* process high-degree vertices at block granularity */

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute3(const int nodes,
              const int* const __restrict__ nidx,
              const int* const __restrict__ nlist,
                    int* const __restrict__ nstat,
              const int* const __restrict__ wl)
{
  __shared__ int vB;
  if (threadIdx.x == 0) {
    const int idx = atomicAdd(&posH, -1);
    vB = (idx > topH) ? wl[idx] : -1;
  }
  __syncthreads();
  while (vB >= 0) {
    const int v = vB;
    __syncthreads();
    int vstat = representative(v, nstat);
    for (int i = nidx[v] + threadIdx.x; i < nidx[v + 1]; i += ThreadsPerBlock) {
      const int nli = nlist[i];
      if (v > nli) {
        int ostat = representative(nli, nstat);
        bool repeat;
        do {
          repeat = false;
          if (vstat != ostat) {
            int ret;
            if (vstat < ostat) {
              if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                ostat = ret;
                repeat = true;
              }
            } else {
              if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                vstat = ret;
                repeat = true;
              }
            }
          }
        } while (repeat);
      }
    }
    if (threadIdx.x == 0) {
      const int idx = atomicAdd(&posH, -1);
      vB = (idx > topH) ? wl[idx] : -1;
    }
    __syncthreads();
  }
}

/* link all vertices to sink */

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void flatten(const int nodes,
             const int* const __restrict__ nidx,
             const int* const __restrict__ nlist,
                   int* const __restrict__ nstat)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    int next, vstat = nstat[v];
    const int old = vstat;
    while (vstat > (next = nstat[vstat])) {
      vstat = next;
    }
    if (old != vstat) nstat[v] = vstat;
  }
}
// ---- end upstream kernels ----------------------------------------------------

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

// Deterministic undirected graph, deliberately sparse (isolated vertices and
// several components), emitted as CSR with ascending column indices.
static int build_graph(int nodes, int picks, int **nidx_out, int **nlist_out) {
  unsigned char *adj = (unsigned char*)calloc((size_t)nodes * nodes, 1);
  for (int u = 0; u < nodes; ++u)
    for (int k = 0; k < picks; ++k) {
      // ~60% of pick slots are skipped to fragment the graph
      if (h01((unsigned)u, 400 + k) < 0.6f) continue;
      int w = (int)(h01((unsigned)u, 500 + k) * (float)nodes);
      if (w > nodes - 1) w = nodes - 1;
      if (w != u) { adj[(size_t)u * nodes + w] = 1; adj[(size_t)w * nodes + u] = 1; }
    }
  int *nidx = (int*)malloc((nodes + 1) * sizeof(int));
  int edges = 0;
  for (int u = 0; u < nodes; ++u)
    for (int w = 0; w < nodes; ++w) edges += adj[(size_t)u * nodes + w];
  int *nlist = (int*)malloc((edges > 0 ? edges : 1) * sizeof(int));
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
  int *nstat = (int*)malloc(nodes * sizeof(int));

  int *d_nidx, *d_nlist, *d_nstat, *d_wl;
  CK(cudaMalloc(&d_nidx, (nodes + 1) * sizeof(int)));
  CK(cudaMalloc(&d_nlist, (edges > 0 ? edges : 1) * sizeof(int)));
  CK(cudaMalloc(&d_nstat, nodes * sizeof(int)));
  CK(cudaMalloc(&d_wl, nodes * sizeof(int)));
  CK(cudaMemcpy(d_nidx, nidx, (nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_nlist, nlist, (edges > 0 ? edges : 1) * sizeof(int), cudaMemcpyHostToDevice));

  const int blocks = 24;
  init<<<blocks, ThreadsPerBlock>>>(nodes, d_nidx, d_nlist, d_nstat);
  compute1<<<blocks, ThreadsPerBlock>>>(nodes, d_nidx, d_nlist, d_nstat, d_wl);
  compute2<<<blocks, ThreadsPerBlock>>>(nodes, d_nidx, d_nlist, d_nstat, d_wl);
  compute3<<<blocks, ThreadsPerBlock>>>(nodes, d_nidx, d_nlist, d_nstat, d_wl);
  flatten<<<blocks, ThreadsPerBlock>>>(nodes, d_nidx, d_nlist, d_nstat);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(nstat, d_nstat, nodes * sizeof(int), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (int i = 0; i < nodes; ++i) fprintf(f, "%d\n", nstat[i]);
  fclose(f);

  cudaFree(d_nidx); cudaFree(d_nlist); cudaFree(d_nstat); cudaFree(d_wl);
  free(nidx); free(nlist); free(nstat);
  return 0;
}
