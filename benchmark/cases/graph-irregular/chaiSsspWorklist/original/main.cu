// Chai SSSP: single-source shortest paths with a double-buffered work queue.
// The kernel drains the input frontier through per-block shared-memory local
// queues (with overflow spill), relaxes edges via atomicMax over negated
// costs, and concatenates the new frontier into the global output queue.
//
// Extracted from HeCBench src/sssp-cuda (origin: Chai heterogeneous
// benchmark suite, University of Cordoba / University of Illinois license
// preserved in the snapshot).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5.
// The SSSP_gpu kernel and the constants/types from support/common.h are
// upstream code verbatim. The host driver keeps upstream's structure
// (host-side first iteration, queue swap by iteration parity, per-iteration
// scalar uploads) but drops the heterogeneous CPU-thread path and the
// file-based graph input: it is GPU-only over a deterministic directed CSR
// graph. Costs are stored negated (INF = -2^31+1), so atomicMax performs
// min-relaxation; final distances are the negated costs.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

// ---- upstream constants and types (support/common.h, verbatim) ---------------
#define INF -2147483647
#define UP_LIMIT 16677216 //2^24
#define WHITE 16677217
#define GRAY 16677218
#define GRAY0 16677219
#define GRAY1 16677220
#define BLACK 16677221
#define W_QUEUE_SIZE 1600

typedef struct {
    int x;
    int y;
} Node;
typedef struct {
    int x;
    int y;
} Edge;

// ---- upstream kernel (verbatim) ----------------------------------------------
// GPU kernel
__global__ void SSSP_gpu(
    const Node *__restrict__ graph_nodes_av,
    const Edge *__restrict__ graph_edges_av,
    int *__restrict__ cost,
    int *__restrict__ color,
    const int *__restrict__ q1,
          int *__restrict__ q2,
    const int *__restrict__ n_t,
    int *__restrict__ head,
    int *__restrict__ tail,
    int *__restrict__ overflow,
    const int *__restrict__ gray_shade,
    int *__restrict__ iter)
{
  __shared__ int l_mem[W_QUEUE_SIZE+2];
  __shared__ int tail_bin;
  int* l_q2 = l_mem;
  int* shift = l_mem + W_QUEUE_SIZE;
  int* base = l_mem + W_QUEUE_SIZE + 1;

  const int tid     = threadIdx.x;
  const int gtid    = blockIdx.x * blockDim.x + threadIdx.x;
  const int WG_SIZE = blockDim.x;

  int n_t_local = *n_t; // atomicAdd(n_t, 0);
  int gray_shade_local = *gray_shade; // atomicAdd(&gray_shade[0], 0);

  if(tid == 0) {
    // Reset queue
    tail_bin = 0;
  }

  // Fetch frontier elements from the queue
  if(tid == 0)
    *base = atomicAdd(&head[0], WG_SIZE);
  __syncthreads();

  int my_base = *base;
  while(my_base < n_t_local) {

    // If local queue might overflow
    if(tail_bin >= W_QUEUE_SIZE / 2) {
      if(tid == 0) {
        // Add local tail_bin to tail
        *shift = atomicAdd(&tail[0], tail_bin);
      }
      __syncthreads();
      int local_shift = tid;
      while(local_shift < tail_bin) {
        q2[*shift + local_shift] = l_q2[local_shift];
        // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
        local_shift += WG_SIZE;
      }
      __syncthreads();
      if(tid == 0) {
        // Reset local queue
        tail_bin = 0;
      }
      __syncthreads();
    }

    if(my_base + tid < n_t_local && *overflow == 0) {
      // Visit a node from the current frontier
      int pid = q1[my_base + tid];
      //////////////// Visit node ///////////////////////////
      atomicExch(&color[pid], BLACK); // Node visited
      int  cur_cost = cost[pid]; // atomicAdd(&cost[pid], 0); // Look up shortest-path distance to this node
      Node cur_node;
      cur_node.x = graph_nodes_av[pid].x;
      cur_node.y = graph_nodes_av[pid].y;
      Edge cur_edge;
      // For each outgoing edge
      for(int i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
        cur_edge.x = graph_edges_av[i].x;
        cur_edge.y = graph_edges_av[i].y;
        int id     = cur_edge.x;
        int cost_local   = cur_edge.y;
        cost_local += cur_cost;
        int orig_cost = atomicMax(&cost[id], cost_local);
        if(orig_cost < cost_local) {
          int old_color = atomicMax(&color[id], gray_shade_local);
          if(old_color != gray_shade_local) {
            // Push to the queue
            int tail_index = atomicAdd(&tail_bin, 1);
            if(tail_index >= W_QUEUE_SIZE) {
              *overflow = 1;
            } else
              l_q2[tail_index] = id;
          }
        }
      }
    }

    if(tid == 0)
      *base = atomicAdd(&head[0], WG_SIZE); // Fetch more frontier elements from the queue
    __syncthreads();
    my_base = *base;
  }
  /////////////////////////////////////////////////////////
  // Compute size of the output and allocate space in the global queue
  if(tid == 0) {
    *shift = atomicAdd(&tail[0], tail_bin);
  }
  __syncthreads();
  ///////////////////// CONCATENATE INTO GLOBAL MEMORY /////////////////////
  int local_shift = tid;
  while(local_shift < tail_bin) {
    q2[*shift + local_shift] = l_q2[local_shift];
    // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
    local_shift += WG_SIZE;
  }
  //////////////////////////////////////////////////////////////////////////

  if(gtid == 0) {
    atomicAdd(&iter[0], 1);
  }
}
// ---- end upstream kernel ------------------------------------------------------

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const int n_nodes = 4096, picks = 4;
  const int n_gpu_blocks = 8, n_gpu_threads = 128;
  const int source = 0;
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";

  // Deterministic directed graph: 'picks' distinct out-edges per node with
  // hash weights in [1,9], emitted as CSR (Node = {start, count}).
  unsigned char *adj = (unsigned char*)calloc((size_t)n_nodes * n_nodes, 1);
  for (int u = 0; u < n_nodes; ++u)
    for (int k = 0; k < picks; ++k) {
      int w = (int)(h01((unsigned)u, 600 + k) * (float)n_nodes);
      if (w > n_nodes - 1) w = n_nodes - 1;
      if (w != u) adj[(size_t)u * n_nodes + w] = 1;
    }
  int n_edges = 0;
  for (size_t i = 0; i < (size_t)n_nodes * n_nodes; ++i) n_edges += adj[i];

  Node *h_nodes = (Node*)malloc(sizeof(Node) * n_nodes);
  Edge *h_edges = (Edge*)malloc(sizeof(Edge) * n_edges);
  int pos = 0;
  for (int u = 0; u < n_nodes; ++u) {
    h_nodes[u].x = pos;
    for (int w = 0; w < n_nodes; ++w)
      if (adj[(size_t)u * n_nodes + w]) {
        h_edges[pos].x = w;
        int cost = 1 + (int)(h01((unsigned)(u * 16 + (pos - h_nodes[u].x)), 610) * 9.0f);
        h_edges[pos].y = -cost;  // upstream reader stores negated costs
        pos++;
      }
    h_nodes[u].y = pos - h_nodes[u].x;
  }
  free(adj);

  int *h_cost = (int*)malloc(sizeof(int) * n_nodes);
  int *h_color = (int*)malloc(sizeof(int) * n_nodes);
  int *h_q1 = (int*)malloc(sizeof(int) * n_nodes);
  int *h_q2 = (int*)malloc(sizeof(int) * n_nodes);
  int h_head, h_tail, h_num_t, h_overflow, h_gray_shade, h_iter;

  Node *d_nodes; Edge *d_edges;
  int *d_cost, *d_color, *d_q1, *d_q2, *d_head, *d_tail, *d_num_t, *d_overflow, *d_gray_shade, *d_iter;
  CK(cudaMalloc(&d_nodes, sizeof(Node) * n_nodes));
  CK(cudaMalloc(&d_edges, sizeof(Edge) * n_edges));
  CK(cudaMalloc(&d_cost, sizeof(int) * n_nodes));
  CK(cudaMalloc(&d_color, sizeof(int) * n_nodes));
  CK(cudaMalloc(&d_q1, sizeof(int) * n_nodes));
  CK(cudaMalloc(&d_q2, sizeof(int) * n_nodes));
  CK(cudaMalloc(&d_head, sizeof(int)));
  CK(cudaMalloc(&d_tail, sizeof(int)));
  CK(cudaMalloc(&d_num_t, sizeof(int)));
  CK(cudaMalloc(&d_overflow, sizeof(int)));
  CK(cudaMalloc(&d_gray_shade, sizeof(int)));
  CK(cudaMalloc(&d_iter, sizeof(int)));
  CK(cudaMemcpy(d_nodes, h_nodes, sizeof(Node) * n_nodes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_edges, h_edges, sizeof(Edge) * n_edges, cudaMemcpyHostToDevice));

  // Initialization (upstream reset block)
  for (int i = 0; i < n_nodes; ++i) h_cost[i] = INF;
  h_cost[source] = 0;
  for (int i = 0; i < n_nodes; ++i) h_color[i] = WHITE;
  h_tail = 0; h_head = 0;
  h_q1[0] = source;
  h_iter = 0;
  h_overflow = 0;
  h_gray_shade = GRAY0;

  // Run first iteration in master CPU thread (upstream, single-threaded)
  h_num_t = 1;
  for (int index_i = 0; index_i < h_num_t; index_i++) {
    int pid = h_q1[index_i];
    h_color[pid] = BLACK;
    int cur_cost = h_cost[pid];
    for (int i = h_nodes[pid].x; i < (h_nodes[pid].y + h_nodes[pid].x); i++) {
      int id = h_edges[i].x;
      int cost = h_edges[i].y;
      cost += cur_cost;
      h_cost[id] = cost;
      h_color[id] = GRAY0;
      int index_o = h_tail++;
      h_q2[index_o] = id;
    }
  }
  h_num_t = h_tail;
  h_tail = 0;
  h_gray_shade = GRAY1;
  h_iter++;

  // Copy state to device once; queues swap on the device afterwards.
  CK(cudaMemcpy(d_cost, h_cost, sizeof(int) * n_nodes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_color, h_color, sizeof(int) * n_nodes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_q1, h_q1, sizeof(int) * n_nodes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_q2, h_q2, sizeof(int) * n_nodes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_overflow, &h_overflow, sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_iter, &h_iter, sizeof(int), cudaMemcpyHostToDevice));

  // Run subsequent iterations on the GPU until the frontier queue is empty
  // (upstream GPU_EXEC path).
  while (h_num_t != 0) {
    int *d_qin, *d_qout;
    if (h_iter % 2 == 0) { d_qin = d_q1; d_qout = d_q2; }
    else                 { d_qin = d_q2; d_qout = d_q1; }

    CK(cudaMemcpy(d_num_t, &h_num_t, sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tail, &h_tail, sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_head, &h_head, sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_gray_shade, &h_gray_shade, sizeof(int), cudaMemcpyHostToDevice));

    dim3 dimGrid(n_gpu_blocks);
    dim3 dimBlock(n_gpu_threads);
    SSSP_gpu<<<dimGrid, dimBlock>>>(d_nodes, d_edges, d_cost, d_color,
                                    d_qin, d_qout, d_num_t,
                                    d_head, d_tail, d_overflow, d_gray_shade, d_iter);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    CK(cudaMemcpy(&h_tail, d_tail, sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&h_iter, d_iter, sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_overflow) { fprintf(stderr, "queue overflow\n"); return 2; }

    h_num_t = h_tail;  // Number of elements in output queue
    h_tail = 0;
    h_head = 0;
    if (h_iter % 2 == 0) h_gray_shade = GRAY0;
    else                 h_gray_shade = GRAY1;
  }

  CK(cudaMemcpy(h_cost, d_cost, sizeof(int) * n_nodes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (int i = 0; i < n_nodes; ++i) fprintf(f, "%d\n", h_cost[i]);
  fclose(f);

  cudaFree(d_nodes); cudaFree(d_edges); cudaFree(d_cost); cudaFree(d_color);
  cudaFree(d_q1); cudaFree(d_q2); cudaFree(d_head); cudaFree(d_tail);
  cudaFree(d_num_t); cudaFree(d_overflow); cudaFree(d_gray_shade); cudaFree(d_iter);
  free(h_nodes); free(h_edges); free(h_cost); free(h_color); free(h_q1); free(h_q2);
  return 0;
}
