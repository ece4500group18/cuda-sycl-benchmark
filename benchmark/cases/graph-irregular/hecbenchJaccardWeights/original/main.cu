// Jaccard edge weights over a CSR graph (nvGRAPH-derived): per-row volume,
// per-edge neighbor-list intersection via binary search with atomic
// accumulation, then the Jaccard weight map — a four-kernel pipeline over
// irregular data.
//
// Extracted from HeCBench src/jaccard-cuda/main.cu (origin: nvGRAPH /
// Rodinia-era kernels, Apache-2.0 upstream header; kernels kept verbatim).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5.
// parallel_prefix_sum, fill, jaccard_row_sum, jaccard_is and jaccard_jw are
// upstream device code verbatim, instantiated unweighted (weighted=false,
// T=float) so every intersection contribution is exactly 1.0f and atomics
// stay order-independent. The host harness builds a deterministic undirected
// CSR graph and keeps upstream's launch geometry.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

#define MAX_KERNEL_THREADS 256
#define mask 0xFFFFFFFF

// ---- upstream device code (verbatim) -----------------------------------------
template<typename T>
__device__
T parallel_prefix_sum(const int n, const int *ind, const T *w)
{

  T sum = 0.0;
  T last;

  int mn =(((n+blockDim.x-1)/blockDim.x)*blockDim.x); //n in multiple of blockDim.x
  for (int i=threadIdx.x; i<mn; i+=blockDim.x) {
    //All threads (especially the last one) must always participate
    //in the shfl instruction, otherwise their sum will be undefined.
    //So, the loop stopping condition is based on multiple of n in loop increments,
    //so that all threads enter into the loop and inside we make sure we do not
    //read out of bounds memory checking for the actual size n.

    //check if the thread is valid
    bool valid  = i<n;

    //Notice that the last thread is used to propagate the prefix sum.
    //For all the threads, in the first iteration the last is 0, in the following
    //iterations it is the value at the last thread of the previous iterations.

    //get the value of the last thread
    last = __shfl_sync(mask, sum, blockDim.x-1, blockDim.x);

    //if you are valid read the value from memory, otherwise set your value to 0
    sum = (valid) ? w[ind[i]] : 0.0;

    //do prefix sum (of size warpSize=blockDim.x =< 32)
    for (int j=1; j<blockDim.x; j*=2) {
      T v = __shfl_up_sync(mask, sum, j, blockDim.x);
      if (threadIdx.x >= j) sum += v;
    }
    //shift by last
    sum += last;
    //notice that no __threadfence or __syncthreads are needed in this implementation
  }
  //get the value of the last thread (to all threads)
  last = __shfl_sync(mask, sum, blockDim.x-1, blockDim.x);

  return last;
}

// Volume of neighboors (*weight_s)
template<bool weighted, typename T>
__global__ void
jaccard_row_sum(const int n,
                const int *__restrict__ csrPtr,
                const int *__restrict__ csrInd,
                const T *__restrict__ w,
                      T *__restrict__ work)
{
  for (int row=threadIdx.y+blockIdx.y*blockDim.y; row<n; row+=gridDim.y*blockDim.y) {
    int start = csrPtr[row];
    int end   = csrPtr[row+1];
    int length= end-start;
    //compute row sums
    if (weighted) {
      T sum = parallel_prefix_sum(length, csrInd + start, w);
      if (threadIdx.x == 0) work[row] = sum;
    }
    else {
      work[row] = (T)length;
    }
  }
}

// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
// Note the number of columns is constrained by the number of rows
template<bool weighted, typename T>
__global__ void
jaccard_is(const int n, const int e,
           const int *__restrict__ csrPtr,
           const int *__restrict__ csrInd,
           const T *__restrict__ v,
           const T *__restrict__ work,
                 T *__restrict__ weight_i,
                 T *__restrict__ weight_s)
{
  for (int row=threadIdx.z+blockIdx.z*blockDim.z; row<n; row+=gridDim.z*blockDim.z) {
    for (int j=csrPtr[row]+threadIdx.y+blockIdx.y*blockDim.y;
             j<csrPtr[row+1]; j+=gridDim.y*blockDim.y) {
      int col = csrInd[j];
      //find which row has least elements (and call it reference row)
      int Ni = csrPtr[row+1] - csrPtr[row];
      int Nj = csrPtr[col+1] - csrPtr[col];
      int ref= (Ni < Nj) ? row : col;
      int cur= (Ni < Nj) ? col : row;

      //compute new sum weights
      weight_s[j] = work[row] + work[col];

      //compute new intersection weights
      //search for the element with the same column index in the reference row
      for (int i=csrPtr[ref]+threadIdx.x+blockIdx.x*blockDim.x; i<csrPtr[ref+1]; i+=gridDim.x*blockDim.x) {
        int match  =-1;
        int ref_col = csrInd[i];
        T ref_val = weighted ? v[ref_col] : (T)1.0;

        //binary search (column indices are sorted within each row)
        int left = csrPtr[cur];
        int right= csrPtr[cur+1]-1;
        while(left <= right){
          int middle = (left+right)>>1;
          int cur_col= csrInd[middle];
          if (cur_col > ref_col) {
            right=middle-1;
          }
          else if (cur_col < ref_col) {
            left=middle+1;
          }
          else {
            match = middle;
            break;
          }
        }

        //if the element with the same column index in the reference row has been found
        if (match != -1){
          atomicAdd(&weight_i[j],ref_val);
        }
      }
    }
  }
}
template<bool weighted, typename T>
__global__ void
jaccard_jw(const int e,
    const T *__restrict__ csrVal,
    const T gamma,
    const T *__restrict__ weight_i,
    const T *__restrict__ weight_s,
          T *__restrict__ weight_j)
{
  for (int j=threadIdx.x+blockIdx.x*blockDim.x; j<e; j+=gridDim.x*blockDim.x) {
    T Wi =  weight_i[j];
    T Ws =  weight_s[j];
    weight_j[j] = (gamma*csrVal[j])* (Wi/(Ws-Wi));
  }
}

template <bool weighted, typename T>
__global__ void
fill(const int e, T* w, const T value)
{
  for (int j=threadIdx.x+blockIdx.x*blockDim.x; j<e; j+=gridDim.x*blockDim.x) {
    // e.g. w[0] is the weight of a non-zeron element when csr_ind[i] equals 0.
    // So multiple non-zero elements on different rows of a matrix may share
    // the same weight value
    w[j] = weighted ? (T)(j+1)/e : value;
  }
}
// ---- end upstream device code -------------------------------------------------

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

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
  typedef float T;
  const bool weighted = false;
  const int n = 2000, picks = 3;
  const T gamma = (T)0.46;  // upstream's arbitrary gamma
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";

  int *csr_ptr, *csr_ind;
  int e = build_graph(n, picks, &csr_ptr, &csr_ind);
  T *csr_val = (T*)malloc(e * sizeof(T));
  for (int j = 0; j < e; ++j) csr_val[j] = (T)1.0;

  T *d_weight_i, *d_weight_s, *d_weight_j, *d_work, *d_csrVal;
  int *d_csrPtr, *d_csrInd;
  CK(cudaMalloc(&d_work, sizeof(T) * n));
  CK(cudaMalloc(&d_weight_i, sizeof(T) * e));
  CK(cudaMalloc(&d_weight_s, sizeof(T) * e));
  CK(cudaMalloc(&d_weight_j, sizeof(T) * e));
  CK(cudaMalloc(&d_csrVal, sizeof(T) * e));
  CK(cudaMalloc(&d_csrPtr, sizeof(int) * (n + 1)));
  CK(cudaMalloc(&d_csrInd, sizeof(int) * e));
  CK(cudaMemcpy(d_csrPtr, csr_ptr, sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_csrInd, csr_ind, sizeof(int) * e, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_csrVal, csr_val, sizeof(T) * e, cudaMemcpyHostToDevice));

  // Upstream launch geometry
  dim3 nthreads, nblocks;
  nthreads = dim3(MAX_KERNEL_THREADS, 1, 1);
  nblocks = dim3((e + MAX_KERNEL_THREADS - 1) / MAX_KERNEL_THREADS, 1, 1);
  fill<weighted, T><<<nblocks, nthreads>>>(e, d_weight_j, (T)1.0);
  fill<false, T><<<nblocks, nthreads>>>(e, d_weight_i, (T)0.0);

  const int y = 4;
  nthreads = dim3(64 / y, y, 1);
  nblocks = dim3(1, (n + nthreads.y - 1) / nthreads.y, 1);
  jaccard_row_sum<weighted, T><<<nblocks, nthreads>>>(n, d_csrPtr, d_csrInd, d_weight_j, d_work);

  nthreads = dim3(32 / y, y, 8);
  nblocks = dim3(1, 1, (n + nthreads.z - 1) / nthreads.z);
  jaccard_is<weighted, T><<<nblocks, nthreads>>>(n, e, d_csrPtr, d_csrInd,
                                                 d_weight_j, d_work, d_weight_i, d_weight_s);

  nthreads = dim3(std::min(e, MAX_KERNEL_THREADS), 1, 1);
  nblocks = dim3((e + nthreads.x - 1) / nthreads.x, 1, 1);
  jaccard_jw<weighted, T><<<nblocks, nthreads>>>(e, d_csrVal, gamma,
                                                 d_weight_i, d_weight_s, d_weight_j);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  T *weight_j = (T*)malloc(sizeof(T) * e);
  CK(cudaMemcpy(weight_j, d_weight_j, sizeof(T) * e, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (int j = 0; j < e; ++j) fprintf(f, "%.9g\n", weight_j[j]);
  fclose(f);

  cudaFree(d_work); cudaFree(d_weight_i); cudaFree(d_weight_s); cudaFree(d_weight_j);
  cudaFree(d_csrVal); cudaFree(d_csrPtr); cudaFree(d_csrInd);
  free(csr_ptr); free(csr_ind); free(csr_val); free(weight_j);
  return 0;
}
