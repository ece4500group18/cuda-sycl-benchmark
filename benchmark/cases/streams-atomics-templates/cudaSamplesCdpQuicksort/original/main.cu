// Quicksort with CUDA Dynamic Parallelism: the kernel recursively launches
// itself on sub-ranges via device-side streams, falling back to selection
// sort at depth/size limits.
//
// Extracted from NVIDIA/cuda-samples 3_CUDA_Features/cdpSimpleQuicksort
// (cdpSimpleQuicksort.cu). Upstream: @ b7c5481c (BSD-3-Clause, NVIDIA).
// selection_sort, cdp_simple_quicksort and the run_qsort wrapper are
// upstream code verbatim. The harness replaces srand-based input with
// deterministic hash values. Requires relocatable device code
// (-rdc=true -lcudadevrt); SYCL has no dynamic-parallelism equivalent,
// making this a deliberately hard migration case.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

#define MAX_DEPTH      16
#define INSERTION_SORT 32

// ---- upstream device code (verbatim) -------------------------------------------
////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
__device__ void selection_sort(unsigned int *data, int left, int right)
{
    for (int i = left; i <= right; ++i) {
        unsigned min_val = data[i];
        int      min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i + 1; j <= right; ++j) {
            unsigned val_j = data[j];

            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx) {
            data[min_idx] = data[i];
            data[i]       = min_val;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Very basic quicksort algorithm, recursively launching the next level.
////////////////////////////////////////////////////////////////////////////////
__global__ void cdp_simple_quicksort(unsigned int *data, int left, int right, int depth)
{
    // If we're too deep or there are few elements left, we use an insertion
    // sort...
    if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT) {
        selection_sort(data, left, right);
        return;
    }

    unsigned int *lptr  = data + left;
    unsigned int *rptr  = data + right;
    unsigned int  pivot = data[(left + right) / 2];

    // Do the partitioning.
    while (lptr <= rptr) {
        // Find the next left- and right-hand values to swap
        unsigned int lval = *lptr;
        unsigned int rval = *rptr;

        // Move the left pointer as long as the pointed element is smaller than the
        // pivot.
        while (lval < pivot) {
            lptr++;
            lval = *lptr;
        }

        // Move the right pointer as long as the pointed element is larger than the
        // pivot.
        while (rval > pivot) {
            rptr--;
            rval = *rptr;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr) {
            *lptr++ = rval;
            *rptr-- = lval;
        }
    }

    // Now the recursive part
    int nright = rptr - data;
    int nleft  = lptr - data;

    // Launch a new block to sort the left part.
    if (left < (rptr - data)) {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, s>>>(data, left, nright, depth + 1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr - data) < right) {
        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<<1, 1, 0, s1>>>(data, nleft, right, depth + 1);
        cudaStreamDestroy(s1);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Call the quicksort kernel from the host.
////////////////////////////////////////////////////////////////////////////////
void run_qsort(unsigned int *data, unsigned int nitems)
{
    // Launch on device
    int left  = 0;
    int right = nitems - 1;
    cdp_simple_quicksort<<<1, 1>>>(data, left, right, 0);
    cudaDeviceSynchronize();
}
// ---- end upstream code -----------------------------------------------------------

static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const unsigned int num_items = 4096;
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";

  unsigned int *h_data = (unsigned int*)malloc(num_items * sizeof(unsigned int));
  // Deterministic values in [0, num_items) like upstream's rand() % nitems.
  for (unsigned int i = 0; i < num_items; ++i)
    h_data[i] = (unsigned int)(h01(i, 141) * (float)num_items);

  unsigned int *d_data;
  CK(cudaMalloc(&d_data, num_items * sizeof(unsigned int)));
  CK(cudaMemcpy(d_data, h_data, num_items * sizeof(unsigned int), cudaMemcpyHostToDevice));

  run_qsort(d_data, num_items);
  CK(cudaGetLastError());

  CK(cudaMemcpy(h_data, d_data, num_items * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (unsigned int i = 0; i < num_items; ++i) fprintf(f, "%u\n", h_data[i]);
  fclose(f);

  cudaFree(d_data); free(h_data);
  return 0;
}
