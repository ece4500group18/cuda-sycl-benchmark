// backpropTreeReduction: shared-memory power-of-two-stride tree reduction
// computing a weighted partial sum per hidden node, from the input-to-hidden
// layer-forward pass of a (toy) backpropagation neural network.
//
// The __global__ kernel below (bpnn_layerforward_CUDA) is reproduced,
// unmodified, from:
//   Rodinia benchmark suite (gpu-rodinia mirror),
//   cuda/backprop/backprop_cuda_kernel.cu
//   https://github.com/yuhc/gpu-rodinia/blob/master/cuda/backprop/backprop_cuda_kernel.cu
//   Copyright (c) 2008-2011 University of Virginia (BSD-style, permissive)
//
// Deterministic inputs (reproduced bit-for-bit by tests/verify.py):
//   input_node[i]       = ((i % 7) - 3) * 0.5      for i in [0, in]
//   weight_matrix[i][j] = (((i + j) % 5) - 2) * 0.1  for i in [0, in], j in [0, hid]
//   in = 8192 (num_blocks = in/16 = 512), hid = WIDTH = 16.
//
// Output: argv[1] (default output/cuda_output.txt), one float per line, the
// 512 x 16 = 8192-element hidden_partial_sum array (one weighted partial sum
// per (input-block, hidden-unit) pair). Correctness is checked by
// tests/verify.py (CPU reference performing the identical tree-reduction order).

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                           \
      return 2;                                                              \
    }                                                                        \
  } while (0)

// --- begin: verbatim from Rodinia cuda/backprop/backprop.h (constants) ---
#define WIDTH 16  // shared memory width
#define HEIGHT 16 // shared memory height
// --- end: verbatim from Rodinia cuda/backprop/backprop.h ---

// --- begin: verbatim from Rodinia cuda/backprop/backprop_cuda_kernel.cu ---

__global__ void
bpnn_layerforward_CUDA(float *input_cuda,
	                   float *output_hidden_cuda,
					   float *input_hidden_cuda,
					   float *hidden_partial_sum,
					   int in,
					   int hid)
{
   int by = blockIdx.y;
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;

   int index_in = HEIGHT * by + ty + 1;

   __shared__ float input_node[HEIGHT];
   __shared__ float weight_matrix[HEIGHT][WIDTH];


   if ( tx == 0 )
   input_node[ty] = input_cuda[index_in] ;

   __syncthreads();

   weight_matrix[ty][tx] = input_hidden_cuda[index];

   __syncthreads();

   weight_matrix[ty][tx] = weight_matrix[ty][tx] * input_node[ty];

   __syncthreads();

   for ( int i = 1 ; i <= __log2f(HEIGHT) ; i++){

	   int power_two = __powf(2, i);

	   if( ty % power_two == 0 )
	   weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];

	   __syncthreads();

   }

   //__syncthreads();

   input_hidden_cuda[index] = weight_matrix[ty][tx];

   __syncthreads();

   if ( tx == 0 ) {
	   hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
   }

}

// --- end: verbatim from Rodinia cuda/backprop/backprop_cuda_kernel.cu ---

// Deterministic input generators (new host code; mirrored in tests/verify.py).
static inline float gen_input(int i) { return (float)((i % 7) - 3) * 0.5f; }
static inline float gen_weight(int i, int j) { return (float)(((i + j) % 5) - 2) * 0.1f; }

int main(int argc, char **argv) {
  const int in = 8192;               // number of input units, multiple of HEIGHT
  const int hid = WIDTH;             // number of hidden units == WIDTH (required by kernel)
  const int num_blocks = in / HEIGHT; // 512
  const char *out = (argc > 1) ? argv[1] : "output/cuda_output.txt";

  const size_t in_bytes = (size_t)(in + 1) * sizeof(float);
  const size_t hidden_bytes = (size_t)(hid + 1) * sizeof(float);
  const size_t weight_bytes = (size_t)(in + 1) * (hid + 1) * sizeof(float);
  const size_t partial_bytes = (size_t)num_blocks * hid * sizeof(float);

  float *h_input = (float *)malloc(in_bytes);
  float *h_weights = (float *)malloc(weight_bytes);
  float *h_partial = (float *)malloc(partial_bytes);

  for (int i = 0; i <= in; ++i) h_input[i] = gen_input(i);
  for (int i = 0; i <= in; ++i)
    for (int j = 0; j <= hid; ++j)
      h_weights[i * (hid + 1) + j] = gen_weight(i, j);

  float *d_input, *d_output_hidden, *d_weights, *d_partial;
  CHECK(cudaMalloc(&d_input, in_bytes));
  CHECK(cudaMalloc(&d_output_hidden, hidden_bytes));
  CHECK(cudaMalloc(&d_weights, weight_bytes));
  CHECK(cudaMalloc(&d_partial, partial_bytes));

  CHECK(cudaMemcpy(d_input, h_input, in_bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_weights, h_weights, weight_bytes, cudaMemcpyHostToDevice));

  dim3 grid(1, num_blocks);
  dim3 threads(WIDTH, HEIGHT);

  bpnn_layerforward_CUDA<<<grid, threads>>>(d_input, d_output_hidden, d_weights,
                                             d_partial, in, hid);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(h_partial, d_partial, partial_bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < num_blocks * hid; ++i) fprintf(f, "%.9g\n", h_partial[i]);
  fclose(f);

  cudaFree(d_input);
  cudaFree(d_output_hidden);
  cudaFree(d_weights);
  cudaFree(d_partial);
  free(h_input);
  free(h_weights);
  free(h_partial);
  return 0;
}
