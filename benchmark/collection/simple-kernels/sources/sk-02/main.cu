// backpropTreeReduction: shared-memory power-of-two-stride tree reduction
// computing a weighted partial sum per hidden node, from the input-to-hidden
// layer-forward pass of a real (if tiny/toy) backpropagation neural network.
//
// The __global__ kernel below is reproduced, unmodified, from:
//   Rodinia benchmark suite (gpu-rodinia mirror), cuda/backprop/backprop_cuda_kernel.cu
//   https://github.com/yuhc/gpu-rodinia/blob/master/cuda/backprop/backprop_cuda_kernel.cu
//   Copyright (c) 2008-2011 University of Virginia
//   License: BSD-style, U. Virginia (permissive; see LICENSE in this directory)
//
// Only `bpnn_layerforward_CUDA` is reproduced here (the suite's second
// kernel, `bpnn_adjust_weights_cuda`, implements the backward weight-update
// pass and is unrelated to the tree-reduction idiom this case isolates, so
// it is omitted -- see README.md).
//
// Each thread block covers HEIGHT=16 input units ("rows", indexed by
// threadIdx.y) and WIDTH=16 hidden units ("columns", indexed by
// threadIdx.x). For its column `tx` (one hidden unit), a block must compute
// the dot product of 16 input-unit activations with the 16 corresponding
// input->hidden weights -- i.e. a 16-wide reduction. The kernel does this
// with the classic shared-memory power-of-two-stride tree reduction:
//   for i = 1..log2(HEIGHT): stride = 2^i;
//     if (ty % stride == 0) weight_matrix[ty][tx] += weight_matrix[ty+stride/2][tx];
//     __syncthreads();
// after which weight_matrix[0][tx] holds the full 16-element weighted sum
// for hidden unit tx, one column computed per thread-block-row, and the
// kernel writes one such partial sum per (block, hidden-unit) pair to
// `hidden_partial_sum`. This is the same reduction *idiom* as this repo's
// warpShuffleReduction case, but expressed as an explicit __shared__-memory,
// __syncthreads()-gated binary tree (not warp shuffles), fixed at a
// compile-time HEIGHT/WIDTH of 16, and embedded in a real application's
// forward pass rather than a standalone microbenchmark.
//
// HEIGHT and WIDTH (both 16, "shared memory height/width") are hardcoded
// locally below exactly as defined in the original suite's backprop.h,
// rather than pulling in the full BPNN struct / training driver / build
// system, which are irrelevant to the kernel under test.
//
// The kernel's own `__log2f(HEIGHT)` / `__powf(2, i)` fast-math calls are
// kept verbatim (they are part of the reproduced kernel). With HEIGHT
// fixed at compile time to 16 -- an exact power of two representable
// exactly in float -- `__log2f(16.0f)` is exactly 4.0f and `__powf(2, i)`
// for i in {1,2,3,4} is exactly {2.0f,4.0f,8.0f,16.0f}: every stride the
// reduction takes is an exact, deterministic power of two, so the sequence
// of tree-reduction steps performed on the GPU is fully determined by the
// fixed HEIGHT/WIDTH constants and does not vary run to run.
//
// Deterministic inputs (replicated by reference.h):
//   input_node[i]     = ((i % 7) - 3) * 0.5      for i in [0, in]
//   weight_matrix[i][j] = ((i + j) % 5 - 2) * 0.1  for i in [0, in], j in [0, hid]
// in = 8192 (multiple of HEIGHT=16, so num_blocks = in/16 = 512 divides
// evenly, matching the original driver's own `num_blocks = in / 16`),
// hid = WIDTH = 16 (the kernel's thread-block x-dimension is fixed at
// WIDTH, so it only computes correctly when the caller's hidden-layer
// width equals WIDTH -- exactly as in the original suite, which always
// launches with `dim3 threads(16, 16)`).
//
// Output: argv[1] (default output/cuda_output.txt), one float per line,
// the 512 x 16 = 8192-element `hidden_partial_sum` array produced by the
// kernel (one weighted partial sum per (input-block, hidden-unit) pair).

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "reference.h"

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                          \
      return 2;                                                             \
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

/*
   for ( unsigned int i = 2 ; i <= HEIGHT ; i *= 2){

	   unsigned int power_two = i - 1;

	   if( (ty & power_two) == 0 ) {
		weight_matrix[ty][tx] = weight_matrix[ty][tx] + weight_matrix[ty + power_two/2][tx];
	   }

   }
   */

   __syncthreads();

   if ( tx == 0 ) {
	   hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty];
   }

}

// --- end: verbatim from Rodinia cuda/backprop/backprop_cuda_kernel.cu ---

int main(int argc, char **argv) {
  const int in = 8192;               // number of input units, multiple of HEIGHT
  const int hid = WIDTH;             // number of hidden units == WIDTH (required by kernel)
  const int num_blocks = in / HEIGHT; // 512
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

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

  // CPU reference check: mirrors the exact same power-of-two-stride tree
  // summation order as the kernel (see reference.h), so the comparison is
  // an exact float match, not a tolerance-bounded one.
  float *href_partial = (float *)malloc(partial_bytes);
  reference_layerforward(h_input, h_weights, href_partial, in, hid);

  double max_abs_err = 0.0;
  for (int i = 0; i < num_blocks * hid; ++i) {
    double d = (double)h_partial[i] - (double)href_partial[i];
    if (d < 0) d = -d;
    if (d > max_abs_err) max_abs_err = d;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < num_blocks * hid; ++i) fprintf(f, "%.9g\n", h_partial[i]);
  fclose(f);

  printf("backpropTreeReduction done: in=%d, hid=%d, num_blocks=%d, "
         "max_abs_error(gpu vs ref)=%.3e -> %s\n",
         in, hid, num_blocks, max_abs_err, out);
  printf("%s\n", (max_abs_err == 0.0) ? "PASS" : "FAIL");

  cudaFree(d_input);
  cudaFree(d_output_hidden);
  cudaFree(d_weights);
  cudaFree(d_partial);
  free(h_input);
  free(h_weights);
  free(h_partial);
  free(href_partial);
  return 0;
}
