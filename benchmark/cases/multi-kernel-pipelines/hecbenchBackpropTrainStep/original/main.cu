// Rodinia backprop training step: shared-memory layer-forward kernel with an
// in-block tree reduction, host-side sigmoid squash over partial sums, then
// the weight-adjustment kernel — a device/host/device pipeline.
//
// Extracted from HeCBench src/backprop-cuda (origin: Rodinia backprop,
// CMU face-recognition NN).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5 (BSD-3-Clause).
// kernel_layerforward and kernel_adjust_weights (bpnn_layerforward.h /
// bpnn_adjust_weights.h) are upstream device code verbatim, with upstream's
// launch geometry and the host partial-sum squash loop kept structurally
// intact. The harness replaces the random network setup and the CPU error
// backpropagation chain with deterministic hash data (documented), then
// dumps hidden units and the adjusted input weights.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

// ---- upstream constants (backprop.h, verbatim subset) -------------------------
#define THREADS 256
#define WIDTH 16  // shared memory width
#define HEIGHT 16 // shared memory height
#define BLOCK_SIZE 16

#define ETA 0.3f       //eta value
#define MOMENTUM 0.3f  //momentum value

// ---- upstream kernels (verbatim) ----------------------------------------------
__global__ void kernel_layerforward(
  const float*__restrict__ input,
        float*__restrict__ input_weights,
        float*__restrict__ hidden_partial_sum,
  const int hid)
{
  __shared__ float input_node[HEIGHT];
  __shared__ float weight_matrix[HEIGHT * WIDTH];

  // gridDim.y << gridDim.x
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int index = ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;

  int index_in = HEIGHT * by + ty + 1;

  if ( tx == 0 )
    input_node[ty] = input[index_in] ;
  __syncthreads();

  weight_matrix[ty * WIDTH + tx] =  input_weights[index];
  __syncthreads();

  weight_matrix[ty * WIDTH + tx]= weight_matrix[ty * WIDTH + tx] * input_node[ty];
  __syncthreads();

  for ( int i = 1 ; i <= HEIGHT ; i=i*2){
    int power_two = i;

    if( ty % power_two == 0 )
      weight_matrix[ty * WIDTH + tx]= weight_matrix[ty * WIDTH + tx] + weight_matrix[(ty + power_two/2)* WIDTH + tx];

    __syncthreads();

  }

  input_weights[index] =  weight_matrix[ty * WIDTH + tx];

  __syncthreads();

  if ( tx == 0 ) {
    hidden_partial_sum[by * hid + ty] = weight_matrix[tx* WIDTH + ty];
  }
}

__global__ void kernel_adjust_weights (
  const float*__restrict__ ly,
       float *__restrict__ w,
  const float*__restrict__ delta,
        float*__restrict__ oldw,
  const int hid)
{
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;
  int index_y = HEIGHT * by + ty + 1;
  int index_x = tx + 1;

  w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
  oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

  __syncthreads();

  if (ty == 0 && by ==0){
    w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
    oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
  }
}
// ---- end upstream kernels -------------------------------------------------------

static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const int in = 4096, hid = 16;
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";
  const unsigned int num_blocks = in / BLOCK_SIZE;
  const size_t wsize = (size_t)(in + 1) * (hid + 1);

  float *input_units = (float*)malloc((in + 1) * sizeof(float));
  float *input_weights_one_dim = (float*)malloc(wsize * sizeof(float));
  float *input_weights_prev_one_dim = (float*)malloc(wsize * sizeof(float));
  float *partial_sum = (float*)malloc((size_t)num_blocks * WIDTH * sizeof(float));
  float *hidden_units = (float*)malloc((hid + 1) * sizeof(float));
  float *hidden_delta = (float*)malloc((hid + 1) * sizeof(float));

  // Deterministic network state (replaces upstream's random setup and, for
  // hidden_delta, the CPU error-backpropagation chain).
  for (int i = 0; i <= in; ++i) input_units[i] = h01((unsigned)i, 91);
  for (size_t i = 0; i < wsize; ++i) {
    input_weights_one_dim[i] = 2.0f * h01((unsigned)i, 92) - 1.0f;
    input_weights_prev_one_dim[i] = 0.1f * (2.0f * h01((unsigned)i, 93) - 1.0f);
  }
  for (int j = 0; j <= hid; ++j) hidden_delta[j] = 0.1f * (2.0f * h01((unsigned)j, 94) - 1.0f);

  float *d_input, *d_input_weights, *d_hidden_partial_sum, *d_hidden_delta, *d_input_prev_weights;
  CK(cudaMalloc(&d_input, sizeof(float) * (in + 1)));
  CK(cudaMalloc(&d_input_weights, sizeof(float) * wsize));
  CK(cudaMalloc(&d_hidden_partial_sum, sizeof(float) * num_blocks * WIDTH));
  CK(cudaMemcpy(d_input, input_units, sizeof(float) * (in + 1), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_input_weights, input_weights_one_dim, sizeof(float) * wsize, cudaMemcpyHostToDevice));

  dim3 grid(1, num_blocks);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

  kernel_layerforward<<<grid, threads>>>(d_input, d_input_weights, d_hidden_partial_sum, hid);
  CK(cudaGetLastError());
  CK(cudaMemcpy(partial_sum, d_hidden_partial_sum, sizeof(float) * num_blocks * WIDTH, cudaMemcpyDeviceToHost));

  // Host squash over block partial sums (upstream loop, verbatim structure).
  for (int j = 1; j <= hid; j++) {
    float sum = 0.f;
    for (unsigned int k = 0; k < num_blocks; k++) {
      sum += partial_sum[k * hid + j - 1];
    }
    sum += input_weights_one_dim[j];  // net->input_weights[0][j]
    hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }

  // input_weights has been written in the first kernel, so it needs to be restored.
  CK(cudaMemcpy(d_input_weights, input_weights_one_dim, sizeof(float) * wsize, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d_hidden_delta, sizeof(float) * (hid + 1)));
  CK(cudaMalloc(&d_input_prev_weights, sizeof(float) * wsize));
  CK(cudaMemcpy(d_hidden_delta, hidden_delta, sizeof(float) * (hid + 1), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_input_prev_weights, input_weights_prev_one_dim, sizeof(float) * wsize, cudaMemcpyHostToDevice));
  kernel_adjust_weights<<<grid, threads>>>(d_input, d_input_weights, d_hidden_delta, d_input_prev_weights, hid);
  CK(cudaGetLastError());
  CK(cudaMemcpy(input_weights_one_dim, d_input_weights, sizeof(float) * wsize, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (int j = 1; j <= hid; ++j) fprintf(f, "%.9g\n", hidden_units[j]);
  for (size_t i = 0; i < wsize; ++i) fprintf(f, "%.9g\n", input_weights_one_dim[i]);
  fclose(f);

  cudaFree(d_input); cudaFree(d_input_weights); cudaFree(d_hidden_partial_sum);
  cudaFree(d_hidden_delta); cudaFree(d_input_prev_weights);
  free(input_units); free(input_weights_one_dim); free(input_weights_prev_one_dim);
  free(partial_sum); free(hidden_units); free(hidden_delta);
  return 0;
}
