// Blocked LU decomposition: diagonal -> perimeter -> internal kernel sweep
// repeated over the matrix diagonal — the classic Rodinia three-kernel
// pipeline with heavy shared-memory tiling.
//
// Extracted from HeCBench src/lud-cuda/lud_kernels.cu (origin: Rodinia LUD).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5 (BSD-3-Clause).
// The three kernels below are upstream device code verbatim (BLOCK_SIZE=16 as
// upstream). The host harness replicates upstream's per-offset launch loop on
// a deterministic diagonally-dominant matrix (no-pivot LU stays stable), and
// the verifier checks the L*U reconstruction residual against the input.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

#define BLOCK_SIZE 16

// ---- upstream kernels (verbatim) --------------------------------------------
__global__ void
lud_diagonal (float *m, const size_t matrix_dim, const int offset) {
  __shared__ float shadow [BLOCK_SIZE*BLOCK_SIZE];
  int i,j;
  int tx = threadIdx.x;

  size_t array_offset = offset * matrix_dim + offset;
  for(i=0; i < BLOCK_SIZE; i++){
    shadow[i * BLOCK_SIZE + tx]=m[array_offset + tx];
    array_offset += matrix_dim;
  }

  __syncthreads();

  for(i=0; i < BLOCK_SIZE-1; i++) {

    if (tx>i){
      for(j=0; j < i; j++)
        shadow[tx * BLOCK_SIZE + i] -= shadow[tx * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + i];
      shadow[tx * BLOCK_SIZE + i] /= shadow[i * BLOCK_SIZE + i];
    }

    __syncthreads();
    if (tx>i){

      for(j=0; j < i+1; j++)
        shadow[(i+1) * BLOCK_SIZE + tx] -= shadow[(i+1) * BLOCK_SIZE + j]*shadow[j * BLOCK_SIZE + tx];
    }

    __syncthreads();
  }

  array_offset = (offset+1)*matrix_dim+offset;
  for(i=1; i < BLOCK_SIZE; i++){
    m[array_offset+tx]=shadow[i * BLOCK_SIZE + tx];
    array_offset += matrix_dim;
  }
}

__global__ void
lud_perimeter (float *m, const size_t matrix_dim, const int offset) {
  __shared__ float dia [BLOCK_SIZE*BLOCK_SIZE];
  __shared__ float peri_row [BLOCK_SIZE*BLOCK_SIZE];
  __shared__ float peri_col [BLOCK_SIZE*BLOCK_SIZE];

  size_t array_offset;
  int i,j;
  int idx;

  int  bx = blockIdx.x;  
  int  tx = threadIdx.x;

  if (tx < BLOCK_SIZE) {
    idx = tx;
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE/2; i++){
      dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }

    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_row[i * BLOCK_SIZE+ idx]=m[array_offset+(bx+1)*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }

  } else {
    idx = tx-BLOCK_SIZE;

    array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
    for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++){
      dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }

    array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_col[i * BLOCK_SIZE + idx] = m[array_offset+idx];
      array_offset += matrix_dim;
    }

  }
  __syncthreads();

  if (tx < BLOCK_SIZE) { //peri-row
    idx=tx;
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i * BLOCK_SIZE + idx]-=dia[i * BLOCK_SIZE+ j]*peri_row[j * BLOCK_SIZE + idx];
    }
  } else { //peri-col
    idx=tx - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx * BLOCK_SIZE + i]-=peri_col[idx * BLOCK_SIZE+ j]*dia[j * BLOCK_SIZE + i];
      peri_col[idx * BLOCK_SIZE + i] /= dia[i * BLOCK_SIZE+ i];
    }
  }

  __syncthreads();

  if (tx < BLOCK_SIZE) { //peri-row
    idx=tx;
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(bx+1)*BLOCK_SIZE+idx] = peri_row[i*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=tx - BLOCK_SIZE;
    array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }
  }
}

__global__ void
lud_internal (float *m, const size_t matrix_dim, const int offset) {
  __shared__ float peri_row [BLOCK_SIZE*BLOCK_SIZE];
  __shared__ float peri_col [BLOCK_SIZE*BLOCK_SIZE];
  int  bx = blockIdx.x;  
  int  by = blockIdx.y;  

  int  tx = threadIdx.x;
  int  ty = threadIdx.y;

  float sum;

  int global_row_id = offset + (by+1)*BLOCK_SIZE;
  int global_col_id = offset + (bx+1)*BLOCK_SIZE;

  peri_row[ty * BLOCK_SIZE + tx] = m[(offset+ty)*matrix_dim+global_col_id+tx];
  peri_col[ty * BLOCK_SIZE + tx] = m[(global_row_id+ty)*matrix_dim+offset+tx];

  __syncthreads();

  int i;
  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++)
    sum += peri_col[ty * BLOCK_SIZE + i] * peri_row[i * BLOCK_SIZE + tx];

  m[(global_row_id+ty)*matrix_dim+global_col_id+tx] -= sum;
}
// ---- end upstream kernels ----------------------------------------------------

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const int matrix_dim = 64;
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)matrix_dim * matrix_dim * sizeof(float);

  float *m = (float*)malloc(bytes);
  for (int i = 0; i < matrix_dim; ++i)
    for (int j = 0; j < matrix_dim; ++j) {
      float v = 0.5f * (2.0f * h01((unsigned)(i * matrix_dim + j), 9) - 1.0f);
      if (i == j) v += (float)matrix_dim;  // diagonally dominant
      m[i * matrix_dim + j] = v;
    }

  float *d_m;
  CK(cudaMalloc(&d_m, bytes));
  CK(cudaMemcpy(d_m, m, bytes, cudaMemcpyHostToDevice));

  // Upstream lud_cuda() driver loop, verbatim geometry.
  int i;
  for (i = 0; i < matrix_dim - BLOCK_SIZE; i += BLOCK_SIZE) {
    lud_diagonal<<<1, BLOCK_SIZE>>>(d_m, matrix_dim, i);
    lud_perimeter<<<(matrix_dim - i) / BLOCK_SIZE - 1, 2 * BLOCK_SIZE>>>(d_m, matrix_dim, i);
    lud_internal<<<dim3((matrix_dim - i) / BLOCK_SIZE - 1, (matrix_dim - i) / BLOCK_SIZE - 1),
                   dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(d_m, matrix_dim, i);
  }
  lud_diagonal<<<1, BLOCK_SIZE>>>(d_m, matrix_dim, i);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(m, d_m, bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (int k = 0; k < matrix_dim * matrix_dim; ++k) fprintf(f, "%.9g\n", m[k]);
  fclose(f);

  cudaFree(d_m); free(m);
  return 0;
}
