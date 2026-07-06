// Integer (int8 -> int32) GEMM on Tensor Cores through the WMMA API:
// fragment declarations, load_matrix_sync, mma_sync, store_matrix_sync.
//
// Extracted from NVIDIA/cuda-samples 3_CUDA_Features/immaTensorCoreGemm
// (immaTensorCoreGemm.cu, the simple_wmma_gemm_imma demonstration kernel).
// Upstream: @ b7c5481c (BSD-3-Clause, NVIDIA).
// The kernel is upstream device code verbatim. The harness shrinks the
// matrices to 64x64x64 (4x4x4 WMMA tiles), feeds deterministic hash int8
// data and dumps the int32 result - integer math makes verification exact.
// Requires a Tensor-Core GPU (sm_72+); build with -arch=native.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <mma.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

using namespace nvcuda;

// WMMA tile dimensions (upstream)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ---- upstream kernel (verbatim) ------------------------------------------------
__global__ void simple_wmma_gemm_imma(const uint8_t *a,
                                      const uint8_t *b,
                                      const int     *c,
                                      int           *d,
                                      int            m_ld,
                                      int            n_ld,
                                      int            k_ld,
                                      int            alpha,
                                      int            beta)
{
    // Leading dimensions. Packed with no transpositions.
    int lda = m_ld;
    int ldb = k_ld;
    int ldc = n_ld;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, uint8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int>                   acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int>                   c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < k_ld; i += WMMA_K) {
        int aCol = i;
        int aRow = warpM * WMMA_M;

        int bCol = i;
        int bRow = warpN * WMMA_N;

        // Bounds checking
        if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result
    // scaled by alpha
    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;

    if (cRow < m_ld && cCol < n_ld) {
        wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
    }
}
// ---- end upstream kernel ---------------------------------------------------------

static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const int M_GLOBAL = 64, N_GLOBAL = 64, K_GLOBAL = 64;
  const int alpha = 2, beta = 3;
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";

  uint8_t *A = (uint8_t*)malloc((size_t)M_GLOBAL * K_GLOBAL);
  uint8_t *B = (uint8_t*)malloc((size_t)K_GLOBAL * N_GLOBAL);
  int *C = (int*)malloc((size_t)M_GLOBAL * N_GLOBAL * sizeof(int));
  int *D = (int*)malloc((size_t)M_GLOBAL * N_GLOBAL * sizeof(int));
  for (int i = 0; i < M_GLOBAL * K_GLOBAL; ++i) A[i] = (uint8_t)(h01((unsigned)i, 151) * 16.0f);
  for (int i = 0; i < K_GLOBAL * N_GLOBAL; ++i) B[i] = (uint8_t)(h01((unsigned)i, 152) * 16.0f);
  for (int i = 0; i < M_GLOBAL * N_GLOBAL; ++i) C[i] = (int)(h01((unsigned)i, 153) * 64.0f);

  uint8_t *d_A, *d_B; int *d_C, *d_D;
  CK(cudaMalloc(&d_A, (size_t)M_GLOBAL * K_GLOBAL));
  CK(cudaMalloc(&d_B, (size_t)K_GLOBAL * N_GLOBAL));
  CK(cudaMalloc(&d_C, (size_t)M_GLOBAL * N_GLOBAL * sizeof(int)));
  CK(cudaMalloc(&d_D, (size_t)M_GLOBAL * N_GLOBAL * sizeof(int)));
  CK(cudaMemcpy(d_A, A, (size_t)M_GLOBAL * K_GLOBAL, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_B, B, (size_t)K_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_C, C, (size_t)M_GLOBAL * N_GLOBAL * sizeof(int), cudaMemcpyHostToDevice));

  // Upstream simple-kernel launch geometry.
  dim3 blockDim(128, 4);
  dim3 gridDim((M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32),
               (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y));

  simple_wmma_gemm_imma<<<gridDim, blockDim>>>(d_A, d_B, d_C, d_D,
                                               M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(D, d_D, (size_t)M_GLOBAL * N_GLOBAL * sizeof(int), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (int i = 0; i < M_GLOBAL * N_GLOBAL; ++i) fprintf(f, "%d\n", D[i]);
  fclose(f);

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
  free(A); free(B); free(C); free(D);
  return 0;
}
