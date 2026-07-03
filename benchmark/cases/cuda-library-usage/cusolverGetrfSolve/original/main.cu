// cuSOLVER dense LU factorization + solve (getrf/getrs with pivoting).
//
// Extracted from CUDALibrarySamples cuSOLVER/getrf (snapshot lib-02 in
// benchmark/collection/cuda-library-usage/sources; Apache-2.0, NVIDIA).
// The cusolverDn API sequence is upstream's verbatim: create handle + bind
// non-blocking stream, getrf_bufferSize workspace query, Dgetrf with pivot
// array, then Dgetrs solve. The harness scales the 3x3 example to a
// deterministic diagonally-dominant 64x64 system and dumps the solution.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#define CUDA_CHECK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(2);}}
#define CUSOLVER_CHECK(x){cusolverStatus_t s=(x);if(s!=CUSOLVER_STATUS_SUCCESS){fprintf(stderr,"cuSOLVER err %d @%d\n",(int)s,__LINE__);exit(2);}}

static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char *argv[]) {
  cusolverDnHandle_t cusolverH = NULL;
  cudaStream_t stream = NULL;

  const int m = 64;
  const int lda = m;
  const int ldb = m;
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";

  // Column-major, diagonally dominant so the system is well-conditioned.
  std::vector<double> A((size_t)lda * m);
  std::vector<double> B(m);
  for (int col = 0; col < m; ++col)
    for (int row = 0; row < m; ++row) {
      double v = 2.0 * (double)h01((unsigned)(col * m + row), 31) - 1.0;
      if (row == col) v += (double)m;
      A[(size_t)col * lda + row] = v;
    }
  for (int i = 0; i < m; ++i) B[i] = 2.0 * (double)h01((unsigned)i, 32) - 1.0;

  std::vector<double> X(m, 0);
  std::vector<int> Ipiv(m, 0);
  int info = 0;

  double *d_A = nullptr;
  double *d_B = nullptr;
  int *d_Ipiv = nullptr;
  int *d_info = nullptr;
  int lwork = 0;
  double *d_work = nullptr;

  const int pivot_on = 1;

  /* step 1: create cusolver handle, bind a stream */
  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

  /* step 2: copy A to device */
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * B.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int) * Ipiv.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(double) * B.size(), cudaMemcpyHostToDevice, stream));

  /* step 3: query working space of getrf */
  CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

  /* step 4: LU factorization */
  if (pivot_on) {
    CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, d_Ipiv, d_info));
  } else {
    CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, NULL, d_info));
  }
  CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (0 > info) {
    fprintf(stderr, "%d-th parameter is wrong \n", -info);
    return 2;
  }

  /* step 5: solve A*X = B */
  if (pivot_on) {
    CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, /* nrhs */
                                    d_A, lda, d_Ipiv, d_B, ldb, d_info));
  } else {
    CUSOLVER_CHECK(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, /* nrhs */
                                    d_A, lda, NULL, d_B, ldb, d_info));
  }
  CUDA_CHECK(cudaMemcpyAsync(X.data(), d_B, sizeof(double) * X.size(), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (int i = 0; i < m; ++i) fprintf(f, "%.12g\n", X[i]);
  fclose(f);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_Ipiv));
  CUDA_CHECK(cudaFree(d_info));
  CUDA_CHECK(cudaFree(d_work));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaDeviceReset());
  return EXIT_SUCCESS;
}
