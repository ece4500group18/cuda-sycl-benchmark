// templatedSharedMemIdiom: a C++ template __global__ kernel that needs a
// dynamically-sized `extern __shared__` array of its own template type T, via
// the standard SharedMemory<T> template-specialization workaround.
//
// The SharedMemory<T> template (+ specializations) and testKernel<T> below are
// reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/simpleTemplates/sharedmem.cuh and
//   simpleTemplates.cu (testKernel<T> only)
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// testKernel<T> computes out[i] = in[i] * num_threads independently per thread
// (the shared-memory round trip only exercises the SharedMemory<T> idiom); no
// cross-thread accumulation. Instantiated for T = int and T = float at
// N = 256 threads / 1 block. num_threads = 256 is a power of two, so the float
// multiply is exact -> exact oracle for both types.
//
// Deterministic input (reproduced by tests/verify.py): g_idata[i] = (T)i, N=256.
//
// Output: argv[1] (default output/cuda_output.txt), 512 lines: the first 256 are
// testKernel<int>'s output (one int per line), the next 256 are
// testKernel<float>'s output (one float per line).

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

// --- begin: verbatim from NVIDIA/cuda-samples simpleTemplates/sharedmem.cuh ---

#ifndef _SHAREDMEM_H_
#define _SHAREDMEM_H_

template <typename T> struct SharedMemory
{
    // Ensure that we won't compile any un-specialized types
    __device__ T *getPointer()
    {
        extern __device__ void error(void);
        error();
        return NULL;
    }
};

template <> struct SharedMemory<int>
{
    __device__ int *getPointer()
    {
        extern __shared__ int s_int[];
        return s_int;
    }
};

template <> struct SharedMemory<unsigned int>
{
    __device__ unsigned int *getPointer()
    {
        extern __shared__ unsigned int s_uint[];
        return s_uint;
    }
};

template <> struct SharedMemory<char>
{
    __device__ char *getPointer()
    {
        extern __shared__ char s_char[];
        return s_char;
    }
};

template <> struct SharedMemory<unsigned char>
{
    __device__ unsigned char *getPointer()
    {
        extern __shared__ unsigned char s_uchar[];
        return s_uchar;
    }
};

template <> struct SharedMemory<short>
{
    __device__ short *getPointer()
    {
        extern __shared__ short s_short[];
        return s_short;
    }
};

template <> struct SharedMemory<unsigned short>
{
    __device__ unsigned short *getPointer()
    {
        extern __shared__ unsigned short s_ushort[];
        return s_ushort;
    }
};

template <> struct SharedMemory<long>
{
    __device__ long *getPointer()
    {
        extern __shared__ long s_long[];
        return s_long;
    }
};

template <> struct SharedMemory<unsigned long>
{
    __device__ unsigned long *getPointer()
    {
        extern __shared__ unsigned long s_ulong[];
        return s_ulong;
    }
};

template <> struct SharedMemory<bool>
{
    __device__ bool *getPointer()
    {
        extern __shared__ bool s_bool[];
        return s_bool;
    }
};

template <> struct SharedMemory<float>
{
    __device__ float *getPointer()
    {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <> struct SharedMemory<double>
{
    __device__ double *getPointer()
    {
        extern __shared__ double s_double[];
        return s_double;
    }
};

#endif //_SHAREDMEM_H_

// --- end: verbatim from NVIDIA/cuda-samples sharedmem.cuh ---

// --- begin: verbatim from NVIDIA/cuda-samples simpleTemplates/simpleTemplates.cu ---

template <class T> __global__ void testKernel(T *g_idata, T *g_odata)
{
    // Shared mem size is determined by the host app at run time
    SharedMemory<T> smem;
    T              *sdata = smem.getPointer();

    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    // read in input data from global memory
    sdata[tid] = g_idata[tid];
    __syncthreads();

    // perform some computations
    sdata[tid] = (T)num_threads * sdata[tid];
    __syncthreads();

    // write data to global memory
    g_odata[tid] = sdata[tid];
}

// --- end: verbatim from NVIDIA/cuda-samples simpleTemplates.cu ---

// Deterministic input generator (new host code; mirrored in tests/verify.py).
template <typename T> static inline T gen_idata(unsigned int i) { return (T)i; }

template <typename T>
static int run_case(unsigned int N, T *h_odata_out) {
  const size_t mem_size = sizeof(T) * N;

  T *h_idata = (T *)malloc(mem_size);
  for (unsigned int i = 0; i < N; ++i) h_idata[i] = gen_idata<T>(i);

  T *d_idata, *d_odata;
  CHECK(cudaMalloc(&d_idata, mem_size));
  CHECK(cudaMalloc(&d_odata, mem_size));
  CHECK(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

  dim3 grid(1, 1, 1);
  dim3 threads(N, 1, 1);
  testKernel<T><<<grid, threads, mem_size>>>(d_idata, d_odata);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(h_odata_out, d_odata, mem_size, cudaMemcpyDeviceToHost));

  cudaFree(d_idata);
  cudaFree(d_odata);
  free(h_idata);
  return 0;
}

int main(int argc, char **argv) {
  const unsigned int N = 256;  // = num_threads for both instantiations
  const char *out = (argc > 1) ? argv[1] : "output/cuda_output.txt";

  int *int_result = (int *)malloc(sizeof(int) * N);
  float *float_result = (float *)malloc(sizeof(float) * N);

  if (run_case<int>(N, int_result) != 0) return 2;
  if (run_case<float>(N, float_result) != 0) return 2;

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (unsigned int i = 0; i < N; ++i) fprintf(f, "%d\n", int_result[i]);
  for (unsigned int i = 0; i < N; ++i) fprintf(f, "%.9g\n", float_result[i]);
  fclose(f);

  free(int_result);
  free(float_result);
  return 0;
}
