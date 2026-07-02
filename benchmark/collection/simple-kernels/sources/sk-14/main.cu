// templatedSharedMemIdiom: a C++ template __global__ kernel that needs a
// dynamically-sized `extern __shared__` array of its own template type T.
//
// CUDA does not allow `extern __shared__ T sdata[];` to be templated
// directly -- nvcc rejects a templated `extern __shared__` declaration,
// since every instantiation would otherwise try to declare a same-named
// external symbol with a different element type. The SharedMemory<T>
// template-specialization idiom below is the standard workaround: an
// unspecialized primary template that intentionally fails to compile if
// instantiated, plus one full specialization per POD type actually used,
// each of which declares its own uniquely-named, non-templated
// `extern __shared__` array and returns a pointer to it (reinterpreted
// as the caller's T*; only one dynamic shared-memory array is ever live
// per kernel launch, so all specializations safely alias the same bank
// at runtime).
//
// The code below is reproduced, unmodified, from:
//   NVIDIA/cuda-samples
//   cpp/0_Introduction/simpleTemplates/sharedmem.cuh (the SharedMemory<T>
//     template and all of its type specializations)
//   cpp/0_Introduction/simpleTemplates/simpleTemplates.cu (testKernel<T> only)
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// testKernel<T> is instantiated here for T = int and T = float (the
// original sample's own `runTest<float>(argc, argv, 32)` and
// `runTest<int>(argc, argv, 64)`; here both are run at N = 256
// threads / 1 block instead, per this repo's index-formula input
// convention). For each type, testKernel<T>:
//   1. obtains an N-element `T *sdata` in shared memory via
//      `SharedMemory<T>::getPointer()`,
//   2. copies g_idata[tid] into sdata[tid] (`__syncthreads()`),
//   3. multiplies every shared-memory element in place by
//      `num_threads` (= blockDim.x, a value only known at kernel-launch
//      time), then syncs again,
//   4. writes sdata[tid] to g_odata[tid].
//
// This is *not* a reduction: every output element is produced
// independently by its own thread (`out[i] = in[i] * num_threads`) --
// the shared-memory round trip exists only to exercise the
// SharedMemory<T> idiom itself, not to combine values across threads.
// Because there is no cross-thread accumulation, the result is exactly
// reproducible regardless of scheduling, and (since `num_threads` here
// is 256, an exact power of two) the float instantiation's multiply
// introduces no rounding error either -- so this case uses an EXACT
// (max_abs_error == 0) correctness oracle for both T = int and T = float.
//
// Deterministic input (replicated by reference.h):
//   g_idata[i] = (T)i, for i in [0, N), N = 256 (matches the original
//   sample's own `h_idata[i] = (T)i;`).
//
// Output: argv[1] (default output/cuda_output.txt), 512 lines -- the
// first 256 are testKernel<int>'s g_odata[] (one int per line), the
// next 256 are testKernel<float>'s g_odata[] (one float per line,
// "%.9g").

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

// --- begin: verbatim from NVIDIA/cuda-samples simpleTemplates/sharedmem.cuh ---

#ifndef _SHAREDMEM_H_
#define _SHAREDMEM_H_

//****************************************************************************
// Because dynamically sized shared memory arrays are declared "extern",
// we can't templatize them directly.  To get around this, we declare a
// simple wrapper struct that will declare the extern array with a different
// name depending on the type.  This avoids compiler errors about duplicate
// definitions.
//
// To use dynamically allocated shared memory in a templatized __global__ or
// __device__ function, just replace code like this:
//
//
//  template<class T>
//  __global__ void
//  foo( T* g_idata, T* g_odata)
//  {
//      // Shared mem size is determined by the host app at run time
//      extern __shared__  T sdata[];
//      ...
//      doStuff(sdata);
//      ...
//   }
//
//   With this
//  template<class T>
//  __global__ void
//  foo( T* g_idata, T* g_odata)
//  {
//      // Shared mem size is determined by the host app at run time
//      SharedMemory<T> smem;
//      T* sdata = smem.getPointer();
//      ...
//      doStuff(sdata);
//      ...
//   }
//****************************************************************************

// This is the un-specialized struct.  Note that we prevent instantiation of
// this
// struct by putting an undefined symbol in the function body so it won't
// compile.
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

// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long, ulong, bool, float, and double
// One could also specialize it for user-defined types.

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

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
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

// Runs testKernel<T> once at N threads / 1 block, and fills `h_odata_out`
// (caller-allocated, N elements) with the GPU result and `max_abs_err`
// with the max abs error against reference_transform<T>() in reference.h.
// Returns 0 on success, non-zero (via CHECK) on any CUDA error.
template <typename T>
static int run_case(unsigned int N, T *h_odata_out, double *max_abs_err) {
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

  T *href = (T *)malloc(mem_size);
  reference_transform<T>(h_idata, href, N);

  double max_err = 0.0;
  for (unsigned int i = 0; i < N; ++i) {
    double d = (double)h_odata_out[i] - (double)href[i];
    if (d < 0) d = -d;
    if (d > max_err) max_err = d;
  }
  *max_abs_err = max_err;

  cudaFree(d_idata);
  cudaFree(d_odata);
  free(h_idata);
  free(href);
  return 0;
}

int main(int argc, char **argv) {
  const unsigned int N = 256;  // = num_threads for both instantiations below
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  int *int_result = (int *)malloc(sizeof(int) * N);
  float *float_result = (float *)malloc(sizeof(float) * N);
  double max_err_int = 0.0, max_err_float = 0.0;

  if (run_case<int>(N, int_result, &max_err_int) != 0) return 2;
  if (run_case<float>(N, float_result, &max_err_float) != 0) return 2;

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (unsigned int i = 0; i < N; ++i) fprintf(f, "%d\n", int_result[i]);
  for (unsigned int i = 0; i < N; ++i) fprintf(f, "%.9g\n", float_result[i]);
  fclose(f);

  printf("templatedSharedMemIdiom done: N=%u, "
         "max_abs_error(int)=%.3e, max_abs_error(float)=%.3e -> %s\n",
         N, max_err_int, max_err_float, out);
  printf("%s\n", (max_err_int == 0.0 && max_err_float == 0.0) ? "PASS" : "FAIL");

  free(int_result);
  free(float_result);
  return 0;
}
