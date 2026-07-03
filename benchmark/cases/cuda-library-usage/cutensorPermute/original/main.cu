// cuTENSOR elementwise permutation: C_{c,w,h,n} = alpha * A_{w,h,c,n}
// (NHWC -> NCHW-style layout change).
//
// Extracted from CUDALibrarySamples cuTENSOR/elementwise_permute.cu
// (snapshot lib-05 in benchmark/collection/cuda-library-usage/sources;
// BSD-3-Clause, NVIDIA). The cuTENSOR 2.x plan-based API sequence is
// upstream's verbatim: descriptors, cutensorCreatePermutation, scalar-type
// query, plan preference, plan (no workspace), cutensorPermute. The harness
// shrinks extents, uses deterministic hash inputs, runs once and dumps C.
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>

#define HANDLE_ERROR(x)                                             \
{ const auto err = x;                                               \
  if( err != CUTENSOR_STATUS_SUCCESS )                              \
  { printf("Error: %s\n", cutensorGetErrorString(err)); exit(-1); } \
};

#define HANDLE_CUDA_ERROR(x)                                    \
{ const auto err = x;                                           \
  if( err != cudaSuccess )                                      \
  { printf("Error: %s\n", cudaGetErrorString(err)); exit(-1); } \
};

static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv)
{
    const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";
    typedef float floatTypeA;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cutensorDataType_t typeA = CUTENSOR_R_32F;
    cutensorDataType_t typeC = CUTENSOR_R_32F;
    const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.0f;

    /* C_{c,w,h,n} = alpha * A_{w,h,c,n} */
    std::vector<int> modeA{'w','h','c','n'};
    std::vector<int> modeC{'c','w','h','n'};
    int nmodeA = modeA.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    extent['h'] = 16;
    extent['w'] = 8;
    extent['c'] = 16;
    extent['n'] = 16;

    std::vector<int64_t> extentA;
    for (auto mode : modeA) extentA.push_back(extent[mode]);
    std::vector<int64_t> extentC;
    for (auto mode : modeC) extentC.push_back(extent[mode]);

    size_t elementsA = 1;
    for (auto mode : modeA) elementsA *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC) elementsC *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    void *A_d, *C_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&A_d, sizeA));
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&C_d, sizeC));

    floatTypeA *A = (floatTypeA*)malloc(sizeA);
    floatTypeC *C = (floatTypeC*)malloc(sizeC);
    if (A == NULL || C == NULL) return -1;

    for (size_t i = 0; i < elementsA; i++) A[i] = h01((unsigned)i, 61) - 0.5f;
    for (size_t i = 0; i < elementsC; i++) C[i] = 0.0f;

    HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));

    const uint32_t kAlignment = 128;

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    /* Create Tensor Descriptors */
    cutensorTensorDescriptor_t  descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descA,
                                                nmodeA,
                                                extentA.data(),
                                                nullptr /* stride */,
                                                typeA,
                                                kAlignment));

    cutensorTensorDescriptor_t  descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descC,
                                                nmodeC,
                                                extentC.data(),
                                                nullptr /* stride */,
                                                typeC,
                                                kAlignment));

    /* Create Permutation Descriptor */
    cutensorOperationDescriptor_t  desc;
    HANDLE_ERROR(cutensorCreatePermutation(handle,
                                           &desc,
                                           descA,
                                           modeA.data(),
                                           CUTENSOR_OP_IDENTITY,
                                           descC,
                                           modeC.data(),
                                           descCompute));

    /* Ensure that the scalar type is correct. */
    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle, desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                         (void*)&scalarType,
                                                         sizeof(scalarType)));
    assert(scalarType == CUTENSOR_R_32F);

    /* Set the algorithm to use */
    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
    cutensorPlanPreference_t  planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(handle,
                                              &planPref,
                                              algo,
                                              CUTENSOR_JIT_MODE_NONE));

    /* Create Plan */
    cutensorPlan_t  plan;
    HANDLE_ERROR(cutensorCreatePlan(handle,
                                    &plan,
                                    desc,
                                    planPref,
                                    0 /* workspaceSizeLimit */));

    /* Run */
    HANDLE_ERROR(cutensorPermute(handle,
                    plan,
                    &alpha, A_d, C_d, nullptr /* stream */));
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

    HANDLE_CUDA_ERROR(cudaMemcpy(C, C_d, sizeC, cudaMemcpyDeviceToHost));

    FILE *f = fopen(out_path, "w");
    if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
    for (size_t i = 0; i < elementsC; i++) fprintf(f, "%.9g\n", C[i]);
    fclose(f);

    HANDLE_ERROR(cutensorDestroy(handle));
    HANDLE_ERROR(cutensorDestroyPlan(plan));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));

    if (A) free(A);
    if (C) free(C);
    if (A_d) cudaFree(A_d);
    if (C_d) cudaFree(C_d);

    return 0;
}
