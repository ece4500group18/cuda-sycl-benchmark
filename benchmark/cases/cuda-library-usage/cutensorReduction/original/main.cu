// cuTENSOR partial tensor reduction: C_{m,v} = alpha * sum_{h,k} A_{m,h,k,v}.
//
// Extracted from CUDALibrarySamples cuTENSOR/reduction.cu (snapshot lib-04
// in benchmark/collection/cuda-library-usage/sources; BSD-3-Clause, NVIDIA).
// The cuTENSOR 2.x plan-based API sequence is upstream's verbatim:
// descriptors, cutensorCreateReduction with CUTENSOR_OP_ADD, plan preference,
// workspace estimate, plan, cutensorReduce. The harness shrinks extents,
// uses deterministic hash inputs, runs once and dumps C.
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

    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.f;

    /* Computing (partial) reduction : C_{m,v} = alpha * A_{m,h,k,v} + beta * C_{m,v} */
    std::vector<int32_t> modeA{'m','h','k','v'};
    std::vector<int32_t> modeC{'m','v'};
    int32_t nmodeA = modeA.size();
    int32_t nmodeC = modeC.size();

    std::unordered_map<int32_t, int64_t> extent;
    extent['m'] = 48;
    extent['v'] = 16;
    extent['h'] = 32;
    extent['k'] = 16;

    std::vector<int64_t> extentC;
    for (auto mode : modeC) extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for (auto mode : modeA) extentA.push_back(extent[mode]);

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

    for (size_t i = 0; i < elementsA; i++) A[i] = h01((unsigned)i, 51) - 0.5f;
    for (size_t i = 0; i < elementsC; i++) C[i] = 0.0f;

    HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));

    const uint32_t kAlignment = 128;

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    /* Create Tensor Descriptors */
    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descA, nmodeA,
                 extentA.data(), NULL /* stride */, typeA, kAlignment));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descC, nmodeC,
                 extentC.data(), NULL /* stride */, typeC, kAlignment));

    const cutensorOperator_t opReduce = CUTENSOR_OP_ADD;

    /* Create Reduction Descriptor */
    cutensorOperationDescriptor_t desc;
    HANDLE_ERROR(cutensorCreateReduction(
                 handle, &desc,
                 descA, modeA.data(), CUTENSOR_OP_IDENTITY,
                 descC, modeC.data(), CUTENSOR_OP_IDENTITY,
                 descC, modeC.data(),
                 opReduce, descCompute));

    /* Set the algorithm to use */
    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
    cutensorPlanPreference_t planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(handle, &planPref, algo,
                 CUTENSOR_JIT_MODE_NONE));

    /* Query workspace estimate */
    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle, desc, planPref,
                 workspacePref, &workspaceSizeEstimate));

    /* Create Plan */
    cutensorPlan_t plan;
    HANDLE_ERROR(cutensorCreatePlan(handle, &plan, desc, planPref,
                 workspaceSizeEstimate));

    uint64_t actualWorkspaceSize = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(handle, plan,
        CUTENSOR_PLAN_REQUIRED_WORKSPACE,
        &actualWorkspaceSize, sizeof(actualWorkspaceSize)));
    assert(actualWorkspaceSize <= workspaceSizeEstimate);

    void *work = nullptr;
    if (actualWorkspaceSize > 0)
    {
        HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
        assert(uintptr_t(work) % 128 == 0);
    }

    /* Run */
    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

    HANDLE_ERROR(cutensorReduce(handle, plan,
            (const void*)&alpha, A_d,
            (const void*)&beta,  C_d,
                                 C_d, work, actualWorkspaceSize, stream));
    HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

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
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

    if (A) free(A);
    if (C) free(C);
    if (A_d) cudaFree(A_d);
    if (C_d) cudaFree(C_d);
    if (work) cudaFree(work);

    return 0;
}
