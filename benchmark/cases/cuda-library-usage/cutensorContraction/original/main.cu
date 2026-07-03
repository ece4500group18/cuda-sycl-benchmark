// cuTENSOR tensor contraction: C_{m,u,n,v} = alpha * A_{m,h,k,n} B_{u,k,v,h}.
//
// Extracted from CUDALibrarySamples cuTENSOR/contraction.cu (snapshot lib-03
// in benchmark/collection/cuda-library-usage/sources; BSD-3-Clause, NVIDIA).
// The cuTENSOR 2.x plan-based API sequence is upstream's verbatim: tensor
// descriptors, contraction operation descriptor, scalar-type query, plan
// preference, workspace estimate, plan, contract. The harness shrinks the
// extents, replaces rand() with a deterministic hash, runs once (no timing
// loop) and dumps C.
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
    typedef float floatTypeB;
    typedef float floatTypeC;

    cutensorDataType_t typeA = CUTENSOR_R_32F;
    cutensorDataType_t typeB = CUTENSOR_R_32F;
    cutensorDataType_t typeC = CUTENSOR_R_32F;
    const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

    /* Computing: C_{m,u,n,v} = alpha * A_{m,h,k,n} B_{u,k,v,h} + beta * C_{m,u,n,v} */
    std::vector<int> modeC{'m','u','n','v'};
    std::vector<int> modeA{'m','h','k','n'};
    std::vector<int> modeB{'u','k','v','h'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    extent['m'] = 16;
    extent['n'] = 16;
    extent['u'] = 16;
    extent['v'] = 8;
    extent['h'] = 8;
    extent['k'] = 8;

    std::vector<int64_t> extentC;
    for (auto mode : modeC) extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for (auto mode : modeA) extentA.push_back(extent[mode]);
    std::vector<int64_t> extentB;
    for (auto mode : modeB) extentB.push_back(extent[mode]);

    size_t elementsA = 1;
    for (auto mode : modeA) elementsA *= extent[mode];
    size_t elementsB = 1;
    for (auto mode : modeB) elementsB *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC) elementsC *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeB = sizeof(floatTypeB) * elementsB;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    void *A_d, *B_d, *C_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &A_d, sizeA));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &B_d, sizeB));
    HANDLE_CUDA_ERROR(cudaMalloc((void**) &C_d, sizeC));

    floatTypeA *A = (floatTypeA*) malloc(sizeA);
    floatTypeB *B = (floatTypeB*) malloc(sizeB);
    floatTypeC *C = (floatTypeC*) malloc(sizeC);
    if (A == NULL || B == NULL || C == NULL) return -1;

    // Deterministic inputs (replaces upstream rand())
    for (size_t i = 0; i < elementsA; i++) A[i] = h01((unsigned)i, 41) - 0.5f;
    for (size_t i = 0; i < elementsB; i++) B[i] = h01((unsigned)i, 42) - 0.5f;
    for (size_t i = 0; i < elementsC; i++) C[i] = 0.0f;

    HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));

    const uint32_t kAlignment = 128;
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(B_d) % kAlignment == 0);
    assert(uintptr_t(C_d) % kAlignment == 0);

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    /* Create Tensor Descriptors */
    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descA, nmodeA,
                 extentA.data(), NULL /*stride*/, typeA, kAlignment));

    cutensorTensorDescriptor_t descB;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descB, nmodeB,
                 extentB.data(), NULL /*stride*/, typeB, kAlignment));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descC, nmodeC,
                 extentC.data(), NULL /*stride*/, typeC, kAlignment));

    /* Create Contraction Descriptor */
    cutensorOperationDescriptor_t desc;
    HANDLE_ERROR(cutensorCreateContraction(handle, &desc,
                 descA, modeA.data(), /* unary operator A*/CUTENSOR_OP_IDENTITY,
                 descB, modeB.data(), /* unary operator B*/CUTENSOR_OP_IDENTITY,
                 descC, modeC.data(), /* unary operator C*/CUTENSOR_OP_IDENTITY,
                 descC, modeC.data(),
                 descCompute));

    /* Ensure that the scalar type is correct. */
    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle, desc,
        CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
        (void*)&scalarType, sizeof(scalarType)));
    assert(scalarType == CUTENSOR_R_32F);
    typedef float floatTypeCompute;
    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.f;

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

    /* Create Contraction Plan */
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

    HANDLE_ERROR(cutensorContract(handle, plan,
                 (void*) &alpha, A_d, B_d,
                 (void*) &beta,  C_d, C_d,
                 work, actualWorkspaceSize, stream));
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
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (work) cudaFree(work);

    return 0;
}
