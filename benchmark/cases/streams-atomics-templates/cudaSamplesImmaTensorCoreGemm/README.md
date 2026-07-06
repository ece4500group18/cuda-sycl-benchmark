# Int8 Tensor-Core GEMM via the WMMA API

uint8 x uint8 -> int32 GEMM on Tensor Cores using wmma fragments with row/col-major layouts and fragment-wise alpha/beta scaling.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/immaTensorCoreGemm/immaTensorCoreGemm.cu

Upstream commit: b7c5481c

License: BSD-3-Clause + CUDA EULA note

Extraction fidelity: extracted

Extraction notes: simple_wmma_gemm_imma kernel verbatim (fragments, load/store_matrix_sync, mma_sync); matrices shrunk to 64^3, deterministic int8 hash data - integer math keeps verification exact. Needs sm_72+; built with -arch=native.
