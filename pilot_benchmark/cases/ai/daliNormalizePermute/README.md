# daliNormalizePermute

- Category: ai
- Operation: DALI normalize and HWC-to-CHW permute
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/fused/normalize_permute.cu
- License: Apache-2.0
- CUDA features: layout transform, per-channel normalization, batched indexing
- Input shape/sizes: [2, 16, 16, 3]
- Output values: 1536 floating-point lines
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 1e-05
- Build command: `nvcc -O2 -std=c++17 original/main.cu -o original/build/app`
- Run command: `original/build/app output/cuda_output.txt`

This is a standalone, deterministic benchmark adaptation for CUDA-to-SYCL migration testing. It keeps the essential indexing, interpolation, pooling, filtering, or geometry pattern from the referenced CUDA operator without requiring the upstream framework runtime.
