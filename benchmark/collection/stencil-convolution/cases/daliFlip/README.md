# daliFlip

- Category: ai
- Operation: DALI batched image flip
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/geometric/flip.cu
- License: Apache-2.0
- CUDA features: geometric transform, branching, batched indexing
- Input shape/sizes: [2, 16, 18, 3]
- Output values: 1728 floating-point lines
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0
- Build command: `nvcc -O2 -std=c++17 original/main.cu -o original/build/app`
- Run command: `original/build/app output/cuda_output.txt`

This is a standalone, deterministic benchmark adaptation for CUDA-to-SYCL migration testing. It keeps the essential indexing, interpolation, pooling, filtering, or geometry pattern from the referenced CUDA operator without requiring the upstream framework runtime.
