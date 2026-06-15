# daliCast

- Category: ai
- Operation: DALI cast with clamp and round
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/util/cast.cu
- License: Apache-2.0
- CUDA features: type conversion, clamp, rounding
- Input shape/sizes: [1024]
- Output values: 1024 floating-point lines
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0
- Build command: `nvcc -O2 -std=c++17 original/main.cu -o original/build/app`
- Run command: `original/build/app output/cuda_output.txt`

This is a standalone, deterministic benchmark adaptation for CUDA-to-SYCL migration testing. It keeps the essential indexing, interpolation, pooling, filtering, or geometry pattern from the referenced CUDA operator without requiring the upstream framework runtime.
