# daliWaterWarp

- Category: ai
- Operation: DALI nonlinear water warp
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/displacement/water.cu
- License: Apache-2.0
- CUDA features: nonlinear warp, bilinear interpolation, border handling
- Input shape/sizes: [1, 24, 24, 3]
- Output values: 1728 floating-point lines
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0001
- Build command: `nvcc -O2 -std=c++17 original/main.cu -o original/build/app`
- Run command: `original/build/app output/cuda_output.txt`

This is a standalone, deterministic benchmark adaptation for CUDA-to-SYCL migration testing. It keeps the essential indexing, interpolation, pooling, filtering, or geometry pattern from the referenced CUDA operator without requiring the upstream framework runtime.
