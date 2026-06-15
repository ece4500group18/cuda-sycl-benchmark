# daliResizeCropMirror

- Category: ai
- Operation: DALI fused resize crop mirror
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/fused/resize_crop_mirror.cu
- License: Apache-2.0
- CUDA features: fused preprocessing, bilinear interpolation, mirror branch
- Input shape/sizes: [1, 20, 20, 3, 12, 12]
- Output values: 432 floating-point lines
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0001
- Build command: `nvcc -O2 -std=c++17 original/main.cu -o original/build/app`
- Run command: `original/build/app output/cuda_output.txt`

This is a standalone, deterministic benchmark adaptation for CUDA-to-SYCL migration testing. It keeps the essential indexing, interpolation, pooling, filtering, or geometry pattern from the referenced CUDA operator without requiring the upstream framework runtime.
