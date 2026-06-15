# daliPaste

- Category: ai
- Operation: DALI paste/composite
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/paste/paste.cu
- License: Apache-2.0
- CUDA features: conditional copy, image compositing, ROI indexing
- Input shape/sizes: [24, 24, 3, 8, 10]
- Output values: 1728 floating-point lines
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 1e-05
- Build command: `nvcc -O2 -std=c++17 original/main.cu -o original/build/app`
- Run command: `original/build/app output/cuda_output.txt`

This is a standalone, deterministic benchmark adaptation for CUDA-to-SYCL migration testing. It keeps the essential indexing, interpolation, pooling, filtering, or geometry pattern from the referenced CUDA operator without requiring the upstream framework runtime.
