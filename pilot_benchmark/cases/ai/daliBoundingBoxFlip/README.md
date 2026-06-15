# daliBoundingBoxFlip

- Category: ai
- Operation: DALI bounding box horizontal flip
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/geometric/bb_flip.cu
- License: Apache-2.0
- CUDA features: bbox transform, coordinate remap
- Input shape/sizes: [16, 4]
- Output values: 64 floating-point lines
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 1e-05
- Build command: `nvcc -O2 -std=c++17 original/main.cu -o original/build/app`
- Run command: `original/build/app output/cuda_output.txt`

This is a standalone, deterministic benchmark adaptation for CUDA-to-SYCL migration testing. It keeps the essential indexing, interpolation, pooling, filtering, or geometry pattern from the referenced CUDA operator without requiring the upstream framework runtime.
