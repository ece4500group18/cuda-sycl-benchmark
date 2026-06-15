# mmcvFurthestPointSample

- Category: ai
- Operation: MMCV furthest point sampling
- Source project: open-mmlab/mmcv
- Source URL: https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/csrc/pytorch/cuda/furthest_point_sample_cuda.cu
- License: Apache-2.0
- CUDA features: point sampling, greedy distance update
- Input shape/sizes: [16, 6]
- Output values: 6 floating-point lines
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0
- Build command: `nvcc -O2 -std=c++17 original/main.cu -o original/build/app`
- Run command: `original/build/app output/cuda_output.txt`

This is a standalone, deterministic benchmark adaptation for CUDA-to-SYCL migration testing. It keeps the essential indexing, interpolation, pooling, filtering, or geometry pattern from the referenced CUDA operator without requiring the upstream framework runtime.
