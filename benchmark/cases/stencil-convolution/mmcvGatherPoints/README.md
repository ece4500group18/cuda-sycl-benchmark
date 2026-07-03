# mmcvGatherPoints

- Category: ai
- Operation: MMCV gather points
- Source project: open-mmlab/mmcv
- Source URL: https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/csrc/pytorch/cuda/gather_points_cuda.cu
- License: Apache-2.0
- CUDA features: gather, batched tensor indexing
- Input shape/sizes: [2, 3, 12, 5]
- Output values: 30 floating-point lines
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0
- Build command: `nvcc -O2 -std=c++17 original/main.cu -o original/build/app`
- Run command: `original/build/app output/cuda_output.txt`

This is a standalone, deterministic benchmark adaptation for CUDA-to-SYCL migration testing. It keeps the essential indexing, interpolation, pooling, filtering, or geometry pattern from the referenced CUDA operator without requiring the upstream framework runtime.
