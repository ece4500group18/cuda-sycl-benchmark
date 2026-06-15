# mmcvRoiPool

- Category: ai
- Operation: MMCV ROIPool
- Source project: open-mmlab/mmcv
- Source URL: https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/csrc/pytorch/cuda/roi_pool_cuda.cu
- License: Apache-2.0
- CUDA features: ROI pooling, max reduction, pooled output
- Input shape/sizes: [1, 3, 16, 16, 4, 2, 2]
- Output values: 48 floating-point lines
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 1e-05
- Build command: `nvcc -O2 -std=c++17 original/main.cu -o original/build/app`
- Run command: `original/build/app output/cuda_output.txt`

This is a standalone, deterministic benchmark adaptation for CUDA-to-SYCL migration testing. It keeps the essential indexing, interpolation, pooling, filtering, or geometry pattern from the referenced CUDA operator without requiring the upstream framework runtime.
