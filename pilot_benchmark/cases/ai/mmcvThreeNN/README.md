# mmcvThreeNN

- Category: ai
- Operation: MMCV three nearest neighbors
- Source project: open-mmlab/mmcv
- Source URL: https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/csrc/pytorch/cuda/three_nn_cuda.cu
- License: Apache-2.0
- CUDA features: nearest neighbor, point cloud distance
- Input shape/sizes: [8, 12]
- Output values: 24 floating-point lines
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 1e-05
- Build command: `nvcc -O2 -std=c++17 original/main.cu -o original/build/app`
- Run command: `original/build/app output/cuda_output.txt`

This is a standalone, deterministic benchmark adaptation for CUDA-to-SYCL migration testing. It keeps the essential indexing, interpolation, pooling, filtering, or geometry pattern from the referenced CUDA operator without requiring the upstream framework runtime.
