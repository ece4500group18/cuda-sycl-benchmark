# pytorchMaxPool2d

- Category: ai
- Operation: PyTorch MaxPool2d CUDA
- Source project: pytorch/pytorch
- Source URL: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MaxPool2d.cu
- License: BSD-style / BSD-3-like
- CUDA features: pooling, window max, NCHW indexing
- Input shape/sizes: [1, 2, 8, 8]
- Output values: 32 floating-point lines
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0
- Build command: `nvcc -O2 -std=c++17 original/main.cu -o original/build/app`
- Run command: `original/build/app output/cuda_output.txt`

This is a standalone, deterministic benchmark adaptation for CUDA-to-SYCL migration testing. It keeps the essential indexing, interpolation, pooling, filtering, or geometry pattern from the referenced CUDA operator without requiring the upstream framework runtime.
