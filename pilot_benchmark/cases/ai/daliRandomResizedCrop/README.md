# daliRandomResizedCrop

- Category: ai
- Operation: DALI random resized crop
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/resize/random_resized_crop.cu
- License: Apache-2.0
- CUDA features: crop window mapping, bilinear interpolation, image preprocessing
- Input shape/sizes: [1, 28, 28, 3, 14, 14]
- Output values: 588 floating-point lines
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0001
- Build command: `nvcc -O2 -std=c++17 original/main.cu -o original/build/app`
- Run command: `original/build/app output/cuda_output.txt`

This is a standalone, deterministic benchmark adaptation for CUDA-to-SYCL migration testing. It keeps the essential indexing, interpolation, pooling, filtering, or geometry pattern from the referenced CUDA operator without requiring the upstream framework runtime.
