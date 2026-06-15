# daliMakeContiguous

- Category: ai
- Operation: DALI make contiguous strided tensor copy
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/util/make_contiguous.cu
- License: Apache-2.0
- CUDA features: strided memory access, layout compaction
- Input shape/sizes: [2, 4, 5, 6]
- Output values: 240 floating-point lines
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0
- Build command: `nvcc -O2 -std=c++17 original/main.cu -o original/build/app`
- Run command: `original/build/app output/cuda_output.txt`

This is a standalone, deterministic benchmark adaptation for CUDA-to-SYCL migration testing. It keeps the essential indexing, interpolation, pooling, filtering, or geometry pattern from the referenced CUDA operator without requiring the upstream framework runtime.
