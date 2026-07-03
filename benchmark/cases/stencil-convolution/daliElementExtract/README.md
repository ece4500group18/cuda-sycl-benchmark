# daliElementExtract

- Category: ai
- Operation: DALI sequence element extract
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/sequence/element_extract.cu
- License: Apache-2.0
- CUDA features: sequence indexing, gather
- Input shape/sizes: [4, 6, 5]
- Output values: 20 floating-point lines
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0
- Build command: `nvcc -O2 -std=c++17 original/main.cu -o original/build/app`
- Run command: `original/build/app output/cuda_output.txt`

This is a standalone, deterministic benchmark adaptation for CUDA-to-SYCL migration testing. It keeps the essential indexing, interpolation, pooling, filtering, or geometry pattern from the referenced CUDA operator without requiring the upstream framework runtime.
