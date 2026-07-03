# Original CUDA Benchmark

This directory contains a standalone CUDA harness for `PyTorch nearest-neighbor upsample`.

The program generates deterministic inputs from a fixed hash seed, launches a CUDA kernel, copies results back to the host, and writes one numeric value per line to the output path passed in `argv[1]`.

Build:

```bash
nvcc -O2 -std=c++17 original/main.cu -o original/build/app
```

Run:

```bash
original/build/app output/cuda_output.txt
```

Upstream source: pytorch/pytorch, https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/UpSampleNearest2d.cu

License: BSD-style / BSD-3-like

Notes: this file is an adapted minimal benchmark, not a vendored copy of the full upstream framework operator.
