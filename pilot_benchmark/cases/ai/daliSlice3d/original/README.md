# Original CUDA case

This standalone CUDA benchmark implements `DALI batched 3D slice` with deterministic hashed inputs and writes one floating-point value per line to the output path given as `argv[1]`.

Build:

```bash
nvcc -O2 -std=c++17 original/main.cu -o original/build/app
```

Run:

```bash
original/build/app output/cuda_output.txt
```

Upstream attribution: adapted from NVIDIA/DALI, `https://github.com/NVIDIA/DALI/blob/main/dali/operators/crop/slice.cu`.

License note: DALI is Apache-2.0. This case is a standalone adaptation of the operator behavior for benchmarking, not a copy of the full framework implementation.
