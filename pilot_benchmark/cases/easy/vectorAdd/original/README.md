# vectorAdd (original CUDA)

Element-wise vector addition `C = A + B` over `n = 100000` floats.

- Kernel: one thread per element, `i = blockIdx.x*blockDim.x + threadIdx.x`.
- Inputs are generated deterministically on the host (see `main.cu`).
- The result vector is written to the path in `argv[1]` (one float per line).

Build (toolchain permitting):

    nvcc -O2 -std=c++17 main.cu -o build/app

Run:

    build/app ../output/cuda_output.txt
