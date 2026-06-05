# transpose (original CUDA)

Matrix transpose out = in^T for a 192x128 row-major matrix.

Inputs are generated deterministically on the host and replicated by
`../tests/verify.py`. The result is written to `argv[1]` (one float per line).

Build (toolchain permitting):

    nvcc -O2 -std=c++17 main.cu -o build/app

Run:

    build/app ../output/cuda_output.txt
