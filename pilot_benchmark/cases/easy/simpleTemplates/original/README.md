# simpleTemplates (original CUDA)

Templated kernel addConst<T>: out = in + 3.0f, n=50000.

Inputs are generated deterministically on the host and replicated by
`../tests/verify.py`. The result is written to `argv[1]` (one float per line).

Build (toolchain permitting):

    nvcc -O2 -std=c++17 main.cu -o build/app

Run:

    build/app ../output/cuda_output.txt
