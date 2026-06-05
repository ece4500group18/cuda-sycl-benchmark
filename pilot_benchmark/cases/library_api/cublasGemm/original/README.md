# cublasGemm (original CUDA)

cuBLAS SGEMM

Inputs are generated deterministically on the host (see `main.cu`) and
replicated by `../tests/verify.py`. The result is written to `argv[1]`
(one value per line).

Build (toolchain permitting):

    nvcc -O2 -std=c++17 main.cu -o build/app -lcublas

Run:

    build/app ../output/cuda_output.txt
