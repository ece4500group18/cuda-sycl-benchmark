# Original CUDA

Standalone Stage 1 CUDA case for `templatedSharedMemIdiom`.

The SharedMemory<T> template with specializations and testKernel<T> are upstream cuda-samples code verbatim; the host driver, gen_idata, and CMakeLists are new. in[i]=(T)i, N=256. Writes testKernel<int>'s 256 outputs then testKernel<float>'s 256 outputs to output/cuda_output.txt; checked by ../tests/verify.py.
