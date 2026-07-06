# Templated shared-memory idiom

CUDA forbids a templated extern __shared__ T sdata[]; the SharedMemory<T> idiom works around it with one full specialization per POD type, each declaring its own uniquely-named extern __shared__ array. testKernel<T> reads in[i] into shared memory, multiplies by num_threads (blockDim.x, known only at launch), and writes it back -- no cross-thread accumulation, so the result is exact for both int and float (num_threads=256 is a power of two).

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleTemplates/simpleTemplates.cu

Snapshot: benchmark/collection/simple-kernels/sources/sk-14

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: SharedMemory<T> template (+ specializations) and testKernel<T> reproduced verbatim from NVIDIA/cuda-samples sharedmem.cuh and simpleTemplates.cu. Host driver, gen_idata, CMakeLists, and Python oracle are new. Snapshot sk-14.
