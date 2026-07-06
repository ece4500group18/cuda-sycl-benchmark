# Stream-ordered allocation vector add

The same c[i]=a[i]+b[i] kernel is fed device buffers two ways: plain cudaMalloc/cudaFree on the default stream, and stream-ordered cudaMallocAsync/cudaFreeAsync (backed by a CUDA memory pool) enqueued within a non-blocking stream. Only how/when the backing memory is obtained and released differs; the per-element addition is identical and exact, so both paths match the CPU reference exactly.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/streamOrderedAllocation/streamOrderedAllocation.cu

Snapshot: benchmark/collection/simple-kernels/sources/sk-12

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: vectorAddGPU reproduced verbatim from NVIDIA/cuda-samples streamOrderedAllocation.cu. Host driver (both classic cudaMalloc and stream-ordered cudaMallocAsync paths), gen_a/gen_b, CMakeLists, and Python oracle are new. Requires CUDA 11.2+. Snapshot sk-12.
