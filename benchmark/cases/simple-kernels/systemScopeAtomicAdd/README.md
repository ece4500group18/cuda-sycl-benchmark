# System-scope atomics

*_system atomics are coherent with concurrent CPU accesses to the same cudaMallocManaged allocation: the GPU atomicKernel and the host atomicKernel_CPU each cover half the logical contributor range and update the same array, and the combined result matches a single closed-form/range oracle. A device-scope twin does the same math on ordinary atomics. Eight order-independent slots are checked exactly; atomicExch/atomicCAS are range-checked to a valid contributor id in [0, len).

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/systemWideAtomics/systemWideAtomics.cu

Snapshot: benchmark/collection/simple-kernels/sources/sk-13

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: atomicKernel and atomicKernel_CPU reproduced verbatim from NVIDIA/cuda-samples systemWideAtomics.cu. atomicKernel_device (device-scope twin), host driver, CMakeLists, and Python oracle are new; the oracle mirrors the sample's own verify() generalized to len/LOOP_NUM. Snapshot sk-13.
