# Packed half2 scalar product

The same packed-half2 dot product is written two ways -- explicit __hfma2/__hadd2 PTX intrinsics vs. half2's native operator+/operator* -- each doing an identical grid-stride multiply-accumulate then an identical 64/32/.../1 shared-memory tree reduction. Inputs are chosen so every intermediate fp16 value is an exactly-representable integer (<= 2048), making a plain integer recompute a bit-exact oracle.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/fp16ScalarProduct/fp16ScalarProduct.cu

Snapshot: benchmark/collection/simple-kernels/sources/sk-08

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: scalarProductKernel_native/_intrinsics and reduceInShared_native/_intrinsics reproduced verbatim from NVIDIA/cuda-samples fp16ScalarProduct.cu. Host driver, gen_a_lane/gen_b_lane, CMakeLists, and Python oracle are new. Inputs chosen so every fp16 intermediate is an exact integer. Snapshot sk-08.
