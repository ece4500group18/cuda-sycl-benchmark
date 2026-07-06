# Atomic intrinsics battery

Every one of 16384 threads (64 blocks x 256 threads) calls the same eleven
global-memory atomic read-modify-write intrinsics (`atomicAdd`, `atomicSub`,
`atomicExch`, `atomicMax`, `atomicMin`, `atomicInc`, `atomicDec`, `atomicCAS`,
`atomicAnd`, `atomicOr`, `atomicXor`) on the same 11-element output array, with
no coordination beyond the atomics themselves. "Simple but not trivial": each
call is a one-line primitive, but reasoning about the guaranteed final value of
a slot 16384 threads raced on -- and recognizing which operations even have one
-- is the substance.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics_kernel.cuh

Snapshot: benchmark/collection/simple-kernels/sources/sk-01

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: `testKernel` is upstream code verbatim. The host driver,
`CMakeLists.txt`, and the Python oracle (`tests/verify.py`) are new; the CPU
reference formulas mirror the sample's own `computeGold`. Nine order-independent
slots are checked exactly; `atomicExch`/`atomicCAS` are execution-order
dependent and only range-checked to a valid thread id in `[0, len)`.
