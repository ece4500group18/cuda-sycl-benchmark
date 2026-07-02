# CUDA library usage collection

Owner: Weixuan Zhang

See `../README.md` for the shared workflow, CSV columns, `SOURCE.txt`
format, and snapshot rules. This file records the coverage matrix for
CUDA library usage examples. Snapshots of every non-excluded candidate
are stored under `sources/<id>/`.

## Sources and licenses (verified 2026-07-02)

| Source | License | Notes |
|---|---|---|
| NVIDIA CUDALibrarySamples | Apache-2.0 | Primary upstream source covering CUDA libraries including cuBLAS, cuBLASLt, cuSOLVER, cuFFT, cuFFTDx, cuRANDDx, cuPQC, cuBLASMp, cuFFTMp, and cuTensorMp. Each snapshot records the exact upstream commit in `SOURCE.txt`. |

## Coverage matrix

Dimensions (a case "covers" a cell if it exercises the corresponding CUDA
library feature):

- **Library**: CUDA library being exercised (`cuBLAS`, `cuBLASLt`,
  `cuSOLVER`, `cuFFT`, `cuFFTDx`, `cuRANDDx`, `cuPQC`,
  `cuBLASMp`, `cuFFTMp`, `cuTensorMp`)
- **API style**: `host`, `device`, `descriptor`, `template`,
  `callback`, `distributed`
- **Execution**: `thread`, `block`, `library-call`, `MPI`
- **Feature**: representative functionality exercised

| Candidate | Library | API style | Execution | Feature |
|---|---|---|---|---|
| lib-01 | cuRANDDx | template | thread | Philox RNG |
| lib-02 | cuRANDDx | template | thread | XORWOW RNG |
| lib-03 | cuRANDDx | template | thread | Sobol quasi-RNG |
| lib-04 | cuRANDDx | template | thread | PCG RNG |
| lib-05 | cuRANDDx | template | thread | Multiple distributions |
| lib-06 | cuRANDDx | template | thread | Random bits generation |
| lib-07 | cuRANDDx | template | library-call | NVRTC integration |
| lib-08 | cuRANDDx | template | thread | Introduction example |
| lib-09 | cuFFTDx | template | block | Device FFT |
| lib-10 | cuPQC | device | thread | SHA2-256 hashing |
| lib-11 | cuPQC | template | block | ML-KEM key encapsulation |
| lib-12 | cuBLASLt | descriptor | library-call | GEMM |
| lib-13 | cuBLASLt | descriptor | library-call | Auto-tuning |
| lib-14 | cuFFT | callback | library-call | Legacy callback |
| lib-15 | cuFFT | callback | device | LTO callback |
| lib-16 | cuBLAS | host | library-call | Batched GEMM |
| lib-17 | cuBLAS | host | library-call | Strided batched GEMM |
| lib-18 | cuBLAS | host | library-call | GemmEx |
| lib-19 | cuBLAS | host | library-call | Grouped Batched GemmEx |
| lib-20 | cuSOLVER | host | library-call | Singular value decomposition |
| lib-21 | cuSOLVER | host | library-call | Symmetric eigendecomposition |
| lib-22 | cuSOLVER | host | library-call | Batched Cholesky |
| lib-23 | cuBLASMp | distributed | MPI | Distributed GEMM |
| lib-24 | cuFFTMp | distributed | MPI | Distributed FFT |
| lib-25 | cuTensorMp | distributed | MPI | Distributed tensor contraction |

Dark cells / known gaps:

- cuSPARSE sparse linear algebra.
- cuDSS sparse direct solver.
- Single-GPU cuTENSOR examples.
- CUDA graph analytics libraries (e.g., cuGraph).
- CUDA communication libraries outside the `*Mp` family.

## Dedup policy

Many CUDA libraries provide multiple samples implementing similar
algorithms (e.g., GEMM or FFT). Multiple variants are retained only when
they exercise different API styles or migration challenges, such as
descriptor-based interfaces, callback mechanisms, template-based
device-side APIs, runtime compilation, batched execution, or distributed
MPI execution. Otherwise, the smallest self-contained sample is
preferred.
