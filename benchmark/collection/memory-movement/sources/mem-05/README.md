# Case: constMemMatrixDims (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | elementwise 2D matrix add, `C = A + B`, matrix dims (`M`, `N`) passed as ordinary kernel arguments vs. read from `__constant__` memory |
| size | M = N = 512 floats (grid = 32x32 blocks of 16x16 threads) |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected |

## Source

The two `__global__` kernels in `main.cu` (`add` and `add_const`) are
reproduced **verbatim** from:

- Project: **CUDAMicroBench**
- File: `ReadOnlyMem_2D_Texture/matadd_2D_cudakernel.cu`
- Repository: https://github.com/passlab/CUDAMicroBench
- Copyright (c) 2021, University of North Carolina at Charlotte and
  Lawrence Livermore National Security, LLC.
- License: BSD-3-Clause (see `LICENSE` in this directory)
- Associated paper: Yi, Xinyao; Stokes, David; Yan, Yonghong; Liao,
  Chunhua. "CUDAMicroBench: Microbenchmarks to Assist CUDA Performance
  Programming." 2021 IEEE IPDPSW, pp. 397-406.
  doi:10.1109/IPDPSW52791.2021.00068

Three other kernels in the original file were omitted: `add_warmingup`
(a throwaway duplicate of `add`, used only to warm up the GPU before
the original's own timing loop, not a distinct technique); and
`add_texture`/`add_texture_constant` (bind `A`/`B` through the legacy
CUDA texture-reference API, `texture<float,2> texMatrixA/texMatrixB`
+ `cudaBindTexture2D`/`tex2D` -- a different memory-movement technique
than constant memory, out of scope for this case, and a legacy API
SYCLomatic/dpct handles very differently from `__constant__` symbols).
The original file's `matadd()` host driver (which binds textures,
launches all five kernels back-to-back with `cudaDeviceSynchronize()`
between them, and has no verification of its own) was replaced;
everything other than the two kernels listed above (host driver,
`reference.h`, `Makefile`, `CMakeLists.txt`, this `README.md`) is new
code written for this repository, following the same minimal
single-file style used by the other `benchmark/memory` cases. One
small correction versus the original driver: the original calls
`cudaMemcpyToSymbol(cons_M, &M, sizeof(float), 0)` to copy an `int`
using `sizeof(float)` (harmless only because both are 4 bytes on every
target platform); this directory's driver uses `sizeof(int)`, the
correct size for the `__constant__ int cons_M`/`cons_N` symbols
declared in the verbatim kernel code.

## What this case demonstrates (methods used)

1. **Constant memory for broadcast, read-only metadata (`memory
   movement` / `memory layout`).** Both kernels compute the identical
   `M x N` row-major elementwise sum, using the identical thread ->
   element mapping (`tidx` = row via `blockDim.x*blockIdx.x+threadIdx.x`,
   `tidy` = column, linear index `tidx * N + tidy`), and differ in
   exactly one respect -- where each thread's copy of the matrix bounds
   `M`/`N` comes from:
   - `add`: `d_M`/`d_N` are ordinary `int` kernel parameters, re-supplied
     by the host in every launch's argument buffer.
   - `add_const`: `cons_M`/`cons_N` are `__constant__` global variables,
     populated once via `cudaMemcpyToSymbol` before either kernel runs.
     Every thread in every block reads the same address out of the
     constant-memory space, which is cached and broadcast to an entire
     warp in a single transaction -- the same hardware path used for
     kernel launch parameters themselves, but here made explicit and
     reusable across launches instead of being re-marshaled by the host
     every time.

   Since `cons_M`/`cons_N` are populated with the same values as `d_M`/
   `d_N` by construction, both kernels touch the same elements, in the
   same order, and write the exact same result; only the storage class
   the bounds are read from differs. This isolates the
   constant-memory-for-metadata technique from bulk-data placement
   (contrast with cases that put an entire data array, rather than a
   couple of scalar dimensions, in `__constant__` memory).

2. **`cudaMemcpyToSymbol`**, the standard host-to-`__constant__`-memory
   upload API, used here to broadcast two `int` scalars once before
   launch rather than passing them as ordinary parameters on every
   launch.

3. **2D thread/block indexing** (`dim3 blocks(M/BLOCK_SIZE,
   N/BLOCK_SIZE, 1)`, `dim3 threadsperblock(BLOCK_SIZE, BLOCK_SIZE, 1)`
   with `BLOCK_SIZE = 16`), the same 2D grid-of-tiles indexing pattern
   used by `tiledMatmulShmem`, applied here to a simple elementwise op
   instead of a reduction.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `A[i] = ((i % 17) - 8) * 0.5`, `B[i] = ((i % 13) - 6) * 0.25`,
    `i` in `[0, M*N)`, `M = N = 512` (both multiples of `BLOCK_SIZE=16`)
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the
  512x512 row-major `C = A+B` matrix computed by `add_const` (the
  `__constant__`-memory variant), one `%.9g` float per line (262144
  lines), plus a `PASS`/`FAIL` line on stdout comparing *both* kernels'
  outputs against `reference_add()` in `reference.h` (exact match
  expected, since each `C[i]` is an independent single-precision add
  regardless of where the bounds used to compute `i` came from).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`__constant__` globals typically migrate to `sycl::ext::oneapi::
experimental` constant-memory properties, a `sycl::constant_buffer`/
`const_mem` wrapper, or plain `__constant__`-annotated global variables
under the SYCLomatic CUDA-to-SYCL constant-memory mapping, depending on
toolkit version -- one of the more interesting migration targets in
this benchmark set; `cudaMemcpyToSymbol` maps to the corresponding
constant-memory initialization/copy call), build the result, run it
with the same `argv[1]` convention (e.g. `output/sycl_output.txt`), and
diff the two output files (exact match expected).
