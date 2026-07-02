# Case: sparseDenseSpmvLayout (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | sparse matrix-vector product `y = M*x`, compressed CSR gather vs. full dense row-major sweep with a zero-skip guard |
| size | num_rows = 256, 5 nonzeros/row (1280 nonzeros total, ~1.95% dense fill), launch = 1 block x 256 threads |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected |

## Source

The two `__global__` kernels in `main.cu` (`spmv_csr` and
`spmv_dense_check_and_compute`) are reproduced **verbatim** from:

- Project: **CUDAMicroBench**
- File: `MiniTransfer_SpMV/SpMV_cudakernel.cu`
- Repository: https://github.com/passlab/CUDAMicroBench
- Copyright (c) 2021, University of North Carolina at Charlotte and
  Lawrence Livermore National Security, LLC.
- License: BSD-3-Clause (see `LICENSE` in this directory)
- Associated paper: Yi, Xinyao; Stokes, David; Yan, Yonghong; Liao,
  Chunhua. "CUDAMicroBench: Microbenchmarks to Assist CUDA Performance
  Programming." 2021 IEEE IPDPSW, pp. 397-406.
  doi:10.1109/IPDPSW52791.2021.00068

The original file's other two kernels (`spmv_dense` -- an unconditional
dense sweep that adds an explicit `matrix * vector` term even where the
matrix is zero, superseded here by the zero-skip variant -- and
`spmv_unified`, a unified-memory variant that does a linear `nnz`-length
scan per row instead of CSR's `O(nnz/num_rows)`-length scan) were
omitted, as was the original `SpMV.h`/`init_csr`/`init_index`/timing
driver (`spmv_cuda_csr_discrete`, `spmv_cuda_dense_discrete`, file I/O
based matrix generation, `read_timer_ms`). Everything else in this
directory (host driver, deterministic CSR/dense input construction,
`reference.h`, `Makefile`, `CMakeLists.txt`, this `README.md`) is new
code written for this repository, replacing the original's file-driven
matrix loader and wall-clock timing harness with a deterministic,
single-file CUDA-vs-SYCL comparison harness.

## What this case demonstrates (methods used)

1. **Compressed vs. dense sparse-matrix storage layout (memory
   movement / memory layout).** Both kernels compute one thread per row
   of the same `y = M*x` product for the same underlying matrix `M`,
   but read it from memory in fundamentally different layouts:
   - `spmv_csr`: row `row`'s nonzeros are **gathered** from three
     compact arrays sized proportional to the number of nonzeros --
     `ptr[row]`/`ptr[row+1]` bound a contiguous slice of `indices`
     (column ids) and `data` (values) -- so a thread reads exactly
     `NNZ_PER_ROW` (5) elements from each array, touching only the
     matrix's actual nonzero footprint.
   - `spmv_dense_check_and_compute`: row `i`'s dot product **sweeps
     the entire row** of the full `num_rows x num_rows` dense matrix
     (`matrix[i*num_rows+j]` for every `j`), reading and testing
     `num_rows` (256) elements per row -- 256/5 = ~51x the memory
     traffic of the CSR path per row -- and skipping the multiply-add
     only when the value read back happens to be exactly `0.0`.

   Both kernels touch the same nonzero (row, column, value) triples and
   the same vector `x`, and this directory's inputs are built so both
   kernels accumulate those triples in the exact same ascending-column
   order per row (`indices`/`data` are sorted by column when the CSR
   arrays are built; the dense sweep visits columns in ascending `j`
   order by construction) -- isolating *how much memory a kernel must
   move and inspect to reconstruct the same sparse structure* as the
   sole difference between the two paths.

2. **Order-and-magnitude-safe exactness.** Every nonzero value is a
   small integer in `{-4,-3,-2,-1,1,2,3,4}` (deliberately excluding 0,
   so the dense kernel's zero-skip guard never discards a term the CSR
   path still includes) and every vector entry is a small integer in
   `[-5, 5]`. With identical terms summed in identical order, every
   partial sum is exactly representable in `float`, so the two GPU
   kernels and the CPU reference are all required to agree bit-for-bit
   (`max_abs_error == 0`), not just within a floating-point tolerance.

3. **CSR (`ptr`/`indices`/`data`) construction and traversal**, one of
   the most common sparse-storage idioms in real-world GPU numerical
   code, contrasted directly with the naive dense fallback it replaces.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `num_rows = 256`, `NNZ_PER_ROW = 5` nonzeros per row (`NNZ = 1280`
    total).
  - Nonzero column `k` (`k` in `[0,5)`) of row `row`:
    `col(row,k) = (row*7 + k*13) % 256` -- always 5 distinct columns
    per row since the offsets `{0,13,26,39,52}` are pairwise distinct
    and span less than 256.
  - Nonzero value: `val(row,k) = (((row + 2*k) % 4) + 1) * (((row+k) % 2 == 0) ? 1 : -1)`,
    i.e. magnitude in `[1,4]` with an alternating sign, never 0.
  - Vector: `x[i] = (i % 11) - 5`, range `[-5, 5]`.
  - The dense `num_rows x num_rows` matrix and the CSR (`ptr`,
    `indices`, `data`) arrays are built from the exact same
    `(row, col, val)` triples, so the two kernels read the identical
    sparse structure, just via different storage layouts.
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the
  256-element `y = M*x` vector computed by `spmv_csr`, one `%.9g`
  float per line, plus a `PASS`/`FAIL` line on stdout comparing both
  kernels' outputs against `reference_spmv()` in `reference.h` (exact
  match expected).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct, build the
result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
