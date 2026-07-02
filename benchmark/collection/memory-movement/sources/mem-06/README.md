# Case: csrScanOrderSpMM (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory access order) |
| operation | sparse-times-sparse matrix product A*B, computed by two kernels that scan the same CSR-shaped operand data in different orders (full linear rescan + range-check vs. bounded nested column-range intersection) |
| size | 32 x 32 sparse matrices, 4 nonzeros per row (and per column), nnzA = nnzB = 128; result = 32x32 = 1024 elements |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected |

## Source

The two `__global__` kernels in `main.cu`
(`spmm_csr_csr_kernel` and `spmm_csc_csr_kernel`) are reproduced
**verbatim** from:

- Project: **CUDAMicroBench**
- File: `CoMem_SpMM/SpMM_cudakernel.cu`
- Repository: https://github.com/passlab/CUDAMicroBench
- Copyright (c) 2021, University of North Carolina at Charlotte and
  Lawrence Livermore National Security, LLC.
- License: BSD-3-Clause (see `LICENSE` in this directory)
- Associated paper: Yi, Xinyao; Stokes, David; Yan, Yonghong; Liao,
  Chunhua. "CUDAMicroBench: Microbenchmarks to Assist CUDA Performance
  Programming." 2021 IEEE IPDPSW, pp. 397-406.
  doi:10.1109/IPDPSW52791.2021.00068

Two other kernels in the original file (`spmm_csr_csr_warmingup`,
`spmm_csc_csr_warmingup`) are byte-for-byte duplicates of the two kept
kernels (the upstream driver calls the "warmingup" copy once to warm
the GPU up before timing the real kernel) and were omitted as
redundant. The original driver (`SpMM_cuda.c`) builds its two 100x100
operand matrices with `srand48()`/`drand48()` and times the kernels
with `read_timer_ms()`/`sys/timeb.h`; none of that is needed for a
deterministic CUDA-vs-SYCL output comparison, so this directory
replaces it with fixed-size (32x32), index-formula-generated circulant
sparse matrices (`reference.h`) and a single-file harness that checks
both kernels against an independent CPU reference. Everything in this
directory other than the two kernels listed above (host driver,
`reference.h`, `Makefile`, `CMakeLists.txt`, this `README.md`) is new
code written for this repository.

Upstream itself confirms the intended data-layout contract this case
reproduces: in `SpMM_cuda.c`'s `main()`, matrix A is built once as CSR
(`ptrA_csr`/`indicesA_csr`/`dataA_csr`) and passed unchanged to both
`spmm_csr_cuda()` and `spmm_csc_cuda()`, while the very same logical
matrix B is built **twice** -- once as CSR (`init_data_csr`/
`init_ptr_csr`, row-major `ptrB_csr`/`indicesB_csr`) for
`spmm_csr_csr_kernel`, and once as CSC (`init_data_csc`/`init_ptr_csc`,
column-major `ptrB_csc`/`indicesB_csc`) for `spmm_csc_csr_kernel` --
and both are checked against the same dense `matmul_serial()` oracle.
This case reproduces exactly that: one shared A, two storage-order
builds of the same B, one shared CPU oracle.

## What this case demonstrates (methods used)

1. **Memory access order for an identical sparse dot product, not a
   storage-format shootout.** Both kernels compute exactly the same
   `result[row][col] = sum_k A[row][k] * B[k][col]` from exactly the
   same two logical matrices A and B -- they are not being compared on
   different math, and this is deliberately **not** framed as "CSR
   matrix vs CSC matrix" in the sense of which format is generally
   faster to store. The two kernels are instead the same intersection
   problem (find the k values where both A's row and B's relevant range
   have a nonzero) solved with two different memory access strategies:
   - `spmm_csr_csr_kernel` receives B in row-major CSR order and, for
     every output column, does a **full unbounded linear scan** over
     *all* `nnzB` entries of `indicesB`/`dataB`, filtering by
     `indicesB[j] == col` combined with a `ptrB[k]..ptrB[k+1]`
     range-check. Every column of every row rescans the entire B array.
   - `spmm_csc_csr_kernel` receives the same logical B in column-major
     CSC order and, for every output column, **bounds the scan** to
     `ptrB[col]..ptrB[col+1]` -- only B's nonzeros in that column --
     and merges that short range directly against A's row range via
     `indicesA[i] == indicesB[j]`, the textbook bounded intersection of
     two short sorted index lists.

2. **Exact-match correctness despite different summation order.** Every
   nonzero of A and B is a small nonzero integer (`val_a`/`val_b` in
   `reference.h`, magnitudes 1-6, values stored in `float` but never
   exceeding a few dozen after multiplication/summation -- well within
   `float`'s 24-bit exact-integer range). Integer addition is exactly
   commutative and associative, so no matter which order (or how many
   redundant zero-filtered passes) each kernel visits the matching
   `(i, k)` pairs in, the final `dot` for each `(row, col)` must be
   bit-for-bit identical between the two kernels and the CPU reference
   -- `max_abs_error == 0` is the expected, not merely hoped-for,
   result.

3. **Circulant sparsity pattern for a self-consistent CSR/CSC pair.**
   `reference.h` builds A and B as circulant matrices: row `i` has a
   nonzero at column `(i + off) % NUM_ROWS` for each `off` in a fixed
   4-element offset set. This guarantees, by construction and without
   any post-hoc balancing, that every row **and** every column has
   exactly 4 nonzeros -- so the CSR-of-B and CSC-of-B builds of the same
   matrix both come out perfectly regular (no empty ranges, no
   variable-length rows/columns to special-case) while still being
   genuinely different physical layouts of the same data.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`, no `rand()`/`drand48()`):
  - `NUM_ROWS = 32`, `NNZ_PER_ROW = 4`, so `nnzA = nnzB = 128`.
  - A's sparsity: row `i` nonzero at columns `(i + off) % 32` for
    `off` in `{1, 5, 11, 19}`.
  - B's sparsity: row `i` nonzero at columns `(i + off) % 32` for
    `off` in `{2, 7, 13, 23}`.
  - `val_a(i, j) = (1 + ((i*13 + j*7 + i*j*3) % 6)) * ((i+j) even ? 1 : -1)`
  - `val_b(i, j) = (1 + ((i*5 + j*11 + i*j*2) % 5)) * ((i+j) even ? -1 : 1)`
  - A is uploaded once, in CSR order, and shared by both kernels; B is
    uploaded twice: once in CSR order (`build_csr_b`) for
    `spmm_csr_csr_kernel`, and once in CSC order (`build_csc_b`) for
    `spmm_csc_csr_kernel`.
- **Output**: `argv[1]` (default `output/cuda_output.txt`), 1024 lines
  (`NUM_ROWS * NUM_ROWS`, row-major), the `spmm_csc_csr_kernel` result,
  one `%.17g` value per line, plus a `PASS`/`FAIL` line on stdout
  comparing **both** kernels' outputs against `reference_spmm()` in
  `reference.h` (exact match expected -- see point 2 above).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct, build the
result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
