# Case: fp16PackedScalarProduct (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | packed half2 dot product: grid-stride multiply-accumulate + shared-memory tree reduction, native half2 operators vs. explicit `__hfma2`/`__hadd2` intrinsics |
| size | 262,144 half2 elements per input vector, launch = 128 blocks x 128 threads |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected (exact match, both per-block and aggregate) |

## Source

The four functions in `main.cu` (`scalarProductKernel_native`,
`scalarProductKernel_intrinsics`, `reduceInShared_native`,
`reduceInShared_intrinsics`) are reproduced **verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/0_Introduction/fp16ScalarProduct/fp16ScalarProduct.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/fp16ScalarProduct/fp16ScalarProduct.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

The original sample's `generateInput()` (host-side `rand() % 4` /
`rand() % 2` values) and `main()` (which relies on `helper_cuda.h`'s
`findCudaDevice`/`checkCudaErrors`, pinned `cudaMallocHost` staging
buffers, and a `< 0.00001f` float tolerance comparing the two kernels'
CPU-summed aggregate results) are not reproduced. This directory's
`main()` replaces them with a deterministic, single-file
CUDA-vs-SYCL comparison harness: index-formula inputs (see below),
plain synchronous `cudaMalloc`/`cudaMemcpy`, the repo's own `CHECK`
error-checking macro, and a CPU reference (`reference.h`) that
independently recomputes each block's expected partial dot product
(rather than only cross-checking the two kernels against each other,
as the original sample does).

## What this case demonstrates (methods used)

1. **Native operator overloads vs. explicit intrinsics for packed
   half2 SIMD arithmetic.** `scalarProductKernel_native` /
   `reduceInShared_native` write the multiply-accumulate and reduction
   as plain `half2` `operator*`/`operator+` expressions;
   `scalarProductKernel_intrinsics` / `reduceInShared_intrinsics` write
   the identical math as explicit `__hfma2` (fused multiply-add) and
   `__hadd2` intrinsic calls. In `<cuda_fp16.hpp>`, `half2`'s
   `operator+`/`operator*` are themselves thin wrappers around
   `__hadd2`/`__hmul2`, so this pair isolates *which PTX instructions
   the compiler emits for two spellings of the same computation* --
   not a difference in numerical semantics. This is "simple but not
   trivial": each kernel is a short, textbook grid-stride-reduce
   pattern, but the two source-level idioms (operator overloading vs.
   explicit intrinsic) are exactly the kind of thing a CUDA-to-SYCL
   migration tool has to map correctly (`sycl::vec<half,2>` operators
   vs. explicit `sycl::ext::oneapi::experimental` / `sycl::fma`-style
   calls, or vendor extension intrinsics).

2. **Grid-stride accumulate + fixed-size shared-memory tree
   reduction.** Both kernels accumulate a per-thread `half2` partial
   over a grid-stride loop across `size` elements, stage it into a
   128-entry `__shared__ half2 shArray`, and then run the identical
   `64/32/16/8/4/2/1` halving tree reduction (`reduceInShared_native`
   / `reduceInShared_intrinsics`) down to a single per-block value,
   which is converted to `float` (`__low2float`/`__high2float` vs.
   explicit `(float)` casts on `.x`/`.y` -- again the same value via
   two spellings) and written to `results[blockIdx.x]`.

3. **Guaranteeing an exact match despite fp16 arithmetic.** fp16
   normally invites "just add a tolerance" -- but this case is
   designed so no tolerance is needed. Inputs are deterministic small
   integers, `a[i].x = a[i].y = i % 4` and `b[i].x = b[i].y = i % 2`
   (see `reference.h`), and the launch configuration
   (`NUM_OF_BLOCKS = NUM_OF_THREADS = 128`, matching the original
   sample, `size = 128*128*16 = 262144`) is chosen so that the
   16,384-wide grid stride is divisible by both 4 and 2. That makes
   every thread's `i % 4` / `i % 2` constant across its 16 grid-stride
   iterations, and (since `128 * blockIdx.x` is itself a multiple of
   4 and 2) makes those residues depend only on `threadIdx.x`, not on
   which block a thread belongs to. Working through the arithmetic,
   every block's 128-way tree reduction sums to *exactly* 2048 in each
   half2 lane -- at, but never over, fp16's exact-integer boundary
   (fp16 represents every integer in `[-2048, 2048]` exactly, and any
   sum of non-negative exactly-representable integers whose running
   total never exceeds that bound is itself exact, regardless of
   summation order). So `reference_block_partials()` in `reference.h`
   can recompute the expected per-block result with plain
   double-precision arithmetic and expect a **bit-exact** match
   against both GPU kernels -- tighter than the original sample's own
   `1e-5` tolerance -- and the final aggregate (summed over blocks in
   the same sequential order the original sample uses) matches
   exactly too, for the same reason.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`'s `gen_a_lane`/`gen_b_lane`, used for both lanes of
  each half2 element):
  - `a[i].x = a[i].y = i % 4`
  - `b[i].x = b[i].y = i % 2`
  - `size = 262144` half2 elements per vector; launch =
    `NUM_OF_BLOCKS = 128` blocks x `NUM_OF_THREADS = 128` threads
    (as in the original sample).
- **Output**: `argv[1]` (default `output/cuda_output.txt`), 128
  lines, the per-block float dot-product partials from
  `scalarProductKernel_intrinsics` (one `%.9g` value per line), plus a
  `PASS`/`FAIL` line on stdout: `PASS` iff both kernels' per-block
  results match `reference_block_partials()` exactly
  (`max_abs_error == 0`) *and* both kernels' sequentially-summed
  aggregate scalars match the reference aggregate exactly.

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`half2` typically migrates to `sycl::marray<sycl::half, 2>` or
`sycl::vec<sycl::half, 2>`; `__hfma2`/`__hadd2`/`__float2half2_rn`/
`__low2float`/`__high2float` migrate to explicit element-wise
`sycl::half`/`sycl::fma` operations or vendor intrinsics depending on
the dpct mapping in use -- this pair is a good stress test for how a
migration tool handles that split), build the result, run it with the
same `argv[1]` convention (e.g. `output/sycl_output.txt`), and diff
the two output files (exact match expected, per the reasoning above).
