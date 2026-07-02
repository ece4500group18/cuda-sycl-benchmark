# Case: histogramSharedPrivatization (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | 64-bin byte histogram: per-block privatized shared-memory histogram + cross-block merge, vs. a naive single-pass global-memory `atomicAdd`-per-element histogram |
| size | byteCount = 1,048,576 bytes (= 65,536 `uint4` words); 64 bins; `histogram64Kernel` launch = 69 blocks x 64 threads, `mergeHistogram64Kernel` launch = 64 blocks x 256 threads, `naiveHistogramKernel` launch = 256 blocks x 256 threads |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected for both GPU paths |

## Source

The two `__global__` kernels in `main.cu` (`histogram64Kernel` and
`mergeHistogram64Kernel`, plus their `addByte`/`addWord` device-function
helpers) are reproduced **verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/2_Concepts_and_Techniques/histogram/histogram64.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/histogram/histogram64.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

The handful of constants/macros/typedefs those kernels rely on
(`HISTOGRAM64_BIN_COUNT`, `HISTOGRAM64_THREADBLOCK_SIZE`,
`SHARED_MEMORY_BANKS`, `UMUL`/`UMAD`, the `uint`/`uchar`/`data_t`
typedefs) are copied from the companion upstream header
`histogram_common.h` in the same directory (also BSD-3-Clause,
NVIDIA/cuda-samples) so this case stays a single `.cu` file.

Omitted from upstream: the host-side `initHistogram64`/`closeHistogram64`/
`histogram64()` driver functions (`extern "C"`, `checkCudaErrors`,
`helper_cuda.h`-dependent), `histogram64CPU()`'s reference implementation
in `histogram_gold.cpp`, and the shared `main.cpp` benchmarking/CLI
harness common to both `histogram64.cu` and `histogram256.cu` in that
directory -- none of these are needed for a deterministic,
single-file CUDA-vs-SYCL output comparison. This directory's `main()`
reimplements the same launch geometry as the upstream `histogram64()`
function (partial-histogram count via the same `iDivUp`/`iSnapDown`
arithmetic) with plain synchronous `cudaMemcpy`, and `reference.h`
provides a fresh, independent CPU oracle (not upstream's `histogram64CPU`)
for the same 64-bin/top-6-bits binning rule used by `addWord`.

`naiveHistogramKernel` in `main.cu` is **new code** written for this
repository -- it does not exist in the upstream sample. It is the
deliberately "dumb" baseline this case contrasts the privatized/merge
kernels against: one `atomicAdd` straight into a 64-entry global-memory
histogram per input byte, with no shared memory and no per-block
privatization at all. Everything else in this directory (the rest of
the host driver, `reference.h`, `Makefile`, `CMakeLists.txt`, this
`README.md`) is also new code written for this repository.

## What this case demonstrates (methods used)

1. **Shared-memory privatization to cut global atomic contention
   (memory movement / memory layout).** `naiveHistogramKernel` issues
   one `atomicAdd` per byte (1,048,576 total) directly against the same
   64 global-memory counters, so every thread that lands on a popular
   bin serializes behind every other thread hitting that bin, across
   the *entire* grid. `histogram64Kernel` instead gives **every
   thread its own private 64-bin sub-histogram in `__shared__` memory**
   and accumulates there (fast, on-chip, no contention with other
   thread blocks, and only contending with the small, fixed number of
   threads sharing that block's shared memory) for its whole
   grid-strided slice of the input. Only once a block has fully
   consumed its data does it reduce its per-thread sub-histograms into
   one 64-bin partial histogram and perform 64 writes to global memory
   -- replacing 1,048,576 contended global atomics with a bounded
   number of uncontended shared-memory updates plus one cheap global
   write per bin per block.

2. **Cross-block reduction of privatized partials.**
   `mergeHistogram64Kernel` launches one thread block per output bin;
   each block sums that bin's value across every block's partial
   histogram from stage 1 using a standard shared-memory tree
   reduction, producing the final 64-bin histogram. This is the
   "merge" half of the privatize-then-merge pattern: private copies
   only pay off if recombining them is itself cheap, and summing 69
   partial counts per bin across 64 tiny thread blocks is exactly that.

3. **Bank-conflict-avoiding index permutation.** Within
   `histogram64Kernel`, `threadPos` permutes `threadIdx.x` bit-by-bit so
   that the `SHARED_MEMORY_BANKS`=16 threads that would otherwise all
   write to the same shared-memory bank (since consecutive bins are
   `HISTOGRAM64_THREADBLOCK_SIZE`=64 bytes apart, a multiple of the
   16-bank stride) instead land on 16 *different*, consecutive banks --
   privatization only pays off if the private per-thread storage itself
   avoids serializing on shared-memory bank conflicts.

4. **Why both paths are guaranteed to agree exactly.** A histogram bin
   count is a sum of `+1` contributions, and integer addition is
   commutative and associative: it does not matter *where* (register,
   shared memory, or global memory), in what order, or by which thread
   each `+1` is applied -- the final count in each of the 64 bins is
   identical. `reference_histogram64()` in `reference.h` computes the
   same 64 bin counts with one sequential CPU pass over the identical
   input bytes, using the identical `(byte >> 2) & 0x3F` binning rule
   `addWord` uses on-device, so `max_abs_error` between either GPU path
   and the CPU reference is exactly 0, not just "small."

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`): `data[i] = (i * 31 + 7) % 256` for
  `i` in `[0, 1048576)` (1 MiB), a byte buffer reinterpreted as `uint4`
  words for `histogram64Kernel` (byteCount is a multiple of
  `sizeof(uint4) == 16`) and as raw `unsigned char` for
  `naiveHistogramKernel` -- both kernels read the exact same underlying
  bytes. Each byte falls into bin `(byte >> 2) & 0x3F`, i.e. its top 6
  bits, giving 64 bins.
- **Output**: `argv[1]` (default `output/cuda_output.txt`), 64 lines,
  the final bin counts from the privatized-shared-memory + merge path
  (`histogram64Kernel` + `mergeHistogram64Kernel`), plus a `PASS`/`FAIL`
  line on stdout comparing *both* GPU histograms (privatized+merge and
  naive) against `reference_histogram64()` in `reference.h` (exact
  match expected for both).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cg::thread_block`/`cg::sync` typically migrate to a SYCL `nd_item`'s
`group()`/`group_barrier`, `__shared__` arrays become `sycl::local_accessor`
or `sycl::local_ptr`-backed local memory, and `atomicAdd` on a raw
pointer typically migrates to `sycl::atomic_ref::fetch_add`), build the
result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
