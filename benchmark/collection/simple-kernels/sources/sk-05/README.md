# Case: cgThreadBlockRankSum (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial) |
| operation | `cooperative_groups::this_thread_block()` partition + shared-memory tree reduction of each thread's rank within the block |
| size | threadsPerBlock = 256, blocksPerGrid = 16 (4,096 threads total) |
| correctness | CPU reference (`reference.h`), exact match expected (`exact_ok`, all 16 blocks) |

## Source

The `sumReduction` device function in `main.cu` is reproduced
**verbatim**, and the `cgkernel` `__global__` kernel is **adapted**,
from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/0_Introduction/simpleCooperativeGroups/simpleCooperativeGroups.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleCooperativeGroups/simpleCooperativeGroups.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

What was kept verbatim: the entire `sumReduction(thread_group g, int *x,
int val)` device function, including its comments -- the generic
shared-memory tree reduction (halve the active-lane count each step,
`x[lane] = val`, `g.sync()`, conditionally add the partner's value,
`g.sync()` again) is untouched.

What was adapted: the original `cgkernel()` creates a whole-thread-block
`this_thread_block()` group, reduces it, prints the result, *then* also
creates 16-thread `tiled_partition<16>` sub-groups of the same block and
reduces each of those too, printing that result as well -- both checks
are self-contained `printf`s inside the kernel with no host-visible
output buffer. This directory's `cgkernel` keeps only the
whole-thread-block partition and its `sumReduction` call (the
`tiled_partition<16>` warp-level-tile idiom is a distinct cooperative-groups
technique, covered by a separate case in this benchmark set, not
duplicated here), and writes the block's reduced sum to
`g_odata[blockIdx.x]` instead of calling `printf`, so the CUDA kernel's
result can be read back on the host and compared byte-for-byte against a
CPU reference and, eventually, a migrated SYCL `sycl::group`/`sycl::sub_group`
port. The reduction algorithm itself -- the `sumReduction` body, and how
`cgkernel` invokes it for the block-wide group -- is unchanged from
upstream. The original file's `main()` (fixed single-block launch,
`cudaDeviceSynchronize` + `printf`-only reporting, no correctness exit
code) is replaced with a deterministic host driver that launches 16
blocks, copies the per-block sums back, and diffs them against a closed-form
CPU reference. Everything else here (host driver, `reference.h`,
`Makefile`, `CMakeLists.txt`, this `README.md`) is new code written for
this repository.

## What this case demonstrates (methods used)

1. **`cooperative_groups::this_thread_block()` partitioning.** Instead of
   raw `threadIdx.x` arithmetic and `__syncthreads()`, `cgkernel` obtains
   a typed `thread_block` handle for the whole block via
   `this_thread_block()`, and passes it (implicitly converted to the base
   `thread_group` type) into the generic `sumReduction` helper, which
   uses the group's own `.thread_rank()`, `.size()`, and `.sync()`
   methods in place of `threadIdx.x`, `blockDim.x`, and
   `__syncthreads()`. This is "simple but not trivial": the CG API calls
   themselves are one-liners, but understanding that `.sync()` here is
   exactly a block-wide barrier (not a smaller, cheaper partition-only
   sync, since the group is the entire block) -- and that the reduction's
   correctness depends on *every* thread in the block reaching each
   `g.sync()` call in lockstep -- is the substantive control-flow
   reasoning this case isolates.

2. **Shared-memory tree reduction with two barriers per step.** Each
   iteration of `sumReduction`'s loop halves the number of "active"
   lanes (`i = g.size()/2; ...; i /= 2`); every thread first stores its
   current value into shared memory (`x[lane] = val`), synchronizes,
   then active lanes (`lane < i`) add their "partner" lane's value
   (`x[lane + i]`), and the group synchronizes again before the next
   iteration overwrites the shared array. With `threadsPerBlock = 256`
   (a power of two), `g.size()/2` halves evenly at every step
   (128, 64, 32, ..., 1), so no thread ever reads a stale or
   out-of-range partner value.

3. **Deterministic, order-independent input.** Each thread's reduction
   input is simply its own `thread_rank()` within the block, so every
   block computes the sum `0 + 1 + ... + (threadsPerBlock - 1)`, which
   has the exact closed-form value `threadsPerBlock * (threadsPerBlock - 1)
   / 2 = 32640` -- an integer computation with no floating-point
   involved and no dependence on scheduling order, so every one of the
   16 blocks must produce this exact value for the case to pass.

## Input / Output

- **Input**: no host-generated data buffers -- each thread's reduction
  input is its own `cooperative_groups` rank (`0..255`), generated
  on-device by the kernel itself; launch = 16 blocks x 256 threads,
  `256 * sizeof(int)` bytes of dynamic shared memory per block.
- **Output**: `argv[1]` (default `output/cuda_output.txt`), 16 lines,
  the reduced rank-sum computed by block `0..15`, plus a `PASS`/`FAIL`
  line on stdout: `PASS` iff every block's value equals
  `reference_block_rank_sum(256) == 32640` from `reference.h` (exact
  match expected).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cooperative_groups::this_thread_block()` / `thread_group::sync()` /
`.thread_rank()` / `.size()` typically migrate to a SYCL `sycl::group`
parameter on the kernel functor plus `group_barrier()` /
`.get_local_linear_id()` / `.get_local_range()`, and the CUDA dynamic
`extern __shared__` workspace array becomes a `local_accessor`), build
the result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
