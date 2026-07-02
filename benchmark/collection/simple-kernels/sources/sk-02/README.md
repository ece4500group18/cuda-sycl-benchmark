# Case: backpropTreeReduction (benchmark/simple)

## Summary

| field | value |
| --- | --- |
| category | simple (simple but not trivial -- shared-memory tree reduction) |
| operation | Rodinia "backprop" layer-forward pass: power-of-two-stride shared-memory tree reduction computing one weighted partial sum per (input-block, hidden-unit) pair |
| size | in = 8192 input units (multiple of HEIGHT=16 -> 512 blocks), hid = WIDTH = 16 hidden units; launch = dim3(1, 512) blocks x dim3(16, 16) threads |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected (float tree-sum, matched order) |

## Source

The `__global__` kernel in `main.cu` (`bpnn_layerforward_CUDA`, together
with the small `WIDTH`/`HEIGHT` constants it depends on) is reproduced
**verbatim** from:

- Project: **Rodinia** benchmark suite (gpu-rodinia mirror)
- File: `cuda/backprop/backprop_cuda_kernel.cu`
- Repository: https://github.com/yuhc/gpu-rodinia/blob/master/cuda/backprop/backprop_cuda_kernel.cu
- Copyright (c) 2008-2011 University of Virginia. All rights reserved.
- License: BSD-style, permissive (see `LICENSE` in this directory for
  the full text; GitHub's automatic license detector tags the upstream
  repository "NOASSERTION" because the wording isn't a byte-exact match
  for the cataloged SPDX `BSD-3-Clause` template, but it grants the same
  three permissive conditions plus the standard "AS IS" disclaimer).

`WIDTH` and `HEIGHT` (both `16`, "shared memory width/height") are taken
from the original suite's `backprop.h` and hardcoded locally in `main.cu`
as plain `#define`s -- the full `BPNN` struct, the training driver
(`bpnn_train_cuda`), and the rest of the suite's build system (`facetrain.c`,
`imagenet.c`, image/weight-file I/O) are irrelevant to the single kernel
under test and were omitted. The upstream file's second kernel,
`bpnn_adjust_weights_cuda` (the backward weight-update pass), is likewise
omitted: it implements a different operation (an elementwise weight
update, no reduction) and is not part of this case's kernel list. This
directory's `main()` is new code that allocates the same device buffers
the upstream `bpnn_train_cuda` allocates around this one kernel call
(`input_cuda`, `output_hidden_cuda`, `input_hidden_cuda`,
`hidden_partial_sum`), replacing upstream's file-loaded/`rand()`-seeded
network weights with deterministic index-formula inputs and upstream's
plain `printf` timing output with a deterministic CUDA-vs-CPU-reference
comparison harness. `reference.h` (deterministic input generators plus a
CPU reference that reproduces the kernel's exact tree-summation order)
is new code written for this repository. Everything else here
(`Makefile`, `CMakeLists.txt`, this `README.md`) is likewise new.

## What this case demonstrates (methods used)

1. **Shared-memory power-of-two-stride tree reduction, applied inside a
   real application's forward pass.** Each thread block covers `HEIGHT`
   (16) input units (rows, `threadIdx.y`) and `WIDTH` (16) hidden units
   (columns, `threadIdx.x`). For its column `tx`, a block must reduce 16
   per-row products (`weight[row][tx] * input[row]`) down to one
   weighted sum. The kernel does this with the classic binary tree
   reduction over `__shared__ float weight_matrix[HEIGHT][WIDTH]`:
   ```c
   for (int i = 1; i <= __log2f(HEIGHT); i++) {
     int power_two = __powf(2, i);
     if (ty % power_two == 0)
       weight_matrix[ty][tx] += weight_matrix[ty + power_two/2][tx];
     __syncthreads();
   }
   ```
   after which `weight_matrix[0][tx]` holds the full reduced sum for
   hidden unit `tx`. This is the same reduction *idiom* as this repo's
   `warpShuffleReduction` case, but expressed as an explicit
   `__shared__`-memory, `__syncthreads()`-gated tree (not warp shuffle
   intrinsics), fixed at a compile-time `HEIGHT`/`WIDTH` of 16, and
   embedded in a genuine neural-network layer-forward computation rather
   than a standalone reduction microbenchmark -- illustrating the same
   technique in a distinct application context.

2. **Deterministic fast-math strides.** The kernel's own
   `__log2f(HEIGHT)` / `__powf(2, i)` calls are kept verbatim (they are
   part of the reproduced kernel, and Rodinia's own upstream code uses
   them unconditionally). Because `HEIGHT` is fixed at compile time to
   16 -- an exact power of two exactly representable in `float` --
   `__log2f(16.0f)` is exactly `4.0f` and `__powf(2, i)` for `i` in
   `{1,2,3,4}` is exactly `{2.0f, 4.0f, 8.0f, 16.0f}`: every stride the
   reduction takes is an exact power of two, fully determined by the
   fixed `HEIGHT`/`WIDTH` constants, so these fast-math intrinsics
   introduce no run-to-run nondeterminism here (unlike, say, calling
   them with a runtime-varying or non-power-of-two argument).

3. **Exact correctness via a matched-order CPU reference.** Tree
   reductions are *not* generally bit-exact against a naive serial sum
   (float addition is not associative), so `reference.h` does not sum
   naively -- it replays the identical stride sequence
   (`stride = 2, 4, 8, ..., HEIGHT`) over the identical per-row products,
   in the identical `float` type and the identical step order, adding
   exactly the same operand pairs the GPU kernel adds at each step (a
   stride's writes only ever touch slots the same stride's reads leave
   untouched, so processing rows sequentially on the CPU within a stride
   is computationally equivalent to the GPU's per-thread parallel
   updates, not just numerically close). This yields a genuine
   bit-for-bit oracle (`max_abs_error == 0`), documented as exact
   *because* the reference mirrors the kernel's tree order -- not a
   fabricated loose tolerance for "just some floats".

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `input_node[i] = ((i % 7) - 3) * 0.5`, for `i` in `[0, in]`
    (`input_cuda`, `in = 8192`)
  - `weight_matrix[i][j] = ((i + j) % 5 - 2) * 0.1`, for `i` in
    `[0, in]`, `j` in `[0, hid]` (`input_hidden_cuda`, flattened
    row-major `(in+1) x (hid+1)`, `hid = WIDTH = 16`)
  - Launch: `dim3 grid(1, 512)`, `dim3 threads(16, 16)` (`num_blocks =
    in / HEIGHT = 512`, matching the original driver's own
    `num_blocks = in / 16`)
- **Output**: `argv[1]` (default `output/cuda_output.txt`), 8192 lines
  (`num_blocks * hid = 512 * 16`), the `hidden_partial_sum` array
  produced by `bpnn_layerforward_CUDA` (one `%.9g` float per line, one
  weighted partial sum per (input-block, hidden-unit) pair), plus a
  `PASS`/`FAIL` line on stdout comparing the GPU output against
  `reference_layerforward()` in `reference.h` (exact match expected).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (the
`__shared__` arrays become `sycl::local_accessor`s / local memory,
`__syncthreads()` becomes `item.barrier(sycl::access::fence_space::local_space)`,
and `__log2f`/`__powf` typically migrate to `sycl::log2`/`sycl::pow` or
can be replaced with the equivalent compile-time constants since `HEIGHT`
is fixed), build the result, run it with the same `argv[1]` convention
(e.g. `output/sycl_output.txt`), and diff the two output files (exact
match expected, given both sides replay the same fixed-stride tree sum).
