# Case: pitchLinearVsCudaArrayTexture (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | periodic (wrap) per-pixel shift of a 2D float grid, read through a point-filtered, wrap-addressed texture bound to pitch-linear memory (`cudaMallocPitch`) vs. a CUDA array (`cudaMallocArray`) |
| size | 64 x 32 floats (2,048 elements), launch = (4,2) blocks x 16x16 threads |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected for both kernels |

## Source

The two `__global__` kernels in `main.cu` (`shiftPitchLinear` and
`shiftArray`) are reproduced **verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/0_Introduction/simplePitchLinearTexture/simplePitchLinearTexture.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simplePitchLinearTexture/simplePitchLinearTexture.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

The original sample's `runTest()` driver -- `helper_cuda`/
`helper_functions` dependency, `NUM_REPS = 100` timed replay loop,
`cudaEvent_t`-based bandwidth/fetch-rate measurement, and a
`compareData(..., 0.0f, 0.15f)` fuzzy/relative-tolerance comparison --
is not reproduced; this directory's `main()` replaces it with a
deterministic, single-file CUDA-vs-SYCL comparison harness that runs
each kernel once and checks an **exact** numeric result. The texture
resource/descriptor setup (`cudaResourceDesc` /
`cudaTextureDesc` with `cudaResourceTypePitch2D` /
`cudaResourceTypeArray`, `cudaFilterModePoint`, `cudaAddressModeWrap`,
normalized coordinates) is carried over because it is required to
correctly invoke the two reproduced kernels, with one deliberate fix:
the original sets `texRes.res.pitch2D.pitchInBytes = h_pitchBytes`
(the *host* row stride) rather than the actual device pitch returned
by `cudaMallocPitch`; this happens to be harmless in the original only
because its 2048-float-wide (8192-byte) rows already satisfy the
allocator's alignment and pick up no padding. This case deliberately
uses a much smaller, non-2048 width (64 floats = 256 bytes/row) to
keep the problem size small, so the two pitches are **not** guaranteed
to coincide; `main.cu` here uses the real `d_pitchBytes` returned by
`cudaMallocPitch` for the texture descriptor, as required for
correctness in general. Host allocation/copy code, the deterministic
input generator, and the CPU reference (`reference.h`) are new code
written for this repository.

## What this case demonstrates (methods used)

1. **Texture-bound memory layout: pitch-linear vs. CUDA array.**
   `shiftPitchLinear` binds its texture object to a plain
   `cudaMallocPitch` linear allocation (`cudaResourceTypePitch2D`);
   `shiftArray` binds its texture object to an opaque `cudaMallocArray`
   CUDA array (`cudaResourceTypeArray`), which CUDA is free to store in
   whatever blocked/swizzled layout is best for 2D spatial locality on
   the given hardware. Both are read with the identical texture-fetch
   expression, so any difference between the two kernels' behavior
   would have to come from the underlying storage layout, not the
   per-thread arithmetic -- this is the point of the comparison.

2. **Point-filtered, wrap-addressed 2D texture fetch as a periodic
   (circular) shift.** Both kernels compute
   `odata[yid][xid] = tex2D<float>(tex, (xid+shiftX)/(float)width,
   (yid+shiftY)/(float)height)` with `cudaFilterModePoint` (nearest
   texel, no interpolation) and `cudaAddressModeWrap` in both
   dimensions (periodic boundary conditions) -- the texture hardware
   itself performs the modulo-wrap addressing that would otherwise
   require an explicit `% width` / `% height` in the kernel body.

3. **Exact-match oracle via power-of-two dimensions.** `width = 64`
   and `height = 32` are both powers of two, so `(xid+shiftX)/(float)width`
   and the wrap unit's internal `frac(.) * width` re-normalization are
   each a single float32 exponent adjustment with **zero** rounding
   error (dividing/multiplying by a power of two is always exact in
   IEEE-754, short of underflow, which cannot occur here). Combined
   with an integer pixel shift, the nearest-texel fetch is therefore
   guaranteed to land exactly on the intended source texel every time,
   so both kernels' output is expected to match the CPU reference
   **exactly** (`max_abs_error == 0`) -- unlike the original sample's
   2048x2048 configuration, which instead compares with a 0.15
   relative tolerance to absorb occasional texture-addressing rounding
   at non-power-of-two-friendly coordinates.

4. **Host/device memory management for 2D data**: `cudaMallocPitch`,
   `cudaMallocArray`, `cudaMemcpy2D`, `cudaMemcpy2DToArray`,
   `cudaMemset2D`, `cudaCreateTextureObject` /
   `cudaDestroyTextureObject`, plus `cudaGetLastError` /
   `cudaDeviceSynchronize` error checking via the repo's `CHECK` macro
   convention.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`):
  - `src[y*w + x] = ((x*3 + y*5) % 17) - 8`, `w = 64`, `h = 32`
  - `shiftX = 5`, `shiftY = 7` (matching the original sample's own
    shift values); launch = `(w/16, h/16) = (4, 2)` blocks of `16x16`
    threads (`TILE_DIM = 16`, matching the original)
- **Output**: `argv[1]` (default `output/output.txt`), `w*h = 2048`
  lines, one `%.9g` float per line in row-major order (`y*w + x`), the
  `shiftPitchLinear` kernel's output, plus a `PASS`/`FAIL` line on
  stdout comparing **both** kernels' outputs against
  `reference_shift()` in `reference.h` (exact match expected, per the
  power-of-two-alignment argument above).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cudaTextureObject_t` + `tex2D<float>` typically migrate to
`dpct::image_wrapper`/`sycl::ext::oneapi::experimental` image objects
or a manual `sycl::sampler` with `sycl::addressing_mode::repeat` and
`sycl::filtering_mode::nearest`; `cudaMallocArray`/`cudaMemcpy2DToArray`
migrate to `dpct::image_matrix`/USM image allocations), build the
result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files (exact match
expected).
