# Case: textureInterpolationGather (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | image rotation, hardware texture-object bilinear gather (`tex2D`, normalized coords, wrap addressing) vs. hand-written software bilinear gather from a plain global-memory array |
| size | 64 x 64 image (4096 pixels), rotation angle = 0.5 rad, launch = 8x8 blocks of 8x8 threads |
| correctness | CPU reference (`reference.h`); texture path: `max_abs_error < 0.1` (hardware fixed-point precision bound); software path: `max_abs_error < 1e-3` (float-rounding bound) -- see "Correctness" note below |

## Source

The `__global__` kernel `transformKernel` in `main.cu` is reproduced
**verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/0_Introduction/simpleTexture/simpleTexture.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleTexture/simpleTexture.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

Kept verbatim: the kernel body (rotation math and the `tex2D<float>(tex, tu
+ 0.5f, tv + 0.5f)` fetch), the fixed rotation angle (`0.5f` radians), and
the texture-object configuration used to launch it
(`normalizedCoords = true`, `filterMode = cudaFilterModeLinear`,
`addressMode[0]/[1] = cudaAddressModeWrap`, `readMode =
cudaReadModeElementType`).

Omitted: the original sample's PGM file I/O (`sdkLoadPGM`/`sdkSavePGM`),
command-line argument parsing, `helper_cuda`/`helper_functions`
dependency, `StopWatchInterface` timing, and its file-based
`compareData()` regression check against `data/ref_rotated.pgm` -- none of
these are needed for (and the PGM/external-dataset dependency would be
disallowed for) a deterministic, single-file CUDA-vs-SYCL output
comparison. In their place, this directory generates a synthetic
deterministic image on the host (see below) and checks both GPU kernels
against an in-process CPU reference.

New code written for this repository: the host driver in `main.cu`
(image generation, texture-object setup, kernel launches, verification),
`reference.h`, `Makefile`, `CMakeLists.txt`, this `README.md`, and the
second kernel, `softwareGatherKernel`. `softwareGatherKernel` was written
specifically for this case to give the texture-hardware path an
apples-to-apples comparison point that uses the *same* memory-movement
technique category (a bilinear, wrap-addressed 2D gather) but goes through
a plain `float*` global-memory array instead of a `cudaArray` +
`cudaTextureObject_t`, with the interpolation done by ordinary IEEE-754
float arithmetic instead of the texture unit's fixed-function hardware.

## What this case demonstrates (methods used)

1. **Hardware texture-object bilinear gather (`memory movement`).**
   `transformKernel` reads the source image exclusively through
   `tex2D<float>(tex, ...)`, where `tex` is bound to a `cudaArray` with
   `normalizedCoords = true` and `filterMode = cudaFilterModeLinear`. For
   every output pixel, the dedicated texture-sampling hardware: (a) wraps
   the normalized coordinate into `[0, 1)` per `cudaAddressModeWrap`,
   (b) converts it to texel space, (c) fetches the surrounding 2x2 texel
   neighborhood, and (d) blends the four samples using on-chip fixed-point
   interpolation weights -- all as a single fetch instruction, with no
   explicit indexing or branching in the kernel source.

2. **Software-emulated bilinear gather from global memory
   (`memory movement` contrast).** `softwareGatherKernel` performs the
   identical rotation, then reproduces the texture unit's
   normalize -> wrap -> unnormalize -> bilinear-blend pipeline by hand:
   it computes the fractional texel coordinate, wraps the two integer
   texel indices into `[0, width)`/`[0, height)` itself (`((i % n) + n) %
   n`), issues four ordinary loads from a plain `float*` array, and blends
   them with explicit float multiply-adds. This isolates *where the
   interpolation happens* (fixed-function texture-sampling hardware with
   quantized weights, vs. general-purpose ALU/load-store hardware with
   full float precision) as the only difference between the two paths --
   both read the same 2x2 neighborhood of the same underlying image.

3. **Correctness oracle with a precision-derived (not heuristic)
   tolerance.** Unlike most cases in `benchmark/memory` (which expect
   `max_abs_error == 0` because both paths perform literally the same
   arithmetic in the same order), this case's two GPU paths do **not**
   perform bit-identical arithmetic: the texture unit's interpolation
   weights are stored as an 8-bit fixed-point fraction (1/256 granularity)
   rather than a full 32-bit float, so `transformKernel`'s result is only
   guaranteed to be *close to*, not bit-identical to, a full-precision
   bilinear blend. `reference.h`'s `reference_rotate_bilinear_wrap()`
   performs that same rotate/wrap/bilinear pipeline in double precision as
   the ground truth, and `main.cu` checks:
   - `transformKernel` (hardware, fixed-point weights) against the
     reference with tolerance **0.1**: this case's image values span a
     range of 22 (from `-11` to `11`), so an 8-bit fixed-point weight's
     worst-case quantization error is bounded by `22 / 256 ~= 0.086`;
     0.1 is a safe, explicitly-derived-from-hardware-precision bound, not
     an arbitrary "floats are involved" fudge factor.
   - `softwareGatherKernel` (global memory, full float precision, same
     formula/operand order as the reference) against the reference with a
     much tighter tolerance **1e-3**: since the arithmetic is the same on
     both sides, the only source of disagreement is float (device
     `cosf`/`sinf`) vs. double (host `cos`/`sin`) rounding in the rotation
     step, not an interpolation-precision difference.

4. **Host/device memory management for two different residency models**:
   `cudaMallocArray`/`cudaMemcpyToArray`/`cudaCreateTextureObject`/
   `cudaDestroyTextureObject`/`cudaFreeArray` for the texture-object path,
   vs. plain `cudaMalloc`/`cudaMemcpy`/`cudaFree` for the software-gather
   path's linear array -- the same source image, resident in two different
   kinds of device memory simultaneously.

## Input / Output

- **Input** (generated deterministically on the host, see `reference.h`):
  - `img[y*width+x] = ((x*3 + y) % 23) - 11`, for `x, y` in `[0, 64)`
    (values range over `[-11, 11]`)
  - rotation angle = `0.5f` radians (matches the original sample's fixed
    `angle` constant)
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the 64x64
  row-major rotated image produced by `transformKernel` (the
  texture-object path), one `%.9g` float per line (4096 lines), plus a
  `PASS`/`FAIL` line on stdout comparing **both** kernels' outputs against
  `reference_rotate_bilinear_wrap()` in `reference.h` (tolerances as
  described above).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cudaTextureObject_t`/`tex2D`/`cudaMallocArray`/`cudaCreateTextureObject`
typically migrate to `sycl::ext::oneapi::experimental` bindless-image or
`sycl::image` APIs -- this is one of the more involved migration targets
in this benchmark set, since SYCL's image/sampler filtering precision and
addressing-mode semantics are implementation-defined and should be
re-validated against the same tolerances after migration), build the
result, run it with the same `argv[1]` convention (e.g.
`output/sycl_output.txt`), and diff the two output files within the same
tolerances (0.1 for the texture-path values, 1e-3 for the software-gather
values -- do not tighten these to an exact match, since the texture
path's fixed-point interpolation precision is a genuine hardware
constraint, not a bug).
