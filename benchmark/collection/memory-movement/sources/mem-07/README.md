# Case: cubemapTextureGather (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | per-element `-src` negation transform, read via a `cudaArrayCubemap` 6-face cubemap texture (`texCubemap`) vs. read directly from a flat, face-major global-memory array |
| size | width = 32 (face size), 6 faces (6,144 elements total), launch = 4x4 blocks of 8x8 threads, one kernel launch covering all 6 faces |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected for both paths |

## Source

The `__global__` kernel `transformKernel` in `main.cu` is reproduced
**verbatim** (formatting only) from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/0_Introduction/simpleCubemapTexture/simpleCubemapTexture.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleCubemapTexture/simpleCubemapTexture.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

Kept verbatim: the `transformKernel` device function body (the `x`/`y`
index computation, the `((x+0.5f)/width)*2-1` texel-center
normalized-coordinate formula, the per-face `cx`/`cy`/`cz` direction-vector
construction for all six cubemap faces, and the
`-texCubemap<float>(tex, cx, cy, cz)` transform), and the overall
host-side texture setup sequence (`cudaCreateChannelDesc`,
`cudaMalloc3DArray(..., cudaArrayCubemap)`, `cudaMemcpy3D`,
`cudaResourceDesc`/`cudaTextureDesc`/`cudaCreateTextureObject`, and the
`dim3 dimBlock(8,8,1)` / `dimGrid(width/8,width/8,1)` launch
configuration, one kernel launch covering all 6 faces since the kernel
itself loops over `face` internally).

Omitted: the original `main()`'s `findCudaDevice`/`helper_cuda.h`/
`helper_functions.h` dependency, its device-property/SM-count query, its
warmup launch and `StopWatchInterface` timing harness, its
`-regression` PGM/`.dat` file-dump option, and its `compareData()`-based
tolerance check (`MIN_EPSILON_ERROR = 5e-3f`) against a host-computed
`-h_data[i]+layer` array -- none of these are needed for a
deterministic, single-file, console-only CUDA-vs-SYCL output comparison.
One deliberate behavioral change: the original configures the texture
with `cudaFilterModeLinear` (bilinear) and therefore needs the `5e-3f`
tolerance above even though it samples at exact face-texel centers (real
texture units use fixed-point interpolation weights, which are not
guaranteed bit-exact even at weight 1.0); this directory instead
configures `texDescr.filterMode = cudaFilterModePoint` (nearest/point
sampling), which returns the backing texel with no interpolation at
all, so the oracle here can be a strict `max_abs_error == 0` instead of
an epsilon-tolerance check.

New code written for this repository: the host driver in `main.cu`
(replacing the original's device-query/timing/regression-file `main`
with a deterministic harness that runs both memory paths and diffs each
against a CPU reference), `reference.h`, `Makefile`, `CMakeLists.txt`,
this `README.md`, and the second kernel `transformKernelFlat`, added
specifically for this case to provide the "flat array" side of the
memory-layout comparison (the original sample only ever demonstrates
the cubemap-texture path; comparing it against a linear-array read of
the identical data is new).

## What this case demonstrates (methods used)

1. **Cubemap-texture memory layout vs. linear array layout.**
   `transformKernel` (upstream, unmodified) fetches its source value
   through the CUDA texture pipeline: a `cudaArrayCubemap` `cudaArray`
   (six faces in an opaque, hardware-defined cubemap layout, addressed
   by a 3D direction vector rather than a linear offset, and backed by
   the dedicated texture cache) accessed via `texCubemap(tex, cx, cy,
   cz)` with a per-face direction vector built from normalized
   texel-center coordinates. `transformKernelFlat` (new, this repo)
   performs the exact same `-src[idx]` arithmetic, but computes
   `idx = face*width*width + y*width + x` and reads `g_idata[idx]` with
   a plain pointer load from an ordinary linear global-memory buffer
   holding the identical data in face-major order. Both kernels visit
   every `(x, y, face)` triple exactly once and apply the identical
   transform; only the memory layout/access mechanism used to fetch the
   source element differs -- isolating "cubemap texture" from "linear
   array" as a pure memory-movement variable.

2. **Point/nearest-filtered texel-center sampling for a bit-exact
   oracle.** Normalized direction-vector components are constructed so
   that, for every `(x, y, face)`, the sampled direction lands exactly
   on that face's texel center. With `cudaFilterModePoint` this
   guarantees the fetched value is exactly the backing texel (no
   interpolation), so the texture path's output can be compared
   bit-for-bit against a flat-array CPU reference instead of requiring
   an interpolation-error tolerance.

3. **Cubemap texture object setup**: `cudaMalloc3DArray` with the
   `cudaArrayCubemap` flag (extent `width x width x 6`), `cudaMemcpy3D`
   to populate all six faces in one call, and
   `cudaResourceDesc`/`cudaTextureDesc` + `cudaCreateTextureObject` to
   build a `cudaTextureObject_t` bound to that array -- the standard
   CUDA cubemap-texture-object idiom, with a single `texCubemap` fetch
   per `(x, y, face)` driven entirely by the 3D direction vector (no
   explicit face index is passed to the texture fetch itself).

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`): `data[i] = (i % 29) - 14` for the flat, face-major
  index `i` in `[0, width*width*6)` (`i = face*width*width + y*width +
  x`), uploaded once to the cubemap texture array (`cudaMemcpy3D`) and
  once to the flat device buffer (`cudaMemcpy`). `width = 32`
  (face size), 6 faces.
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the
  texture-path result (`h_odata_tex`), `width*width*6 = 6144` lines, one
  `%.9g` float per line (face-major, then row-major within a face,
  matching `face*width*width + y*width + x`), plus a `PASS`/`FAIL` line
  on stdout comparing **both** the texture-path and flat-path outputs
  against `reference_cubemap_transform()` in `reference.h` (exact match
  expected for both).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cudaTextureObject_t`/`texCubemap` typically migrate to
`dpct::image_wrapper`/`sycl::ext::oneapi::experimental` image APIs with
a cubemap-compatible sampler, or may require manual porting since SYCL's
image support for cubemap-style 3D-direction addressing is less
standardized than 2D/layered images -- use
`sycl::filtering_mode::nearest` to preserve the point-sampling behavior
this case relies on for an exact oracle), build the result, run it with
the same `argv[1]` convention (e.g. `output/sycl_output.txt`), and diff
the two output files (exact match expected).
