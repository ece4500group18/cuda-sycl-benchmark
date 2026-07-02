# Case: layeredTextureGather (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | per-element `-src[layer]+layer` transform, read via a `cudaArrayLayered` 3D-layered texture (`tex2DLayered`) vs. read directly from a flat, linear global-memory array |
| size | width = height = 32, num_layers = 4 (4,096 elements total), launch = 4x4 blocks of 8x8 threads per layer |
| correctness | CPU reference (`reference.h`), `max_abs_error == 0` expected for both paths |

## Source

The `__global__` kernel `transformKernel` in `main.cu` is reproduced
**verbatim** (formatting only) from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/0_Introduction/simpleLayeredTexture/simpleLayeredTexture.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleLayeredTexture/simpleLayeredTexture.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

Kept verbatim: the `transformKernel` device function body (the
`x`/`y` index computation, the `(x+0.5f)/width` texel-center
normalized-coordinate formula, and the
`-tex2DLayered<float>(tex,u,v,layer) + layer` transform), and the
overall host-side texture setup sequence (`cudaCreateChannelDesc`,
`cudaMalloc3DArray(..., cudaArrayLayered)`, `cudaMemcpy3D`,
`cudaResourceDesc`/`cudaTextureDesc`/`cudaCreateTextureObject`, and the
`dim3 dimBlock(8,8,1)` / `dimGrid(width/8,height/8,1)` launch
configuration, one kernel launch per layer).

Omitted: the original `main()`'s `findCudaDevice`/`helper_cuda.h`/
`helper_functions.h` dependency, its warmup launch and
`StopWatchInterface` timing harness, its `-regression` file-dump
option, and its `compareData()`-based tolerance check
(`MIN_EPSILON_ERROR = 5e-3f`) against a host-computed
`-h_data[i]+layer` array -- none of these are needed for a
deterministic, single-file CUDA-vs-SYCL output comparison, and the
timing/regression-file code pulls in the `helper_functions.h`
dependency this repo avoids. One deliberate behavioral change: the
original configures the texture with `cudaFilterModeLinear` (bilinear)
and therefore needs the `5e-3f` tolerance above even though it samples
at exact texel centers (real texture units use fixed-point
interpolation weights, which are not guaranteed bit-exact even at
weight 1.0); this directory instead configures
`texDescr.filterMode = cudaFilterModePoint` (nearest/point sampling),
which returns the backing texel with no interpolation at all, so the
oracle here can be a strict `max_abs_error == 0` instead of an
epsilon-tolerance check.

New code written for this repository: the host driver in `main.cu`
(replacing the original's device-query/timing/regression-file `main`
with a deterministic harness that runs both memory paths and diffs
each against a CPU reference), `reference.h`, `Makefile`,
`CMakeLists.txt`, this `README.md`, and the second kernel
`transformKernelFlat`, added specifically for this case to provide the
"flat array" side of the memory-layout comparison (the original sample
only ever demonstrates the texture path; comparing it against a
linear-array read of the identical data is new).

## What this case demonstrates (methods used)

1. **Layered-texture memory layout vs. linear array layout.**
   `transformKernel` (upstream, unmodified) fetches its source value
   through the CUDA texture pipeline: a `cudaArrayLayered`
   `cudaArray` (an opaque, hardware-defined layout, potentially
   tiled/swizzled for locality and backed by the dedicated texture
   cache) accessed via `tex2DLayered(tex, u, v, layer)` with
   normalized texel-center coordinates. `transformKernelFlat` (new,
   this repo) performs the exact same `-src[idx] + layer` arithmetic,
   but computes `idx = layer*width*height + y*width + x` and reads
   `g_idata[idx]` with a plain pointer load from an ordinary linear
   global-memory buffer holding the identical data. Both kernels visit
   every `(x, y, layer)` triple exactly once and apply the identical
   transform; only the memory layout/access mechanism used to fetch
   the source element differs -- isolating "layered texture" from
   "linear array" as a pure memory-movement variable.

2. **Point/nearest-filtered texel-center sampling for a bit-exact
   oracle.** Normalized coordinates `u = (x+0.5)/width`,
   `v = (y+0.5)/height` land exactly on a texel center. With
   `cudaFilterModePoint` this guarantees the fetched value is exactly
   the backing texel (no interpolation), so the texture path's output
   can be compared bit-for-bit against a flat-array CPU reference
   instead of requiring an interpolation-error tolerance.

3. **3D-layered texture object setup**: `cudaMalloc3DArray` with the
   `cudaArrayLayered` flag, `cudaMemcpy3D` to populate all layers in
   one call, and `cudaResourceDesc`/`cudaTextureDesc` +
   `cudaCreateTextureObject` to build a `cudaTextureObject_t` bound to
   that array -- the standard CUDA texture-object idiom, one
   `tex2DLayered` fetch per thread with an explicit `layer` argument.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`): `data[i] = (i % 19) - 9` for the in-layer flat index
  `i` in `[0, width*height)`, replicated identically into all
  `num_layers` layers before being uploaded once to the texture array
  (`cudaMemcpy3D`) and once to the flat device buffer (`cudaMemcpy`).
  `width = height = 32`, `num_layers = 4`.
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the
  texture-path result (`h_odata_tex`), `width*height*num_layers = 4096`
  lines, one `%.9g` float per line (layer-major, then row-major within
  a layer, matching `layer*width*height + y*width + x`), plus a
  `PASS`/`FAIL` line on stdout comparing **both** the texture-path and
  flat-path outputs against `reference_layered_transform()` in
  `reference.h` (exact match expected for both).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cudaTextureObject_t`/`tex2DLayered` typically migrate to
`dpct::image_wrapper`/`sycl::ext::oneapi::experimental` image APIs or
an equivalent layered-image sampler with `sycl::filtering_mode::nearest`
to preserve the point-sampling behavior this case relies on for an
exact oracle), build the result, run it with the same `argv[1]`
convention (e.g. `output/sycl_output.txt`), and diff the two output
files (exact match expected).
