# Case: chunkedStreamPipelineIncrement (benchmark/memory)

## Summary

| field | value |
| --- | --- |
| category | memory (memory movement / memory layout) |
| operation | per-element increment `out[i] = in[i] + 1`, buffer split across 4 pinned-staging CUDA streams (chunked/pipelined) vs. one pinned single-stream pass over the whole buffer (monolithic) |
| size | N = 1<<22 = 4,194,304 ints; STREAM_COUNT = 4 (chunk_size = 1,048,576); block = 512 threads |
| correctness | CPU reference (`reference.h`), exact match expected (`mismatches == 0` for both paths) |

## Source

The `__global__` kernel in `main.cu` (`incKernel`) is reproduced
**verbatim** from:

- Project: **NVIDIA/cuda-samples**
- File: `cpp/0_Introduction/simpleMultiCopy/simpleMultiCopy.cu`
- Repository: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleMultiCopy/simpleMultiCopy.cu
- Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
- License: BSD-3-Clause (see `LICENSE` in this directory)

Kept verbatim: the `incKernel` body (`g_out[idx] = g_in[idx] + 1` inside
an `inner_reps`-iteration loop, guarded by `if (idx < N)`), and the
overall enqueue order of upstream's `processWithStreams` (H2D copy,
then kernel, then D2H copy, all issued on the same per-chunk stream).

Omitted, and why: upstream's `cudaEvent_t`-based timing
(`memcpy_h2d_time`, `memcpy_d2h_time`, `kernel_time`, the `cycleDone[]`
events used to pace a 10-repetition (`nreps`) streaming loop), its
`cudaDeviceGetAttribute(cudaDevAttrGpuOverlap, ...)` /
`deviceProp.asyncEngineCount` overlap-capability queries used only to
print diagnostic text, its GFLOPS-based device auto-selection
(`gpuGetMaxGflopsDeviceId`) and `helper_cuda`/`helper_functions`
dependency, and its `SIMULATE_IO` host-side `memcpy` option -- none of
this affects correctness, and this benchmark isolates the chunked/
pipelined memory-movement technique itself, not wall-clock timing or
device selection. Upstream also reuses the same 4 chunk buffers across
10 repetitions purely to amortize allocation cost during timing; since
this case runs the increment exactly once (`inner_reps` fixed at 1,
`nreps` dropped entirely), each chunk gets its own single-use buffer
pair, which simplifies the code without changing the memory-movement
pattern being demonstrated.

New code in this directory: the host driver in `main.cu` (buffer
allocation, deterministic input generation, the two comparison paths,
and CPU-reference verification), `reference.h`, `Makefile`,
`CMakeLists.txt`, and this `README.md` -- replacing upstream's
timing/throughput-measurement harness with a deterministic,
single-file CUDA-vs-SYCL comparison harness.

## What this case demonstrates (methods used)

1. **Chunked, multi-stream pipelining of H2D copy / kernel / D2H copy
   (memory movement).** The N-element buffer is split into
   `STREAM_COUNT = 4` equal chunks. Each chunk gets its own pinned
   (`cudaHostAlloc`) input/output staging buffers, its own device
   input/output buffers, and its own `cudaStream_t`. For every chunk
   `c`, an H2D `cudaMemcpyAsync`, an `incKernel` launch, and a D2H
   `cudaMemcpyAsync` are all enqueued on `stream[c]` -- the same
   enqueue order upstream's `processWithStreams` uses. Because each
   chunk's three operations live entirely on that chunk's own stream,
   a GPU with independent copy engines can execute chunk `c`'s copy
   concurrently with chunk `c-1`'s or `c+1`'s kernel/copy, instead of
   serializing all of them behind one global data dependency.

2. **Single monolithic stream, for contrast.** The identical N-element
   buffer (also pinned, so the *only* variable being isolated is chunk
   count, not pinned-vs-pageable memory -- that contrast is covered by
   this repo's separate `pinnedAsyncIncrement` case) is copied H2D in
   one `cudaMemcpyAsync`, processed by one `incKernel` launch over the
   full `N`, and copied D2H in one `cudaMemcpyAsync`, all on a single
   stream. With nothing else enqueued concurrently, there is no
   opportunity for copy/compute overlap -- this is the baseline the
   chunked/pipelined path is contrasted against.

3. **Why the two paths are guaranteed to match exactly.** `incKernel`
   is unmodified in both paths and computes `g_out[idx] = g_in[idx] +
   1` with `inner_reps` fixed at 1 -- a per-element update with no
   cross-element dependency and no floating-point accumulation. Every
   index in `[0, N)` is processed by exactly one chunk in the chunked
   path (chunk `c` owns indices `[c*chunk_size, (c+1)*chunk_size)`) and
   by the single pass in the monolithic path; splitting the buffer into
   differently-staged, differently-scheduled pieces cannot change the
   value computed for any individual index. The CPU reference
   (`reference_inc` in `reference.h`) computes the same
   `out[i] = in[i] + 1` directly, so both GPU paths must match it
   exactly (`mismatches == 0`), with integer arithmetic ruling out any
   floating-point tolerance question entirely.

## Input / Output

- **Input** (generated deterministically on the host, see
  `reference.h`): `in[i] = i`, for `i` in `[0, N)`, `N = 1<<22 =
  4,194,304`. (Upstream zero-initializes its test input, which cannot
  distinguish "processed the wrong chunk" from "processed the right
  chunk" since `0 + 1 == 1` either way; using `in[i] = i` here makes
  any chunk-offset or stream-buffer mixup produce a detectable,
  non-zero mismatch.)
- **Output**: `argv[1]` (default `output/cuda_output.txt`), the
  chunked/pipelined path's output buffer, one `%d` int per line
  (4,194,304 lines, `out[i] = in[i] + 1` in original index order), plus
  a `PASS`/`FAIL` line on stdout comparing both the chunked/pipelined
  path's and the monolithic path's output against `reference_inc()` in
  `reference.h` (exact match expected).

## Build & run

```bash
make run            # nvcc build, writes output/cuda_output.txt
```

For the SYCL side, migrate `main.cu` with SYCLomatic/dpct (note:
`cudaStreamCreate`/`cudaMemcpyAsync`/`cudaStreamSynchronize` typically
migrate to `sycl::queue` construction and `queue::memcpy`/
`queue::submit` calls against distinct in-order queues, and
`cudaHostAlloc` to `sycl::malloc_host`), build the result, run it with
the same `argv[1]` convention (e.g. `output/sycl_output.txt`), and diff
the two output files (exact match expected).
