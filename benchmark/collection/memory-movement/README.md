# memory movement and memory layout collection

Owner: yuepan (migrated from the `yuepan` branch `benchmark/memory/` cases)

See `../README.md` for the shared workflow, CSV columns, SOURCE.txt
format, and snapshot rules. Fill in the coverage matrix below, then
register candidates in `candidates.csv`. Adapted (runnable) cases go to
`benchmark/cases/<this category's slug>/<case>/` (not here) and use the
shared `benchmark/tools/verify_lib.py`.

Snapshots of every candidate below are in `sources/<id>/`; each pairs
one or more upstream-derived CUDA kernels with a new, deterministic
CPU-checked harness (see each case's own `README.md` and `SOURCE.txt`
for exact provenance).

## Coverage matrix

Dimensions:

- **Mechanism**: the specific memory-movement/layout feature under test
  (shared memory, texture, constant memory, pinned/zero-copy/unified
  host memory, streams, alignment, access order, tiling).
- **Contrast**: the two techniques the candidate's kernels compare.
- **Kernel shape**: the compute pattern the mechanism is applied to.

| Candidate | Mechanism | Contrast | Kernel shape |
|---|---|---|---|
| mem-01 asyncCopySingleStage | async-copy | shared-mem tiling | matmul |
| mem-02 bankConflictReduction | shared-mem layout | bank-conflict addressing | reduction |
| mem-03 chunkedStreamPipelineIncrement | streams | pinned staging / chunking | elementwise |
| mem-04 coalescedAxpyDistribution | global access pattern | coalesced vs chunked | elementwise (AXPY) |
| mem-05 constMemMatrixDims | constant memory | read-only cache | elementwise |
| mem-06 csrScanOrderSpMM | access order | sparse (CSR) scan order | SpMM |
| mem-07 cubemapTextureGather | texture (cubemap) | hw gather vs flat array | elementwise |
| mem-08 histogramSharedPrivatization | shared-mem layout | privatization + merge | histogram |
| mem-09 layeredTextureGather | texture (layered) | hw gather vs flat array | elementwise |
| mem-10 memAlign | global access pattern | alignment | elementwise (AXPY) |
| mem-11 pinnedAsyncIncrement | pinned host memory | async vs sync copy | elementwise |
| mem-12 pitchLinearVsCudaArrayTexture | texture (pitch-linear/array) | hw gather vs flat array | elementwise |
| mem-13 pyramidTiledPathfinderDP | temporal tiling | pyramid batching | DP recurrence |
| mem-14 separableConvHaloTiling | shared-mem layout | halo tiling | stencil/convolution |
| mem-15 sharedScanVsShuffleScan | shared-mem vs register | scan (Hillis-Steele vs shuffle) | scan |
| mem-16 sparseDenseSpmvLayout | access order | sparse (CSR) vs dense | SpMV |
| mem-17 textureInterpolationGather | texture (plain) | hw interpolation vs sw gather | elementwise |
| mem-18 tiledMatmulShmem | shared-mem layout | tiling | matmul |
| mem-19 unifiedMemoryAccess | unified memory | discrete vs managed | strided gather |
| mem-20 zeroCopyMappedVectorAdd | zero-copy mapped memory | mapped vs explicit copy | elementwise (vector add) |

Dark cells / known gaps: no float-atomic-driven memory contention case
(covered instead in `simple-kernels`/`streams-atomics-templates`); no
multi-GPU or peer-to-peer memory-movement case.

## Dedup policy

Same algorithm from multiple suites: include more than one variant only
when they cover different matrix cells; otherwise prefer the smaller,
cleaner source.
