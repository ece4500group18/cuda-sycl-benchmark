# simple-but-not-trivial kernels collection

Owner: yuepan (migrated from the `yuepan` branch `benchmark/simple/` cases)

See `../README.md` for the shared workflow, CSV columns, SOURCE.txt
format, and snapshot rules. Fill in the coverage matrix below, then
register candidates in `candidates.csv`. Adapted (runnable) cases, if
any, go under `cases/<name>/` and use the shared
`../tools/verify_lib.py`.

Snapshots of every candidate below are in `sources/<id>/`; each pairs
one or more upstream-derived CUDA kernels with a new, deterministic
CPU-checked harness (see each case's own `README.md` and `SOURCE.txt`
for exact provenance).

## Coverage matrix

Dimensions:

- **Technique**: the specific CUDA feature/idiom under test (atomics,
  cooperative groups, warp intrinsics, templates, dynamic shared memory,
  device asserts, occupancy API, stream-ordered allocation).
- **Contrast**: the two implementations the candidate's kernels compare.
- **Data type**: the element type the kernels operate on.

| Candidate | Technique | Contrast | Data type |
|---|---|---|---|
| sk-01 atomicIntrinsics | atomics (11 RMW ops) | device-wide race | int |
| sk-02 backpropTreeReduction | shared-mem tree reduction | real-app kernel (backprop) | float |
| sk-03 binaryPartitionOddEvenReduce | cooperative groups (binary partition) | cg reduce vs naive atomics | int |
| sk-04 bitonicVsOddEvenMergeSort | sorting network (shared mem) | bitonic vs odd-even merge | key/value |
| sk-05 cgThreadBlockRankSum | cooperative groups (thread block) | generic reduce helper | int |
| sk-06 deviceAssertGuard | device-side assert | assert vs flag-recording | int |
| sk-07 dynamicSharedMinReduction | dynamic shared memory | tree-halving vs linear scan | float |
| sk-08 fp16PackedScalarProduct | fp16 packed SIMD (half2) | native ops vs explicit intrinsics | half2 |
| sk-09 gridSyncCGReduction | cooperative groups (grid sync) | single-pass vs two-launch reduction | double |
| sk-10 occupancyTunedLaunch | occupancy API | suggested vs fixed launch config | uint32 |
| sk-11 shflScanWarpPrefixSum | warp shuffle scan | single-warp vs multi-warp config | int |
| sk-12 streamOrderedAllocVectorAdd | stream-ordered allocation | cudaMallocAsync vs cudaMalloc | float |
| sk-13 systemScopeAtomicAdd | atomics (system scope) | system-scope vs device-scope | int |
| sk-14 templatedSharedMemIdiom | C++ templates (shared mem idiom) | int vs float instantiation | int/float |
| sk-15 threadFenceSinglePassReduction | __threadfence + atomic ticket | single-pass vs two-launch reduction | float |
| sk-16 voteAnyAll | warp vote intrinsics | any vs all | bool |
| sk-17 warpAggregatedAtomicCompaction | warp-aggregated atomics | aggregated vs naive atomicAdd | int (compaction) |
| sk-18 warpDivergence | warp divergence | branchy vs branch-free | float |
| sk-19 warpShuffleReduction | warp shuffle reduction | shared-mem tree vs shuffle | float |

Dark cells / known gaps: no `curand`/library-RNG-driven simple kernel
(covered in `molecular-dynamics`); no multi-GPU or peer-to-peer case.

## Dedup policy

Same algorithm from multiple suites: include more than one variant only
when they cover different matrix cells; otherwise prefer the smaller,
cleaner source.
