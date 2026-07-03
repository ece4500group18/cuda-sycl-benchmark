# graph / irregular access collection

Owner: liqui

See `../README.md` for the shared workflow, CSV columns, SOURCE.txt
format, and snapshot rules. This file holds the coverage matrix and the
category-specific notes. Snapshots of every non-excluded candidate are in
`sources/<id>/`.

## Sources and licenses (verified 2026-06-11)

| Source | License | Notes |
|---|---|---|
| HeCBench (zjin-lcf/HeCBench) | BSD-3-Clause | Each case also ships an official SYCL port — useful as oracle, but a training-data contamination risk if used to evaluate LLM-based migration (team decision pending). Most shortlisted cases have built-in CPU verification; per-case provenance (ECL suite, Chai, cuGraph) is in each `SOURCE.txt`. |
| Rodinia (yuhc/gpu-rodinia mirror) | BSD-style (UVA) | Datasets are a separate download. |
| Galois / LonestarGPU (IntelligentSoftwareSystems/Galois) | BSD-3 (UT Austin) | Worklist apps; hardest tier. GPU apps compile against the in-repo `libgpu` (gg/IrGL) runtime + cub — `libgpu` is vendored in `sources/_deps/galois-libgpu/`; cub/moderngpu are Galois `external/` submodules, not vendored. |
| Pannotia (pannotia/pannotia) | BSD-style (AMD) | **Excluded as direct source**: upstream kernels are OpenCL `.cl`, not CUDA. CUDA equivalents come via HeCBench ports. |
| Gunrock | Apache-2.0 | Template-heavy framework; whole-app migration out of scope. |

## Coverage matrix

Dimensions (a case "covers" a cell if its kernels exercise it):

- **Rep**: input representation — `CSR`, `COO/edge-list`, `dense-adj`
- **Par**: parallelization — `topo` (topology-driven, all vertices each
  iter), `edge` (edge-centric), `wl` (data-driven worklist/frontier)
- **Sync**: `flag` (host-loop convergence flag), `atomic`
  (add/min/CAS), `warp` (shfl/ballot/vote)
- **LB**: load balancing — `tpv` (thread-per-vertex), `coop`
  (warp/block-cooperative), `dp` (dynamic parallelism)
- **Mem**: `smem` (shared-memory staging/queues), `ldg` (texture/__ldg),
  `dynalloc` (in-kernel allocation)
- **RNG**: randomness source (added after finding curand in independentset)

Matrix reflects desk review (2026-06-11), not just registration guesses.

| Candidate | Rep | Par | Sync | LB | Mem | RNG |
|---|---|---|---|---|---|---|
| graph-01 rodinia/bfs | CSR | topo | flag | tpv | — | — |
| graph-02 hecbench/sssp (Chai) | CSR | wl | atomic(min,add,CAS) | tpv | smem | — |
| graph-03 hecbench/page-rank | CSR | topo | flag | tpv | — | — |
| graph-05 hecbench/mis (ECL-MIS) | CSR | topo | flag | tpv | — | hash priorities |
| graph-06 hecbench/cc (ECL-CC) | CSR/edge | edge | atomic(CAS,add)+warp(shfl) | tpv | smem | — |
| graph-07 hecbench/jaccard (cuGraph) | CSR | edge | atomic(add)+warp(shfl) | coop | — | — |
| graph-08 hecbench/floydwarshall | dense-adj | — | — | tpv | — (fw2 variant: smem tiled) | — |
| graph-09 hecbench/bh (ECL-BH) | tree | wl | atomic(CAS)+warp(ballot) | coop | smem | — |
| graph-10 lonestar/bfs-wl | CSR | wl | warp(ballot) | coop | smem queue | — |
| graph-11 lonestar/sssp-wl | CSR | wl | atomic(min)+warp(ballot) | coop | smem queue | — |
| graph-12 lonestar/spanningtree | CSR | wl | flag | coop | smem | — |
| graph-15 lonestar/triangle-counting | CSR | edge | atomic(add) | coop | smem | — |
| graph-16 lonestar/independentset | CSR | topo | flag | tpv | — | curand |
| graph-13 lonestar/dmr | mesh | wl | atomic(add,min) | coop | dynalloc | — |

Dark cells / gaps:
- Dynamic parallelism (`dp`) — only if a clean candidate shows up; not
  worth forcing.
- Graph coloring (Pannotia color is OpenCL-only; ECL-GC is a CUDA
  alternative, license unverified) and betweenness centrality (Pannotia bc
  also OpenCL-only) — acceptable gaps unless a clean candidate appears.

Excluded (in `candidates.csv`, no snapshot): graph-04 Pannotia color
(OpenCL-only), graph-14 Gunrock (whole-framework, out of scope).

## Dedup policy

Same algorithm from multiple suites (e.g. bfs in Rodinia, HeCBench,
Lonestar): include at most two variants and only if they cover different
matrix cells (e.g. graph-01 is topo+flag, graph-10 is
wl+ballot+smem-queue — both stay). Otherwise prefer the smaller, cleaner
source.

## Cases already adapted

Four candidates are adapted into full case units under
`benchmark/cases/graph-irregular/` (2026-07-03), with upstream kernels kept
verbatim and deterministic CSR-graph harnesses + CPU oracles added:

- graph-03 → `hecbenchPagerankMapReduce` (map+reduce power iteration)
- graph-05 → `hecbenchMisPriority` (ECL-MIS lock-free prioritized selection)
- graph-06 → `hecbenchEclConnectedComponents` (ECL-CC 5-kernel hooking +
  pointer jumping)
- graph-07 → `hecbenchJaccardWeights` (nvGRAPH Jaccard: warp prefix sum,
  binary-search intersections, atomics)

- graph-02 → `chaiSsspWorklist` (Chai worklist SSSP: per-block
  shared-memory queues, double-buffered global frontier — the dataset's
  only worklist-pattern case)

All five pass `verify.py --selftest`; GPU validation pending. graph-01
(bfs) and graph-08 (floyd-warshall) were NOT re-adapted — the dataset
already covers those patterns (bfs, hecbenchBfsFrontier/RelaxEdges,
hecbenchFloydWarshallStep/MinPlus2). Galois-based candidates
(graph-10..16) need runtime-library extraction and remain sources-only
for now.
