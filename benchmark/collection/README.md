# Benchmark Collection: graph/irregular + MD/simulation

Collection workspace for the two categories owned by liqui. Strategy:
coverage-matrix-driven, not count-driven. We stop collecting when new
candidates no longer light up new cells in the matrices below.

Candidates are registered in `candidates.csv` (one row per candidate;
registration != inclusion). Workflow:

1. Round 1 (desk review): fill id/repo/url/license/name/domain/features/
   difficulty by reading code. No building.
2. Round 2 (validation): for shortlisted rows only, fill build/run status
   and design the correctness oracle. Requires an nvcc machine (local
   machine has none).
3. Final: mark include/exclude; included cases get adapted to the
   `pilot_benchmark/` case format (deterministic inputs, verify.py,
   metadata.json).

## Sources and licenses (verified 2026-06-11)

| Source | License | Notes |
|---|---|---|
| HeCBench (zjin-lcf/HeCBench) | BSD-3-Clause | Each case also ships an official SYCL port — useful as oracle, but a training-data contamination risk if used to evaluate LLM-based migration. Team decision pending. Desk review confirmed nearly all shortlisted cases have built-in CPU verification (`reference.h`); provenance per case recorded in CSV (ECL suite, Chai, cuGraph, SHOC, ising-gpu, motionsim). |
| Rodinia (rodinia.cs.virginia.edu) | BSD-style (UVA) | Permissive; some sub-apps carry their own licenses — check per app. Datasets are a separate download. |
| Pannotia (pannotia/pannotia) | BSD-style (AMD) | **Excluded as direct source** (desk review 2026-06-11): upstream kernels are OpenCL `.cl`, not CUDA. CUDA equivalents come via HeCBench ports. |
| Galois / LonestarGPU (IntelligentSoftwareSystems/Galois) | BSD-3 (UT Austin) | Worklist-based irregular apps; hardest tier. GPU apps depend on the in-repo `libgpu` (gg/IrGL) runtime + cub — extracting a case means vendoring those headers. |
| NVIDIA cuda-samples | BSD-style (NVIDIA) | Clean kernels, idiomatic CUDA. Repo restructured 2025: samples live under `cpp/<tier>/`. |
| CoMD-CUDA (NVIDIA/CoMD-CUDA) | BSD-style (LANL+NVIDIA) | Full MD mini-app; richest CUDA feature set of all candidates (shfl, ballot, streams, async copies, vendored cub). |
| Gunrock | Apache-2.0 | Template-heavy framework; whole-app migration likely out of scope. |
| miniMD (Mantevo) | LGPL-3 | **Excluded**: copyleft + no plain-CUDA variant (Kokkos/OpenMP-target only). |

Local working copies for desk review live in `sources/` (gitignored,
sparse checkouts).

## Coverage matrix: graph / irregular access

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

Matrix below reflects desk review (2026-06-11), not just registration
guesses. `RNG` column added after finding curand in independentset.

| Candidate | Rep | Par | Sync | LB | Mem | RNG |
|---|---|---|---|---|---|---|
| rodinia/bfs | CSR | topo | flag | tpv | — | — |
| hecbench/sssp (Chai) | CSR | wl | atomic(min,add,CAS) | tpv | smem | — |
| hecbench/page-rank | CSR | topo | flag | tpv | — | — |
| hecbench/mis (ECL-MIS) | CSR | topo | flag | tpv | — | hash priorities |
| hecbench/cc (ECL-CC) | CSR/edge | edge | atomic(CAS,add)+warp(shfl) | tpv | smem | — |
| hecbench/jaccard (cuGraph) | CSR | edge | atomic(add)+warp(shfl) | coop | — | — |
| hecbench/floydwarshall | dense-adj | — | — | tpv | — (fw2 variant: smem tiled) | — |
| hecbench/bh (ECL-BH) | tree | wl | atomic(CAS)+warp(ballot) | coop | smem | — |
| lonestar/bfs-wl | CSR | wl | warp(ballot) | coop | smem queue | — |
| lonestar/sssp-wl | CSR | wl | atomic(min)+warp(ballot) | coop | smem queue | — |
| lonestar/spanningtree | CSR | wl | flag | coop | smem | — |
| lonestar/triangle-counting | CSR | edge | atomic(add) | coop | smem | — |
| lonestar/independentset | CSR | topo | flag | tpv | — | curand |
| lonestar/dmr | mesh | wl | atomic(add,min) | coop | dynalloc | — |

Cells still dark after this set: dynamic parallelism (`dp`) — only if a
clean candidate shows up; not worth forcing. **Algorithm gaps**: graph
coloring (Pannotia color is OpenCL-only; ECL-GC is a CUDA alternative,
license unverified) and betweenness centrality (Pannotia bc also
OpenCL-only) — acceptable gaps unless a clean candidate appears.

## Coverage matrix: MD / simulation

- **Neigh**: interaction handling — `all-pairs`, `cell-list`,
  `verlet-list`
- **Force**: `LJ`, `EAM`, `gravity`, `SPH`, `spin-MC` (Metropolis)
- **Pipe**: `single-k` vs `multi-k` (multi-kernel + host time loop)
- **Red**: energy/virial reduction present
- **RNG**: `none`, `hash` (inline LCG/hash), `curand`
- **Layout**: `AoS`, `SoA`, `vec` (float4 etc.)
- **FPatomic**: atomic accumulation on floating point

Matrix below reflects desk review (2026-06-11). Corrections vs
registration: sph has NO atomics and is double-precision throughout;
particle-diffusion RNG is host-pregenerated (kernel deterministic);
fdtd3d turned out to be the cuda-samples FDTD3d port (pure stencil →
proposed transfer); mcmd excluded (16.8K-LOC full application).

| Candidate | Neigh | Force | Pipe | Red | RNG | Layout | Notable |
|---|---|---|---|---|---|---|---|
| cuda-samples/nbody | all-pairs | gravity | multi-k | — | none | vec (float4) | precision templates (51), __constant__ |
| rodinia/lavaMD | cell-list | LJ-like | single-k | — | none | AoS+vec | — |
| hecbench/md (SHOC) | verlet-list | LJ | single-k | yes | none | vec | precision templates |
| hecbench/haccmk | all-pairs (cutoff) | poly-fit | single-k | yes | none | SoA | clean A-tier |
| hecbench/ising | lattice | spin-MC | multi-k | yes | **curand** | SoA | library-RNG cell |
| hecbench/sph | cell-list | SPH | multi-k (4 kernels) | — | none | AoS, double | no built-in verify — oracle to design |
| cuda-samples/particles | cell-list (thrust sort) | DEM springs | multi-k | — | none | vec | thrust→oneDPL crossover |
| hecbench/particle-diffusion | — | random walk | single-k | yes | host-pregen | SoA | deterministic kernel, exact-match oracle |
| CoMD-CUDA | cell-list | LJ+EAM | multi-k | yes | none | SoA | shfl×48, ballot, streams×39, cub |

Notes: `particles` uses thrust sort (library-usage crossover — coordinate
with the CUDA-library category owner). `fdtd3d` moved to
`transfer-to-stencil` in the CSV. `bh` overlaps graph and n-body; counted
under graph. Float-atomic force accumulation is now a dark cell (sph
doesn't have it) — CoMD's force kernels may cover it; re-check during
build round.

## Dedup policy

Same algorithm from multiple suites (e.g. bfs in Rodinia, HeCBench,
Lonestar): include at most two variants and only if they cover different
matrix cells (e.g. rodinia/bfs is topo+flag, lonestar/bfs is
wl+atomic+smem-queue — both stay). Otherwise prefer the smaller,
cleaner source.
