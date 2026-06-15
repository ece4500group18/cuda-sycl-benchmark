# molecular dynamics / simulation collection

Owner: liqui

See `../README.md` for the shared workflow, CSV columns, SOURCE.txt
format, and snapshot rules. This file holds the coverage matrix and the
category-specific notes. Snapshots of every non-excluded candidate are in
`sources/<id>/`.

## Sources and licenses (verified 2026-06-11)

| Source | License | Notes |
|---|---|---|
| HeCBench (zjin-lcf/HeCBench) | BSD-3-Clause | Official SYCL port per case (contamination caveat, see graph README). Per-case provenance (SHOC, HACC, NVIDIA ising-gpu, Intel motionsim) in each `SOURCE.txt`. |
| NVIDIA cuda-samples | BSD-style (NVIDIA) | Clean, idiomatic CUDA. Repo restructured 2025: samples live under `cpp/<tier>/`. OpenGL renderers stripped during adaptation. |
| Rodinia (yuhc/gpu-rodinia mirror) | BSD-style (UVA) | lavaMD; large output dumps excluded from snapshot. |
| CoMD-CUDA (NVIDIA/CoMD-CUDA) | BSD-style (LANL+NVIDIA) | Full MD mini-app; richest CUDA feature set of all candidates (shfl, ballot, streams, async copies, vendored cub). License ships inside `sources/md-09/LICENSE.md`. |
| miniMD (Mantevo) | LGPL-3 | **Excluded**: copyleft + no plain-CUDA variant (Kokkos/OpenMP-target only). |

## Coverage matrix

Dimensions:

- **Neigh**: interaction handling — `all-pairs`, `cell-list`,
  `verlet-list`
- **Force**: `LJ`, `EAM`, `gravity`, `SPH`, `spin-MC` (Metropolis)
- **Pipe**: `single-k` vs `multi-k` (multi-kernel + host time loop)
- **Red**: energy/virial reduction present
- **RNG**: `none`, `hash` (inline LCG/hash), `host-pregen`, `curand`
- **Layout**: `AoS`, `SoA`, `vec` (float4 etc.)

Matrix reflects desk review (2026-06-11). Corrections vs registration: sph
has NO atomics and is double-precision throughout; particle-diffusion RNG
is host-pregenerated (kernel deterministic); fdtd3d turned out to be the
cuda-samples FDTD3d port (pure stencil → proposed transfer); mcmd excluded
(16.8K-LOC full application).

| Candidate | Neigh | Force | Pipe | Red | RNG | Layout | Notable |
|---|---|---|---|---|---|---|---|
| md-01 cuda-samples/nbody | all-pairs | gravity | multi-k | — | none | vec (float4) | precision templates (51), __constant__ |
| md-02 rodinia/lavaMD | cell-list | LJ-like | single-k | — | none | AoS+vec | — |
| md-03 hecbench/md (SHOC) | verlet-list | LJ | single-k | yes | none | vec | precision templates |
| md-04 hecbench/haccmk | all-pairs (cutoff) | poly-fit | single-k | yes | none | SoA | clean A-tier |
| md-05 hecbench/ising | lattice | spin-MC | multi-k | yes | **curand** | SoA | library-RNG cell |
| md-06 hecbench/sph | cell-list | SPH | multi-k (4 kernels) | — | none | AoS, double | no built-in verify — oracle to design |
| md-07 cuda-samples/particles | cell-list (thrust sort) | DEM springs | multi-k | — | none | vec | thrust→oneDPL crossover |
| md-08 hecbench/particle-diffusion | — | random walk | single-k | yes | host-pregen | SoA | deterministic kernel, exact-match oracle |
| md-09 CoMD-CUDA | cell-list | LJ+EAM | multi-k | yes | none | SoA | shfl×48, ballot, streams×39, cub |

Notes:
- `md-07 particles` uses thrust sort (library-usage crossover — coordinate
  with the CUDA-library category owner).
- `md-10 fdtd3d` is snapshotted here but marked `transfer-to-stencil` in
  `candidates.csv`: it is the cuda-samples FDTD3d port = pure stencil.
  Hand off to the stencil category owner.
- `bh` overlaps graph and n-body; it is counted under graph (graph-09).
- Float-atomic force accumulation is a dark cell (sph doesn't have it) —
  CoMD's force kernels may cover it; re-check during build round.

Excluded (in `candidates.csv`, no snapshot): md-11 miniMD (LGPL +
no CUDA variant), md-12 mcmd (16.8K-LOC full application).

## Dedup policy

Same algorithm from multiple suites: include more than one variant only
when they cover different matrix cells; otherwise prefer the smaller,
cleaner source. (e.g. lavaMD exists in both Rodinia and HeCBench — keep
one.)
