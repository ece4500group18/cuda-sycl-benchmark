# Benchmark cases

The unified, runnable Stage 1 dataset. One folder per collection
category, one folder per case:

```
benchmark/cases/<category>/<case>/
├── original/main.cu (+ CMakeLists.txt, README.md)
├── tests/verify.py          <- resolves ../../../../tools/verify_lib.py,
│                               i.e. benchmark/tools/verify_lib.py
├── metadata.json            <- includes category, domain, difficulty
├── metadata.stage1.json     <- strict Stage 1 sidecar (generated)
└── logs/                    <- committed build/run/verify/perf results
```

## Categories

The ten categories are the collection categories defined in
`../collection/README.md`, keyed by the dominant CUDA/algorithmic
pattern of the kernel. `domain` (application area) and `difficulty` are
orthogonal axes kept in metadata, not in the directory tree.

| Category | Cases |
|---|---|
| stencil-convolution | 73 |
| simple-kernels | 56 |
| reductions-scans | 34 |
| memory-movement | 21 |
| graph-irregular | 18 |
| linear-algebra | 14 |
| cuda-library-usage | 14 |
| streams-atomics-templates | 12 |
| molecular-dynamics | 12 |
| multi-kernel-pipelines | 2 |
| **Total** | **256** |

2026-07-03: six real-extraction MD cases were adapted from the
molecular-dynamics collection snapshots (md-03..md-08): SHOC LJ force,
HACC short-range force, Ising Metropolis, SPH pipeline, cuda-samples
particles collision, motionsim random walk. Their kernels are upstream
device code verbatim; each passes `verify.py --selftest`. CUDA
build/run/perf validation on the team GPU machine is pending
(status `verify_ready`, not `perf_ready`).

## Provenance

This tree consolidates (2026-07-03) the two previous case locations:

- `pilot_benchmark/cases/<group>/<case>` (216 cases; the legacy groups
  `ai/easy/hpc/library_api/medium` mixed domain and difficulty axes and
  were retired),
- `benchmark/collection/stencil-convolution/cases/<case>` (34 cases).

Every case was re-categorized by its dominant kernel pattern; the
per-case mapping is recorded in the git rename history of the
consolidation commit. Case contents (sources, inputs, verifiers, logs)
are unchanged; `metadata.json` gained/updated `category`, `domain`, and
`difficulty` fields so classification no longer depends on directory
names. All 250 cases pass strict Stage 1 metadata validation and
`verify.py --selftest` after the move.

The per-category counts also expose current collection gaps (e.g.
multi-kernel-pipelines and molecular-dynamics are thin) — useful input
for the coverage-driven collection phase.
