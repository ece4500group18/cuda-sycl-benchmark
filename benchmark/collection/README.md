**English** | [中文](README.zh-CN.md)

# Collection workspace

This is the **collection-phase working area** for the CUDA-to-SYCL
benchmark. It holds the candidate registries and CUDA source snapshots we
gather before adapting selected cases into the final case-unit format
(`original/main.cu` + `CMakeLists` + deterministic inputs +
`tests/verify.py` + `metadata.json`). Adapted (runnable) cases do **not**
live here: they go to `benchmark/cases/<category-slug>/<case>/` and use
the shared `benchmark/tools/verify_lib.py`.

**One folder per category. One owner per category.** Each member works
inside their own category folder, so registries never collide.
Registration in a `candidates.csv` is *not* a decision to include a case —
the `final_decision` column is.

## Layout

```
collection/
├── README.md                 <- this file: the shared contribution guide
├── _TEMPLATE/                <- copy this to start a new category
│   ├── README.md
│   └── candidates.csv
└── <category-slug>/          <- one per category (see table below)
    ├── README.md             <- owner + coverage matrix + gaps for this category
    ├── candidates.csv         <- this category's registry (columns defined below)
    └── sources/
        ├── <id>/             <- one snapshot per non-excluded candidate
        │   ├── <upstream CUDA source, trimmed>
        │   └── SOURCE.txt    <- provenance (format below)
        ├── _deps/            <- shared headers a case compiles against (optional)
        └── _licenses/        <- upstream license texts, referenced by SOURCE.txt
```

## How to add your category

1. `cp -r _TEMPLATE <your-category-slug>` (use the slug from the table below).
2. Edit `<slug>/README.md`: put your name as owner and define your
   coverage matrix (the dimensions your category should span). Collection
   is **coverage-driven, not count-driven** — you stop when new candidates
   stop lighting up new matrix cells, not at a fixed number.
3. Register every candidate as a row in `<slug>/candidates.csv` using the
   columns below. IDs are `<slug-prefix>-NN` (e.g. `graph-01`, `md-01`).
4. For each candidate whose `final_decision` is not `exclude`, snapshot
   its CUDA source under `<slug>/sources/<id>/` and add a `SOURCE.txt`
   (format below).
5. Open a PR. A teammate reviews format compliance before merge.

## candidates.csv columns

`id, source_repo, source_url, license, kernel_application_name, domain,
cuda_features_used, estimated_difficulty, build_status, run_status,
correctness_oracle, input_size, reason_selected, migration_notes,
final_decision`

- `estimated_difficulty`: A (straightforward) / B / C (hardest).
- `build_status` / `run_status`: `not_attempted` until validated on a
  machine with the toolchain (see note below); then `ok` / `fail:<why>` /
  `skipped`.
- `correctness_oracle`: how a migrated version is checked (built-in
  reference, CPU recompute, checksum, tolerance/statistical check). A case
  with no designable oracle should be excluded.
- `final_decision`: `candidate` / `exclude` / or a routing note such as
  `transfer-to-stencil` when a case belongs to another category.

## SOURCE.txt format

One per snapshot, recording exactly where the code came from so anyone can
re-fetch or extend it:

```
id:        graph-01
upstream:  <repo URL> @ <short commit sha>
path:      <subpath within the upstream repo>
license:   <name> (full text: ../_licenses/<source>-LICENSE.txt)
retrieved: YYYY-MM-DD
notes:     <inputs location, what was stripped, build caveats>
```

## Snapshot rules

- Pin the upstream **commit** in `SOURCE.txt`; sparse-checkout of the
  recorded `path` is enough to reproduce a snapshot.
- Keep it minimal: strip `.git`, `doc/`, IDE configs, and large output
  dumps. Keep small upstream-shipped inputs (e.g. a sample graph);
  otherwise record the download URL in `notes`.
- Put the upstream license text once in `sources/_licenses/` and reference
  it from each `SOURCE.txt`. Preserve any per-case `LICENSE` file that
  ships inside the source.
- Headers shared by several cases (e.g. a runtime library) go in
  `sources/_deps/` and are referenced from the cases' `SOURCE.txt`.

## Validation note

Build/run validation needs an nvcc + SYCL toolchain. Members without a
local toolchain leave `build_status`/`run_status` as `not_attempted` and
validate on the team's designated GPU machine. Desk review (sources,
licenses, features, difficulty, oracle plan) needs no GPU and comes first.

## Categories and owners

| Slug | Category | Owner |
|---|---|---|
| `simple-kernels` | simple-but-not-trivial kernels | yuepan |
| `memory-movement` | memory movement & layout | yuepan |
| `stencil-convolution` | stencil / convolution / image processing | Zijian |
| `reductions-scans` | reductions and scans | Zijian |
| `graph-irregular` | graph / irregular access | liqui |
| `molecular-dynamics` | molecular dynamics / simulation | liqui |
| `linear-algebra` | linear algebra | TBD |
| `multi-kernel-pipelines` | multi-kernel pipelines | TBD |
| `cuda-library-usage` | CUDA library usage | TBD |
| `streams-atomics-templates` | streams, events, shared memory, atomics, templates, macros | TBD |

Cross-category cases (e.g. a simulation case that is really a stencil) are
assigned to exactly one owner; note the hand-off in `final_decision` and
raise it at the weekly sync.
