# Benchmark

Unified home of the CUDA-to-SYCL migration benchmark.

```
benchmark/
├── cases/                   <- the runnable dataset: one folder per
│   └── <category>/<case>/      collection category, one per case
├── tools/
│   └── verify_lib.py        <- shared verifier library used by every
│                               case's tests/verify.py (resolved 4 levels
│                               up from tests/, so case depth is fixed)
└── collection/              <- collection-phase workspace: candidate
    └── <category>/             registries (candidates.csv) and raw
                                upstream CUDA snapshots (sources/<id>/).
                                No runnable cases live here.
```

Each case under `cases/<category>/<case>/` is a complete benchmark unit:
`original/main.cu` + `CMakeLists.txt` + deterministic inputs +
`tests/verify.py` + `metadata.json` + `logs/`. `<category>` is one of the
ten collection categories defined in `collection/README.md`; `domain` and
`difficulty` are metadata fields, not directory levels.

Stage 1 tooling (build / run / verify / benchmark / report) lives in the
repository-root `tools/` and discovers cases under `benchmark/cases/`.
See `STAGE1_CUDA_DATASET.md` at the repository root.
