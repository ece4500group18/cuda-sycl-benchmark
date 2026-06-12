# Source snapshots

One directory per candidate id from `../candidates.csv` (only candidates
with `final_decision != exclude`). Each directory is a minimal snapshot
of the upstream CUDA source plus a `SOURCE.txt` recording:

    upstream repo URL + commit, subpath, license, retrieval date, notes

Excluded from snapshots: `.git` metadata, `doc/`, IDE configs, large
output dumps. Input datasets are kept only when small and shipped
upstream (e.g. `graph-05/internet.egr`); otherwise `SOURCE.txt` gives
the download location.

Shared pieces:

- `_deps/galois-libgpu/` — the Galois gg/IrGL GPU runtime headers that
  all `graph-1x` Lonestar cases compile against. They additionally need
  cub/moderngpu (Galois `external/` submodules, not vendored here).
- `_licenses/` — full upstream license texts, one per source family,
  referenced from each `SOURCE.txt`.

Regenerating or extending a snapshot: clone the upstream repo at the
commit in `SOURCE.txt` (sparse checkout of the recorded path suffices),
copy the directory, and update `SOURCE.txt`.
