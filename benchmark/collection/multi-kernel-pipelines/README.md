# multi-kernel pipelines collection

Owner: TBD

See `../README.md` for the shared workflow, CSV columns, SOURCE.txt
format, and snapshot rules. Fill in the coverage matrix below, then
register candidates in `candidates.csv`. Adapted (runnable) cases go to
`benchmark/cases/<this category's slug>/<case>/` (not here) and use the
shared `benchmark/tools/verify_lib.py`.

## Coverage matrix

Define the dimensions this category should span (the axes that distinguish
how kernels in this domain are written). A candidate "covers" a cell if
its kernels exercise it. Collection is coverage-driven, not count-driven:
stop when new candidates stop covering new cells.

| Candidate | <dim 1> | <dim 2> | <dim 3> | ... |
|---|---|---|---|---|
| <id / name> | | | | |

List dark cells / known gaps explicitly.

## Dedup policy

Same algorithm from multiple suites: include more than one variant only
when they cover different matrix cells; otherwise prefer the smaller,
cleaner source.
