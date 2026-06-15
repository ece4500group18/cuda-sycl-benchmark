# <Category name> collection

Owner: <your name>

Copy of the category template. See `../README.md` for the shared workflow,
CSV columns, SOURCE.txt format, and snapshot rules. Fill in the coverage
matrix below, then register candidates in `candidates.csv`.

## Coverage matrix

Define the dimensions this category should span: pick the axes that
distinguish how kernels in this domain are written (data layout, sync
primitives, memory features, library calls, etc.). A candidate "covers" a
cell if its kernels exercise it. Collection stops when new candidates stop
covering new cells.

| Candidate | <dim 1> | <dim 2> | <dim 3> | ... |
|---|---|---|---|---|
| <id / name> | | | | |

List dark cells / known gaps explicitly, so the team can see what is still
missing and decide whether a gap is acceptable.

## Dedup policy

Same algorithm from multiple suites: include more than one variant only
when they cover different matrix cells; otherwise prefer the smaller,
cleaner source.
