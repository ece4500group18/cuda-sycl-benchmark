# pad-slm-arrays

**Class:** Cache  ·  **Priority:** High  ·  **Impact:** 10–25%  ·  **Arch:** BMG / heuristic on CRI

## When to apply
SLM **bank-conflict rate** is high (`MemoryProfile` on BMG; heuristic on CRI) — typically a 2D SLM array
whose row length is a multiple of the bank count, so lanes accessing the same column all map to the same
bank and serialize. Classic with power-of-two tile widths and column/transpose access.

## Transformation
**Pad the leading dimension by one element** so consecutive rows land in different banks, spreading
column accesses across banks.

Before — `[TILE][TILE]`, column access all-conflict:
```cpp
sycl::local_accessor<float, 2> t({TILE, TILE}, h);      // stride = TILE (bank-aligned)
// ...
t[lr][lc] = g[...];
sycl::group_barrier(grp);
float v = t[lc][lr];                                     // transpose read → bank conflicts
```

After — pad the row by 1 (`[TILE][TILE+1]`):
```cpp
sycl::local_accessor<float, 2> t({TILE, TILE + 1}, h);  // stride = TILE+1 (bank-offset)
// ...
t[lr][lc] = g[...];
sycl::group_barrier(grp);
float v = t[lc][lr];                                     // now conflict-free
```

## Correctness invariants
- Padding changes only the **storage stride**, not the logical shape — indexing stays `[i][j]`; never
  read the padding column as data.
- SLM footprint grows by one column per row — recheck the SLM/occupancy budget (see
  `reduce-slm-allocation`).
- Results are identical; this is a layout-only edit.

## Verify it took effect
- SLM bank-conflict rate drops on the re-profile; SLM access latency for the conflicting phase falls.
- No change in DRAM traffic or occupancy beyond the small SLM increase.

## Pitfalls / conflicts
- **Conflicts:** `reduce-slm-allocation` — padding *adds* SLM and can push a group over the next
  allocation bucket, cutting residency; balance the two.
- **Synergizes:** `slm-cache-reuse`, `tile-data-access`.
- Padding by 1 is the usual fix; if conflicts persist, the access pattern (not the stride) may be the
  issue — see `optimize-access-pattern`.
