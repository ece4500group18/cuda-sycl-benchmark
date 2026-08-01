# optimize-access-pattern

**Class:** Cache  ·  **Priority:** Medium  ·  **Impact:** 10–40%

## When to apply
Cache **hit rate is good** but achieved bandwidth is low because the data layout is GPU-unfriendly —
typically **Array-of-Structs (AoS)** where each work-item touches one field, so each cache line is mostly
wasted (only the used field is consumed per line). Converting to **Struct-of-Arrays (SoA)** makes each
line fully useful and lanes contiguous.

## Transformation
Store each field in its own contiguous array (SoA) instead of interleaved structs (AoS). Lanes then read
stride-1 within a field.

Before — AoS: `x` reads stride by `sizeof(Particle)`:
```cpp
struct Particle { float x, y, z, w; };
q.parallel_for(sycl::nd_range<1>(g, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n) out[i] = p[i].x * 2.f;      // line holds x,y,z,w; 3/4 wasted
});
```

After — SoA: `x` is its own contiguous array:
```cpp
// px, py, pz, pw are separate float* arrays
q.parallel_for(sycl::nd_range<1>(g, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n) out[i] = px[i] * 2.f;       // contiguous, full-line utilization
});
```

## Correctness invariants
- The AoS→SoA conversion must be applied **consistently** everywhere the data is produced and consumed —
  a half-converted layout is a correctness bug.
- Field values and their association with element `i` are unchanged; only storage layout moves.
- If conversion happens at a boundary, do it once (not per kernel) and account for its cost.

## Verify it took effect
- Effective bandwidth rises at the same hit rate; useful bytes per cache line increase.
- Memory transactions become contiguous/coalesced in the profile.

## Pitfalls / conflicts
- **Conflicts:** none.
- **Synergizes:** `coalesce-memory-access` (SoA makes coalescing natural), `tile-data-access`,
  `vectorize-vec4`.
- This is an **enabling transform** (Phase 3): the layout conversion has upfront cost and only pays once
  a downstream strategy (coalesce/vectorize) exploits the contiguous fields — allow a bounded regression
  to reach it.
