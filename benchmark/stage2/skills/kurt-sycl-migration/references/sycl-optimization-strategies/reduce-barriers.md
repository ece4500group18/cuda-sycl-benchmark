# reduce-barriers

**Class:** Synchronization  ·  **Priority:** High  ·  **Impact:** 10–30%

## When to apply
`SyncStall` is > 20% of stall time, or the barrier count is high for the algorithm — often a reduction
or scan written with a barrier on **every** tree step, or barriers guarding data that never crosses
sub-group boundaries. Each `group_barrier` synchronizes the whole work-group and serializes it.

## Transformation
Remove barriers that aren't needed: do the intra-sub-group part with **sub-group primitives** (no
barrier), and barrier only when data actually crosses sub-groups.

Before — barrier per reduction step across the work-group:
```cpp
for (int off = wg / 2; off > 0; off >>= 1) {
    if (lid < off) s[lid] += s[lid + off];
    sycl::group_barrier(grp);          // barrier every step → SyncStall
}
```

After — barrier-free sub-group reduce, then one barrier to combine sub-groups:
```cpp
auto sg = it.get_sub_group();
float v = sycl::reduce_over_group(sg, partial, sycl::plus<float>());  // no barrier
if (sg.get_local_id() == 0) s[sg.get_group_id()] = v;                 // one partial / sub-group
sycl::group_barrier(grp);                                             // single barrier
if (lid == 0) {
    float total = 0.f;
    for (int i = 0; i < it.get_sub_group().get_group_range()[0]; ++i) total += s[i];
    out[grp_id] = total;
}
```

## Correctness invariants
- A barrier may only be removed if the data it guarded **never crosses sub-group boundaries**; keep the
  one barrier that combines sub-group partials.
- All work-items must reach every remaining `group_barrier` (no barrier inside divergent control flow).
- Reduction result must match the baseline (mind float reassociation tolerance).

## Verify it took effect
- `SyncStall` share and executed-barrier count drop in the re-profile.
- The reduction/scan timeline shows fewer whole-group sync points.

## Pitfalls / conflicts
- **Conflicts:** none — but never delete a barrier that guards genuine cross-sub-group sharing (data
  race).
- **Synergizes:** `subgroup-reduction`, `lock-free-atomics`.
- **Anti-pattern reminder:** `reduce_over_group` can lose to a hand-tuned butterfly shuffle on Intel —
  benchmark both when the reduction is hot.
