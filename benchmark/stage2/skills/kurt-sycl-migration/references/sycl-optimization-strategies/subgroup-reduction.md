# subgroup-reduction

**Class:** Synchronization  ·  **Priority:** Medium  ·  **Impact:** 10–30% (profile vs shuffle)

## When to apply
A work-group reduction is used where the reduction scope is actually **sub-group local**, or the final
combine spans only a handful of sub-group partials. Sub-group reductions run in registers with no SLM
and no `group_barrier`, avoiding the whole-group synchronization.

## Transformation
Replace the SLM-tree reduction with `sycl::reduce_over_group(sub_group, ...)`; if the result must span
the work-group, reduce per sub-group then combine the few partials.

Before — SLM + barriers:
```cpp
s[lid] = partial;
sycl::group_barrier(grp);
for (int off = wg / 2; off > 0; off >>= 1) {
    if (lid < off) s[lid] += s[lid + off];
    sycl::group_barrier(grp);
}
float total = s[0];
```

After — sub-group reduce (no barrier, no SLM):
```cpp
auto sg = it.get_sub_group();
float sgsum = sycl::reduce_over_group(sg, partial, sycl::plus<float>());
// if a single work-group total is needed, write one partial per sub-group and combine:
if (sg.get_local_id() == 0) s[sg.get_group_id()] = sgsum;
sycl::group_barrier(grp);                            // one barrier for the cross-sub-group combine
float total = (lid == 0) ? combine(s, nsg) : 0.f;
```

## Correctness invariants
- All lanes of a sub-group must participate in `reduce_over_group` (no divergent early-exit around it).
- Float addition reassociates — validate the reduced result within tolerance against the baseline.
- If the true scope is the whole work-group, keep the single cross-sub-group combine step.

## Verify it took effect
- `SyncStall` and barrier count drop; SLM usage for the reduction disappears.
- The reduction shows up as register shuffles in the IGC asm, not SLM traffic.

## Pitfalls / conflicts
- **Conflicts:** the `reduce_over_group` anti-pattern — on Intel a manual XOR butterfly shuffle can beat
  the built-in; **profile both** when the reduction is hot.
- **Synergizes:** `reduce-barriers`, `float2-accumulation` (reduce paired stats at once).
- Sub-group width is 16/32 on Xe — a reduction wider than one sub-group still needs the combine step.
