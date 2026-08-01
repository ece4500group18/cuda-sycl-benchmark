# coalesce-memory-access

**Class:** Memory-bandwidth  ·  **Priority:** High  ·  **Impact:** 2–8× on bad strides

## When to apply
High DRAM traffic (`MemoryProfile` read/write bytes) but **low effective bandwidth** vs device peak,
because neighboring lanes hit **strided** addresses. Each sub-group memory message is fragmented into
many partial transactions instead of one contiguous burst.

## Transformation
Re-map the work so **adjacent work-items touch adjacent addresses** (stride-1 within a sub-group). Often
this means iterating the contiguous dimension across lanes and the strided dimension in a loop —
frequently just swapping which index the global id drives.

Before — each item owns a whole block; lanes stride by `block_size`:
```cpp
q.parallel_for(sycl::nd_range<1>(gBlocks, wg), [=](sycl::nd_item<1> it) {
    const int b = it.get_global_id(0);
    if (b >= blocks) return;
    float sum = 0.f;
    for (int i = 0; i < block_size; ++i)
        sum += in[b * block_size + i];   // lane b and b+1 are block_size apart → uncoalesced
    out[b] = sum;
});
```

After — lanes stride the contiguous axis; the block index is the loop:
```cpp
q.parallel_for(sycl::nd_range<2>({blocksP, (size_t)wgX}, {1, (size_t)wgX}),
    [=](sycl::nd_item<2> it) {
    const int b   = it.get_global_id(0);
    const int lid = it.get_local_id(1);
    if (b >= blocks) return;
    float partial = 0.f;
    for (int i = lid; i < block_size; i += wgX)   // lane i, i+1 → addresses i, i+1 (coalesced)
        partial += in[b * block_size + i];
    // sub-group / work-group reduce `partial` into out[b] (see subgroup-reduction)
    float sum = sycl::reduce_over_group(it.get_group(), partial, sycl::plus<float>());
    if (lid == 0) out[b] = sum;
});
```

## Correctness invariants
- The reassignment must cover every element exactly once — check the strided loop bounds and the tail.
- If you add a group reduction, only one lane writes the output.
- Result value is unchanged; only the iteration→lane mapping moves.

## Verify it took effect
- Effective bandwidth rises toward peak at the same byte count; `SendStall` drops.
- Memory transaction size increases (fewer, wider LSC messages in the profile).

## Pitfalls / conflicts
- **Conflicts:** `uniform-control-flow` if re-bucketing lanes to fix divergence re-introduces strides —
  re-profile after either.
- **Synergizes:** `vectorize-vec4` (coalesce first, then widen), `optimize-access-pattern` (AoS→SoA
  makes the contiguous axis lane-friendly).
- **Anti-pattern reminder:** avoid 1D `nd_range` with manual index decomposition for multi-D data — map
  dimensions directly with `nd_range<2/3>`.
