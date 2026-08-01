# fuse-passes

**Class:** Memory-bandwidth  ·  **Priority:** High  ·  **Impact:** 1.5–3×

## When to apply
The kernel is memory-bound and **multiple kernels re-read the same arrays** — a pipeline of element-wise
/ map passes each streaming the full tensor from DRAM. Every extra pass pays the full memory round-trip
for the same data.

## Transformation
Merge the passes into **one kernel** so each element is loaded once, transformed through the whole chain
in registers, and stored once. Collapses `read→write→read→write` into `read→compute→write`.

Before — three passes, three full DRAM round-trips:
```cpp
q.parallel_for(sycl::nd_range<1>(g, wg), [=](sycl::nd_item<1> it){ int i=it.get_global_id(0); if(i<n) t1[i]=x[i]+bias[i]; });
q.parallel_for(sycl::nd_range<1>(g, wg), [=](sycl::nd_item<1> it){ int i=it.get_global_id(0); if(i<n) t2[i]=sycl::max(t1[i],0.f); });
q.parallel_for(sycl::nd_range<1>(g, wg), [=](sycl::nd_item<1> it){ int i=it.get_global_id(0); if(i<n) y[i]=t2[i]*scale; });
```

After — one kernel, one load + one store per element:
```cpp
q.parallel_for(sycl::nd_range<1>(g, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n) {
        float v = x[i] + bias[i];        // pass 1
        v = sycl::max(v, 0.f);           // pass 2  (stays in registers)
        y[i] = v * scale;                // pass 3
    }
});
```

## Correctness invariants
- Fusion is valid only when passes are **element-local** (no cross-element dependency between them); a
  reduction or transpose in the middle breaks it.
- The intermediate tensors (`t1`, `t2`) disappear — remove their allocations and any other readers.
- Result is bit-identical when the math order per element is preserved.

## Verify it took effect
- DRAM read+write bytes (`MemoryProfile`) drop ~proportionally to passes removed; kernel-launch count
  falls.
- End-to-end wall time drops even if per-kernel occupancy is unchanged.

## Pitfalls / conflicts
- **Conflicts:** none — but do not fuse across a needed synchronization boundary (e.g. a global
  reduction) or you change semantics.
- **Synergizes:** `float2-accumulation` (fuse the sum + sum-of-squares passes of a normalization),
  `multi-output-per-item`.
- Fusing too many passes can raise register pressure — if occupancy drops, split the chain.
