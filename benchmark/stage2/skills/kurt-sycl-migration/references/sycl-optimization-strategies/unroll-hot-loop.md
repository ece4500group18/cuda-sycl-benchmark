# unroll-hot-loop

**Class:** Compute-bound  ·  **Priority:** High  ·  **Impact:** 10–25%

## When to apply
`PipeStall`/`DistStall` is dominant (ALU dependency / register-distance stalls) on a **hot inner loop
with a trip count ≥ 8**. Unrolling exposes independent work between iterations and cuts loop-overhead
instructions, letting the scheduler fill dependency gaps.

## Transformation
Apply `#pragma unroll` with an explicit factor on the hot loop only. Prefer a factor that divides the
trip count; keep a remainder path if it does not.

Before — tight dependent loop:
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i >= n) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)          // K >= 8, ALU-bound, PipeStall
        acc += coef[k] * src[i * K + k];
    dst[i] = acc;
});
```

After — unroll by 4 (independent partials break the dependency chain):
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i >= n) return;
    float acc = 0.f;
#pragma unroll 4
    for (int k = 0; k < K; ++k)
        acc += coef[k] * src[i * K + k];
    dst[i] = acc;
});
```

## Correctness invariants
- Only unroll when the **trip count ≥ 8**; below that the overhead outweighs the benefit (see
  anti-patterns).
- Preserve accumulation order if bit-exactness is required; if you split into multiple partial sums,
  document the (usually harmless) reassociation.
- Keep the tail guard; ensure a non-divisible trip count is still fully covered.

## Verify it took effect
- `PipeStall`/`DistStall` share drops; the loop body in the IGC asm shows replicated iterations with
  interleaved independent ops.
- Watch `InstrFetchStall` — over-unrolling bloats the icache footprint.

## Pitfalls / conflicts
- **Conflicts:** `reduce-register-pressure` (unrolling raises live values), and an `InstrFetchStall`
  risk from code bloat — if either appears, lower the factor.
- **Synergizes:** `increase-ilp`, `native-math`.
- **Anti-pattern reminder:** unrolling loops with a small trip count (< 8) regressed −11 to −32% on
  Intel Xe — do not do it.
