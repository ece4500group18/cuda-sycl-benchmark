# float2-accumulation

**Class:** Memory-bandwidth  ·  **Priority:** Medium  ·  **Impact:** 1.2–1.5×

## When to apply
A normalization/reduction kernel makes **two passes** over the same data to compute two related
statistics — classically `sum` and `sum-of-squares` for mean/variance (LayerNorm, RMSNorm, BatchNorm).
Each pass streams the tensor from DRAM again.

## Transformation
Compute both accumulators in a **single pass**, packing them in a `sycl::float2` so the data is read
once. `x.x()` accumulates `Σv`, `x.y()` accumulates `Σv²`.

Before — two reads of `row`:
```cpp
// pass 1: mean
float s = 0.f;
for (int j = lid; j < D; j += wg) s += row[j];
// ... group-reduce s, barrier ...
// pass 2: variance (re-reads row from DRAM)
float sq = 0.f;
for (int j = lid; j < D; j += wg) { float d = row[j] - mean; sq += d * d; }
```

After — one read, paired accumulators:
```cpp
sycl::float2 acc{0.f, 0.f};                       // {Σv, Σv²}
for (int j = lid; j < D; j += wg) {
    float v = row[j];                             // single load
    acc.x() += v;
    acc.y() += v * v;
}
sycl::float2 tot = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<sycl::float2>());
float mean = tot.x() / D;
float var  = tot.y() / D - mean * mean;           // E[x²] − E[x]²
```

## Correctness invariants
- `var = E[x²] − mean²` is more sensitive to cancellation than the two-pass form; keep accumulators in
  **FP32** (or FP64 for very large `D` / high dynamic range) and validate variance against the two-pass
  baseline.
- The group reduction must sum **both** components — use a `float2` plus-reduction, not two scalar ones
  unless separately reduced.

## Verify it took effect
- DRAM reads of the normalized tensor drop ~2× (`MemoryProfile`); one fewer barrier phase.
- Correctness gate on mean/variance passes within tolerance.

## Pitfalls / conflicts
- **Conflicts:** none.
- **Synergizes:** `fuse-passes` (fold the normalize-apply step in too), `subgroup-reduction`.
- If the one-pass variance loses too much precision on your data, fall back to two-pass — measure the
  accuracy, not just the speed.
