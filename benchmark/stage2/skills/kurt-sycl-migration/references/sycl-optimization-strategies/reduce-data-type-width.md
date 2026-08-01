# reduce-data-type-width

**Class:** Memory-bandwidth  ·  **Priority:** Medium  ·  **Impact:** up to 2× bandwidth

## When to apply
The kernel is **bandwidth-bound** (achieved BW ≥ 70% of peak) and uses **FP32** where **FP16/BF16** is
numerically acceptable — e.g. inference activations, weights, or intermediate buffers that tolerate
reduced precision. Halving the element size halves the bytes moved.

## Transformation
Store the data in the narrower type and compute in FP32 (load → convert → math → convert → store), or
carry the narrow type through when precision allows. Keep the **accumulator in FP32** to bound error.

Before — FP32 storage and math:
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n) dst[i] = a * src[i] + b[i];    // 3× FP32 traffic
});
```

After — BF16 storage, FP32 compute (half the bytes):
```cpp
// src, dst, b are sycl::bfloat16*
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n) {
        float x = float(src[i]);              // widen for math
        float y = float(b[i]);
        dst[i] = sycl::bfloat16(a * x + y);   // store narrow
    }
});
```

## Correctness invariants
- **Accumulate in FP32**; only inputs/outputs are narrowed. Never sum long series in BF16.
- Validate against the FP32 baseline with a **relaxed tolerance** appropriate to the type (BF16 ~1e-2
  relative); document the tolerance in the run record.
- Watch for overflow/underflow of the reduced-range type on extreme inputs.

## Verify it took effect
- DRAM bytes (`MemoryProfile`) drop ~2×; achieved BW headroom improves.
- Correctness gate passes at the relaxed tolerance.

## Pitfalls / conflicts
- **Conflicts:** unsupported XMX input type only — do not narrow to a type the target DPAS path cannot
  consume.
- **Synergizes:** `offload-xmx-library` (narrow types feed XMX at higher throughput), `vectorize-vec4`
  (2× elements per vector), `vectorize-low-bitwidth-stores`.
- This is often an **enabling transform** (Phase 3): it may add convert overhead and only pay off once a
  downstream strategy exploits the narrower data — allow a bounded regression to reach the follow-up.
