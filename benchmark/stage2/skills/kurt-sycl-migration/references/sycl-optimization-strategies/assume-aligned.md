# assume-aligned

**Class:** Memory-bandwidth (enabling)  ·  **Priority:** Medium  ·  **Impact:** enables 1.5–4×

## When to apply
A widening strategy (`vectorize-vec4`, `vectorize-low-bitwidth-stores`) was applied but the IGC asm
**still shows narrow scalar messages** — the compiler can't prove the pointer is sufficiently aligned to
emit a wide (128-bit) load/store, so it falls back to scalar or issues a slow unaligned path. The data
*is* aligned; the compiler just doesn't know it.

## Transformation
Communicate the alignment: allocate on a wide boundary (USM `malloc_device` is 16-byte aligned) and pass
the guarantee to the compiler with `sycl::multi_ptr` alignment / `assume_aligned` before the hot loop.

Before — alignment unknown, widening not emitted:
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n4) {
        auto* s4 = reinterpret_cast<const sycl::float4*>(src);
        d4[i] = a * s4[i];         // compiler unsure src is 16B-aligned → may stay scalar
    }
});
```

After — assert 16-byte alignment on the pointers:
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n4) {
        const float* sa = std::assume_aligned<16>(src);   // promise: 16B aligned
        float*       da = std::assume_aligned<16>(dst);
        auto* s4 = reinterpret_cast<const sycl::float4*>(sa);
        auto* d4 = reinterpret_cast<sycl::float4*>(da);
        d4[i] = a * s4[i];          // now free to emit a 128-bit vector load/store
    }
});
```

## Correctness invariants
- **Only assert an alignment the pointer actually has.** A false `assume_aligned` is undefined behavior;
  verify the allocation boundary and that any offset into it preserves the alignment.
- Sub-buffer / offset pointers (`base + k`) are aligned only if `k * sizeof(elem)` is a multiple of the
  asserted alignment — check before asserting.
- Values are unchanged; only the emitted load/store width changes.

## Verify it took effect
- IGC asm now shows wide vector loads/stores where it previously emitted scalar/unaligned messages.
- Achieved bandwidth rises; the widening strategy's expected speedup actually materializes.

## Pitfalls / conflicts
- **Conflicts:** none.
- **Synergizes:** `vectorize-vec4`, `vectorize-low-bitwidth-stores`, `coalesce-memory-access`.
- This is an **enabling transform**: neutral on its own — it only pays once a widening strategy emits the
  wide access it permits. Pair them; don't expect a win from alignment alone.
