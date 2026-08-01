# vectorize-vec4

**Class:** Memory-bandwidth  ·  **Priority:** Critical  ·  **Impact:** 1.5–4× bandwidth

## When to apply
`SendStall` is the dominant `VectorEngineStalls` entry on a **bandwidth-bound** element-wise kernel that
issues **scalar** 32-bit loads/stores (one element per work-item). Each lane emits a narrow memory
message; the memory pipe is message-rate limited rather than byte limited. Fetching 4 contiguous
elements per message amortizes the per-message overhead into one wide transaction.

## Transformation
Load/store `sycl::float4` (128-bit) per work-item over the vectorizable bulk, and handle the remainder
scalar-wise. Requires the base pointer and length to be 4-element friendly (align allocations; guard the
tail).

Before — scalar element-wise:
```cpp
void scale(sycl::queue &q, const float* src, float* dst, float a, int n) {
    const int global = ((n + 255) / 256) * 256;
    q.parallel_for(sycl::nd_range<1>(global, 256), [=](sycl::nd_item<1> it) {
        const int i = it.get_global_id(0);
        if (i < n) dst[i] = a * src[i];          // 1 element / SendStall-bound message
    });
}
```

After — `float4` bulk + scalar tail:
```cpp
void scale(sycl::queue &q, const float* src, float* dst, float a, int n) {
    const int n4 = n / 4;                          // whole vec4 groups
    const int global = ((n4 + 255) / 256) * 256;
    auto* s4 = reinterpret_cast<const sycl::float4*>(src);
    auto* d4 = reinterpret_cast<sycl::float4*>(dst);
    q.parallel_for(sycl::nd_range<1>(global, 256), [=](sycl::nd_item<1> it) {
        const int i = it.get_global_id(0);
        if (i < n4) d4[i] = a * s4[i];             // 4 elements / one wide message
    });
    const int tail = n4 * 4;                       // scalar remainder (n % 4)
    if (tail < n) {
        const int rem = n - tail;
        q.parallel_for(sycl::nd_range<1>(((rem + 255) / 256) * 256, 256),
            [=](sycl::nd_item<1> it) {
                const int i = it.get_global_id(0);
                if (i < rem) dst[tail + i] = a * src[tail + i];
            });
    }
}
```

## Correctness invariants
- `src`/`dst` must be **16-byte aligned** for the `float4` reinterpret (USM `malloc_device` is; verify
  for offset sub-buffers).
- The **scalar tail** (`n % 4`) must still be processed — do not drop the remainder.
- `float4` arithmetic is component-wise; the numerical result is bit-identical to the scalar path.

## Verify it took effect
- `SendStall` share drops and achieved read/write bandwidth (`MemoryProfile` / `ComputeBasic`) rises
  toward device peak.
- IGC asm shows 128-bit vector loads/stores replacing the scalar `d32` messages.

## Pitfalls / conflicts
- **Conflicts:** `reduce-register-pressure`, `switch-grf-mode:small` — `float4` uses 4× the register
  width per value; if occupancy drops or spills appear, this can regress.
- **Synergizes:** `coalesce-memory-access` (align + contiguous first), `increase-ilp`.
- **Anti-pattern reminder:** do **not** replace `float4`+`sycl::dot()` with scalar code — the vector
  form generates better SIMD. Manual vectorized 64-bit loads are a separate known regression.
