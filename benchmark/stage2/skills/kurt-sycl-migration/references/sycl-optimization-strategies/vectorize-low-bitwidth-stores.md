# vectorize-low-bitwidth-stores

**Class:** Memory-bandwidth  ·  **Priority:** Critical  ·  **Impact:** 3–5× (bf16), 4–17× (int8)

## When to apply
The kernel stores **sub-32-bit** elements (bf16/fp16/int8/fp8/int4) one at a time, and the IGC asm dump
shows narrow `d16u32` / `d8u32` store encodings. Each scalar sub-word store still occupies a full memory
message, so the write pipe moves a fraction of a message's worth of useful bytes per instruction.

## Transformation
Pack several low-bitwidth elements into a wide store (e.g. a `sycl::vec` or a 32-/128-bit word) so one
message carries many elements. Each work-item produces a contiguous group.

Before — scalar bf16 stores (`d16u32` per element):
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n) dst[i] = sycl::bfloat16(a * float(src[i]));   // one narrow store / item
});
```

After — each item writes a contiguous vec of 8 bf16 (one wide message):
```cpp
constexpr int V = 8;                                   // 8 * 16-bit = 128-bit store
const int nV = n / V;
const int global = ((nV + wg - 1) / wg) * wg;
using vec8 = sycl::vec<sycl::bfloat16, 8>;
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int j = it.get_global_id(0);
    if (j < nV) {
        vec8 v;
        for (int k = 0; k < V; ++k) v[k] = sycl::bfloat16(a * float(src[j * V + k]));
        reinterpret_cast<vec8*>(dst)[j] = v;           // one 128-bit store / 8 elements
    }
});
// scalar tail for n % V (same as the Before body)
```

## Correctness invariants
- Handle the `n % V` **tail** scalar-wise; do not drop it.
- `dst` must be aligned for the wide store (`V * sizeof(elem)`); USM `malloc_device` is 16-byte aligned.
- Numerical values are identical — only the store width changes, not the arithmetic.

## Verify it took effect
- IGC asm shows wide vector stores replacing `d16u32`/`d8u32`; write-message count drops ~`V×`.
- Achieved write bandwidth (`MemoryProfile`) rises; `SendStall` on the store path shrinks.

## Pitfalls / conflicts
- **Conflicts:** none intrinsic (orthogonal to most strategies).
- **Synergizes:** `reduce-data-type-width` (produce the narrow type in the first place), `vectorize-vec4`.
- Applies to **stores** specifically; low-bitwidth *loads* are usually already widened by the compiler —
  measure before adding manual load packing.
