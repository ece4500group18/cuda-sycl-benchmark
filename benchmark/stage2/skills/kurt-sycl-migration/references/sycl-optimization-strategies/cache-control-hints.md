# cache-control-hints

**Class:** Cache  ·  **Priority:** Medium  ·  **Impact:** 10–30%

## When to apply
The L2/L3 hit rate is hurt by **streaming traffic** that shouldn't be cached: a write-once output that
pollutes the cache and evicts genuinely reused data, or a read-once input streamed past reused tiles.
The default caching policy treats all accesses the same; Intel's LSC lets you mark specific accesses as
**streaming/uncached** so they bypass the cache and leave it for the data that benefits.

## Transformation
Apply Intel LSC cache-control properties to the specific load/store — `streaming`/`uncached` for
read-once or write-once data, keep the default `cached` for reused data.

Before — streaming output pollutes the cache:
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n) out[i] = compute(reuse_tile, i);   // out[] written once but caches, evicting reuse_tile
});
```

After — mark the write-once store streaming (bypass cache):
```cpp
namespace ie = sycl::ext::intel::experimental;
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n) {
        float v = compute(reuse_tile, i);
        // write-back at L1/L2 with streaming (write-once) policy so it does not evict reused data
        ie::prefetch /* or the cache_control store property set on the pointer/access */;
        sycl::ext::oneapi::experimental::properties store_props{
            ie::cache_control<ie::cache_mode::streaming, ie::cache_level::L2>};
        // apply store_props to the store of out[i] via the annotated pointer API
        out[i] = v;
    }
});
```
(Exact spelling depends on the toolchain's `cache_control` / annotated-pointer API version — the intent
is: **streaming/uncached for read-once and write-once accesses; cached for reused accesses**. Confirm
the property names against the installed oneAPI headers.)

## Correctness invariants
- Cache hints are a **performance** control only — results are identical regardless of the policy.
- Never mark **reused** data uncached/streaming — that defeats `prefetch-to-slm`/`slm-cache-reuse` and
  regresses.
- Verify the target device/toolchain supports the LSC cache-control property; fall back to default
  caching if unavailable.

## Verify it took effect
- L2/L3 hit rate for the **reused** data rises (it is no longer evicted by streaming traffic).
- Streaming access latency is unchanged, but overall cache pressure and miss rate drop.

## Pitfalls / conflicts
- **Conflicts:** `prefetch-to-slm` / `slm-cache-reuse` if you accidentally mark the staged/reused data
  uncached — apply hints only to the genuinely read-once/write-once streams.
- **Synergizes:** `optimize-access-pattern`, `tile-data-access` (protect the tiled reuse set from
  streaming eviction).
- The API surface is toolchain-version-sensitive; treat spelling as version-dependent and validate it
  compiles for the target before relying on it.
