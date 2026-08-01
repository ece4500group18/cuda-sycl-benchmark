# SYCL Optimization Strategy Cards (Intel Xe2/Xe3)

One card per **strategy ID** used by `sycl-optimization-catalog.md` (Part 2). The catalog is the
*decision layer* (which strategy, when, priority, conflicts); each card here is the *example database*
(how to actually apply it). The join key is the strategy ID:

> catalog row `increase-ilp` → `sycl-optimization-strategies/increase-ilp.md`

The `sycl-optimization` search loop reads **only the card for the strategy it selected** in the APPLY
step — do not preload all of them.

## Card format

Every card follows the same fixed sections so the agent can consume them uniformly:

```markdown
# <strategy-id>

**Class:** <bottleneck class>  ·  **Priority:** <Critical/High/Medium>  ·  **Impact:** <empirical range>

## When to apply
The profiling trigger (in this framework's stall/metric vocabulary) that justifies this strategy.

## Transformation
Before → after SYCL (USM + `nd_range`, per `sycl-kernel-patterns.md` conventions). Keep it minimal and
correctness-preserving.

## Correctness invariants
What must stay true after the edit (types, tail guards, accumulation order, precision).

## Verify it took effect
The specific profiler/asm change that confirms the strategy landed (not just wall-clock).

## Pitfalls / conflicts
Framework anti-patterns and conflicting strategies to re-profile against.
```

## Index (strategy ID → card)

### Memory-latency
- `increase-ilp` · `software-pipeline` · `prefetch-to-slm` · `slm-cache-reuse` ·
  `prefetch-global-to-l3`

### Memory-bandwidth
- `vectorize-vec4` · `coalesce-memory-access` · `vectorize-low-bitwidth-stores` ·
  `reduce-data-type-width` · `fuse-passes` · `float2-accumulation` · `assume-aligned`

### Compute-bound
- `offload-xmx-library` · `native-math` · `unroll-hot-loop` · `compile-time-specialization` ·
  `fast-integer-division` · `balance-alu-pipes` · `restrict-kernel-args`

### Occupancy-bound
- `reduce-register-pressure` · `switch-grf-mode` · `tune-work-group-size` · `adaptive-block-size` ·
  `reduce-slm-allocation`

### Synchronization
- `double-buffer-slm` · `reduce-barriers` · `subgroup-reduction` · `lock-free-atomics`

### Cache
- `tile-data-access` · `pad-slm-arrays` · `optimize-access-pattern` · `cache-control-hints`

### Control-flow
- `eliminate-branches` · `uniform-control-flow`

### Launch / dispatch
- `multi-output-per-item` · `sycl-graphs`

### Host / dispatch overlap
- `overlap-copy-compute`

> A card is authoritative for *how* to apply its strategy; the catalog stays authoritative for *whether*
> and *when*. Keep them in sync by ID — never inline a card back into the catalog.
