# software-pipeline

**Class:** Memory-latency  ·  **Priority:** High  ·  **Impact:** 10–30%

## When to apply
`SbidStall` is the dominant `VectorEngineStalls` entry and the hot loop has **few async ops in flight**:
it loads tile *k*, waits, computes tile *k*, then loads tile *k+1*. Each iteration serializes
load-then-compute, so the scoreboard idles during every load.

## Transformation
Overlap iterations: prefetch the **next** iteration's data while computing the **current** one (a
load/compute software pipeline). Keep one extra register buffer for the look-ahead load.

Before — load then compute, serialized per iteration:
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        float x = src[i * K + k];      // load
        acc += f(x);                   // compute waits on the load → SbidStall
    }
    if (i < n) dst[i] = acc;
});
```

After — prologue load, then compute(k) overlapped with load(k+1):
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    float acc = 0.f;
    float cur = src[i * K + 0];                     // prologue: prime the pipeline
    for (int k = 0; k < K; ++k) {
        float nxt = (k + 1 < K) ? src[i * K + k + 1] : 0.f;  // issue next load early...
        acc += f(cur);                              // ...compute overlaps its latency
        cur = nxt;
    }
    if (i < n) dst[i] = acc;
});
```

## Correctness invariants
- The prologue must load the first element exactly once; guard the look-ahead (`k + 1 < K`).
- Accumulation order is unchanged, so results are bit-identical to the serial loop.
- Only one extra live value (`nxt`) — keep the buffer depth minimal to avoid register growth.

## Verify it took effect
- `SbidStall` share drops in the re-profiled `VectorEngineStalls` mix.
- IGC asm shows the next-iteration `send` issued **before** the current iteration's dependent ALU.

## Pitfalls / conflicts
- **Conflicts:** `reduce-register-pressure` — deeper pipelines hold more live values; if spills appear,
  reduce the look-ahead depth.
- **Synergizes:** `increase-ilp` (independent lanes) and `prefetch-to-slm` (stage the look-ahead tile
  in SLM for multi-item reuse).
- Do not deepen the pipeline past 1–2 stages without measuring; latency hiding saturates, spills do not.
