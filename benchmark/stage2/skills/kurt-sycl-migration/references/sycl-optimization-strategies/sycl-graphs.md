# sycl-graphs

**Class:** Launch / dispatch overhead  ·  **Priority:** Medium  ·  **Impact:** overhead-bound only

## When to apply
A **repeated sequence of small kernels** is dominated by per-submit overhead — the same graph of kernels
runs every iteration (an inference step, a solver iteration), and host-side submit/dependency cost rivals
the kernel time. A SYCL graph records the sequence once and replays it with a single submit, cutting the
per-launch host overhead.

## Transformation
Record the kernel sequence into a `command_graph`, finalize it once, then `ext_oneapi_graph` the
executable graph each iteration.

Before — resubmit every kernel every iteration:
```cpp
for (int step = 0; step < steps; ++step) {
    q.parallel_for(/* k1 */ ...);
    q.parallel_for(/* k2 */ ...);
    q.parallel_for(/* k3 */ ...);      // 3 submits × steps → host overhead
}
q.wait();
```

After — record once, replay:
```cpp
namespace se = sycl::ext::oneapi::experimental;
se::command_graph graph(q.get_context(), q.get_device());
graph.begin_recording(q);
q.parallel_for(/* k1 */ ...);
q.parallel_for(/* k2 */ ...);
q.parallel_for(/* k3 */ ...);
graph.end_recording();
auto exec = graph.finalize();
for (int step = 0; step < steps; ++step)
    q.ext_oneapi_graph(exec);          // one submit / iteration
q.wait();
```

## Correctness invariants
- The recorded topology must be **static** across iterations — same kernels, same dependency structure;
  only buffer *contents* change between replays, not the graph shape.
- **No mid-graph `q.wait()`** — it is incompatible with graph recording/replay.
- SYCL graphs are **single-device**; do not span devices in one graph.
- Requires the `ext_oneapi_limited_graph` aspect on the target — check device support.

## Verify it took effect
- Host-side submit/dispatch time per iteration drops; the timeline shows one replay instead of N submits.
- Only helps when the workload was **overhead-bound** — compute-bound sequences see little change.

## Pitfalls / conflicts
- **Conflicts:** mid-graph `wait()`; multi-device use; dynamic per-iteration topology.
- **Synergizes:** none listed — apply after per-kernel optimizations; graphs cut *launch* cost, not
  kernel cost.
- If the sequence shape varies per iteration, graphs don't apply — reduce kernel count instead
  (`fuse-passes`, `multi-output-per-item`).
