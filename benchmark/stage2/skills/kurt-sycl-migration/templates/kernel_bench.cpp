// Standalone single-kernel benchmark driver for a migrated SYCL kernel.
// This is the measurement + profiling target for `sycl-optimization` (NOT the full e2e workload):
// it drives ONE kernel in isolation over a set of representative shapes so timing is fast, stable,
// and safe to profile with unitrace (metric counters + stall sampling) without overflowing the tracer.
//
// Build it JIT (do NOT add -fsycl-targets=spir64_gen) when it will be the IGC shader-dump / stall
// target — an AOT build dumps shaders at build time and the runtime dump is empty (see profiler §4).
//
// Adapt: the kernel entry point, the I/O allocation, and the SHAPES table to the kernel's real
// tensor shapes. Regenerate inputs deterministically from the kernel detail's seed+shape+dtype.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

// Declare the migrated kernel entry point (defined in src/<id>.cpp).
// void run_kernel(sycl::queue& q, const float* in, float* out, /* dims... */);

struct Shape {
    const char* group;   // "model" | "general" | "edge"
    const char* name;    // human label, e.g. "llama-4096" or "edge-odd-1023"
    std::int64_t n;      // adapt: replace with the kernel's real dimension set (M,N,K / rows,cols / ...)
};

// Representative shapes. Pick each group deliberately (see sycl-profiler §2 "Kernel-level benchmark"):
//   model   — sizes from real models the kernel serves (hidden dims, head dim, seq len, batch/tokens)
//   general — a couple of round, saturating sizes for steady-state throughput
//   edge    — tiny, non-power-of-2 / non-multiple-of-sub-group, masked/boundary, and near-memory-limit
static const std::vector<Shape> SHAPES = {
    {"model",   "hidden-768",   768LL * 1024},
    {"model",   "hidden-4096",  4096LL * 1024},
    {"general", "round-1M",     1LL << 20},
    {"general", "round-16M",    1LL << 24},
    {"edge",    "tiny-1",       1},
    {"edge",    "odd-1023",     1023},
    {"edge",    "large-256M",   1LL << 28},
};

// Warm-up + timed iterations for one shape; returns median device time in ms.
static double time_shape(sycl::queue& q, const Shape& s, int warmup, int iters) {
    // 1. Allocate USM inputs/outputs for shape s (regenerate inputs from seed+shape+dtype).
    // 2. Warm up:
    //    for (int i = 0; i < warmup; ++i) { run_kernel(q, /*...*/); }
    //    q.wait();
    // 3. Time `iters` runs; prefer sycl::event device-profiling over host wall-clock:
    //    std::vector<double> ms;
    //    for (int i = 0; i < iters; ++i) {
    //        auto t0 = std::chrono::high_resolution_clock::now();
    //        run_kernel(q, /*...*/); q.wait();
    //        auto t1 = std::chrono::high_resolution_clock::now();
    //        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    //    }
    //    return median(ms);
    (void)q; (void)s; (void)warmup; (void)iters;
    return 0.0;
}

int main(int argc, char** argv) {
    int warmup = 3, iters = 20;
    std::string only_group;                 // optional: restrict to one group (model|general|edge)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--group" && i + 1 < argc) only_group = argv[++i];
    }

    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // One JSON object per shape so the trial log / bench.sh can parse per-shape results.
    for (const auto& s : SHAPES) {
        if (!only_group.empty() && only_group != s.group) continue;
        double med = time_shape(q, s, warmup, iters);
        std::printf("{\"group\":\"%s\",\"shape\":\"%s\",\"median_ms\":%.4f}\n", s.group, s.name, med);
    }
    return 0;
}
