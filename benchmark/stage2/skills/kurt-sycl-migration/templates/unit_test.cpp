// Golden-reference unit-test harness for a migrated SYCL kernel.
// Pattern: load/produce fixed inputs -> run SYCL kernel -> compare vs golden within rtol/atol.
// Adapt the kernel call and I/O to the specific kernel. Returns non-zero on failure (CTest fails).
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cmath>
#include <vector>

// Declare the migrated kernel entry point (defined in src/<id>.cpp).
// void run_kernel(sycl::queue& q, /* args */);

static bool allclose(const std::vector<float>& a, const std::vector<float>& b,
                     float rtol = 1e-5f, float atol = 1e-6f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > atol + rtol * std::fabs(b[i])) {
            std::printf("mismatch at %zu: got %g expected %g (diff %g)\n", i, a[i], b[i], diff);
            return false;
        }
    }
    return true;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // 1. Prepare fixed inputs (load from .sycl/state/kernels/<id>/ref/ or generate deterministically).
    // 2. Compute the golden output (CUDA-captured, CPU reference, or analytical).
    // 3. Run the SYCL kernel:  run_kernel(q, ...); q.wait();
    // 4. Compare:
    //    std::vector<float> got = ...; std::vector<float> golden = ...;
    //    if (!allclose(got, golden, /*rtol*/1e-5f, /*atol*/1e-6f)) return 1;

    std::printf("PASS\n");
    return 0;
}
