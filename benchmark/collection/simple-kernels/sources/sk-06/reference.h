// reference.h
//
// CPU reference for the deviceAssertGuard case: the same predicate
// (gtid < N) that testKernel asserts on and testKernelFlag records.
#ifndef DEVICEASSERTGUARD_REFERENCE_H
#define DEVICEASSERTGUARD_REFERENCE_H

// Predicate matching testKernel's assert(gtid < N) / testKernelFlag's
// g_flag[gtid] = (gtid < N) ? 1 : 0. Purely a function of the absolute
// thread id gtid and N -- no rand()/drand48()/curand anywhere.
inline int reference_predicate(int gtid, int N) {
  return (gtid < N) ? 1 : 0;
}

// Fills out[0..total) with reference_predicate(i, N) for every absolute
// thread id i in [0, total). Each entry is an independent, order-free
// evaluation of the predicate, so the result is exact regardless of
// which order threads/blocks execute in on the GPU.
inline void reference_predicate_array(int *out, int total, int N) {
  for (int i = 0; i < total; ++i) {
    out[i] = reference_predicate(i, N);
  }
}

#endif  // DEVICEASSERTGUARD_REFERENCE_H
