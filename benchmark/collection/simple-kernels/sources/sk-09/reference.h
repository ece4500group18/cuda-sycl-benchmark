// reference.h
//
// CPU reference implementation for the gridSyncCGReduction case
// (sum reduction of n doubles).
#ifndef GRIDSYNCCGREDUCTION_REFERENCE_H
#define GRIDSYNCCGREDUCTION_REFERENCE_H

// Deterministic input generator:
//   input[i] = ((i % 23) - 11) * 0.5
// Every value is an exact multiple of 0.5 (a dyadic rational) with
// |input[i]| <= 5.5.
inline double gen_val(long i) {
  return (double)((i % 23) - 11) * 0.5;
}

// Plain, sequential left-to-right sum of gen_val(0..n-1).
//
// Why this is an EXACT oracle for two GPU reductions that combine the same
// values in *different groupings/orders* (warp-shuffle tree, then a
// sequential per-block leader loop, then either a grid-synced single-kernel
// combine or a second kernel launch's combine loop):
//
// Every input value is an exact multiple of 0.5, i.e. of the form k/2 for
// an integer k. For n = 1<<18 = 262144 elements with |k| <= 11, every
// *partial* sum produced by any subset/order of additions has magnitude
// bounded by n * 5.5 ~= 1.44e6, i.e. |2*partial_sum| well under 2^53. Since
// IEEE-754 double addition is exact whenever the true mathematical result
// is itself representable (round-to-nearest of an exactly-representable
// value is that value, with no rounding error), and every dyadic rational
// with denominator <=2 and magnitude far below 2^53 is exactly
// representable in double, *every single addition step* in *any*
// grouping/order of these particular values is computed without rounding
// error. By induction, the final double is therefore the unique exact
// mathematical sum, independent of association/commutation order -- so a
// plain sequential CPU sum, a per-warp shuffle-tree GPU sum, and a
// two-kernel-launch GPU sum must all produce the bit-identical double.
inline double reference_reduce_sum(long n) {
  double sum = 0.0;
  for (long i = 0; i < n; ++i) {
    sum += gen_val(i);
  }
  return sum;
}

#endif  // GRIDSYNCCGREDUCTION_REFERENCE_H
