// reference.h
//
// CPU reference implementation for the binaryPartitionOddEvenReduce case
// (odd/even element count + odd-sum + even-sum over a fixed-size int array).
#ifndef BINARYPARTITIONODDEVENREDUCE_REFERENCE_H
#define BINARYPARTITIONODDEVENREDUCE_REFERENCE_H

// Deterministic input generator, replacing the upstream sample's own
// `rand() % 50`:
//   gen(i) = (i * 7 + 3) % 50
inline int gen(unsigned int i) {
  return (int)((i * 7u + 3u) % 50u);
}

// Counts odd elements and sums odd/even elements separately over
// gen(0..size-1). Every element contributes independently to exactly one
// of three plain integer accumulators (count of odds, sum of odds, sum of
// evens) via addition, which is commutative and associative -- so the
// totals are identical regardless of the order in which elements are
// visited or how they are grouped before being combined.
inline void reference_odd_even_count_and_sum(unsigned int size,
                                              long long *numOfOdds,
                                              long long *sumOfOdds,
                                              long long *sumOfEvens) {
  long long odds = 0, oddSum = 0, evenSum = 0;
  for (unsigned int i = 0; i < size; ++i) {
    int elem = gen(i);
    if (elem & 1) {
      odds += 1;
      oddSum += elem;
    } else {
      evenSum += elem;
    }
  }
  *numOfOdds = odds;
  *sumOfOdds = oddSum;
  *sumOfEvens = evenSum;
}

#endif  // BINARYPARTITIONODDEVENREDUCE_REFERENCE_H
