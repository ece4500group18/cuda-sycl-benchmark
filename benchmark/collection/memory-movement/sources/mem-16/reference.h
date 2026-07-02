// reference.h
//
// CPU reference implementation for the sparseDenseSpmvLayout case
// (sparse matrix-vector product, y = M * x, compressed CSR gather vs.
// full dense row-major sweep).
#ifndef SPARSEDENSESPMVLAYOUT_REFERENCE_H
#define SPARSEDENSESPMVLAYOUT_REFERENCE_H

#include <algorithm>

#define NUM_ROWS 256
#define NNZ_PER_ROW 5
#define NNZ (NUM_ROWS * NNZ_PER_ROW)

// Deterministic dense vector: x[i] = (i % 11) - 5, range [-5, 5].
inline double gen_x(int i) {
  return (double)((i % 11) - 5);
}

// Deterministic column index of the k-th (k in [0, NNZ_PER_ROW)) nonzero
// placed in `row`. The offsets {0, 13, 26, 39, 52} added to `row*7` are
// all pairwise distinct and span less than 256, so reducing mod 256 (a
// bijection for any fixed shift) always yields NNZ_PER_ROW distinct
// columns per row, in whatever order the shift happens to rotate them to.
inline int col_of(int row, int k) {
  return (row * 7 + k * 13) % NUM_ROWS;
}

// Deterministic nonzero value of the k-th nonzero in `row`. Restricted to
// the integers {-4,-3,-2,-1,1,2,3,4} (never 0): spmv_dense_check_and_compute
// skips any matrix entry that reads back as exactly 0.0, so if one of our
// "nonzero" entries were 0 the dense path would silently drop a term the
// CSR path still includes, breaking the exact-match guarantee below.
inline double val_of(int row, int k) {
  double mag = (double)((row + 2 * k) % 4) + 1.0;  // 1..4
  double sign = (((row + k) % 2) == 0) ? 1.0 : -1.0;
  return mag * sign;
}

struct Nz {
  int col;
  double val;
};

// Build the row's NNZ_PER_ROW (column, value) pairs sorted by ascending
// column. This is the accumulation order both kernels use: spmv_csr sums
// `data[row_start..row_end)` in whatever order `indices` lists them (we
// sort `indices`/`data` by column when building the CSR arrays, below),
// and spmv_dense_check_and_compute sums over its plain
// `j = 0 .. num_rows-1` sweep, which visits columns in ascending order.
// So the two kernels add the exact same terms in the exact same order.
inline void row_nonzeros_sorted(int row, Nz out[NNZ_PER_ROW]) {
  for (int k = 0; k < NNZ_PER_ROW; ++k) {
    out[k].col = col_of(row, k);
    out[k].val = val_of(row, k);
  }
  std::sort(out, out + NNZ_PER_ROW,
            [](const Nz &a, const Nz &b) { return a.col < b.col; });
}

// CPU reference: y[row] = sum_k val_of(row,k) * x[col_of(row,k)], summed
// in ascending column order (matching both GPU kernels). Every value and
// vector entry is a small integer (|val| <= 4, |x| <= 5), so every partial
// sum is exactly representable in a float/double -- this is not a
// "generous tolerance" situation, the two kernels must agree bit-for-bit.
inline void reference_spmv(const double *x, double *y_out) {
  for (int row = 0; row < NUM_ROWS; ++row) {
    Nz nz[NNZ_PER_ROW];
    row_nonzeros_sorted(row, nz);
    double dot = 0.0;
    for (int k = 0; k < NNZ_PER_ROW; ++k) dot += nz[k].val * x[nz[k].col];
    y_out[row] = dot;
  }
}

#endif  // SPARSEDENSESPMVLAYOUT_REFERENCE_H
