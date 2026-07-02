// reference.h
//
// Deterministic input construction and CPU reference implementation for
// the csrScanOrderSpMM case (sparse-times-sparse matrix product A*B).
//
// A and B are both NUM_ROWS x NUM_ROWS circulant sparse integer matrices:
// row i of A has a nonzero at column (i + off) % NUM_ROWS for each offset
// `off` in OFFA, and row i of B has a nonzero at column (i + off) %
// NUM_ROWS for each offset `off` in OFFB. Because the offsets within each
// set are distinct mod NUM_ROWS, every row has exactly NNZ_PER_ROW
// nonzeros; because the pattern is circulant (a fixed offset applied to
// every row), every *column* also has exactly NNZ_PER_ROW nonzeros -- so
// a CSR-of-B build and a CSC-of-B build of the very same logical matrix
// B both come out perfectly regular, with no padding or empty rows/cols.
//
// All nonzero values are small nonzero integers (see val_a/val_b), so
// every dot product in A*B is a sum of a handful of exact integer
// products -- exactly representable in `float`/`double` and exactly
// commutative/associative regardless of the order terms are added in.
#ifndef CSRSCANORDERSPMM_REFERENCE_H
#define CSRSCANORDERSPMM_REFERENCE_H

static const int NUM_ROWS = 32;
static const int NNZ_PER_ROW = 4;
static const int NNZ_A = NUM_ROWS * NNZ_PER_ROW;  // 128
static const int NNZ_B = NUM_ROWS * NNZ_PER_ROW;  // 128

// Fixed, distinct-mod-32 column offsets defining the two circulant
// sparsity patterns.
static const int OFFA[NNZ_PER_ROW] = {1, 5, 11, 19};
static const int OFFB[NNZ_PER_ROW] = {2, 7, 13, 23};

inline int mod_nr(int x) {
  int r = x % NUM_ROWS;
  if (r < 0) r += NUM_ROWS;
  return r;
}

// A[i][j] value when nonzero: magnitude in [1,6], alternating sign.
inline int val_a(int i, int j) {
  int mag = 1 + ((i * 13 + j * 7 + i * j * 3) % 6);
  int sign = ((i + j) % 2 == 0) ? 1 : -1;
  return mag * sign;
}

// B[i][j] value when nonzero: magnitude in [1,5], alternating sign
// (opposite phase from val_a so A and B differ).
inline int val_b(int i, int j) {
  int mag = 1 + ((i * 5 + j * 11 + i * j * 2) % 5);
  int sign = ((i + j) % 2 == 0) ? -1 : 1;
  return mag * sign;
}

inline bool is_nonzero_a(int i, int j) {
  for (int t = 0; t < NNZ_PER_ROW; ++t) {
    if (j == mod_nr(i + OFFA[t])) return true;
  }
  return false;
}

inline bool is_nonzero_b(int i, int j) {
  for (int t = 0; t < NNZ_PER_ROW; ++t) {
    if (j == mod_nr(i + OFFB[t])) return true;
  }
  return false;
}

inline void sort4(int *a) {
  for (int i = 1; i < NNZ_PER_ROW; ++i) {
    int key = a[i];
    int j = i - 1;
    while (j >= 0 && a[j] > key) {
      a[j + 1] = a[j];
      --j;
    }
    a[j + 1] = key;
  }
}

// CSR of A: ptr[NUM_ROWS+1], indices[NNZ_A], data[NNZ_A]. Row-major,
// column indices ascending within each row.
template <typename REAL>
inline void build_csr_a(int *ptr, int *indices, REAL *data) {
  int pos = 0;
  ptr[0] = 0;
  for (int i = 0; i < NUM_ROWS; ++i) {
    int cols[NNZ_PER_ROW];
    for (int t = 0; t < NNZ_PER_ROW; ++t) cols[t] = mod_nr(i + OFFA[t]);
    sort4(cols);
    for (int t = 0; t < NNZ_PER_ROW; ++t) {
      indices[pos] = cols[t];
      data[pos] = (REAL)val_a(i, cols[t]);
      ++pos;
    }
    ptr[i + 1] = pos;
  }
}

// CSR of B (row-major: ptr indexed by B's row, indices give B's column).
// Fed to spmm_csr_csr_kernel, which expects exactly this layout.
template <typename REAL>
inline void build_csr_b(int *ptr, int *indices, REAL *data) {
  int pos = 0;
  ptr[0] = 0;
  for (int i = 0; i < NUM_ROWS; ++i) {
    int cols[NNZ_PER_ROW];
    for (int t = 0; t < NNZ_PER_ROW; ++t) cols[t] = mod_nr(i + OFFB[t]);
    sort4(cols);
    for (int t = 0; t < NNZ_PER_ROW; ++t) {
      indices[pos] = cols[t];
      data[pos] = (REAL)val_b(i, cols[t]);
      ++pos;
    }
    ptr[i + 1] = pos;
  }
}

// CSC of B (column-major: ptr indexed by B's column, indices give B's
// row within that column) -- the very same logical matrix B as
// build_csr_b, just transposed in storage order. Fed to
// spmm_csc_csr_kernel, which expects exactly this layout.
template <typename REAL>
inline void build_csc_b(int *ptr, int *indices, REAL *data) {
  int pos = 0;
  ptr[0] = 0;
  for (int j = 0; j < NUM_ROWS; ++j) {
    int rows[NNZ_PER_ROW];
    for (int t = 0; t < NNZ_PER_ROW; ++t) rows[t] = mod_nr(j - OFFB[t]);
    sort4(rows);
    for (int t = 0; t < NNZ_PER_ROW; ++t) {
      indices[pos] = rows[t];
      data[pos] = (REAL)val_b(rows[t], j);
      ++pos;
    }
    ptr[j + 1] = pos;
  }
}

// Independent CPU reference: dense result[i][j] = sum_k A[i][k]*B[k][j],
// recomputed directly from the val_a/val_b/is_nonzero_* formulas (not
// from the CSR/CSC arrays), so it cannot share a bug with either GPU
// kernel's scan logic. Every term is an exact small-integer product, so
// this is an exact (not approximate) reference regardless of summation
// order.
template <typename REAL>
inline void reference_spmm(REAL *result) {
  for (int i = 0; i < NUM_ROWS; ++i) {
    for (int j = 0; j < NUM_ROWS; ++j) {
      double dot = 0.0;
      for (int k = 0; k < NUM_ROWS; ++k) {
        double a = is_nonzero_a(i, k) ? (double)val_a(i, k) : 0.0;
        double b = is_nonzero_b(k, j) ? (double)val_b(k, j) : 0.0;
        dot += a * b;
      }
      result[i * NUM_ROWS + j] = (REAL)dot;
    }
  }
}

#endif  // CSRSCANORDERSPMM_REFERENCE_H
