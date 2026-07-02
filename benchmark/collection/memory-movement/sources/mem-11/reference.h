// reference.h
//
// CPU reference implementation for the pinnedAsyncIncrement case
// (per-element increment, g_data[i] += inc_value).
#ifndef PINNEDASYNCINCREMENT_REFERENCE_H
#define PINNEDASYNCINCREMENT_REFERENCE_H

// Deterministic input generator: g_data[i] = i.
inline int gen_data(int i) {
  return i;
}

// g_data[i] after the increment: i + inc_value. Each element is an
// independent update, so the result is exact regardless of which thread
// performs it or in what order threads execute.
inline int reference_increment(int i, int inc_value) {
  return gen_data(i) + inc_value;
}

#endif  // PINNEDASYNCINCREMENT_REFERENCE_H
