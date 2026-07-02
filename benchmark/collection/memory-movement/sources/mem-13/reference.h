// reference.h
//
// CPU reference implementation for the pyramidTiledPathfinderDP case
// (Rodinia "pathfinder" shortest-path dynamic-programming recurrence).
#ifndef PYRAMIDTILEDPATHFINDERDP_REFERENCE_H
#define PYRAMIDTILEDPATHFINDERDP_REFERENCE_H

#include <cstdlib>

// Deterministic replacement for the original sample's
// `wall[i][j] = rand() % 10;` (seeded with a fixed `srand(9)`, which is
// not reproducible across libc versions/platforms):
//   wall[i][j] = (i*31 + j*17) % 10
// `data[]` is the row-major (rows x cols) flattened array passed to the
// CUDA driver exactly as upstream's `init()` builds it: row 0 is the
// initial "source" row (copied into gpuResult[0] before the first
// kernel launch); rows 1..rows-1 are the per-step "wall" costs
// (copied into gpuWall, which upstream's driver receives as `data+cols`).
inline int gen_wall(int i, int j) {
  return (i * 31 + j * 17) % 10;
}

inline void gen_data(int rows, int cols, int *data) {
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      data[i * cols + j] = gen_wall(i, j);
}

// Sequential (one row at a time) CPU reference for the exact recurrence
// computed by `dynproc_kernel`, regardless of how many rows a given GPU
// kernel launch batches together (`iteration`/pyramid height):
//
//   result[0][j]  = data[j]                                  (row 0, the
//                                                              initial
//                                                              source row)
//   result[t][j]  = data[t*cols+j] +
//                   min(result[t-1][j-1], result[t-1][j], result[t-1][j+1])
//
// with edge-clamped neighbors (result[t-1][-1] treated as result[t-1][0],
// result[t-1][cols] treated as result[t-1][cols-1]) -- exactly the
// boundary behavior `dynproc_kernel` produces via its `validXmin`/
// `validXmax`-clamped `W`/`E` indices at the true left/right edges of the
// row. Every step is an integer MIN + integer ADD, fully associative and
// independent of how many rows are grouped into one kernel launch, so the
// same final `result[cols]` array is expected bit-for-bit regardless of
// pyramid height.
inline void reference_pathfinder(const int *data, int rows, int cols,
                                  int *result_out) {
  int *cur = new int[cols];
  int *nxt = new int[cols];

  for (int j = 0; j < cols; ++j) cur[j] = data[j];  // row 0 (source row)

  for (int t = 1; t < rows; ++t) {
    const int *wall_row = data + (size_t)t * cols;
    for (int j = 0; j < cols; ++j) {
      int left = (j > 0) ? cur[j - 1] : cur[j];
      int up = cur[j];
      int right = (j < cols - 1) ? cur[j + 1] : cur[j];
      int shortest = left < up ? left : up;
      shortest = (right < shortest) ? right : shortest;
      nxt[j] = shortest + wall_row[j];
    }
    int *tmp = cur;
    cur = nxt;
    nxt = tmp;
  }

  for (int j = 0; j < cols; ++j) result_out[j] = cur[j];

  delete[] cur;
  delete[] nxt;
}

#endif  // PYRAMIDTILEDPATHFINDERDP_REFERENCE_H
