// pyramidTiledPathfinderDP: Rodinia "pathfinder" shortest-path dynamic-
// programming recurrence, run two ways with the *same* __global__ kernel:
// batched pyramid/temporal tiling (multiple DP row-steps per kernel
// launch) vs. a straightforward one-row-per-launch recurrence.
//
// The __global__ kernel below (and the IN_RANGE/MIN macros and BLOCK_SIZE/
// HALO constants it depends on) is reproduced, unmodified, from:
//   Rodinia benchmark suite (gpu-rodinia mirror), cuda/pathfinder/pathfinder.cu
//   https://github.com/yuhc/gpu-rodinia/blob/master/cuda/pathfinder/pathfinder.cu
//   Copyright (c) 2008-2011 University of Virginia
//   License: BSD-style, permissive (see LICENSE in this directory)
//
// `dynproc_kernel` advances the DP recurrence
//   result[t][j] = wall[t][j] + min(result[t-1][j-1], result[t-1][j], result[t-1][j+1])
// (edge-clamped at j=0 and j=cols-1) by `iteration` rows in a single
// launch: each thread block loads one row-slice of the current DP state
// into __shared__ memory once, then loops `iteration` times *entirely out
// of shared memory*, re-syncing only between rows, before writing the
// final row back to global memory. A `border = iteration*HALO` halo of
// extra columns is loaded on each side of a block's "small block" so that
// the shared-memory recurrence at the block's edges stays consistent with
// its neighbors after `iteration` steps -- the classic "pyramid" temporal
// tiling trick.
//
// This case launches the identical kernel two ways over the identical
// input:
//   - "batched": pyramid height (iteration) = 4 -- 16 kernel launches
//     amortize the global-memory halo exchange over 4 DP rows each.
//   - "singlestep": pyramid height (iteration) = 1 -- 63 kernel launches,
//     one row's worth of work (and one halo exchange) per launch, the
//     straightforward un-tiled recurrence.
// Both paths read/write the same row-major `wall` cost array and compute
// the exact same integer MIN-then-ADD recurrence in the exact same
// row-by-row order; only *how many DP time-steps are batched behind one
// halo exchange / kernel launch* differs. This isolates temporal tiling's
// memory-traffic reduction (fewer global-memory round trips for the same
// number of row-updates) as a technique distinct from spatial
// shared-memory tiling (see tiledMatmulShmem in this repo), where the
// tile always corresponds to one fixed output region rather than several
// sequential time-steps of the same region.
//
// Deterministic inputs (replicated by reference.h):
//   wall[i][j] = (i*31 + j*17) % 10, i in [0,rows), j in [0,cols)
// rows = 64, cols = 512, BLOCK_SIZE = 256 (fixed, matches upstream).
//
// Output: argv[1] (default output/cuda_output.txt), 512 lines, one final
// integer DP value per column, from the batched (pyramid height 4) path.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "reference.h"

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                          \
      return 2;                                                             \
    }                                                                        \
  } while (0)

// --- begin: verbatim from Rodinia cuda/pathfinder/pathfinder.cu ---

#define BLOCK_SIZE 256
#define HALO 1  // halo width along one direction when advancing to the next iteration

#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

__global__ void dynproc_kernel(
                int iteration,
                int *gpuWall,
                int *gpuSrc,
                int *gpuResults,
                int cols,
                int rows,
                int startStep,
                int border)
{

        __shared__ int prev[BLOCK_SIZE];
        __shared__ int result[BLOCK_SIZE];

	int bx = blockIdx.x;
	int tx=threadIdx.x;

        // each block finally computes result for a small block
        // after N iterations.
        // it is the non-overlapping small blocks that cover
        // all the input data

        // calculate the small block size
	int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

        // calculate the boundary for the block according to
        // the boundary of its small block
        int blkX = small_block_cols*bx-border;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
	int xidx = blkX+tx;

        // effective range within this block that falls within
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

        int W = tx-1;
        int E = tx+1;

        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols-1)){
            prev[tx] = gpuSrc[xidx];
	}
	__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        bool computed;
        for (int i=0; i<iteration ; i++){
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  isValid){
                  computed = true;
                  int left = prev[W];
                  int up = prev[tx];
                  int right = prev[E];
                  int shortest = MIN(left, up);
                  shortest = MIN(shortest, right);
                  int index = cols*(startStep+i)+xidx;
                  result[tx] = shortest + gpuWall[index];

            }
            __syncthreads();
            if(i==iteration-1)
                break;
            if(computed)	 //Assign the computation range
                prev[tx]= result[tx];
	    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
      }

      // update the global memory
      // after the last iteration, only threads coordinated within the
      // small block perform the calculation and switch on ``computed''
      if (computed){
          gpuResults[xidx]=result[tx];
      }
}

// --- end: verbatim from Rodinia cuda/pathfinder/pathfinder.cu ---

int main(int argc, char **argv) {
  const int rows = 64;
  const int cols = 512;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  const size_t data_bytes = (size_t)rows * cols * sizeof(int);
  const size_t wall_bytes = (size_t)(rows - 1) * cols * sizeof(int);
  const size_t row_bytes = (size_t)cols * sizeof(int);

  int *h_data = (int *)malloc(data_bytes);
  gen_data(rows, cols, h_data);

  // gpuWall (rows-1 rows of per-step cost) and the row-0 initial source
  // row, laid out exactly as upstream's init()/run(): row 0 is the
  // initial DP state, rows 1..rows-1 (h_data + cols) are the wall costs.
  int *dWall;
  CHECK(cudaMalloc(&dWall, wall_bytes));
  CHECK(cudaMemcpy(dWall, h_data + cols, wall_bytes, cudaMemcpyHostToDevice));

  // --- Path 1: batched pyramid tiling, pyramid height (iteration) = 4 ---
  const int ph_batched = 4;
  const int border_batched = ph_batched * HALO;
  const int small_block_batched = BLOCK_SIZE - ph_batched * HALO * 2;
  const int blockCols_batched =
      cols / small_block_batched + ((cols % small_block_batched == 0) ? 0 : 1);

  int *dResultBatched[2];
  CHECK(cudaMalloc(&dResultBatched[0], row_bytes));
  CHECK(cudaMalloc(&dResultBatched[1], row_bytes));
  CHECK(cudaMemcpy(dResultBatched[0], h_data, row_bytes, cudaMemcpyHostToDevice));

  {
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols_batched);
    int src = 1, dst = 0;
    for (int t = 0; t < rows - 1; t += ph_batched) {
      int tmp = src; src = dst; dst = tmp;
      int iter = MIN(ph_batched, rows - t - 1);
      dynproc_kernel<<<dimGrid, dimBlock>>>(iter, dWall, dResultBatched[src],
                                             dResultBatched[dst], cols, rows,
                                             t, border_batched);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
    }
    int *h_result_batched = (int *)malloc(row_bytes);
    CHECK(cudaMemcpy(h_result_batched, dResultBatched[dst], row_bytes,
                      cudaMemcpyDeviceToHost));

    // --- Path 2: one row per launch, pyramid height (iteration) = 1 ---
    const int ph_single = 1;
    const int border_single = ph_single * HALO;
    const int small_block_single = BLOCK_SIZE - ph_single * HALO * 2;
    const int blockCols_single =
        cols / small_block_single + ((cols % small_block_single == 0) ? 0 : 1);

    int *dResultSingle[2];
    CHECK(cudaMalloc(&dResultSingle[0], row_bytes));
    CHECK(cudaMalloc(&dResultSingle[1], row_bytes));
    CHECK(cudaMemcpy(dResultSingle[0], h_data, row_bytes, cudaMemcpyHostToDevice));

    dim3 dimBlock2(BLOCK_SIZE);
    dim3 dimGrid2(blockCols_single);
    int src2 = 1, dst2 = 0;
    for (int t = 0; t < rows - 1; t += ph_single) {
      int tmp = src2; src2 = dst2; dst2 = tmp;
      int iter = MIN(ph_single, rows - t - 1);
      dynproc_kernel<<<dimGrid2, dimBlock2>>>(iter, dWall, dResultSingle[src2],
                                               dResultSingle[dst2], cols, rows,
                                               t, border_single);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
    }
    int *h_result_single = (int *)malloc(row_bytes);
    CHECK(cudaMemcpy(h_result_single, dResultSingle[dst2], row_bytes,
                      cudaMemcpyDeviceToHost));

    // CPU reference check.
    int *h_ref = (int *)malloc(row_bytes);
    reference_pathfinder(h_data, rows, cols, h_ref);

    long max_abs_err_batched = 0, max_abs_err_single = 0, max_abs_err_cross = 0;
    for (int j = 0; j < cols; ++j) {
      long d1 = (long)h_result_batched[j] - (long)h_ref[j];
      long d2 = (long)h_result_single[j] - (long)h_ref[j];
      long d3 = (long)h_result_batched[j] - (long)h_result_single[j];
      if (d1 < 0) d1 = -d1;
      if (d2 < 0) d2 = -d2;
      if (d3 < 0) d3 = -d3;
      if (d1 > max_abs_err_batched) max_abs_err_batched = d1;
      if (d2 > max_abs_err_single) max_abs_err_single = d2;
      if (d3 > max_abs_err_cross) max_abs_err_cross = d3;
    }

    FILE *f = fopen(out, "w");
    if (!f) {
      fprintf(stderr, "cannot open %s for writing\n", out);
      return 2;
    }
    for (int j = 0; j < cols; ++j) fprintf(f, "%d\n", h_result_batched[j]);
    fclose(f);

    printf("pyramidTiledPathfinderDP done: rows=%d, cols=%d, pyramid_height=%d, "
           "max_abs_error(batched vs ref)=%ld, max_abs_error(singlestep vs ref)=%ld, "
           "max_abs_error(batched vs singlestep)=%ld -> %s\n",
           rows, cols, ph_batched, max_abs_err_batched, max_abs_err_single,
           max_abs_err_cross, out);
    printf("%s\n", (max_abs_err_batched == 0 && max_abs_err_single == 0 &&
                    max_abs_err_cross == 0)
                       ? "PASS"
                       : "FAIL");

    cudaFree(dResultSingle[0]);
    cudaFree(dResultSingle[1]);
    free(h_result_batched);
    free(h_result_single);
    free(h_ref);
  }

  cudaFree(dWall);
  cudaFree(dResultBatched[0]);
  cudaFree(dResultBatched[1]);
  free(h_data);
  return 0;
}
