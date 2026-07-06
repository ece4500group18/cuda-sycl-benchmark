// Eigenvalues of a symmetric tridiagonal matrix by parallel bisection with
// Sturm-sequence counting — AMD's classic eigenvalue sample.
//
// Extracted from HeCBench src/eigenvalue-cuda (kernels.cu + reference.cu +
// main.cu driver loop; origin: AMD APP SDK sample).
// Upstream: https://github.com/zjin-lcf/HeCBench @ 01f58fc5 (AMD
// BSD-style license preserved in the snapshot).
// The two kernels (calNumEigenValueInterval, recalculateEigenIntervals),
// the device Sturm counter, and the host helpers (isComplete,
// computeGerschgorinInterval) are upstream code verbatim, as is the
// double-buffered host convergence loop. The harness uses deterministic
// hash-generated diagonals and dumps the converged eigen intervals.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
typedef unsigned int uint;
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

// ---- upstream device code (kernels.cu, verbatim) ------------------------------
__device__
float calNumEigenValuesLessThan(
   const float x,
   const uint width,
   const float *__restrict__ diagonal,
   const float *__restrict__ offDiagonal)
{
  uint count = 0;

  float prev_diff = (diagonal[0] - x);
  count += (prev_diff < 0)? 1 : 0;
  for(uint i = 1; i < width ; i += 1)
  {
    float diff = (diagonal[i] - x) - ((offDiagonal[i-1] * offDiagonal[i-1]) / prev_diff);

    count += (diff < 0) ? 1 : 0;
    prev_diff = diff;
  }
  return count;
}

__global__
void calNumEigenValueInterval(
    uint  *__restrict__ numEigenIntervals,
    const float *__restrict__ eigenIntervals,
    const float *__restrict__ diagonal,
    const float *__restrict__ offDiagonal,
    const uint     width)
{
  uint gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint lowerId = 2 * gid;
  uint upperId = lowerId + 1;
  float lowerLimit = eigenIntervals[lowerId];
  float upperLimit = eigenIntervals[upperId];
  uint lower = calNumEigenValuesLessThan(lowerLimit, width, diagonal, offDiagonal);
  uint upper = calNumEigenValuesLessThan(upperLimit, width, diagonal, offDiagonal);
  numEigenIntervals[gid] = upper - lower;
}

__global__
void recalculateEigenIntervals(
          float *__restrict__ newEigenIntervals,
    const float *__restrict__ eigenIntervals,
    const uint  *__restrict__ numEigenIntervals,
    const float *__restrict__ diagonal,
    const float *__restrict__ offDiagonal,
    const    uint    width,
    const    float   tolerance)
{
  uint gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint lowerId = 2 * gid;
  uint upperId = lowerId + 1;
  uint currentIndex = gid;

  uint index = 0;
  while(currentIndex >= numEigenIntervals[index])
  {
    currentIndex -= numEigenIntervals[index];
    ++index;
  }

  uint lId = 2 * index;
  uint uId = lId + 1;

  /* if the number of eigenvalues in the interval is just 1 */
  if(numEigenIntervals[index] == 1)
  {
    float midValue = (eigenIntervals[uId] + eigenIntervals[lId])/2;
    float n        = calNumEigenValuesLessThan(midValue, width, diagonal, offDiagonal);
    n -= calNumEigenValuesLessThan(eigenIntervals[lId], width, diagonal, offDiagonal);

    /* check if the interval size is less than tolerance levels */
    if(eigenIntervals[uId] - eigenIntervals[lId] < tolerance)
    {
      newEigenIntervals[lowerId] = eigenIntervals[lId];
      newEigenIntervals[upperId] = eigenIntervals[uId];
    }
    else if(n == 0) /* if the eigenvalue lies in the right half of the interval */
    {
      newEigenIntervals[lowerId] = midValue;
      newEigenIntervals[upperId] = eigenIntervals[uId];
    }
    else           /* if the eigenvalue lies in the left half of the interval */
    {
      newEigenIntervals[lowerId] = eigenIntervals[lId];
      newEigenIntervals[upperId] = midValue;
    }
  }
  /* split the intervals into equal intervals of size divisionWidth */
  else /* (numEigenIntervals[index] > 1) */
  {
    float divisionWidth = (eigenIntervals[uId] - eigenIntervals[lId]) / numEigenIntervals[index];
    newEigenIntervals[lowerId] = eigenIntervals[lId] + divisionWidth * currentIndex;
    newEigenIntervals[upperId] = newEigenIntervals[lowerId] + divisionWidth;
  }
}
// ---- end upstream device code ---------------------------------------------------

// ---- upstream host helpers (reference.cu, verbatim) ----------------------------
int isComplete(float * eigenIntervals, const int length, const float tolerance)
{
  for(int i=0; i< length; i++)
  {
    uint lid = 2*i;
    uint uid = lid + 1;
    if(eigenIntervals[uid] - eigenIntervals[lid] >= tolerance)
    {
      return 1;
    }
  }
  return 0;
}

void computeGerschgorinInterval(float * lLimit,
                                float * uLimit,
                                const float * diagonal,
                                const float * offDiagonal,
                                const uint length)
{

  float lowerLimit = diagonal[0] - fabs(offDiagonal[0]);
  float upperLimit = diagonal[0] + fabs(offDiagonal[0]);

  for(uint i = 1; i < length-1; ++i)
  {
    float r =  fabs(offDiagonal[i-1]) + fabs(offDiagonal[i]);
    lowerLimit = (lowerLimit > (diagonal[i] - r))? (diagonal[i] - r): lowerLimit;
    upperLimit = (upperLimit < (diagonal[i] + r))? (diagonal[i] + r): upperLimit;
  }

  lowerLimit = (lowerLimit > (diagonal[length-1] - fabs(offDiagonal[length-2])))?
    (diagonal[length-1] - fabs(offDiagonal[length-2])): lowerLimit;
  upperLimit = (upperLimit < (diagonal[length-1] + fabs(offDiagonal[length-2])))?
    (diagonal[length-1] + fabs(offDiagonal[length-2])): upperLimit;

  *lLimit = lowerLimit;
  *uLimit = upperLimit;
}
// ---- end upstream host helpers ---------------------------------------------------

static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char *argv[]) {
  const int length = 256;
  const float tolerance = 1e-4f;
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";

  float *diagonal = (float*)malloc(length * sizeof(float));
  float *offDiagonal = (float*)malloc((length - 1) * sizeof(float));
  for (int i = 0; i < length; ++i) diagonal[i] = 2.0f * h01((unsigned)i, 111) - 1.0f;
  for (int i = 0; i < length - 1; ++i) offDiagonal[i] = 0.5f * (2.0f * h01((unsigned)i, 112) - 1.0f);

  float *eigenIntervals[2];
  eigenIntervals[0] = (float*)malloc(2 * length * sizeof(float));
  eigenIntervals[1] = (float*)malloc(2 * length * sizeof(float));

  // Initial interval: the Gershgorin bound split evenly across gid slots
  // (upstream initialization).
  float lowerLimit, upperLimit;
  computeGerschgorinInterval(&lowerLimit, &upperLimit, diagonal, offDiagonal, length);
  for (int i = 0; i < length; ++i) {
    eigenIntervals[0][2 * i] = lowerLimit;
    eigenIntervals[0][2 * i + 1] = upperLimit;
  }

  float *d_diagonal, *d_offDiagonal, *d_intervals[2];
  uint *d_numIntervals;
  CK(cudaMalloc(&d_diagonal, length * sizeof(float)));
  CK(cudaMalloc(&d_offDiagonal, (length - 1) * sizeof(float)));
  CK(cudaMalloc(&d_intervals[0], 2 * length * sizeof(float)));
  CK(cudaMalloc(&d_intervals[1], 2 * length * sizeof(float)));
  CK(cudaMalloc(&d_numIntervals, length * sizeof(uint)));
  CK(cudaMemcpy(d_diagonal, diagonal, length * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_offDiagonal, offDiagonal, (length - 1) * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_intervals[0], eigenIntervals[0], 2 * length * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_intervals[1], eigenIntervals[0], 2 * length * sizeof(float), cudaMemcpyHostToDevice));

  dim3 grid(length / 256);
  dim3 block(256);

  // Upstream double-buffered convergence loop.
  uint in = 0;
  while (isComplete(eigenIntervals[in], length, tolerance)) {
    calNumEigenValueInterval<<<grid, block>>>(d_numIntervals, d_intervals[in],
                                              d_diagonal, d_offDiagonal, length);
    recalculateEigenIntervals<<<grid, block>>>(d_intervals[1 - in], d_intervals[in],
                                               d_numIntervals, d_diagonal, d_offDiagonal,
                                               length, tolerance);
    CK(cudaGetLastError());
    in = 1 - in;
    CK(cudaMemcpy(eigenIntervals[in], d_intervals[in], 2 * length * sizeof(float),
                  cudaMemcpyDeviceToHost));
  }

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (int i = 0; i < 2 * length; ++i) fprintf(f, "%.9g\n", eigenIntervals[in][i]);
  fclose(f);

  cudaFree(d_diagonal); cudaFree(d_offDiagonal);
  cudaFree(d_intervals[0]); cudaFree(d_intervals[1]); cudaFree(d_numIntervals);
  free(diagonal); free(offDiagonal); free(eigenIntervals[0]); free(eigenIntervals[1]);
  return 0;
}
