// thrustReduce: sum of n=1048576 hashed floats with thrust::reduce.
//   in[i] = h01(i, 123).  Output: the single sum.
#include <cstdio>
#include <cstdlib>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

__host__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const int n = 1048576;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  thrust::host_vector<float> h(n);
  for (int i = 0; i < n; ++i) h[i] = h01(i, 123);
  thrust::device_vector<float> d = h;
  float s = thrust::reduce(d.begin(), d.end(), 0.0f);
  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  fprintf(f, "%.9g\n", s); fclose(f);
  printf("thrustReduce done: n=%d sum=%.6f -> %s\n", n, s, out);
  return 0;
}
