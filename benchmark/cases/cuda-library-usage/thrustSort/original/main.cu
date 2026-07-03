// thrustSort: ascending sort of n=1048576 hashed floats with Thrust.
//   in[i] = h01(i, 123)
#include <cstdio>
#include <cstdlib>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

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
  thrust::sort(d.begin(), d.end());
  thrust::copy(d.begin(), d.end(), h.begin());
  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", h[i]); fclose(f);
  printf("thrustSort done: n=%d -> %s\n", n, out);
  return 0;
}
