#!/usr/bin/env python3
"""Add a small curated batch of Stage 1 CUDA benchmark cases.

The generated cases are intentionally modest, standalone kernels that fill the
current modern_ml and hpc gaps while staying compatible with the legacy pilot
case layout.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = ROOT / "pilot_benchmark" / "cases"

COMMON_CUDA = r"""
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__host__ __device__ static inline float hs(unsigned i, unsigned s) {
  return 2.0f * h01(i, s) - 1.0f;
}

static void write_vec(const char *path, const float *data, int n) {
  FILE *f = fopen(path, "w");
  if (!f) { fprintf(stderr, "open %s\n", path); exit(2); }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", data[i]);
  fclose(f);
}
""".strip()


def meta(case_id, name, category, domain, difficulty, features, sizes, tolerance, notes):
    return {
        "case_id": case_id,
        "name": name,
        "category": category,
        "domain": domain,
        "difficulty": difficulty,
        "source": {
            "type": "authored",
            "url": "https://github.com/ece4500group18/cuda-sycl-benchmark",
            "license": "MIT",
            "original_path": "original/main.cu",
        },
        "cuda_features": features,
        "libraries": [],
        "input": {"type": "hashed", "sizes": sizes, "seed": 123},
        "build": {
            "cuda_build_command": "nvcc -O2 -std=c++17 original/main.cu -o original/build/app",
            "sycl_build_command": "icpx -fsycl -O2 -std=c++17 syclomatic/main.dp.cpp -o build_sycl/app",
        },
        "run": {
            "cuda_run_command": "original/build/app output/cuda_output.txt",
            "sycl_run_command": "build_sycl/app output/sycl_output.txt",
        },
        "correctness": {
            "method": "cpu_reference",
            "metric": "max_abs_error",
            "tolerance": tolerance,
            "expected_pass_string": "PASS",
        },
        "syclomatic": {
            "status": "not_attempted",
            "command": "",
            "warnings_count": None,
            "manual_fixes_required": None,
        },
        "status": {
            "cuda_compile": "not_attempted",
            "cuda_run": "not_attempted",
            "cuda_verify": "not_attempted",
            "syclomatic_migrate": "not_attempted",
            "sycl_compile": "not_attempted",
            "sycl_run": "not_attempted",
            "sycl_verify": "not_attempted",
        },
        "notes": notes,
    }


CASES = {
    ("ai", "reluActivation"): {
        "metadata": meta(
            "reluActivation", "ReLU activation", "ai", "modern_ml", "easy",
            ["__global__", "elementwise", "activation"], [1048576], 1e-6,
            "Elementwise ReLU over deterministic signed inputs.",
        ),
        "main": r"""
__global__ void relu_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float v = x[i]; y[i] = v > 0.0f ? v : 0.0f; }
}

int main(int argc, char **argv) {
  const int n = 1048576;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hx=(float*)malloc(bytes), *hy=(float*)malloc(bytes);
  for (int i=0;i<n;++i) hx[i] = 4.0f * hs(i, 123);
  float *dx,*dy; CK(cudaMalloc(&dx,bytes)); CK(cudaMalloc(&dy,bytes));
  CK(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  relu_kernel<<<blocks,tpb>>>(dx,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,bytes,cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
        "verify": r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123) * V.F32(4.0)
    return np.maximum(x, V.F32(0.0))
""",
    },
    ("ai", "siluActivation"): {
        "metadata": meta(
            "siluActivation", "SiLU activation", "ai", "modern_ml", "medium",
            ["__global__", "elementwise", "expf", "activation"], [1048576], 1e-5,
            "Elementwise SiLU / swish activation.",
        ),
        "main": r"""
__global__ void silu_kernel(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { float v = x[i]; y[i] = v / (1.0f + expf(-v)); }
}

int main(int argc, char **argv) {
  const int n = 1048576;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hx=(float*)malloc(bytes), *hy=(float*)malloc(bytes);
  for (int i=0;i<n;++i) hx[i] = 6.0f * hs(i, 123);
  float *dx,*dy; CK(cudaMalloc(&dx,bytes)); CK(cudaMalloc(&dy,bytes));
  CK(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  silu_kernel<<<blocks,tpb>>>(dx,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,bytes,cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
        "verify": r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123).astype(np.float32) * np.float32(6.0)
    return x / (np.float32(1.0) + np.exp(-x, dtype=np.float32))
""",
    },
    ("ai", "biasAddRelu"): {
        "metadata": meta(
            "biasAddRelu", "Bias add plus ReLU", "ai", "modern_ml", "medium",
            ["__global__", "2D_indexing", "activation", "broadcast"], [512, 256], 1e-6,
            "Fused row-major bias add and ReLU, a common neural-network epilogue.",
        ),
        "main": r"""
__global__ void bias_relu(const float *x, const float *b, float *y, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = rows * cols;
  if (idx < n) {
    float v = x[idx] + b[idx % cols];
    y[idx] = v > 0.0f ? v : 0.0f;
  }
}

int main(int argc, char **argv) {
  const int rows = 512, cols = 256, n = rows * cols;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hb=(float*)malloc((size_t)cols*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) hx[i] = hs(i, 123);
  for (int i=0;i<cols;++i) hb[i] = 0.25f * hs(i, 321);
  float *dx,*db,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&db,(size_t)cols*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db,hb,(size_t)cols*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  bias_relu<<<blocks,tpb>>>(dx,db,dy,rows,cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dx); cudaFree(db); cudaFree(dy); free(hx); free(hb); free(hy); return 0;
}
""",
        "verify": r"""
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    b = V.gen_hashsigned(cols, 321) * V.F32(0.25)
    return np.maximum(x + b.reshape(1, cols), V.F32(0.0)).reshape(-1)
""",
    },
    ("ai", "residualScaleAdd"): {
        "metadata": meta(
            "residualScaleAdd", "Residual scaled add", "ai", "modern_ml", "easy",
            ["__global__", "elementwise", "residual"], [1048576], 1e-6,
            "Transformer-style residual addition y = x + alpha*r.",
        ),
        "main": r"""
__global__ void residual_add(const float *x, const float *r, float *y, int n, float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + alpha * r[i];
}

int main(int argc, char **argv) {
  const int n = 1048576; const float alpha = 0.125f;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hx=(float*)malloc(bytes), *hr=(float*)malloc(bytes), *hy=(float*)malloc(bytes);
  for (int i=0;i<n;++i) { hx[i] = hs(i, 123); hr[i] = hs(i, 777); }
  float *dx,*dr,*dy; CK(cudaMalloc(&dx,bytes)); CK(cudaMalloc(&dr,bytes)); CK(cudaMalloc(&dy,bytes));
  CK(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dr,hr,bytes,cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  residual_add<<<blocks,tpb>>>(dx,dr,dy,n,alpha);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,bytes,cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dx); cudaFree(dr); cudaFree(dy); free(hx); free(hr); free(hy); return 0;
}
""",
        "verify": r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    return V.gen_hashsigned(n, 123) + V.F32(0.125) * V.gen_hashsigned(n, 777)
""",
    },
    ("ai", "adamUpdate"): {
        "metadata": meta(
            "adamUpdate", "Adam optimizer update", "ai", "modern_ml", "medium",
            ["__global__", "optimizer", "sqrtf", "multiple_arrays"], [262144], 1e-5,
            "One Adam optimizer step over parameter, moment, variance, and gradient arrays.",
        ),
        "main": r"""
__global__ void adam_update(const float *p, const float *m, const float *v, const float *g, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float m2 = 0.9f * m[i] + 0.1f * g[i];
    float v2 = 0.999f * v[i] + 0.001f * g[i] * g[i];
    out[i] = p[i] - 0.001f * m2 / (sqrtf(v2) + 1.0e-8f);
  }
}

int main(int argc, char **argv) {
  const int n = 262144;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hp=(float*)malloc(bytes), *hm=(float*)malloc(bytes), *hv=(float*)malloc(bytes), *hg=(float*)malloc(bytes), *hy=(float*)malloc(bytes);
  for (int i=0;i<n;++i) { hp[i]=hs(i,11); hm[i]=0.1f*hs(i,22); hv[i]=0.01f+0.01f*h01(i,33); hg[i]=hs(i,44); }
  float *dp,*dm,*dv,*dg,*dy; CK(cudaMalloc(&dp,bytes)); CK(cudaMalloc(&dm,bytes)); CK(cudaMalloc(&dv,bytes)); CK(cudaMalloc(&dg,bytes)); CK(cudaMalloc(&dy,bytes));
  CK(cudaMemcpy(dp,hp,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dm,hm,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv,hv,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg,hg,bytes,cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  adam_update<<<blocks,tpb>>>(dp,dm,dv,dg,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,bytes,cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dp); cudaFree(dm); cudaFree(dv); cudaFree(dg); cudaFree(dy); free(hp); free(hm); free(hv); free(hg); free(hy); return 0;
}
""",
        "verify": r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    p = V.gen_hashsigned(n, 11)
    m = V.F32(0.1) * V.gen_hashsigned(n, 22)
    v = V.F32(0.01) + V.F32(0.01) * V.gen_hash01(n, 33)
    g = V.gen_hashsigned(n, 44)
    m2 = V.F32(0.9) * m + V.F32(0.1) * g
    v2 = V.F32(0.999) * v + V.F32(0.001) * g * g
    return p - V.F32(0.001) * m2 / (np.sqrt(v2, dtype=np.float32) + V.F32(1.0e-8))
""",
    },
    ("hpc", "particleUpdate"): {
        "metadata": meta(
            "particleUpdate", "Particle position/velocity update", "hpc", "hpc", "medium",
            ["__global__", "particle_simulation", "multiple_arrays"], [262144], 1e-6,
            "Euler update of particle position and velocity from acceleration.",
        ),
        "main": r"""
__global__ void particle_update(const float *x, const float *v, const float *a, float *out, int n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float vn = v[i] + a[i] * dt;
    float xn = x[i] + vn * dt;
    out[2*i] = xn; out[2*i+1] = vn;
  }
}

int main(int argc, char **argv) {
  const int n = 262144; const float dt = 0.01f;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hx=(float*)malloc(bytes), *hv=(float*)malloc(bytes), *ha=(float*)malloc(bytes), *hy=(float*)malloc((size_t)2*n*sizeof(float));
  for (int i=0;i<n;++i) { hx[i]=hs(i,1); hv[i]=0.1f*hs(i,2); ha[i]=0.01f*hs(i,3); }
  float *dx,*dv,*da,*dy; CK(cudaMalloc(&dx,bytes)); CK(cudaMalloc(&dv,bytes)); CK(cudaMalloc(&da,bytes)); CK(cudaMalloc(&dy,(size_t)2*n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv,hv,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(da,ha,bytes,cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  particle_update<<<blocks,tpb>>>(dx,dv,da,dy,n,dt);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)2*n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, 2*n);
  cudaFree(dx); cudaFree(dv); cudaFree(da); cudaFree(dy); free(hx); free(hv); free(ha); free(hy); return 0;
}
""",
        "verify": r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 1)
    v = V.F32(0.1) * V.gen_hashsigned(n, 2)
    a = V.F32(0.01) * V.gen_hashsigned(n, 3)
    vn = v + a * V.F32(0.01)
    xn = x + vn * V.F32(0.01)
    out = np.empty(2 * n, dtype=np.float32)
    out[0::2] = xn
    out[1::2] = vn
    return out
""",
    },
    ("hpc", "tridiagResidual"): {
        "metadata": meta(
            "tridiagResidual", "Tridiagonal residual", "hpc", "hpc", "medium",
            ["__global__", "stencil", "linear_system"], [262144], 1e-6,
            "Computes r = b - A*x for a 1D tridiagonal Laplacian operator.",
        ),
        "main": r"""
__global__ void residual(const float *x, const float *b, float *r, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float xm = x[i > 0 ? i - 1 : i];
    float xp = x[i + 1 < n ? i + 1 : i];
    r[i] = b[i] - (2.0f * x[i] - xm - xp);
  }
}

int main(int argc, char **argv) {
  const int n = 262144;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hx=(float*)malloc(bytes), *hb=(float*)malloc(bytes), *hr=(float*)malloc(bytes);
  for (int i=0;i<n;++i) { hx[i]=hs(i,123); hb[i]=0.5f*hs(i,456); }
  float *dx,*db,*dr; CK(cudaMalloc(&dx,bytes)); CK(cudaMalloc(&db,bytes)); CK(cudaMalloc(&dr,bytes));
  CK(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(db,hb,bytes,cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  residual<<<blocks,tpb>>>(dx,db,dr,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hr,dr,bytes,cudaMemcpyDeviceToHost));
  write_vec(out, hr, n);
  cudaFree(dx); cudaFree(db); cudaFree(dr); free(hx); free(hb); free(hr); return 0;
}
""",
        "verify": r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123)
    b = V.F32(0.5) * V.gen_hashsigned(n, 456)
    i = np.arange(n)
    xm = x[np.maximum(i - 1, 0)]
    xp = x[np.minimum(i + 1, n - 1)]
    return b - (V.F32(2.0) * x - xm - xp)
""",
    },
    ("hpc", "wave1dStep"): {
        "metadata": meta(
            "wave1dStep", "1D wave equation step", "hpc", "hpc", "medium",
            ["__global__", "stencil", "time_step"], [262144], 1e-6,
            "Single explicit update for a 1D wave equation with clamped endpoints.",
        ),
        "main": r"""
__global__ void wave_step(const float *prev, const float *cur, float *next, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float left = cur[i > 0 ? i - 1 : i];
    float right = cur[i + 1 < n ? i + 1 : i];
    next[i] = 2.0f * cur[i] - prev[i] + 0.1f * (left - 2.0f * cur[i] + right);
  }
}

int main(int argc, char **argv) {
  const int n = 262144;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hp=(float*)malloc(bytes), *hc=(float*)malloc(bytes), *hn=(float*)malloc(bytes);
  for (int i=0;i<n;++i) { hp[i]=hs(i,13); hc[i]=hs(i,14); }
  float *dp,*dc,*dn; CK(cudaMalloc(&dp,bytes)); CK(cudaMalloc(&dc,bytes)); CK(cudaMalloc(&dn,bytes));
  CK(cudaMemcpy(dp,hp,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dc,hc,bytes,cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  wave_step<<<blocks,tpb>>>(dp,dc,dn,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hn,dn,bytes,cudaMemcpyDeviceToHost));
  write_vec(out, hn, n);
  cudaFree(dp); cudaFree(dc); cudaFree(dn); free(hp); free(hc); free(hn); return 0;
}
""",
        "verify": r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    prev = V.gen_hashsigned(n, 13)
    cur = V.gen_hashsigned(n, 14)
    i = np.arange(n)
    left = cur[np.maximum(i - 1, 0)]
    right = cur[np.minimum(i + 1, n - 1)]
    return V.F32(2.0) * cur - prev + V.F32(0.1) * (left - V.F32(2.0) * cur + right)
""",
    },
}


VERIFY_PREFIX = """#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

"""

CMAKELISTS = """cmake_minimum_required(VERSION 3.18)
project(stage1_case CUDA)
add_executable(app main.cu)
set_target_properties(app PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
"""


def write_case(category: str, case_id: str, spec: dict) -> None:
    case_dir = CASES_ROOT / category / case_id
    for rel in ("original", "tests", "input", "expected", "output", "logs", "migrated"):
        (case_dir / rel).mkdir(parents=True, exist_ok=True)
    (case_dir / "metadata.json").write_text(json.dumps(spec["metadata"], indent=2) + "\n", encoding="utf-8")
    title = spec["metadata"]["name"]
    notes = spec["metadata"]["notes"]
    (case_dir / "README.md").write_text(f"# {title}\n\n{notes}\n", encoding="utf-8")
    (case_dir / "original" / "README.md").write_text(f"# Original CUDA\n\nStandalone CUDA implementation for `{case_id}`.\n", encoding="utf-8")
    (case_dir / "original" / "CMakeLists.txt").write_text(CMAKELISTS, encoding="utf-8")
    (case_dir / "original" / "main.cu").write_text(COMMON_CUDA + "\n\n" + spec["main"].strip() + "\n", encoding="utf-8")
    (case_dir / "tests" / "verify.py").write_text(VERIFY_PREFIX + spec["verify"].strip() + "\n\nif __name__ == \"__main__\":\n    V.run(reference)\n", encoding="utf-8")


def main() -> int:
    for (category, case_id), spec in CASES.items():
        write_case(category, case_id, spec)
        print(f"[ok] {category}/{case_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
