#!/usr/bin/env python3
"""Add a real-project-adapted Stage 1 CUDA case batch.

The cases are standalone CUDA extractions/simplifications of kernel patterns
from real open-source CUDA projects. They intentionally keep dependencies out
of the case directories so Stage 1 can build, run, verify, and benchmark the
original CUDA ground truth directly.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = ROOT / "pilot_benchmark" / "cases"

LLAMA = {
    "project": "ggml-org/llama.cpp",
    "license": "MIT",
    "base": "https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda",
}
VLLM = {
    "project": "vllm-project/vllm",
    "license": "Apache-2.0",
    "base": "https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable",
}
BNB = {
    "project": "bitsandbytes-foundation/bitsandbytes",
    "license": "MIT",
    "base": "https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/csrc",
}
FLASH = {
    "project": "Dao-AILab/flash-attention",
    "license": "BSD-3-Clause",
    "base": "https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src",
}
XFORMERS = {
    "project": "facebookresearch/xformers",
    "license": "BSD-3-Clause",
    "base": "https://github.com/facebookresearch/xformers/blob/main",
}
HECBENCH = {
    "project": "ORNL/HeCBench",
    "license": "BSD-3-Clause",
    "base": "https://github.com/ORNL/HeCBench/blob/master/src",
}

COMMON_CUDA = r"""
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 2; \
} } while (0)

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

VERIFY_PREFIX = """#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

"""

CMAKELISTS = """cmake_minimum_required(VERSION 3.18)
project(stage1_real_project_case CUDA)
add_executable(app main.cu)
set_target_properties(app PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
"""


def source_url(source: dict[str, str], rel: str) -> str:
    return f"{source['base']}/{rel}"


def metadata(
    case_id: str,
    name: str,
    category: str,
    domain: str,
    difficulty: str,
    source: dict[str, str],
    rel: str,
    features: list[str],
    sizes: list[int],
    tolerance: float,
    description: str,
    notes: str,
) -> dict:
    url = source_url(source, rel)
    return {
        "case_id": case_id,
        "name": name,
        "category": category,
        "domain": domain,
        "difficulty": difficulty,
        "source_project": source["project"],
        "source_url": url,
        "license": source["license"],
        "adaptation_type": "simplified",
        "description": description,
        "source": {
            "type": "simplified",
            "url": url,
            "license": source["license"],
            "original_path": rel,
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


def case(
    case_id: str,
    name: str,
    category: str,
    domain: str,
    difficulty: str,
    source: dict[str, str],
    rel: str,
    features: list[str],
    sizes: list[int],
    tolerance: float,
    description: str,
    notes: str,
    main: str,
    verify: str,
) -> dict:
    return {
        "category": category,
        "metadata": metadata(
            case_id, name, category, domain, difficulty, source, rel,
            features, sizes, tolerance, description, notes,
        ),
        "main": main,
        "verify": verify,
    }


def elemwise_main(kernel_name: str, kernel_body: str, n: int, scale_x: float = 4.0, scale_g: float = 2.0) -> str:
    return f"""
__global__ void {kernel_name}(const float *x, const float *g, float *y, int n) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {{
    {kernel_body}
  }}
}}

int main(int argc, char **argv) {{
  const int n = {n};
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hx=(float*)malloc(bytes), *hg=(float*)malloc(bytes), *hy=(float*)malloc(bytes);
  for (int i=0;i<n;++i) {{ hx[i] = {scale_x}f * hs(i, 123); hg[i] = {scale_g}f * hs(i, 321); }}
  float *dx,*dg,*dy; CK(cudaMalloc(&dx,bytes)); CK(cudaMalloc(&dg,bytes)); CK(cudaMalloc(&dy,bytes));
  CK(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg,hg,bytes,cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  {kernel_name}<<<blocks,tpb>>>(dx,dg,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,bytes,cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dx); cudaFree(dg); cudaFree(dy); free(hx); free(hg); free(hy); return 0;
}}
"""


CASES: dict[str, dict] = {}


def add(spec: dict) -> None:
    CASES[spec["metadata"]["case_id"]] = spec


add(case(
    "ggmlRmsNormBlock", "ggml RMSNorm block reduction", "ai", "modern_ml", "hard",
    LLAMA, "norm.cu",
    ["__global__", "__shared__", "block_reduction", "rsqrtf"],
    [256, 512], 1e-4,
    "Per-row RMS normalization with learned weight, adapted from ggml CUDA normalization kernels.",
    "Standalone simplification of llama.cpp/ggml norm.cu using deterministic row-major tensors and a CPU reference.",
    r"""
__global__ void rmsnorm_kernel(const float *x, const float *w, float *y, int rows, int cols) {
  extern __shared__ float s[];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  float sum = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    float v = x[row * cols + c];
    sum += v * v;
  }
  s[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  float inv = rsqrtf(s[0] / (float)cols + 1.0e-6f);
  for (int c = tid; c < cols; c += blockDim.x) {
    y[row * cols + c] = x[row * cols + c] * inv * w[c];
  }
}

int main(int argc, char **argv) {
  const int rows = 256, cols = 512, n = rows * cols;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hw=(float*)malloc((size_t)cols*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) hx[i] = 3.0f * hs(i, 123);
  for (int i=0;i<cols;++i) hw[i] = 1.0f + 0.1f * hs(i, 77);
  float *dx,*dw,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dw,(size_t)cols*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dw,hw,(size_t)cols*sizeof(float),cudaMemcpyHostToDevice));
  rmsnorm_kernel<<<rows,256,256*sizeof(float)>>>(dx,dw,dy,rows,cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dx); cudaFree(dw); cudaFree(dy); free(hx); free(hw); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = (V.F32(3.0) * V.gen_hashsigned(rows * cols, 123)).reshape(rows, cols)
    w = V.F32(1.0) + V.F32(0.1) * V.gen_hashsigned(cols, 77)
    inv = V.F32(1.0) / np.sqrt(np.mean(x * x, axis=1, dtype=np.float32) + V.F32(1.0e-6), dtype=np.float32)
    return (x * inv.reshape(rows, 1) * w.reshape(1, cols)).reshape(-1)
""",
))

add(case(
    "ggmlRopeInterleaved", "ggml interleaved RoPE", "ai", "modern_ml", "hard",
    LLAMA, "rope.cu",
    ["__global__", "sinf", "cosf", "tensor_layout"],
    [512, 128], 2e-4,
    "Interleaved rotary position embedding over token rows, adapted from ggml CUDA RoPE kernels.",
    "Standalone simplification of llama.cpp/ggml rope.cu; host code fixes deterministic positions and tensor dimensions.",
    r"""
__global__ void rope_kernel(const float *x, float *y, int tokens, int dim) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs = tokens * (dim / 2);
  if (pair < pairs) {
    int t = pair / (dim / 2);
    int p = pair % (dim / 2);
    int i0 = t * dim + 2 * p;
    int i1 = i0 + 1;
    float theta = (float)((t * 17) % 2048) * powf(10000.0f, -2.0f * (float)p / (float)dim);
    float c = cosf(theta), s = sinf(theta);
    float a = x[i0], b = x[i1];
    y[i0] = a * c - b * s;
    y[i1] = a * s + b * c;
  }
}

int main(int argc, char **argv) {
  const int tokens = 512, dim = 128, n = tokens * dim;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) hx[i] = hs(i, 123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  int total = tokens * (dim / 2); int tpb=256, blocks=(total+tpb-1)/tpb;
  rope_kernel<<<blocks,tpb>>>(dx,dy,tokens,dim);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    tokens, dim = meta["input"]["sizes"]
    x = V.gen_hashsigned(tokens * dim, 123).reshape(tokens, dim)
    y = np.empty_like(x)
    for t in range(tokens):
        pos = (t * 17) % 2048
        for p in range(dim // 2):
            theta = V.F32(pos) * np.float32(10000.0) ** np.float32(-2.0 * p / dim)
            c = np.cos(theta, dtype=np.float32)
            s = np.sin(theta, dtype=np.float32)
            a = x[t, 2 * p]
            b = x[t, 2 * p + 1]
            y[t, 2 * p] = a * c - b * s
            y[t, 2 * p + 1] = a * s + b * c
    return y.reshape(-1)
""",
))

add(case(
    "ggmlQ8Dequant", "ggml q8 block dequantization", "ai", "modern_ml", "medium",
    LLAMA, "dequantize.cuh",
    ["__global__", "int8_t", "quantization", "block_layout"],
    [4096, 32], 1e-5,
    "q8 block dequantization with one scale per 32-value block, adapted from ggml dequantization helpers.",
    "Standalone simplified q8_0-style block layout from llama.cpp/ggml dequantize.cuh.",
    r"""
__global__ void dequant_q8(const int8_t *q, const float *scale, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (float)q[i] * scale[i / 32];
}

int main(int argc, char **argv) {
  const int blocks_q = 4096, qk = 32, n = blocks_q * qk;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  int8_t *hq=(int8_t*)malloc((size_t)n*sizeof(int8_t));
  float *hscl=(float*)malloc((size_t)blocks_q*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int b=0;b<blocks_q;++b) hscl[b] = 0.02f + 0.03f * h01(b, 77);
  for (int i=0;i<n;++i) hq[i] = (int8_t)lrintf(127.0f * hs(i, 123));
  int8_t *dq; float *ds,*dy; CK(cudaMalloc(&dq,(size_t)n)); CK(cudaMalloc(&ds,(size_t)blocks_q*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dq,hq,(size_t)n,cudaMemcpyHostToDevice)); CK(cudaMemcpy(ds,hscl,(size_t)blocks_q*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(n+tpb-1)/tpb; dequant_q8<<<grid,tpb>>>(dq,ds,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dq); cudaFree(ds); cudaFree(dy); free(hq); free(hscl); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    blocks, qk = meta["input"]["sizes"]
    n = blocks * qk
    q = np.rint(V.F32(127.0) * V.gen_hashsigned(n, 123)).astype(np.int8).astype(np.float32)
    scale = V.F32(0.02) + V.F32(0.03) * V.gen_hash01(blocks, 77)
    return q * np.repeat(scale, qk)
""",
))

add(case(
    "ggmlQ4Dequant", "ggml q4 nibble dequantization", "ai", "modern_ml", "hard",
    LLAMA, "dequantize.cuh",
    ["__global__", "uint8_t", "bit_unpack", "quantization"],
    [4096, 32], 1e-5,
    "q4 block dequantization with packed nibbles and per-block scale, adapted from ggml quantized block helpers.",
    "Standalone q4-style nibble unpacking from llama.cpp/ggml dequantize.cuh.",
    r"""
__global__ void dequant_q4(const uint8_t *q, const float *scale, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint8_t packed = q[i / 2];
    int nibble = (i & 1) ? (packed >> 4) : (packed & 15);
    y[i] = ((float)nibble - 8.0f) * scale[i / 32];
  }
}

int main(int argc, char **argv) {
  const int blocks_q = 4096, qk = 32, n = blocks_q * qk;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  uint8_t *hq=(uint8_t*)calloc((size_t)n/2, sizeof(uint8_t));
  float *hscl=(float*)malloc((size_t)blocks_q*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int b=0;b<blocks_q;++b) hscl[b] = 0.04f + 0.02f * h01(b, 88);
  for (int i=0;i<n;++i) {
    uint8_t nib = (uint8_t)min(15, max(0, (int)floorf(16.0f * h01(i, 123))));
    if (i & 1) hq[i/2] |= (uint8_t)(nib << 4); else hq[i/2] |= nib;
  }
  uint8_t *dq; float *ds,*dy; CK(cudaMalloc(&dq,(size_t)n/2)); CK(cudaMalloc(&ds,(size_t)blocks_q*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dq,hq,(size_t)n/2,cudaMemcpyHostToDevice)); CK(cudaMemcpy(ds,hscl,(size_t)blocks_q*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(n+tpb-1)/tpb; dequant_q4<<<grid,tpb>>>(dq,ds,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dq); cudaFree(ds); cudaFree(dy); free(hq); free(hscl); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    blocks, qk = meta["input"]["sizes"]
    n = blocks * qk
    q = np.floor(V.F32(16.0) * V.gen_hash01(n, 123)).astype(np.int32)
    q = np.clip(q, 0, 15).astype(np.float32)
    scale = V.F32(0.04) + V.F32(0.02) * V.gen_hash01(blocks, 88)
    return (q - V.F32(8.0)) * np.repeat(scale, qk)
""",
))

add(case(
    "ggmlQuantizeQ8", "ggml block q8 quantize-dequantize", "ai", "modern_ml", "hard",
    LLAMA, "quantize.cu",
    ["__global__", "__shared__", "block_reduction", "quantization"],
    [4096, 32], 1e-5,
    "Per-block q8 quantization with max-abs scale and dequantized output, adapted from ggml quantization kernels.",
    "Standalone quantize/dequantize round trip inspired by llama.cpp/ggml quantize.cu.",
    r"""
__global__ void quant_q8_roundtrip(const float *x, float *y, int blocks_q) {
  __shared__ float s[32];
  int b = blockIdx.x;
  int t = threadIdx.x;
  float v = x[b * 32 + t];
  s[t] = fabsf(v);
  __syncthreads();
  for (int stride=16; stride>0; stride>>=1) {
    if (t < stride) s[t] = fmaxf(s[t], s[t + stride]);
    __syncthreads();
  }
  float scale = s[0] / 127.0f + 1.0e-12f;
  int q = (int)lrintf(v / scale);
  q = max(-127, min(127, q));
  y[b * 32 + t] = (float)q * scale;
}

int main(int argc, char **argv) {
  const int blocks_q = 4096, qk = 32, n = blocks_q * qk;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) hx[i] = 5.0f * hs(i, 123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  quant_q8_roundtrip<<<blocks_q,32>>>(dx,dy,blocks_q);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    blocks, qk = meta["input"]["sizes"]
    x = (V.F32(5.0) * V.gen_hashsigned(blocks * qk, 123)).reshape(blocks, qk)
    scale = np.max(np.abs(x), axis=1).astype(np.float32) / V.F32(127.0) + V.F32(1.0e-12)
    q = np.rint(x / scale.reshape(blocks, 1)).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.float32)
    return (q * scale.reshape(blocks, 1)).reshape(-1)
""",
))

add(case(
    "ggmlSoftcap", "ggml softcap activation", "ai", "modern_ml", "medium",
    LLAMA, "softcap.cu",
    ["__global__", "tanhf", "activation"],
    [1048576], 1e-5,
    "Softcap activation y = cap * tanh(x / cap), adapted from ggml softcap kernels.",
    "Standalone elementwise softcap extraction from llama.cpp/ggml softcap.cu.",
    elemwise_main("softcap_kernel", "float cap = 30.0f; y[i] = cap * tanhf(x[i] / cap);", 1048576, 40.0, 1.0),
    r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.F32(40.0) * V.gen_hashsigned(n, 123)
    cap = V.F32(30.0)
    return cap * np.tanh(x / cap, dtype=np.float32)
""",
))

add(case(
    "ggmlCausalSoftmax", "ggml causal row softmax", "ai", "modern_ml", "hard",
    LLAMA, "softmax.cu",
    ["__global__", "__shared__", "softmax", "causal_mask", "reduction"],
    [512, 128], 1e-4,
    "Row-wise softmax with a causal mask, adapted from ggml CUDA softmax kernels.",
    "Standalone masked softmax inspired by llama.cpp/ggml softmax.cu.",
    r"""
__global__ void causal_softmax(const float *x, float *y, int rows, int cols) {
  int row = blockIdx.x;
  if (threadIdx.x == 0) {
    int limit = row % cols;
    float m = -3.402823e38f;
    for (int c=0; c<=limit; ++c) m = fmaxf(m, x[row * cols + c]);
    float denom = 0.0f;
    for (int c=0; c<=limit; ++c) denom += expf(x[row * cols + c] - m);
    for (int c=0; c<cols; ++c) {
      y[row * cols + c] = (c <= limit) ? expf(x[row * cols + c] - m) / denom : 0.0f;
    }
  }
}

int main(int argc, char **argv) {
  const int rows=512, cols=128, n=rows*cols;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) hx[i] = 3.0f * hs(i, 123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  causal_softmax<<<rows,128>>>(dx,dy,rows,cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = (V.F32(3.0) * V.gen_hashsigned(rows * cols, 123)).reshape(rows, cols)
    y = np.zeros_like(x)
    for r in range(rows):
        limit = r % cols
        vals = x[r, :limit + 1]
        m = np.max(vals)
        e = np.exp(vals - m, dtype=np.float32)
        y[r, :limit + 1] = e / np.sum(e, dtype=np.float32)
    return y.reshape(-1)
""",
))

add(case(
    "ggmlMeanRows", "ggml row mean reduction", "ai", "modern_ml", "medium",
    LLAMA, "mean.cu",
    ["__global__", "__shared__", "reduction"],
    [512, 256], 1e-5,
    "Per-row mean reduction, adapted from ggml CUDA mean kernels.",
    "Standalone row mean reduction inspired by llama.cpp/ggml mean.cu.",
    r"""
__global__ void mean_rows(const float *x, float *y, int rows, int cols) {
  extern __shared__ float s[];
  int row = blockIdx.x, tid = threadIdx.x;
  float sum = 0.0f;
  for (int c=tid;c<cols;c+=blockDim.x) sum += x[row*cols+c];
  s[tid] = sum;
  __syncthreads();
  for (int stride=blockDim.x/2; stride>0; stride>>=1) {
    if (tid < stride) s[tid] += s[tid+stride];
    __syncthreads();
  }
  if (tid == 0) y[row] = s[0] / (float)cols;
}

int main(int argc, char **argv) {
  const int rows=512, cols=256, n=rows*cols;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)rows*sizeof(float));
  for (int i=0;i<n;++i) hx[i] = hs(i, 123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)rows*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  mean_rows<<<rows,128,128*sizeof(float)>>>(dx,dy,rows,cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)rows*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, rows);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    return np.mean(x, axis=1, dtype=np.float32)
""",
))

add(case(
    "ggmlConcatRows", "ggml row concat", "ai", "modern_ml", "medium",
    LLAMA, "concat.cu",
    ["__global__", "tensor_layout", "strided_copy"],
    [512, 192, 64], 1e-6,
    "Concatenate two row-major tensors along the column dimension, adapted from ggml CUDA concat kernels.",
    "Standalone tensor layout transformation inspired by llama.cpp/ggml concat.cu.",
    r"""
__global__ void concat_rows(const float *a, const float *b, float *y, int rows, int ca, int cb) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * (ca + cb);
  if (idx < total) {
    int row = idx / (ca + cb);
    int col = idx % (ca + cb);
    y[idx] = (col < ca) ? a[row * ca + col] : b[row * cb + (col - ca)];
  }
}

int main(int argc, char **argv) {
  const int rows=512, ca=192, cb=64, na=rows*ca, nb=rows*cb, n=rows*(ca+cb);
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  float *ha=(float*)malloc((size_t)na*sizeof(float)), *hb=(float*)malloc((size_t)nb*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<na;++i) ha[i] = hs(i, 123);
  for (int i=0;i<nb;++i) hb[i] = 2.0f * hs(i, 321);
  float *da,*db,*dy; CK(cudaMalloc(&da,(size_t)na*sizeof(float))); CK(cudaMalloc(&db,(size_t)nb*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(da,ha,(size_t)na*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(db,hb,(size_t)nb*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(n+tpb-1)/tpb; concat_rows<<<grid,tpb>>>(da,db,dy,rows,ca,cb);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(da); cudaFree(db); cudaFree(dy); free(ha); free(hb); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    rows, ca, cb = meta["input"]["sizes"]
    a = V.gen_hashsigned(rows * ca, 123).reshape(rows, ca)
    b = (V.F32(2.0) * V.gen_hashsigned(rows * cb, 321)).reshape(rows, cb)
    return np.concatenate([a, b], axis=1).reshape(-1)
""",
))

add(case(
    "ggmlPool2dAvg", "ggml 2D average pooling", "ai", "modern_ml", "medium",
    LLAMA, "pool2d.cu",
    ["__global__", "2D_indexing", "pooling", "tensor_layout"],
    [4, 3, 64, 64], 1e-6,
    "2D average pooling over NCHW tensors, adapted from ggml CUDA pool2d kernels.",
    "Standalone avg-pool extraction inspired by llama.cpp/ggml pool2d.cu.",
    r"""
__global__ void avg_pool2d(const float *x, float *y, int n, int c, int h, int w) {
  int oh = h / 2, ow = w / 2;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * c * oh * ow;
  if (idx < total) {
    int tmp = idx;
    int ox = tmp % ow; tmp /= ow;
    int oy = tmp % oh; tmp /= oh;
    int ch = tmp % c; int batch = tmp / c;
    int base = ((batch * c + ch) * h + 2 * oy) * w + 2 * ox;
    y[idx] = 0.25f * (x[base] + x[base + 1] + x[base + w] + x[base + w + 1]);
  }
}

int main(int argc, char **argv) {
  const int n=4, c=3, h=64, w=64, in_n=n*c*h*w, out_n=n*c*(h/2)*(w/2);
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  float *hx=(float*)malloc((size_t)in_n*sizeof(float)), *hy=(float*)malloc((size_t)out_n*sizeof(float));
  for (int i=0;i<in_n;++i) hx[i] = hs(i, 123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)in_n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)out_n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)in_n*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(out_n+tpb-1)/tpb; avg_pool2d<<<grid,tpb>>>(dx,dy,n,c,h,w);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)out_n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, out_n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    n, c, h, w = meta["input"]["sizes"]
    x = V.gen_hashsigned(n * c * h * w, 123).reshape(n, c, h, w)
    y = V.F32(0.25) * (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2])
    return y.reshape(-1)
""",
))

add(case(
    "vllmSiluMul", "vLLM SiLU multiply activation", "ai", "modern_ml", "medium",
    VLLM, "activation_kernels.cu",
    ["__global__", "expf", "fused_activation"],
    [1048576], 1e-5,
    "Fused SiLU and multiply activation, adapted from vLLM CUDA activation kernels.",
    "Standalone activation epilogue inspired by vLLM csrc/libtorch_stable/activation_kernels.cu.",
    elemwise_main("silu_mul_kernel", "float v = x[i]; y[i] = (v / (1.0f + expf(-v))) * g[i];", 1048576, 6.0, 2.0),
    r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.F32(6.0) * V.gen_hashsigned(n, 123)
    g = V.F32(2.0) * V.gen_hashsigned(n, 321)
    return (x / (V.F32(1.0) + np.exp(-x, dtype=np.float32))) * g
""",
))

add(case(
    "vllmGeluTanhApprox", "vLLM GELU tanh approximation", "ai", "modern_ml", "medium",
    VLLM, "activation_kernels.cu",
    ["__global__", "tanhf", "fused_activation"],
    [1048576], 1e-5,
    "GELU tanh approximation, adapted from vLLM activation kernels.",
    "Standalone GELU approximation inspired by vLLM csrc/libtorch_stable/activation_kernels.cu.",
    elemwise_main("gelu_tanh_kernel", "float v=x[i]; float u=0.7978845608f*(v+0.044715f*v*v*v); y[i]=0.5f*v*(1.0f+tanhf(u));", 1048576, 5.0, 1.0),
    r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.F32(5.0) * V.gen_hashsigned(n, 123)
    u = V.F32(0.7978845608) * (x + V.F32(0.044715) * x * x * x)
    return V.F32(0.5) * x * (V.F32(1.0) + np.tanh(u, dtype=np.float32))
""",
))

add(case(
    "vllmFusedAddRmsNorm", "vLLM fused add RMSNorm", "ai", "modern_ml", "hard",
    VLLM, "layernorm_kernels.cu",
    ["__global__", "__shared__", "residual", "rmsnorm", "reduction"],
    [256, 512], 1e-4,
    "Fused residual add and RMSNorm, adapted from vLLM layernorm CUDA kernels.",
    "Standalone simplification of vLLM fused_add_rms_norm-style behavior.",
    r"""
__global__ void add_rmsnorm(const float *x, const float *r, const float *w, float *y, int rows, int cols) {
  extern __shared__ float s[];
  int row=blockIdx.x, tid=threadIdx.x;
  float sum=0.0f;
  for (int c=tid;c<cols;c+=blockDim.x) {
    float v = x[row*cols+c] + r[row*cols+c];
    sum += v * v;
  }
  s[tid] = sum; __syncthreads();
  for (int stride=blockDim.x/2; stride>0; stride>>=1) {
    if (tid < stride) s[tid] += s[tid+stride];
    __syncthreads();
  }
  float inv = rsqrtf(s[0] / (float)cols + 1.0e-6f);
  for (int c=tid;c<cols;c+=blockDim.x) {
    float v = x[row*cols+c] + r[row*cols+c];
    y[row*cols+c] = v * inv * w[c];
  }
}

int main(int argc, char **argv) {
  const int rows=256, cols=512, n=rows*cols;
  const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hr=(float*)malloc((size_t)n*sizeof(float)), *hw=(float*)malloc((size_t)cols*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) { hx[i]=2.0f*hs(i,123); hr[i]=0.5f*hs(i,777); }
  for (int i=0;i<cols;++i) hw[i]=1.0f+0.1f*hs(i,44);
  float *dx,*dr,*dw,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dr,(size_t)n*sizeof(float))); CK(cudaMalloc(&dw,(size_t)cols*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dr,hr,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dw,hw,(size_t)cols*sizeof(float),cudaMemcpyHostToDevice));
  add_rmsnorm<<<rows,256,256*sizeof(float)>>>(dx,dr,dw,dy,rows,cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dr); cudaFree(dw); cudaFree(dy); free(hx); free(hr); free(hw); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = (V.F32(2.0) * V.gen_hashsigned(rows * cols, 123)).reshape(rows, cols)
    r = (V.F32(0.5) * V.gen_hashsigned(rows * cols, 777)).reshape(rows, cols)
    w = V.F32(1.0) + V.F32(0.1) * V.gen_hashsigned(cols, 44)
    v = x + r
    inv = V.F32(1.0) / np.sqrt(np.mean(v * v, axis=1, dtype=np.float32) + V.F32(1.0e-6), dtype=np.float32)
    return (v * inv.reshape(rows, 1) * w.reshape(1, cols)).reshape(-1)
""",
))

add(case(
    "vllmRotaryPaged", "vLLM paged rotary embedding", "ai", "modern_ml", "hard",
    VLLM, "pos_encoding_kernels.cu",
    ["__global__", "sinf", "cosf", "paged_positions", "tensor_layout"],
    [256, 4, 64], 1e-4,
    "Rotary embedding using per-token positions across heads, adapted from vLLM positional encoding kernels.",
    "Standalone paged-position RoPE simplification from vLLM pos_encoding_kernels.cu.",
    r"""
__global__ void rotary_paged(const float *x, float *y, int tokens, int heads, int dim) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs = tokens * heads * (dim / 2);
  if (pair < pairs) {
    int p = pair % (dim / 2);
    int tmp = pair / (dim / 2);
    int h = tmp % heads;
    int t = tmp / heads;
    int base = (t * heads + h) * dim + 2 * p;
    int pos = (t * 37 + h * 11) % 4096;
    float theta = (float)pos * powf(10000.0f, -2.0f * (float)p / (float)dim);
    float c=cosf(theta), s=sinf(theta), a=x[base], b=x[base+1];
    y[base] = a*c - b*s; y[base+1] = a*s + b*c;
  }
}

int main(int argc, char **argv) {
  const int tokens=256, heads=4, dim=64, n=tokens*heads*dim;
  const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) hx[i]=hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  int pairs=tokens*heads*(dim/2), tpb=256, grid=(pairs+tpb-1)/tpb; rotary_paged<<<grid,tpb>>>(dx,dy,tokens,heads,dim);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    tokens, heads, dim = meta["input"]["sizes"]
    x = V.gen_hashsigned(tokens * heads * dim, 123).reshape(tokens, heads, dim)
    y = np.empty_like(x)
    for t in range(tokens):
        for h in range(heads):
            pos = (t * 37 + h * 11) % 4096
            for p in range(dim // 2):
                theta = V.F32(pos) * np.float32(10000.0) ** np.float32(-2.0 * p / dim)
                c = np.cos(theta, dtype=np.float32)
                s = np.sin(theta, dtype=np.float32)
                a = x[t, h, 2 * p]
                b = x[t, h, 2 * p + 1]
                y[t, h, 2 * p] = a * c - b * s
                y[t, h, 2 * p + 1] = a * s + b * c
    return y.reshape(-1)
""",
))

add(case(
    "vllmKvCacheReshape", "vLLM KV cache reshape", "ai", "modern_ml", "medium",
    VLLM, "cache_kernels.cu",
    ["__global__", "tensor_layout", "cache_write", "strided_copy"],
    [512, 8, 64], 1e-6,
    "Token-major to head-major KV cache layout transform, adapted from vLLM cache kernels.",
    "Standalone KV cache reshape inspired by vLLM cache_kernels.cu.",
    r"""
__global__ void kv_reshape(const float *x, float *cache, int tokens, int heads, int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = tokens * heads * dim;
  if (idx < n) {
    int d = idx % dim;
    int tmp = idx / dim;
    int h = tmp % heads;
    int t = tmp / heads;
    int out = (h * tokens + t) * dim + d;
    cache[out] = x[idx];
  }
}

int main(int argc, char **argv) {
  const int tokens=512, heads=8, dim=64, n=tokens*heads*dim;
  const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) hx[i]=hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(n+tpb-1)/tpb; kv_reshape<<<grid,tpb>>>(dx,dy,tokens,heads,dim);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    tokens, heads, dim = meta["input"]["sizes"]
    x = V.gen_hashsigned(tokens * heads * dim, 123).reshape(tokens, heads, dim)
    return np.transpose(x, (1, 0, 2)).reshape(-1)
""",
))

add(case(
    "vllmPagedAttentionScore", "vLLM paged attention scores", "ai", "modern_ml", "hard",
    VLLM, "attention/attention_generic.cuh",
    ["__global__", "__shared__", "attention", "dot_product", "reduction"],
    [128, 64, 64], 1e-4,
    "Query-key dot scores for a paged attention tile, adapted from vLLM attention helpers.",
    "Standalone score tile simplification inspired by vLLM csrc/attention/attention_generic.cuh.",
    r"""
__global__ void attn_scores(const float *q, const float *k, float *scores, int queries, int keys, int dim) {
  int pair = blockIdx.x;
  int qi = pair / keys;
  int ki = pair % keys;
  extern __shared__ float s[];
  int tid = threadIdx.x;
  float sum = 0.0f;
  for (int d=tid; d<dim; d+=blockDim.x) sum += q[qi*dim+d] * k[ki*dim+d];
  s[tid] = sum; __syncthreads();
  for (int stride=blockDim.x/2; stride>0; stride>>=1) {
    if (tid < stride) s[tid] += s[tid+stride];
    __syncthreads();
  }
  if (tid == 0) scores[pair] = s[0] * rsqrtf((float)dim);
}

int main(int argc, char **argv) {
  const int queries=128, keys=64, dim=64, nq=queries*dim, nk=keys*dim, ns=queries*keys;
  const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hq=(float*)malloc((size_t)nq*sizeof(float)), *hk=(float*)malloc((size_t)nk*sizeof(float)), *hy=(float*)malloc((size_t)ns*sizeof(float));
  for (int i=0;i<nq;++i) hq[i]=hs(i,123);
  for (int i=0;i<nk;++i) hk[i]=hs(i,321);
  float *dq,*dk,*dy; CK(cudaMalloc(&dq,(size_t)nq*sizeof(float))); CK(cudaMalloc(&dk,(size_t)nk*sizeof(float))); CK(cudaMalloc(&dy,(size_t)ns*sizeof(float)));
  CK(cudaMemcpy(dq,hq,(size_t)nq*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dk,hk,(size_t)nk*sizeof(float),cudaMemcpyHostToDevice));
  attn_scores<<<ns,128,128*sizeof(float)>>>(dq,dk,dy,queries,keys,dim);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)ns*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,ns);
  cudaFree(dq); cudaFree(dk); cudaFree(dy); free(hq); free(hk); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    queries, keys, dim = meta["input"]["sizes"]
    q = V.gen_hashsigned(queries * dim, 123).reshape(queries, dim)
    k = V.gen_hashsigned(keys * dim, 321).reshape(keys, dim)
    return (q @ k.T * (V.F32(1.0) / np.sqrt(V.F32(dim), dtype=np.float32))).astype(np.float32).reshape(-1)
""",
))

add(case(
    "vllmLayernormResidual", "vLLM residual LayerNorm", "ai", "modern_ml", "hard",
    VLLM, "layernorm_kernels.cu",
    ["__global__", "__shared__", "layernorm", "mean_variance", "residual"],
    [256, 512], 1e-4,
    "Residual add plus LayerNorm, adapted from vLLM layernorm CUDA kernels.",
    "Standalone layernorm simplification from vLLM csrc/libtorch_stable/layernorm_kernels.cu.",
    r"""
__global__ void residual_layernorm(const float *x, const float *r, const float *gamma, const float *beta, float *y, int rows, int cols) {
  extern __shared__ float s[];
  int row=blockIdx.x, tid=threadIdx.x;
  float sum=0.0f, sum2=0.0f;
  for (int c=tid;c<cols;c+=blockDim.x) {
    float v=x[row*cols+c]+r[row*cols+c];
    sum += v; sum2 += v*v;
  }
  s[tid] = sum; s[tid + blockDim.x] = sum2; __syncthreads();
  for (int stride=blockDim.x/2; stride>0; stride>>=1) {
    if (tid < stride) { s[tid] += s[tid+stride]; s[tid+blockDim.x] += s[tid+blockDim.x+stride]; }
    __syncthreads();
  }
  float mean=s[0]/(float)cols;
  float var=s[blockDim.x]/(float)cols - mean*mean;
  float inv=rsqrtf(var + 1.0e-5f);
  for (int c=tid;c<cols;c+=blockDim.x) {
    float v=x[row*cols+c]+r[row*cols+c];
    y[row*cols+c] = (v - mean) * inv * gamma[c] + beta[c];
  }
}

int main(int argc, char **argv) {
  const int rows=256, cols=512, n=rows*cols;
  const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hr=(float*)malloc((size_t)n*sizeof(float)), *hg=(float*)malloc((size_t)cols*sizeof(float)), *hb=(float*)malloc((size_t)cols*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) { hx[i]=2.0f*hs(i,123); hr[i]=0.25f*hs(i,321); }
  for (int i=0;i<cols;++i) { hg[i]=1.0f+0.1f*hs(i,55); hb[i]=0.01f*hs(i,66); }
  float *dx,*dr,*dg,*db,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dr,(size_t)n*sizeof(float))); CK(cudaMalloc(&dg,(size_t)cols*sizeof(float))); CK(cudaMalloc(&db,(size_t)cols*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dr,hr,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg,hg,(size_t)cols*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(db,hb,(size_t)cols*sizeof(float),cudaMemcpyHostToDevice));
  residual_layernorm<<<rows,256,512*sizeof(float)>>>(dx,dr,dg,db,dy,rows,cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dr); cudaFree(dg); cudaFree(db); cudaFree(dy); free(hx); free(hr); free(hg); free(hb); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = (V.F32(2.0) * V.gen_hashsigned(rows * cols, 123)).reshape(rows, cols)
    r = (V.F32(0.25) * V.gen_hashsigned(rows * cols, 321)).reshape(rows, cols)
    g = V.F32(1.0) + V.F32(0.1) * V.gen_hashsigned(cols, 55)
    b = V.F32(0.01) * V.gen_hashsigned(cols, 66)
    v = x + r
    mean = np.mean(v, axis=1, dtype=np.float32)
    var = np.mean(v * v, axis=1, dtype=np.float32) - mean * mean
    return ((v - mean.reshape(rows, 1)) / np.sqrt(var.reshape(rows, 1) + V.F32(1.0e-5), dtype=np.float32) * g.reshape(1, cols) + b.reshape(1, cols)).reshape(-1)
""",
))

add(case(
    "vllmPerTokenInt8Quant", "vLLM per-token int8 quantization", "ai", "modern_ml", "hard",
    VLLM, "quantization/activation_kernels.cu",
    ["__global__", "__shared__", "per_token_scale", "quantization"],
    [512, 256], 1e-5,
    "Per-token int8 quantize/dequantize round trip, adapted from vLLM activation quantization kernels.",
    "Standalone per-row activation quantization inspired by vLLM csrc/libtorch_stable/quantization/activation_kernels.cu.",
    r"""
__global__ void per_token_q8(const float *x, float *y, int rows, int cols) {
  extern __shared__ float s[];
  int row=blockIdx.x, tid=threadIdx.x;
  float m=0.0f;
  for (int c=tid;c<cols;c+=blockDim.x) m=fmaxf(m, fabsf(x[row*cols+c]));
  s[tid]=m; __syncthreads();
  for (int stride=blockDim.x/2; stride>0; stride>>=1) {
    if (tid < stride) s[tid]=fmaxf(s[tid],s[tid+stride]);
    __syncthreads();
  }
  float scale=s[0]/127.0f + 1.0e-12f;
  for (int c=tid;c<cols;c+=blockDim.x) {
    int q=(int)lrintf(x[row*cols+c]/scale);
    q=max(-127,min(127,q));
    y[row*cols+c]=(float)q*scale;
  }
}

int main(int argc, char **argv) {
  const int rows=512, cols=256, n=rows*cols;
  const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) hx[i]=4.0f*hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  per_token_q8<<<rows,256,256*sizeof(float)>>>(dx,dy,rows,cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = (V.F32(4.0) * V.gen_hashsigned(rows * cols, 123)).reshape(rows, cols)
    scale = np.max(np.abs(x), axis=1).astype(np.float32) / V.F32(127.0) + V.F32(1.0e-12)
    q = np.rint(x / scale.reshape(rows, 1)).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.float32)
    return (q * scale.reshape(rows, 1)).reshape(-1)
""",
))

add(case(
    "bnbBlockwiseAbsmax", "bitsandbytes blockwise absmax", "ai", "modern_ml", "medium",
    BNB, "kernels.cu",
    ["__global__", "__shared__", "block_reduction", "quantization"],
    [4096, 64], 1e-5,
    "Blockwise absolute maximum reduction used for quantization scales, adapted from bitsandbytes kernels.",
    "Standalone absmax scale extraction inspired by bitsandbytes csrc/kernels.cu.",
    r"""
__global__ void block_absmax(const float *x, float *out, int blocks_n) {
  __shared__ float s[64];
  int b=blockIdx.x, t=threadIdx.x;
  float v=fabsf(x[b*64+t]);
  s[t]=v; __syncthreads();
  for (int stride=32; stride>0; stride>>=1) {
    if (t < stride) s[t]=fmaxf(s[t],s[t+stride]);
    __syncthreads();
  }
  if (t==0) out[b]=s[0];
}

int main(int argc, char **argv) {
  const int blocks_n=4096, block=64, n=blocks_n*block;
  const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)blocks_n*sizeof(float));
  for (int i=0;i<n;++i) hx[i]=7.0f*hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)blocks_n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  block_absmax<<<blocks_n,64>>>(dx,dy,blocks_n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)blocks_n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,blocks_n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    blocks, block = meta["input"]["sizes"]
    x = (V.F32(7.0) * V.gen_hashsigned(blocks * block, 123)).reshape(blocks, block)
    return np.max(np.abs(x), axis=1).astype(np.float32)
""",
))

add(case(
    "bnbInt8VectorQuant", "bitsandbytes int8 vector quantize", "ai", "modern_ml", "hard",
    BNB, "kernels.cu",
    ["__global__", "__shared__", "int8_quantization", "block_scale"],
    [4096, 64], 1e-5,
    "Blockwise int8 quantize/dequantize round trip, adapted from bitsandbytes quantization kernels.",
    "Standalone blockwise int8 quantization inspired by bitsandbytes csrc/kernels.cu.",
    r"""
__global__ void bnb_q8(const float *x, float *y, int blocks_n) {
  __shared__ float s[64];
  int b=blockIdx.x, t=threadIdx.x;
  float v=x[b*64+t];
  s[t]=fabsf(v); __syncthreads();
  for (int stride=32; stride>0; stride>>=1) { if (t<stride) s[t]=fmaxf(s[t],s[t+stride]); __syncthreads(); }
  float scale=s[0]/127.0f + 1.0e-12f;
  int q=(int)lrintf(v/scale); q=max(-127,min(127,q));
  y[b*64+t]=(float)q*scale;
}

int main(int argc, char **argv) {
  const int blocks_n=4096, block=64, n=blocks_n*block;
  const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) hx[i]=5.0f*hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  bnb_q8<<<blocks_n,64>>>(dx,dy,blocks_n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    blocks, block = meta["input"]["sizes"]
    x = (V.F32(5.0) * V.gen_hashsigned(blocks * block, 123)).reshape(blocks, block)
    scale = np.max(np.abs(x), axis=1).astype(np.float32) / V.F32(127.0) + V.F32(1.0e-12)
    q = np.rint(x / scale.reshape(blocks, 1)).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.float32)
    return (q * scale.reshape(blocks, 1)).reshape(-1)
""",
))

add(case(
    "bnbDequantizeBlockwise", "bitsandbytes blockwise dequantize", "ai", "modern_ml", "medium",
    BNB, "kernels.cu",
    ["__global__", "int8_t", "dequantization", "block_scale"],
    [4096, 64], 1e-5,
    "Blockwise int8 dequantization with one scale per block, adapted from bitsandbytes kernels.",
    "Standalone dequantization inspired by bitsandbytes csrc/kernels.cu.",
    r"""
__global__ void bnb_dequant(const int8_t *q, const float *scale, float *y, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) y[i]=(float)q[i]*scale[i/64];
}

int main(int argc, char **argv) {
  const int blocks_n=4096, block=64, n=blocks_n*block;
  const char *out=(argc>1)?argv[1]:"output/output.txt";
  int8_t *hq=(int8_t*)malloc((size_t)n); float *hscl=(float*)malloc((size_t)blocks_n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) hq[i]=(int8_t)lrintf(127.0f*hs(i,123));
  for (int b=0;b<blocks_n;++b) hscl[b]=0.01f+0.05f*h01(b,321);
  int8_t *dq; float *ds,*dy; CK(cudaMalloc(&dq,(size_t)n)); CK(cudaMalloc(&ds,(size_t)blocks_n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dq,hq,(size_t)n,cudaMemcpyHostToDevice)); CK(cudaMemcpy(ds,hscl,(size_t)blocks_n*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(n+tpb-1)/tpb; bnb_dequant<<<grid,tpb>>>(dq,ds,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dq); cudaFree(ds); cudaFree(dy); free(hq); free(hscl); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    blocks, block = meta["input"]["sizes"]
    n = blocks * block
    q = np.rint(V.F32(127.0) * V.gen_hashsigned(n, 123)).astype(np.int8).astype(np.float32)
    scale = V.F32(0.01) + V.F32(0.05) * V.gen_hash01(blocks, 321)
    return q * np.repeat(scale, block)
""",
))

add(case(
    "bnbAdam8bitMoments", "bitsandbytes Adam 8-bit moment update", "ai", "modern_ml", "hard",
    BNB, "ops.cu",
    ["__global__", "optimizer", "int8_dequantization", "sqrtf"],
    [262144], 1e-5,
    "Adam update using dequantized 8-bit moment estimates, adapted from bitsandbytes optimizer kernels.",
    "Standalone optimizer-step simplification inspired by bitsandbytes csrc/ops.cu and csrc/kernels.cu.",
    r"""
__global__ void adam8bit(const float *p, const float *g, const int8_t *mq, const int8_t *vq, float *out, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) {
    float m=(float)mq[i]*0.002f;
    float v=fabsf((float)vq[i])*0.0002f + 0.001f;
    float m2=0.9f*m + 0.1f*g[i];
    float v2=0.999f*v + 0.001f*g[i]*g[i];
    out[i]=p[i] - 0.001f*m2/(sqrtf(v2)+1.0e-8f);
  }
}

int main(int argc, char **argv) {
  const int n=262144; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hp=(float*)malloc((size_t)n*sizeof(float)), *hg=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  int8_t *hm=(int8_t*)malloc((size_t)n), *hv=(int8_t*)malloc((size_t)n);
  for (int i=0;i<n;++i) { hp[i]=hs(i,11); hg[i]=hs(i,22); hm[i]=(int8_t)lrintf(60.0f*hs(i,33)); hv[i]=(int8_t)lrintf(60.0f*hs(i,44)); }
  float *dp,*dg,*dy; int8_t *dm,*dv; CK(cudaMalloc(&dp,(size_t)n*sizeof(float))); CK(cudaMalloc(&dg,(size_t)n*sizeof(float))); CK(cudaMalloc(&dm,(size_t)n)); CK(cudaMalloc(&dv,(size_t)n)); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dp,hp,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg,hg,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dm,hm,(size_t)n,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv,hv,(size_t)n,cudaMemcpyHostToDevice));
  int tpb=256, grid=(n+tpb-1)/tpb; adam8bit<<<grid,tpb>>>(dp,dg,dm,dv,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dp); cudaFree(dg); cudaFree(dm); cudaFree(dv); cudaFree(dy); free(hp); free(hg); free(hm); free(hv); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    p = V.gen_hashsigned(n, 11)
    g = V.gen_hashsigned(n, 22)
    m = np.rint(V.F32(60.0) * V.gen_hashsigned(n, 33)).astype(np.int8).astype(np.float32) * V.F32(0.002)
    v = np.abs(np.rint(V.F32(60.0) * V.gen_hashsigned(n, 44)).astype(np.int8).astype(np.float32)) * V.F32(0.0002) + V.F32(0.001)
    m2 = V.F32(0.9) * m + V.F32(0.1) * g
    v2 = V.F32(0.999) * v + V.F32(0.001) * g * g
    return p - V.F32(0.001) * m2 / (np.sqrt(v2, dtype=np.float32) + V.F32(1.0e-8))
""",
))

add(case(
    "bnbPercentileClipping", "bitsandbytes gradient clipping", "ai", "modern_ml", "medium",
    BNB, "kernels.cu",
    ["__global__", "__shared__", "block_reduction", "clamp"],
    [4096, 64], 1e-6,
    "Blockwise gradient clipping using a scale derived from absmax, adapted from bitsandbytes optimizer helpers.",
    "Standalone percentile-style clipping simplification inspired by bitsandbytes csrc/kernels.cu.",
    r"""
__global__ void clip_blockwise(const float *g, float *y, int blocks_n) {
  __shared__ float s[64];
  int b=blockIdx.x, t=threadIdx.x;
  float v=g[b*64+t];
  s[t]=fabsf(v); __syncthreads();
  for (int stride=32; stride>0; stride>>=1) { if(t<stride) s[t]=fmaxf(s[t],s[t+stride]); __syncthreads(); }
  float thr=0.7f*s[0];
  y[b*64+t]=fminf(thr,fmaxf(-thr,v));
}

int main(int argc, char **argv) {
  const int blocks_n=4096, block=64, n=blocks_n*block; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hg=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for(int i=0;i<n;++i) hg[i]=8.0f*hs(i,123);
  float *dg,*dy; CK(cudaMalloc(&dg,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dg,hg,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  clip_blockwise<<<blocks_n,64>>>(dg,dy,blocks_n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dg); cudaFree(dy); free(hg); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    blocks, block = meta["input"]["sizes"]
    g = (V.F32(8.0) * V.gen_hashsigned(blocks * block, 123)).reshape(blocks, block)
    thr = V.F32(0.7) * np.max(np.abs(g), axis=1).astype(np.float32)
    return np.minimum(thr.reshape(blocks, 1), np.maximum(-thr.reshape(blocks, 1), g)).reshape(-1)
""",
))

add(case(
    "bnbL2NormBlockReduce", "bitsandbytes block L2 norm", "ai", "modern_ml", "medium",
    BNB, "kernels.cu",
    ["__global__", "__shared__", "block_reduction", "sqrtf"],
    [4096, 64], 1e-5,
    "Blockwise L2 norm reduction, adapted from bitsandbytes optimizer/statistics kernels.",
    "Standalone norm reduction inspired by bitsandbytes csrc/kernels.cu.",
    r"""
__global__ void l2_blocks(const float *x, float *y, int blocks_n) {
  __shared__ float s[64];
  int b=blockIdx.x, t=threadIdx.x;
  float v=x[b*64+t]; s[t]=v*v; __syncthreads();
  for (int stride=32; stride>0; stride>>=1) { if(t<stride) s[t]+=s[t+stride]; __syncthreads(); }
  if(t==0) y[b]=sqrtf(s[0]);
}

int main(int argc, char **argv) {
  const int blocks_n=4096, block=64, n=blocks_n*block; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)blocks_n*sizeof(float));
  for(int i=0;i<n;++i) hx[i]=hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)blocks_n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  l2_blocks<<<blocks_n,64>>>(dx,dy,blocks_n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)blocks_n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,blocks_n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    blocks, block = meta["input"]["sizes"]
    x = V.gen_hashsigned(blocks * block, 123).reshape(blocks, block)
    return np.sqrt(np.sum(x * x, axis=1, dtype=np.float32), dtype=np.float32)
""",
))

add(case(
    "flashOnlineSoftmax", "FlashAttention online softmax", "ai", "modern_ml", "hard",
    FLASH, "softmax.h",
    ["__global__", "online_softmax", "expf", "rowwise"],
    [512, 128], 1e-4,
    "Online row softmax recurrence, adapted from FlashAttention softmax helpers.",
    "Standalone online-softmax simplification inspired by Dao-AILab/flash-attention csrc/flash_attn/src/softmax.h.",
    r"""
__global__ void online_softmax(const float *x, float *y, int rows, int cols) {
  int row=blockIdx.x;
  float m=-3.402823e38f, l=0.0f;
  for(int c=0;c<cols;++c) {
    float v=x[row*cols+c];
    float nm=fmaxf(m,v);
    l = l * expf(m - nm) + expf(v - nm);
    m = nm;
  }
  for(int c=threadIdx.x;c<cols;c+=blockDim.x) y[row*cols+c]=expf(x[row*cols+c]-m)/l;
}

int main(int argc, char **argv) {
  const int rows=512, cols=128, n=rows*cols; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for(int i=0;i<n;++i) hx[i]=3.0f*hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  online_softmax<<<rows,128>>>(dx,dy,rows,cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = (V.F32(3.0) * V.gen_hashsigned(rows * cols, 123)).reshape(rows, cols)
    m = np.max(x, axis=1)
    e = np.exp(x - m.reshape(rows, 1), dtype=np.float32)
    return (e / np.sum(e, axis=1, dtype=np.float32).reshape(rows, 1)).reshape(-1)
""",
))

add(case(
    "flashCausalMaskSoftmax", "FlashAttention causal masked softmax", "ai", "modern_ml", "hard",
    FLASH, "mask.h",
    ["__global__", "__shared__", "causal_mask", "softmax", "reduction"],
    [512, 128], 1e-4,
    "Causal masked softmax over attention scores, adapted from FlashAttention mask and softmax helpers.",
    "Standalone masked softmax inspired by FlashAttention mask.h and softmax.h.",
    r"""
__global__ void flash_causal(const float *x, float *y, int rows, int cols) {
  int row=blockIdx.x;
  if(threadIdx.x==0){
    int limit=(row*7)%cols;
    float m=-3.402823e38f;
    for(int c=0;c<=limit;++c) m=fmaxf(m,x[row*cols+c]);
    float denom=0.0f;
    for(int c=0;c<=limit;++c) denom += expf(x[row*cols+c]-m);
    for(int c=0;c<cols;++c) y[row*cols+c]=(c<=limit)?expf(x[row*cols+c]-m)/denom:0.0f;
  }
}

int main(int argc, char **argv) {
  const int rows=512, cols=128, n=rows*cols; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for(int i=0;i<n;++i) hx[i]=3.0f*hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  flash_causal<<<rows,128>>>(dx,dy,rows,cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = (V.F32(3.0) * V.gen_hashsigned(rows * cols, 123)).reshape(rows, cols)
    y = np.zeros_like(x)
    for r in range(rows):
        limit = (r * 7) % cols
        vals = x[r, :limit + 1]
        m = np.max(vals)
        e = np.exp(vals - m, dtype=np.float32)
        y[r, :limit + 1] = e / np.sum(e, dtype=np.float32)
    return y.reshape(-1)
""",
))

add(case(
    "flashAttentionScoreValue", "FlashAttention score-value tile", "ai", "modern_ml", "hard",
    FLASH, "flash_fwd_kernel.h",
    ["__global__", "__shared__", "attention", "softmax", "value_accumulation"],
    [64, 64, 32], 2e-4,
    "Single-query attention tile computing softmax(QK^T)V, adapted from FlashAttention forward kernel structure.",
    "Standalone forward attention tile inspired by csrc/flash_attn/src/flash_fwd_kernel.h.",
    r"""
__global__ void attention_tile(const float *q, const float *k, const float *v, float *out, int queries, int keys, int dim) {
  __shared__ float score[64];
  int qi=blockIdx.x, tid=threadIdx.x;
  if(tid<keys){
    float s=0.0f;
    for(int d=0;d<dim;++d) s += q[qi*dim+d] * k[tid*dim+d];
    score[tid]=s*rsqrtf((float)dim);
  }
  __syncthreads();
  float m=-3.402823e38f;
  for(int i=0;i<keys;++i) m=fmaxf(m,score[i]);
  float denom=0.0f;
  for(int i=0;i<keys;++i){ score[i]=expf(score[i]-m); denom += score[i]; }
  if(tid<dim){
    float acc=0.0f;
    for(int kk=0;kk<keys;++kk) acc += (score[kk]/denom) * v[kk*dim+tid];
    out[qi*dim+tid]=acc;
  }
}

int main(int argc, char **argv) {
  const int queries=64, keys=64, dim=32, nq=queries*dim, nk=keys*dim, no=queries*dim;
  const char *outp=(argc>1)?argv[1]:"output/output.txt";
  float *hq=(float*)malloc((size_t)nq*sizeof(float)), *hk=(float*)malloc((size_t)nk*sizeof(float)), *hv=(float*)malloc((size_t)nk*sizeof(float)), *hy=(float*)malloc((size_t)no*sizeof(float));
  for(int i=0;i<nq;++i) hq[i]=hs(i,123);
  for(int i=0;i<nk;++i){ hk[i]=hs(i,321); hv[i]=hs(i,777); }
  float *dq,*dk,*dv,*dy; CK(cudaMalloc(&dq,(size_t)nq*sizeof(float))); CK(cudaMalloc(&dk,(size_t)nk*sizeof(float))); CK(cudaMalloc(&dv,(size_t)nk*sizeof(float))); CK(cudaMalloc(&dy,(size_t)no*sizeof(float)));
  CK(cudaMemcpy(dq,hq,(size_t)nq*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dk,hk,(size_t)nk*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv,hv,(size_t)nk*sizeof(float),cudaMemcpyHostToDevice));
  attention_tile<<<queries,64>>>(dq,dk,dv,dy,queries,keys,dim);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)no*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(outp,hy,no);
  cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dy); free(hq); free(hk); free(hv); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    queries, keys, dim = meta["input"]["sizes"]
    q = V.gen_hashsigned(queries * dim, 123).reshape(queries, dim)
    k = V.gen_hashsigned(keys * dim, 321).reshape(keys, dim)
    v = V.gen_hashsigned(keys * dim, 777).reshape(keys, dim)
    scores = (q @ k.T) * (V.F32(1.0) / np.sqrt(V.F32(dim), dtype=np.float32))
    m = np.max(scores, axis=1)
    e = np.exp(scores - m.reshape(queries, 1), dtype=np.float32)
    p = e / np.sum(e, axis=1, dtype=np.float32).reshape(queries, 1)
    return (p @ v).astype(np.float32).reshape(-1)
""",
))

add(case(
    "flashDropoutMaskApply", "FlashAttention dropout mask apply", "ai", "modern_ml", "medium",
    FLASH, "dropout.h",
    ["__global__", "dropout", "masking", "hash_rng"],
    [1048576], 1e-6,
    "Deterministic dropout mask application with scale compensation, adapted from FlashAttention dropout helpers.",
    "Standalone dropout/mask helper inspired by csrc/flash_attn/src/dropout.h.",
    elemwise_main("dropout_apply", "float keep = h01((unsigned)i, 999u) > 0.1f ? 1.0f : 0.0f; y[i] = x[i] * keep / 0.9f;", 1048576, 2.0, 1.0),
    r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.F32(2.0) * V.gen_hashsigned(n, 123)
    keep = (V.gen_hash01(n, 999) > V.F32(0.1)).astype(np.float32)
    return x * keep / V.F32(0.9)
""",
))

add(case(
    "xformersSwiGLU", "xFormers SwiGLU activation", "ai", "modern_ml", "medium",
    XFORMERS, "xformers/ops/swiglu_op.py",
    ["__global__", "expf", "gated_activation"],
    [1048576], 1e-5,
    "SwiGLU-style gated activation, adapted from xFormers SwiGLU operator behavior.",
    "Standalone CUDA version of the xFormers SwiGLU operation shape.",
    elemwise_main("swiglu_kernel", "float v=x[i]; y[i] = (v / (1.0f + expf(-v))) * g[i];", 1048576, 6.0, 2.0),
    r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.F32(6.0) * V.gen_hashsigned(n, 123)
    g = V.F32(2.0) * V.gen_hashsigned(n, 321)
    return (x / (V.F32(1.0) + np.exp(-x, dtype=np.float32))) * g
""",
))

add(case(
    "xformersSparse24Mask", "xFormers 2:4 sparse mask", "ai", "modern_ml", "hard",
    XFORMERS, "xformers/csrc/sparse24/sparse24_largest_mask_2d.cu",
    ["__global__", "structured_sparsity", "top2", "branching"],
    [65536, 4], 1e-6,
    "Apply a 2:4 structured sparsity mask retaining the two largest magnitudes per group, adapted from xFormers sparse24 kernels.",
    "Standalone structured sparsity mask inspired by xformers/csrc/sparse24/sparse24_largest_mask_2d.cu.",
    r"""
__global__ void sparse24_mask(const float *x, float *y, int groups) {
  int g=blockIdx.x*blockDim.x+threadIdx.x;
  if(g<groups){
    float v0=x[4*g+0], v1=x[4*g+1], v2=x[4*g+2], v3=x[4*g+3];
    float a0=fabsf(v0), a1=fabsf(v1), a2=fabsf(v2), a3=fabsf(v3);
    int m0=0, m1=1; float b0=a0, b1=a1;
    if(b1>b0){float tb=b0; b0=b1; b1=tb; m0=1; m1=0;}
    if(a2>b0){b1=b0; m1=m0; b0=a2; m0=2;} else if(a2>b1){b1=a2; m1=2;}
    if(a3>b0){b1=b0; m1=m0; b0=a3; m0=3;} else if(a3>b1){b1=a3; m1=3;}
    y[4*g+0]=(m0==0||m1==0)?v0:0.0f;
    y[4*g+1]=(m0==1||m1==1)?v1:0.0f;
    y[4*g+2]=(m0==2||m1==2)?v2:0.0f;
    y[4*g+3]=(m0==3||m1==3)?v3:0.0f;
  }
}

int main(int argc, char **argv) {
  const int groups=65536, n=groups*4; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for(int i=0;i<n;++i) hx[i]=hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(groups+tpb-1)/tpb; sparse24_mask<<<grid,tpb>>>(dx,dy,groups);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    groups, width = meta["input"]["sizes"]
    x = V.gen_hashsigned(groups * width, 123).reshape(groups, width)
    order = np.argsort(-np.abs(x), axis=1)
    keep = np.zeros_like(x)
    rows = np.arange(groups)
    keep[rows, order[:, 0]] = V.F32(1.0)
    keep[rows, order[:, 1]] = V.F32(1.0)
    return (x * keep).reshape(-1)
""",
))

add(case(
    "hecbenchBfsFrontier", "HeCBench BFS frontier step", "hpc", "hpc", "hard",
    HECBENCH, "bfs-cuda/bfs.cu",
    ["__global__", "graph", "frontier", "atomicMin"],
    [4096, 3], 0.0,
    "Single BFS frontier relaxation over a deterministic sparse graph, adapted from HeCBench/Rodinia BFS.",
    "Standalone graph frontier step inspired by ORNL/HeCBench src/bfs-cuda/bfs.cu.",
    r"""
__global__ void bfs_step(const int *src, const int *dst, const int *frontier, int *dist, int *next, int edges) {
  int e=blockIdx.x*blockDim.x+threadIdx.x;
  if(e<edges){
    int u=src[e], v=dst[e];
    if(frontier[u]){
      int old=atomicMin(&dist[v], dist[u]+1);
      if(old>dist[u]+1) next[v]=1;
    }
  }
}

int main(int argc, char **argv) {
  const int nodes=4096, deg=3, edges=nodes*deg; const char *out=(argc>1)?argv[1]:"output/output.txt";
  int *hs=(int*)malloc((size_t)edges*sizeof(int)), *hd=(int*)malloc((size_t)edges*sizeof(int)), *hf=(int*)calloc(nodes,sizeof(int)), *hdis=(int*)malloc((size_t)nodes*sizeof(int)), *hn=(int*)calloc(nodes,sizeof(int));
  for(int i=0;i<nodes;++i){ hdis[i]=1000000; if(i%97==0){hf[i]=1; hdis[i]=0;} }
  for(int i=0;i<nodes;++i){ hs[3*i]=i; hd[3*i]=(i+1)%nodes; hs[3*i+1]=i; hd[3*i+1]=(i+17)%nodes; hs[3*i+2]=i; hd[3*i+2]=(i*13+7)%nodes; }
  int *ds,*dd,*df,*ddi,*dn; CK(cudaMalloc(&ds,(size_t)edges*sizeof(int))); CK(cudaMalloc(&dd,(size_t)edges*sizeof(int))); CK(cudaMalloc(&df,(size_t)nodes*sizeof(int))); CK(cudaMalloc(&ddi,(size_t)nodes*sizeof(int))); CK(cudaMalloc(&dn,(size_t)nodes*sizeof(int)));
  CK(cudaMemcpy(ds,hs,(size_t)edges*sizeof(int),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dd,hd,(size_t)edges*sizeof(int),cudaMemcpyHostToDevice)); CK(cudaMemcpy(df,hf,(size_t)nodes*sizeof(int),cudaMemcpyHostToDevice)); CK(cudaMemcpy(ddi,hdis,(size_t)nodes*sizeof(int),cudaMemcpyHostToDevice)); CK(cudaMemset(dn,0,(size_t)nodes*sizeof(int)));
  int tpb=256, grid=(edges+tpb-1)/tpb; bfs_step<<<grid,tpb>>>(ds,dd,df,ddi,dn,edges);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hdis,ddi,(size_t)nodes*sizeof(int),cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hn,dn,(size_t)nodes*sizeof(int),cudaMemcpyDeviceToHost));
  float *outv=(float*)malloc((size_t)nodes*sizeof(float)); for(int i=0;i<nodes;++i) outv[i]=(float)(hdis[i]<1000000?hdis[i]:-1) + 0.01f*(float)hn[i];
  write_vec(out,outv,nodes);
  cudaFree(ds); cudaFree(dd); cudaFree(df); cudaFree(ddi); cudaFree(dn); free(hs); free(hd); free(hf); free(hdis); free(hn); free(outv); return 0;
}
""",
    r"""
def reference(meta):
    nodes, deg = meta["input"]["sizes"]
    dist = np.full(nodes, 1000000, dtype=np.int32)
    frontier = np.zeros(nodes, dtype=np.int32)
    for i in range(nodes):
        if i % 97 == 0:
            frontier[i] = 1
            dist[i] = 0
    nextf = np.zeros(nodes, dtype=np.int32)
    for i in range(nodes):
        dests = [(i + 1) % nodes, (i + 17) % nodes, (i * 13 + 7) % nodes]
        if frontier[i]:
            for v in dests:
                if dist[v] > dist[i] + 1:
                    dist[v] = dist[i] + 1
                    nextf[v] = 1
    out = np.where(dist < 1000000, dist, -1).astype(np.float32) + V.F32(0.01) * nextf.astype(np.float32)
    return out
""",
))

add(case(
    "hecbenchBackpropForward", "HeCBench backprop layer forward", "hpc", "hpc", "medium",
    HECBENCH, "backprop-cuda/bpnn_layerforward.h",
    ["__global__", "matrix_vector", "sigmoid"],
    [512, 256], 1e-4,
    "Neural-network layer forward pass from the HeCBench/Rodinia backprop benchmark.",
    "Standalone matrix-vector sigmoid layer inspired by ORNL/HeCBench src/backprop-cuda.",
    r"""
__global__ void layer_forward(const float *input, const float *weights, float *out, int in_n, int hid_n) {
  int h=blockIdx.x*blockDim.x+threadIdx.x;
  if(h<hid_n){
    float sum=weights[h*(in_n+1)];
    for(int i=0;i<in_n;++i) sum += input[i]*weights[h*(in_n+1)+i+1];
    out[h]=1.0f/(1.0f+expf(-sum));
  }
}

int main(int argc, char **argv) {
  const int in_n=512, hid_n=256; const char *outp=(argc>1)?argv[1]:"output/output.txt";
  float *hi=(float*)malloc((size_t)in_n*sizeof(float)), *hw=(float*)malloc((size_t)hid_n*(in_n+1)*sizeof(float)), *hy=(float*)malloc((size_t)hid_n*sizeof(float));
  for(int i=0;i<in_n;++i) hi[i]=hs(i,123);
  for(int i=0;i<hid_n*(in_n+1);++i) hw[i]=0.01f*hs(i,321);
  float *di,*dw,*dy; CK(cudaMalloc(&di,(size_t)in_n*sizeof(float))); CK(cudaMalloc(&dw,(size_t)hid_n*(in_n+1)*sizeof(float))); CK(cudaMalloc(&dy,(size_t)hid_n*sizeof(float)));
  CK(cudaMemcpy(di,hi,(size_t)in_n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dw,hw,(size_t)hid_n*(in_n+1)*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=128, grid=(hid_n+tpb-1)/tpb; layer_forward<<<grid,tpb>>>(di,dw,dy,in_n,hid_n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)hid_n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(outp,hy,hid_n);
  cudaFree(di); cudaFree(dw); cudaFree(dy); free(hi); free(hw); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    in_n, hid_n = meta["input"]["sizes"]
    x = V.gen_hashsigned(in_n, 123)
    w = (V.F32(0.01) * V.gen_hashsigned(hid_n * (in_n + 1), 321)).reshape(hid_n, in_n + 1)
    s = w[:, 0] + w[:, 1:] @ x
    return V.F32(1.0) / (V.F32(1.0) + np.exp(-s, dtype=np.float32))
""",
))

add(case(
    "hecbenchFdtd3dStep", "HeCBench FDTD 3D step", "hpc", "hpc", "hard",
    HECBENCH, "fdtd3d-cuda/main.cu",
    ["__global__", "3D_indexing", "stencil", "time_step"],
    [48, 48, 32], 1e-5,
    "Single 3D FDTD-style stencil update, adapted from HeCBench FDTD3D.",
    "Standalone 3D stencil inspired by ORNL/HeCBench src/fdtd3d-cuda/main.cu.",
    r"""
__global__ void fdtd3d(const float *x, float *y, int nx, int ny, int nz) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int n=nx*ny*nz;
  if(idx<n){
    int k=idx%nz; int j=(idx/nz)%ny; int i=idx/(ny*nz);
    int im=max(i-1,0), jm=max(j-1,0), km=max(k-1,0);
    y[idx]=x[idx]+0.1f*(x[(im*ny+j)*nz+k]+x[(i*ny+jm)*nz+k]+x[(i*ny+j)*nz+km]-3.0f*x[idx]);
  }
}

int main(int argc, char **argv) {
  const int nx=48, ny=48, nz=32, n=nx*ny*nz; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for(int i=0;i<n;++i) hx[i]=hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(n+tpb-1)/tpb; fdtd3d<<<grid,tpb>>>(dx,dy,nx,ny,nz);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    nx, ny, nz = meta["input"]["sizes"]
    x = V.gen_hashsigned(nx * ny * nz, 123).reshape(nx, ny, nz)
    y = np.empty_like(x)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                y[i, j, k] = x[i, j, k] + V.F32(0.1) * (x[max(i-1,0), j, k] + x[i, max(j-1,0), k] + x[i, j, max(k-1,0)] - V.F32(3.0) * x[i, j, k])
    return y.reshape(-1)
""",
))

add(case(
    "hecbenchConvolution3d", "HeCBench 3D convolution", "hpc", "hpc", "hard",
    HECBENCH, "convolution3D-cuda/main.cu",
    ["__global__", "3D_indexing", "convolution", "stencil"],
    [32, 32, 32], 1e-5,
    "3D 7-point convolution/stencil, adapted from HeCBench convolution3D.",
    "Standalone 3D convolution inspired by ORNL/HeCBench src/convolution3D-cuda/main.cu.",
    r"""
__global__ void conv3d7(const float *x, float *y, int n) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int total=n*n*n;
  if(idx<total){
    int k=idx%n; int j=(idx/n)%n; int i=idx/(n*n);
    int im=max(i-1,0), ip=min(i+1,n-1), jm=max(j-1,0), jp=min(j+1,n-1), km=max(k-1,0), kp=min(k+1,n-1);
    y[idx]=0.4f*x[idx]+0.1f*(x[(im*n+j)*n+k]+x[(ip*n+j)*n+k]+x[(i*n+jm)*n+k]+x[(i*n+jp)*n+k]+x[(i*n+j)*n+km]+x[(i*n+j)*n+kp]);
  }
}

int main(int argc, char **argv) {
  const int n=32, total=n*n*n; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)total*sizeof(float)), *hy=(float*)malloc((size_t)total*sizeof(float));
  for(int i=0;i<total;++i) hx[i]=hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)total*sizeof(float))); CK(cudaMalloc(&dy,(size_t)total*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)total*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(total+tpb-1)/tpb; conv3d7<<<grid,tpb>>>(dx,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)total*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,total);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n * n * n, 123).reshape(n, n, n)
    y = np.empty_like(x)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                y[i,j,k] = V.F32(0.4)*x[i,j,k] + V.F32(0.1)*(x[max(i-1,0),j,k]+x[min(i+1,n-1),j,k]+x[i,max(j-1,0),k]+x[i,min(j+1,n-1),k]+x[i,j,max(k-1,0)]+x[i,j,min(k+1,n-1)])
    return y.reshape(-1)
""",
))

add(case(
    "hecbenchBitonicPass", "HeCBench bitonic sort pass", "hpc", "hpc", "medium",
    HECBENCH, "bitonic-sort-cuda/main.cu",
    ["__global__", "compare_swap", "branching"],
    [262144], 0.0,
    "One bitonic compare-swap pass, adapted from HeCBench bitonic-sort.",
    "Standalone compare-swap pass inspired by ORNL/HeCBench src/bitonic-sort-cuda/main.cu.",
    r"""
__global__ void bitonic_pass(const float *x, float *y, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){
    int ixj=i^1;
    float a=x[i], b=x[ixj];
    bool up=((i & 2)==0);
    if((up && a>b) || (!up && a<b)) y[i]=b; else y[i]=a;
  }
}

int main(int argc, char **argv) {
  const int n=262144; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for(int i=0;i<n;++i) hx[i]=hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(n+tpb-1)/tpb; bitonic_pass<<<grid,tpb>>>(dx,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123)
    y = np.empty_like(x)
    for i in range(n):
        ixj = i ^ 1
        up = (i & 2) == 0
        a = x[i]
        b = x[ixj]
        y[i] = b if ((up and a > b) or ((not up) and a < b)) else a
    return y
""",
))

add(case(
    "hecbenchCsrSpmv", "HeCBench CSR SpMV", "hpc", "hpc", "hard",
    HECBENCH, "amgmk-cuda/csr_matvec.cu",
    ["__global__", "csr", "sparse_matrix", "irregular_memory"],
    [4096, 7], 1e-5,
    "CSR sparse matrix-vector multiply over a deterministic 7-point row pattern, adapted from HeCBench AMG/matvec kernels.",
    "Standalone CSR SpMV inspired by ORNL/HeCBench src/amgmk-cuda/csr_matvec.cu.",
    r"""
__global__ void csr_spmv(const int *row, const int *col, const float *val, const float *x, float *y, int rows) {
  int r=blockIdx.x*blockDim.x+threadIdx.x;
  if(r<rows){
    float sum=0.0f;
    for(int p=row[r]; p<row[r+1]; ++p) sum += val[p] * x[col[p]];
    y[r]=sum;
  }
}

int main(int argc, char **argv) {
  const int rows=4096, nnz_per=7, nnz=rows*nnz_per; const char *out=(argc>1)?argv[1]:"output/output.txt";
  int *hr=(int*)malloc((size_t)(rows+1)*sizeof(int)), *hc=(int*)malloc((size_t)nnz*sizeof(int));
  float *hv=(float*)malloc((size_t)nnz*sizeof(float)), *hx=(float*)malloc((size_t)rows*sizeof(float)), *hy=(float*)malloc((size_t)rows*sizeof(float));
  for(int r=0;r<=rows;++r) hr[r]=r*nnz_per;
  for(int r=0;r<rows;++r){ for(int j=0;j<nnz_per;++j){ int p=r*nnz_per+j; hc[p]=(r+j*13+rows-39)%rows; hv[p]=0.1f+0.01f*(float)j; } hx[r]=hs(r,123); }
  int *dr,*dc; float *dv,*dx,*dy; CK(cudaMalloc(&dr,(size_t)(rows+1)*sizeof(int))); CK(cudaMalloc(&dc,(size_t)nnz*sizeof(int))); CK(cudaMalloc(&dv,(size_t)nnz*sizeof(float))); CK(cudaMalloc(&dx,(size_t)rows*sizeof(float))); CK(cudaMalloc(&dy,(size_t)rows*sizeof(float)));
  CK(cudaMemcpy(dr,hr,(size_t)(rows+1)*sizeof(int),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dc,hc,(size_t)nnz*sizeof(int),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv,hv,(size_t)nnz*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx,hx,(size_t)rows*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(rows+tpb-1)/tpb; csr_spmv<<<grid,tpb>>>(dr,dc,dv,dx,dy,rows);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)rows*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,rows);
  cudaFree(dr); cudaFree(dc); cudaFree(dv); cudaFree(dx); cudaFree(dy); free(hr); free(hc); free(hv); free(hx); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    rows, nnz_per = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows, 123)
    y = np.zeros(rows, dtype=np.float32)
    vals = V.F32(0.1) + V.F32(0.01) * np.arange(nnz_per, dtype=np.float32)
    for r in range(rows):
        for j in range(nnz_per):
            c = (r + j * 13 + rows - 39) % rows
            y[r] += vals[j] * x[c]
    return y
""",
))

add(case(
    "hecbenchBabelStreamTriad", "HeCBench BabelStream triad", "hpc", "hpc", "medium",
    HECBENCH, "babelstream-cuda/main.cu",
    ["__global__", "streaming_memory", "triad"],
    [1048576], 1e-6,
    "STREAM triad memory-bandwidth kernel, adapted from HeCBench BabelStream.",
    "Standalone triad kernel inspired by ORNL/HeCBench src/babelstream-cuda/main.cu.",
    elemwise_main("triad_kernel", "y[i] = x[i] + 3.0f * g[i];", 1048576, 1.0, 1.0),
    r"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123)
    g = V.gen_hashsigned(n, 321)
    return x + V.F32(3.0) * g
""",
))

add(case(
    "hecbenchD2q9BgkStep", "HeCBench D2Q9 BGK step", "hpc", "hpc", "hard",
    HECBENCH, "d2q9-bgk-cuda/main.cu",
    ["__global__", "2D_indexing", "lattice_boltzmann", "multi_array"],
    [128, 128, 9], 1e-5,
    "One simplified D2Q9 BGK lattice update, adapted from HeCBench D2Q9 BGK.",
    "Standalone lattice update inspired by ORNL/HeCBench src/d2q9-bgk-cuda/main.cu.",
    r"""
__global__ void d2q9_step(const float *f, float *rho_out, int h, int w) {
  int cell=blockIdx.x*blockDim.x+threadIdx.x;
  int cells=h*w;
  if(cell<cells){
    float rho=0.0f;
    for(int q=0;q<9;++q) rho += f[q*cells + cell];
    float ux=(f[1*cells+cell]-f[3*cells+cell]+f[5*cells+cell]-f[6*cells+cell]-f[7*cells+cell]+f[8*cells+cell])/rho;
    float uy=(f[2*cells+cell]-f[4*cells+cell]+f[5*cells+cell]+f[6*cells+cell]-f[7*cells+cell]-f[8*cells+cell])/rho;
    rho_out[cell]=rho + 0.01f*(ux+uy);
  }
}

int main(int argc, char **argv) {
  const int h=128, w=128, q=9, cells=h*w, n=q*cells; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hf=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)cells*sizeof(float));
  for(int i=0;i<n;++i) hf[i]=0.2f+0.01f*h01(i,123);
  float *df,*dy; CK(cudaMalloc(&df,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)cells*sizeof(float)));
  CK(cudaMemcpy(df,hf,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(cells+tpb-1)/tpb; d2q9_step<<<grid,tpb>>>(df,dy,h,w);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)cells*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,cells);
  cudaFree(df); cudaFree(dy); free(hf); free(hy); return 0;
}
""",
    r"""
def reference(meta):
    h, w, q = meta["input"]["sizes"]
    cells = h * w
    f = (V.F32(0.2) + V.F32(0.01) * V.gen_hash01(q * cells, 123)).reshape(q, cells)
    rho = np.sum(f, axis=0, dtype=np.float32)
    ux = (f[1] - f[3] + f[5] - f[6] - f[7] + f[8]) / rho
    uy = (f[2] - f[4] + f[5] + f[6] - f[7] - f[8]) / rho
    return rho + V.F32(0.01) * (ux + uy)
""",
))


def write_case(case_id: str, spec: dict) -> None:
    category = spec["category"]
    case_dir = CASES_ROOT / category / case_id
    for rel in ("original", "tests", "input", "expected", "output", "logs", "migrated"):
        (case_dir / rel).mkdir(parents=True, exist_ok=True)

    meta = spec["metadata"]
    (case_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (case_dir / "README.md").write_text(
        f"# {meta['name']}\n\n"
        f"{meta['description']}\n\n"
        f"Source project: {meta['source_project']}\n\n"
        f"Source URL: {meta['source_url']}\n\n"
        f"License: {meta['license']}\n\n"
        f"Extraction notes: {meta['notes']}\n",
        encoding="utf-8",
    )
    (case_dir / "original" / "README.md").write_text(
        f"# Original CUDA\n\n"
        f"Standalone Stage 1 CUDA extraction/adaptation for `{case_id}`.\n"
        f"The source pattern is attributed in metadata.json.\n",
        encoding="utf-8",
    )
    (case_dir / "original" / "CMakeLists.txt").write_text(CMAKELISTS, encoding="utf-8")
    (case_dir / "original" / "main.cu").write_text(
        COMMON_CUDA + "\n\n" + spec["main"].strip() + "\n",
        encoding="utf-8",
    )
    (case_dir / "tests" / "verify.py").write_text(
        VERIFY_PREFIX + spec["verify"].strip() + "\n\nif __name__ == \"__main__\":\n    V.run(reference)\n",
        encoding="utf-8",
    )


def main() -> int:
    for case_id in sorted(CASES):
        write_case(case_id, CASES[case_id])
        meta = CASES[case_id]["metadata"]
        print(f"[ok] {meta['category']}/{case_id} <- {meta['source_project']}")
    print(f"Wrote {len(CASES)} real-project Stage 1 cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
