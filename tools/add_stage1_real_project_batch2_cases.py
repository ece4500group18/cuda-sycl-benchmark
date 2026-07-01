#!/usr/bin/env python3
"""Add Stage 1 real-project batch 2 cases, covering cases 131-172.

This batch is intentionally generated from compact templates so the case
structure stays consistent while each case records its own source project,
URL, license, extraction fidelity, and CPU-reference verifier.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = ROOT / "pilot_benchmark" / "cases"

SOURCES = {
    "llama": {
        "project": "ggml-org/llama.cpp",
        "license": "MIT",
        "base": "https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda",
    },
    "vllm": {
        "project": "vllm-project/vllm",
        "license": "Apache-2.0",
        "base": "https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable",
    },
    "vllm_csrc": {
        "project": "vllm-project/vllm",
        "license": "Apache-2.0",
        "base": "https://github.com/vllm-project/vllm/blob/main/csrc",
    },
    "bnb": {
        "project": "bitsandbytes-foundation/bitsandbytes",
        "license": "MIT",
        "base": "https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/csrc",
    },
    "flash": {
        "project": "Dao-AILab/flash-attention",
        "license": "BSD-3-Clause",
        "base": "https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src",
    },
    "cutlass": {
        "project": "NVIDIA/cutlass",
        "license": "BSD-3-Clause",
        "base": "https://github.com/NVIDIA/cutlass/blob/main/examples",
    },
    "dali": {
        "project": "NVIDIA/DALI",
        "license": "Apache-2.0",
        "base": "https://github.com/NVIDIA/DALI/blob/main",
    },
    "hecbench": {
        "project": "ORNL/HeCBench",
        "license": "BSD-3-Clause",
        "base": "https://github.com/ORNL/HeCBench/blob/master/src",
    },
    "cuda_samples": {
        "project": "NVIDIA/cuda-samples",
        "license": "BSD-3-Clause + CUDA EULA note",
        "base": "https://github.com/NVIDIA/cuda-samples/blob/master",
    },
}

COMMON_CUDA = r"""
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 2; } } while (0)

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
project(stage1_real_project_batch2 CUDA)
add_executable(app main.cu)
set_target_properties(app PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
"""


def url(source_key: str, rel: str) -> str:
    return f"{SOURCES[source_key]['base']}/{rel}"


def metadata(spec: dict) -> dict:
    src = SOURCES[spec["source"]]
    source_url = url(spec["source"], spec["rel"])
    return {
        "case_id": spec["id"],
        "name": spec["name"],
        "category": spec["category"],
        "domain": spec["domain"],
        "difficulty": spec["difficulty"],
        "source_project": src["project"],
        "source_url": source_url,
        "license": src["license"],
        "adaptation_type": spec.get("fidelity", "inspired_by"),
        "extraction_fidelity": spec.get("fidelity", "inspired_by"),
        "extraction_notes": spec["notes"],
        "description": spec["description"],
        "source": {
            "type": spec.get("fidelity", "inspired_by"),
            "url": source_url,
            "license": src["license"],
            "original_path": spec["rel"],
        },
        "cuda_features": spec["features"],
        "libraries": spec.get("libraries", []),
        "input": {"type": "hashed", "sizes": spec["sizes"], "seed": 123},
        "build": {
            "cuda_build_command": spec.get("build", "nvcc -O2 -std=c++17 original/main.cu -o original/build/app"),
            "sycl_build_command": "icpx -fsycl -O2 -std=c++17 syclomatic/main.dp.cpp -o build_sycl/app",
        },
        "run": {
            "cuda_run_command": "original/build/app output/cuda_output.txt",
            "sycl_run_command": "build_sycl/app output/sycl_output.txt",
        },
        "correctness": {
            "method": "cpu_reference",
            "metric": "max_abs_error",
            "tolerance": spec["tol"],
            "expected_pass_string": "PASS",
        },
        "syclomatic": {"status": "not_attempted", "command": "", "warnings_count": None, "manual_fixes_required": None},
        "status": {
            "cuda_compile": "not_attempted",
            "cuda_run": "not_attempted",
            "cuda_verify": "not_attempted",
            "syclomatic_migrate": "not_attempted",
            "sycl_compile": "not_attempted",
            "sycl_run": "not_attempted",
            "sycl_verify": "not_attempted",
        },
        "notes": spec["notes"],
    }


def elemwise_main(kernel: str, expr: str, n: int, xscale: float = 4.0, gscale: float = 2.0) -> str:
    return f"""
__global__ void {kernel}(const float *x, const float *g, float *y, int n) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {{
    {expr}
  }}
}}
int main(int argc, char **argv) {{
  const int n = {n};
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hg=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) {{ hx[i] = {xscale}f * hs(i, 123); hg[i] = {gscale}f * hs(i, 321); }}
  float *dx,*dg,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dg,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg,hg,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(n+tpb-1)/tpb; {kernel}<<<grid,tpb>>>(dx,dg,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dg); cudaFree(dy); free(hx); free(hg); free(hy); return 0;
}}
"""


def elemwise_ref(expr: str) -> str:
    return f"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.F32(meta.get("xscale", 1.0)) * V.gen_hashsigned(n, 123)
    g = V.F32(meta.get("gscale", 1.0)) * V.gen_hashsigned(n, 321)
    return {expr}
"""


def add_case(cases: list[dict], spec: dict, main: str, verify: str) -> None:
    spec = dict(spec)
    spec["main"] = main
    spec["verify"] = verify
    cases.append(spec)


CASES: list[dict] = []

# Modern ML / LLM / attention / quantization, mostly from prior source families.
add_case(CASES, {
    "id": "ggmlClampKernel", "name": "ggml clamp activation", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "llama", "rel": "clamp.cu", "fidelity": "simplified", "sizes": [1048576], "tol": 1e-6,
    "features": ["__global__", "branching", "elementwise"],
    "description": "Clamp tensor values to a fixed interval, following ggml CUDA clamp kernels.",
    "notes": "Simplified standalone clamp extracted from the llama.cpp/ggml clamp.cu operator pattern.",
}, elemwise_main("clamp_kernel", "float v=x[i]; y[i]=fminf(1.25f,fmaxf(-1.25f,v));", 1048576, 4.0, 1.0),
"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.F32(4.0) * V.gen_hashsigned(n, 123)
    return np.minimum(V.F32(1.25), np.maximum(V.F32(-1.25), x))
""")

add_case(CASES, {
    "id": "ggmlDiagMask", "name": "ggml diagonal mask", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "llama", "rel": "diagmask.cu", "fidelity": "simplified", "sizes": [512, 128], "tol": 1e-6,
    "features": ["__global__", "2D_indexing", "masking"],
    "description": "Apply an upper-triangular attention-style diagonal mask.",
    "notes": "Simplified standalone diagonal masking inspired by llama.cpp/ggml diagmask.cu.",
}, r"""
__global__ void diag_mask(const float *x, float *y, int rows, int cols) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x, n=rows*cols;
  if(idx<n){ int r=idx/cols, c=idx%cols; y[idx]=(c > (r%cols)) ? -10000.0f : x[idx]; }
}
int main(int argc, char **argv) {
  const int rows=512, cols=128, n=rows*cols; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for(int i=0;i<n;++i) hx[i]=hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(n+tpb-1)/tpb; diag_mask<<<grid,tpb>>>(dx,dy,rows,cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out,hy,n); cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    y = x.copy()
    for r in range(rows):
        y[r, np.arange(cols) > (r % cols)] = V.F32(-10000.0)
    return y.reshape(-1)
""")

add_case(CASES, {
    "id": "ggmlArgmaxRows", "name": "ggml row argmax", "category": "ai", "domain": "modern_ml", "difficulty": "hard",
    "source": "llama", "rel": "argmax.cu", "fidelity": "simplified", "sizes": [1024, 128], "tol": 0.0,
    "features": ["__global__", "__shared__", "reduction", "argmax"],
    "description": "Compute per-row argmax indices for logits.",
    "notes": "Standalone row argmax following llama.cpp/ggml argmax.cu reduction behavior.",
}, r"""
__global__ void argmax_rows(const float *x, float *y, int rows, int cols) {
  extern __shared__ float s[];
  int row=blockIdx.x, tid=threadIdx.x;
  float best=-3.402823e38f; int bi=0;
  for(int c=tid;c<cols;c+=blockDim.x){ float v=x[row*cols+c]; if(v>best){best=v; bi=c;} }
  s[tid]=best; s[tid+blockDim.x]=(float)bi; __syncthreads();
  for(int stride=blockDim.x/2;stride>0;stride>>=1){ if(tid<stride && s[tid+stride]>s[tid]){s[tid]=s[tid+stride]; s[tid+blockDim.x]=s[tid+blockDim.x+stride];} __syncthreads(); }
  if(tid==0) y[row]=s[blockDim.x];
}
int main(int argc,char**argv){ const int rows=1024,cols=128,n=rows*cols; const char*out=(argc>1)?argv[1]:"output/output.txt";
float *hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)rows*sizeof(float));
for(int i=0;i<n;++i) hx[i]=hs(i,123)+0.0001f*(float)(i%cols);
float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)rows*sizeof(float)));
CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); argmax_rows<<<rows,128,256*sizeof(float)>>>(dx,dy,rows,cols);
CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hy,dy,(size_t)rows*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,rows);
cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0; }
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols) + V.F32(0.0001) * np.arange(cols, dtype=np.float32).reshape(1, cols)
    return np.argmax(x, axis=1).astype(np.float32)
""")

add_case(CASES, {
    "id": "ggmlCumsumRows", "name": "ggml row cumulative sum", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "llama", "rel": "cumsum.cu", "fidelity": "simplified", "sizes": [1024, 128], "tol": 1e-5,
    "features": ["__global__", "prefix_sum", "rowwise"],
    "description": "Compute per-row prefix sums over short rows.",
    "notes": "Simplified cumsum row kernel based on llama.cpp/ggml cumsum.cu behavior.",
}, r"""
__global__ void cumsum_rows(const float *x, float *y, int rows, int cols) {
  int row=blockIdx.x, c=threadIdx.x;
  if(c<cols){ float s=0.0f; for(int i=0;i<=c;++i) s += x[row*cols+i]; y[row*cols+c]=s; }
}
int main(int argc,char**argv){ const int rows=1024,cols=128,n=rows*cols; const char*out=(argc>1)?argv[1]:"output/output.txt";
float *hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));
for(int i=0;i<n;++i) hx[i]=0.1f*hs(i,123);
float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); cumsum_rows<<<rows,128>>>(dx,dy,rows,cols);
CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0; }
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = (V.F32(0.1) * V.gen_hashsigned(rows * cols, 123)).reshape(rows, cols)
    return np.cumsum(x, axis=1, dtype=np.float32).reshape(-1)
""")

add_case(CASES, {
    "id": "vllmGeluMul", "name": "vLLM GELU multiply activation", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "vllm", "rel": "activation_kernels.cu", "fidelity": "simplified", "sizes": [1048576], "tol": 1e-5,
    "features": ["__global__", "erff", "gated_activation"],
    "description": "Fused GELU and multiply activation.",
    "notes": "Standalone GELU-multiply epilogue inspired by vLLM activation kernels.",
}, elemwise_main("gelu_mul", "float v=x[i]; float gelu=0.5f*v*(1.0f+erff(v*0.70710678118f)); y[i]=gelu*g[i];", 1048576, 5.0, 2.0),
"""
def reference(meta):
    import math
    n = meta["input"]["sizes"][0]
    x = V.F32(5.0) * V.gen_hashsigned(n, 123)
    g = V.F32(2.0) * V.gen_hashsigned(n, 321)
    gelu = V.F32(0.5) * x * (V.F32(1.0) + np.vectorize(math.erf, otypes=[np.float32])(x * V.F32(0.70710678118)).astype(np.float32))
    return gelu * g
""")

add_case(CASES, {
    "id": "vllmTop1MoEGate", "name": "vLLM top-1 MoE gate", "category": "ai", "domain": "modern_ml", "difficulty": "hard",
    "source": "vllm_csrc", "rel": "moe/dynamic_4bit_int_moe_cpu.cpp", "fidelity": "inspired_by", "sizes": [2048, 16], "tol": 0.0,
    "features": ["__global__", "topk", "routing"],
    "description": "Select the top-1 expert index per token from routing logits.",
    "notes": "CUDA standalone inspired by vLLM MoE routing logic; source file is CPU-side but belongs to the vLLM MoE component.",
}, r"""
__global__ void top1_gate(const float *x, float *y, int tokens, int experts) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t<tokens){ float best=-3.402823e38f; int bi=0; for(int e=0;e<experts;++e){float v=x[t*experts+e]; if(v>best){best=v; bi=e;}} y[t]=(float)bi; }
}
int main(int argc,char**argv){ const int tokens=2048,experts=16,n=tokens*experts; const char*out=(argc>1)?argv[1]:"output/output.txt";
float *hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)tokens*sizeof(float)); for(int i=0;i<n;++i) hx[i]=hs(i,123)+0.001f*(float)(i%experts);
float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)tokens*sizeof(float))); CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
int tpb=256,grid=(tokens+tpb-1)/tpb; top1_gate<<<grid,tpb>>>(dx,dy,tokens,experts); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
CK(cudaMemcpy(hy,dy,(size_t)tokens*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,tokens); cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0; }
""", """
def reference(meta):
    tokens, experts = meta["input"]["sizes"]
    x = V.gen_hashsigned(tokens * experts, 123).reshape(tokens, experts) + V.F32(0.001) * np.arange(experts, dtype=np.float32).reshape(1, experts)
    return np.argmax(x, axis=1).astype(np.float32)
""")

add_case(CASES, {
    "id": "vllmPagedKvGather", "name": "vLLM paged KV gather", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "vllm", "rel": "cache_kernels.cu", "fidelity": "simplified", "sizes": [512, 8, 64], "tol": 1e-6,
    "features": ["__global__", "gather", "cache_layout"],
    "description": "Gather KV cache pages into token-major layout.",
    "notes": "Standalone page gather based on vLLM cache_kernels.cu layout movement.",
}, r"""
__global__ void kv_gather(const float *cache, float *out, int tokens, int heads, int dim) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x, n=tokens*heads*dim;
  if(idx<n){ int d=idx%dim; int tmp=idx/dim; int h=tmp%heads; int t=tmp/heads; int page=(t*37)%tokens; out[idx]=cache[(h*tokens+page)*dim+d]; }
}
int main(int argc,char**argv){ const int tokens=512,heads=8,dim=64,n=tokens*heads*dim; const char*outp=(argc>1)?argv[1]:"output/output.txt";
float *hc=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float)); for(int i=0;i<n;++i) hc[i]=hs(i,123);
float *dc,*dy; CK(cudaMalloc(&dc,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float))); CK(cudaMemcpy(dc,hc,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
int tpb=256,grid=(n+tpb-1)/tpb; kv_gather<<<grid,tpb>>>(dc,dy,tokens,heads,dim); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(outp,hy,n); cudaFree(dc); cudaFree(dy); free(hc); free(hy); return 0; }
""", """
def reference(meta):
    tokens, heads, dim = meta["input"]["sizes"]
    cache = V.gen_hashsigned(tokens * heads * dim, 123).reshape(heads, tokens, dim)
    y = np.empty((tokens, heads, dim), dtype=np.float32)
    for t in range(tokens):
        page = (t * 37) % tokens
        y[t] = cache[:, page, :]
    return y.reshape(-1)
""")

add_case(CASES, {
    "id": "bnbNf4Dequant", "name": "bitsandbytes NF4 dequantization", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "bnb", "rel": "kernels.cu", "fidelity": "simplified", "sizes": [131072], "tol": 1e-6,
    "features": ["__global__", "lookup_table", "4bit_dequantization"],
    "description": "NF4-style 4-bit lookup-table dequantization.",
    "notes": "Standalone lookup dequantization inspired by bitsandbytes quantization kernels.",
}, r"""
__constant__ float lut[16];
__global__ void nf4_dequant(const unsigned char *q, float *y, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){ unsigned char p=q[i/2]; int nib=(i&1)?(p>>4):(p&15); y[i]=lut[nib]; }
}
int main(int argc,char**argv){ const int n=131072; const char*out=(argc>1)?argv[1]:"output/output.txt";
float hlut[16]={-1.0f,-0.696f,-0.525f,-0.394f,-0.284f,-0.184f,-0.091f,0.0f,0.079f,0.161f,0.246f,0.338f,0.441f,0.563f,0.723f,1.0f};
unsigned char *hq=(unsigned char*)calloc((size_t)n/2,1); float *hy=(float*)malloc((size_t)n*sizeof(float));
for(int i=0;i<n;++i){ unsigned char nib=(unsigned char)min(15,max(0,(int)floorf(16.0f*h01(i,123)))); if(i&1) hq[i/2]|=(nib<<4); else hq[i/2]|=nib; }
CK(cudaMemcpyToSymbol(lut,hlut,16*sizeof(float))); unsigned char*dq; float*dy; CK(cudaMalloc(&dq,(size_t)n/2)); CK(cudaMalloc(&dy,(size_t)n*sizeof(float))); CK(cudaMemcpy(dq,hq,(size_t)n/2,cudaMemcpyHostToDevice));
int tpb=256,grid=(n+tpb-1)/tpb; nf4_dequant<<<grid,tpb>>>(dq,dy,n); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
write_vec(out,hy,n); cudaFree(dq); cudaFree(dy); free(hq); free(hy); return 0; }
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    lut = np.array([-1.0,-0.696,-0.525,-0.394,-0.284,-0.184,-0.091,0.0,0.079,0.161,0.246,0.338,0.441,0.563,0.723,1.0], dtype=np.float32)
    q = np.floor(V.F32(16.0) * V.gen_hash01(n, 123)).astype(np.int32)
    q = np.clip(q, 0, 15)
    return lut[q]
""")

add_case(CASES, {
    "id": "bnbStableEmbeddingGrad", "name": "bitsandbytes embedding gradient accumulate", "category": "ai", "domain": "modern_ml", "difficulty": "hard",
    "source": "bnb", "rel": "kernels.cu", "fidelity": "inspired_by", "sizes": [4096, 64, 1024], "tol": 1e-5,
    "features": ["__global__", "atomicAdd", "embedding"],
    "description": "Accumulate embedding gradients with atomics.",
    "notes": "Standalone atomic gradient accumulation inspired by bitsandbytes optimizer/embedding support kernels.",
}, r"""
__global__ void embed_grad(const int *idx, const float *grad, float *table, int tokens, int dim, int vocab) {
  int p=blockIdx.x*blockDim.x+threadIdx.x, n=tokens*dim;
  if(p<n){ int t=p/dim, d=p%dim; atomicAdd(&table[idx[t]*dim+d], grad[p]); }
}
int main(int argc,char**argv){ const int tokens=4096,dim=64,vocab=1024,n=tokens*dim,total=vocab*dim; const char*out=(argc>1)?argv[1]:"output/output.txt";
int *hi=(int*)malloc((size_t)tokens*sizeof(int)); float *hg=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)calloc((size_t)total,sizeof(float));
for(int t=0;t<tokens;++t) hi[t]=(int)floorf(h01(t,77)*vocab); for(int i=0;i<n;++i) hg[i]=0.01f*hs(i,123);
int*di; float*dg,*dt; CK(cudaMalloc(&di,(size_t)tokens*sizeof(int))); CK(cudaMalloc(&dg,(size_t)n*sizeof(float))); CK(cudaMalloc(&dt,(size_t)total*sizeof(float)));
CK(cudaMemcpy(di,hi,(size_t)tokens*sizeof(int),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg,hg,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemset(dt,0,(size_t)total*sizeof(float)));
int tpb=256,grid=(n+tpb-1)/tpb; embed_grad<<<grid,tpb>>>(di,dg,dt,tokens,dim,vocab); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
CK(cudaMemcpy(hy,dt,(size_t)total*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,total); cudaFree(di); cudaFree(dg); cudaFree(dt); free(hi); free(hg); free(hy); return 0; }
""", """
def reference(meta):
    tokens, dim, vocab = meta["input"]["sizes"]
    idx = np.floor(V.gen_hash01(tokens, 77) * vocab).astype(np.int64)
    grad = (V.F32(0.01) * V.gen_hashsigned(tokens * dim, 123)).reshape(tokens, dim)
    out = np.zeros((vocab, dim), dtype=np.float32)
    for t in range(tokens):
        out[idx[t]] += grad[t]
    return out.reshape(-1)
""")

add_case(CASES, {
    "id": "flashAlibiBias", "name": "FlashAttention ALiBi bias", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "flash", "rel": "alibi.h", "fidelity": "simplified", "sizes": [512, 128], "tol": 1e-6,
    "features": ["__global__", "attention_bias", "2D_indexing"],
    "description": "Add ALiBi-style linear positional bias to attention scores.",
    "notes": "Standalone ALiBi bias application inspired by FlashAttention alibi.h.",
}, r"""
__global__ void alibi(const float *x, float *y, int rows, int cols) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x,n=rows*cols;
  if(idx<n){ int r=idx/cols,c=idx%cols; float slope=0.01f*(float)((r%8)+1); y[idx]=x[idx]-slope*fabsf((float)(c-(r%cols))); }
}
int main(int argc,char**argv){ const int rows=512,cols=128,n=rows*cols; const char*out=(argc>1)?argv[1]:"output/output.txt";
float *hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float)); for(int i=0;i<n;++i) hx[i]=hs(i,123);
float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float))); CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
int tpb=256,grid=(n+tpb-1)/tpb; alibi<<<grid,tpb>>>(dx,dy,rows,cols); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
write_vec(out,hy,n); cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0; }
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    y = np.empty_like(x)
    c = np.arange(cols, dtype=np.float32)
    for r in range(rows):
        slope = V.F32(0.01) * V.F32((r % 8) + 1)
        y[r] = x[r] - slope * np.abs(c - V.F32(r % cols))
    return y.reshape(-1)
""")

add_case(CASES, {
    "id": "flashRotaryHalf2Pair", "name": "FlashAttention rotary pair helper", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "flash", "rel": "rotary.h", "fidelity": "simplified", "sizes": [512, 64], "tol": 2e-4,
    "features": ["__global__", "sinf", "cosf", "rotary_embedding"],
    "description": "Rotary transform over adjacent vector pairs.",
    "notes": "Standalone helper inspired by FlashAttention rotary.h.",
}, r"""
__global__ void rotary_pair(const float *x, float *y, int rows, int dim) {
  int pair=blockIdx.x*blockDim.x+threadIdx.x,total=rows*(dim/2);
  if(pair<total){ int r=pair/(dim/2), p=pair%(dim/2), base=r*dim+2*p; float th=(float)(r%2048)*powf(10000.0f,-2.0f*(float)p/(float)dim); float c=cosf(th),s=sinf(th),a=x[base],b=x[base+1]; y[base]=a*c-b*s; y[base+1]=a*s+b*c; }
}
int main(int argc,char**argv){ const int rows=512,dim=64,n=rows*dim; const char*out=(argc>1)?argv[1]:"output/output.txt";
float *hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float)); for(int i=0;i<n;++i) hx[i]=hs(i,123);
float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float))); CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
int total=rows*(dim/2),tpb=256,grid=(total+tpb-1)/tpb; rotary_pair<<<grid,tpb>>>(dx,dy,rows,dim); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
write_vec(out,hy,n); cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0; }
""", """
def reference(meta):
    rows, dim = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * dim, 123).reshape(rows, dim)
    y = np.empty_like(x)
    for r in range(rows):
        for p in range(dim // 2):
            th = V.F32(r % 2048) * np.float32(10000.0) ** np.float32(-2.0 * p / dim)
            c = np.cos(th, dtype=np.float32); s = np.sin(th, dtype=np.float32)
            a = x[r, 2*p]; b = x[r, 2*p+1]
            y[r, 2*p] = a*c - b*s
            y[r, 2*p+1] = a*s + b*c
    return y.reshape(-1)
""")

add_case(CASES, {
    "id": "cutlassBasicGemmTile", "name": "CUTLASS basic GEMM tile", "category": "ai", "domain": "modern_ml", "difficulty": "hard",
    "source": "cutlass", "rel": "00_basic_gemm/basic_gemm.cu", "fidelity": "inspired_by", "sizes": [128, 128, 64], "tol": 1e-4,
    "features": ["__global__", "__shared__", "tiled_gemm"],
    "description": "Shared-memory tiled GEMM inspired by CUTLASS basic GEMM examples.",
    "notes": "Standalone tiled GEMM using CUTLASS example dimensions but no CUTLASS dependency.",
}, r"""
__global__ void gemm_tile(const float *A,const float *B,float*C,int M,int N,int K){
  __shared__ float As[16][16], Bs[16][16]; int row=blockIdx.y*16+threadIdx.y, col=blockIdx.x*16+threadIdx.x; float acc=0.0f;
  for(int t=0;t<K;t+=16){ As[threadIdx.y][threadIdx.x]=(row<M&&t+threadIdx.x<K)?A[row*K+t+threadIdx.x]:0.0f; Bs[threadIdx.y][threadIdx.x]=(t+threadIdx.y<K&&col<N)?B[(t+threadIdx.y)*N+col]:0.0f; __syncthreads(); for(int k=0;k<16;++k) acc+=As[threadIdx.y][k]*Bs[k][threadIdx.x]; __syncthreads(); }
  if(row<M&&col<N) C[row*N+col]=acc;
}
int main(int argc,char**argv){ const int M=128,N=128,K=64,nc=M*N; const char*out=(argc>1)?argv[1]:"output/output.txt";
float *ha=(float*)malloc((size_t)M*K*sizeof(float)),*hb=(float*)malloc((size_t)K*N*sizeof(float)),*hc=(float*)malloc((size_t)nc*sizeof(float));
for(int i=0;i<M*K;++i)ha[i]=0.1f*hs(i,123); for(int i=0;i<K*N;++i)hb[i]=0.1f*hs(i,321);
float *da,*db,*dc; CK(cudaMalloc(&da,(size_t)M*K*sizeof(float))); CK(cudaMalloc(&db,(size_t)K*N*sizeof(float))); CK(cudaMalloc(&dc,(size_t)nc*sizeof(float)));
CK(cudaMemcpy(da,ha,(size_t)M*K*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(db,hb,(size_t)K*N*sizeof(float),cudaMemcpyHostToDevice)); gemm_tile<<<dim3((N+15)/16,(M+15)/16),dim3(16,16)>>>(da,db,dc,M,N,K);
CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hc,dc,(size_t)nc*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hc,nc);
cudaFree(da); cudaFree(db); cudaFree(dc); free(ha); free(hb); free(hc); return 0; }
""", """
def reference(meta):
    M, N, K = meta["input"]["sizes"]
    a = (V.F32(0.1) * V.gen_hashsigned(M * K, 123)).reshape(M, K)
    b = (V.F32(0.1) * V.gen_hashsigned(K * N, 321)).reshape(K, N)
    return (a @ b).astype(np.float32).reshape(-1)
""")

add_case(CASES, {
    "id": "cutlassBiasReluEpilogue", "name": "CUTLASS bias ReLU epilogue", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "cutlass", "rel": "12_gemm_bias_relu/gemm_bias_relu.cu", "fidelity": "inspired_by", "sizes": [262144], "tol": 1e-6,
    "features": ["__global__", "epilogue", "activation", "bias"],
    "description": "Bias + ReLU epilogue operation common in CUTLASS examples.",
    "notes": "Standalone epilogue inspired by CUTLASS gemm_bias_relu example.",
}, elemwise_main("bias_relu_epi", "float v=x[i]+0.125f*g[i]; y[i]=v>0.0f?v:0.0f;", 262144, 2.0, 1.0),
"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.F32(2.0) * V.gen_hashsigned(n, 123)
    g = V.gen_hashsigned(n, 321)
    return np.maximum(x + V.F32(0.125) * g, V.F32(0.0))
""")

add_case(CASES, {
    "id": "cutlassBatchedGemmSmall", "name": "CUTLASS small batched GEMM", "category": "ai", "domain": "modern_ml", "difficulty": "hard",
    "source": "cutlass", "rel": "05_batched_gemm/batched_gemm.cu", "fidelity": "inspired_by", "sizes": [32, 16, 16, 16], "tol": 1e-4,
    "features": ["__global__", "batched_gemm", "3D_grid"],
    "description": "Small batched GEMM using one CUDA block per batch.",
    "notes": "Standalone batched GEMM inspired by CUTLASS batched_gemm example.",
}, r"""
__global__ void bgemm(const float*A,const float*B,float*C,int batch,int M,int N,int K){ int b=blockIdx.z,row=threadIdx.y,col=threadIdx.x; if(row<M&&col<N){float acc=0.0f; for(int k=0;k<K;++k) acc+=A[(b*M+row)*K+k]*B[(b*K+k)*N+col]; C[(b*M+row)*N+col]=acc;}}
int main(int argc,char**argv){const int batch=32,M=16,N=16,K=16,nc=batch*M*N; const char*out=(argc>1)?argv[1]:"output/output.txt";
float *ha=(float*)malloc((size_t)batch*M*K*sizeof(float)),*hb=(float*)malloc((size_t)batch*K*N*sizeof(float)),*hc=(float*)malloc((size_t)nc*sizeof(float));
for(int i=0;i<batch*M*K;++i)ha[i]=0.2f*hs(i,123); for(int i=0;i<batch*K*N;++i)hb[i]=0.2f*hs(i,321);
float*da,*db,*dc; CK(cudaMalloc(&da,(size_t)batch*M*K*sizeof(float))); CK(cudaMalloc(&db,(size_t)batch*K*N*sizeof(float))); CK(cudaMalloc(&dc,(size_t)nc*sizeof(float)));
CK(cudaMemcpy(da,ha,(size_t)batch*M*K*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(db,hb,(size_t)batch*K*N*sizeof(float),cudaMemcpyHostToDevice)); bgemm<<<dim3(1,1,batch),dim3(N,M)>>>(da,db,dc,batch,M,N,K);
CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hc,dc,(size_t)nc*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hc,nc);
cudaFree(da); cudaFree(db); cudaFree(dc); free(ha); free(hb); free(hc); return 0;}
""", """
def reference(meta):
    batch, M, N, K = meta["input"]["sizes"]
    a = (V.F32(0.2) * V.gen_hashsigned(batch * M * K, 123)).reshape(batch, M, K)
    b = (V.F32(0.2) * V.gen_hashsigned(batch * K * N, 321)).reshape(batch, K, N)
    return np.matmul(a, b).astype(np.float32).reshape(-1)
""")

add_case(CASES, {
    "id": "cutlassGatherScatterFusion", "name": "CUTLASS gather scatter fusion", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "cutlass", "rel": "36_gather_scatter_fusion/gather_scatter_fusion.cu", "fidelity": "inspired_by", "sizes": [262144], "tol": 1e-6,
    "features": ["__global__", "gather", "scatter", "permutation"],
    "description": "Gather-scatter permutation with fused scale.",
    "notes": "Standalone memory movement pattern inspired by CUTLASS gather_scatter_fusion example.",
}, r"""
__global__ void gather_scatter(const float*x,float*y,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ int src=(i*17+13)%n; int dst=(i*29+7)%n; y[dst]=2.0f*x[src]; }}
int main(int argc,char**argv){const int n=262144; const char*out=(argc>1)?argv[1]:"output/output.txt"; float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)calloc((size_t)n,sizeof(float)); for(int i=0;i<n;++i)hx[i]=hs(i,123);
float*dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float))); CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemset(dy,0,(size_t)n*sizeof(float)));
int tpb=256,grid=(n+tpb-1)/tpb; gather_scatter<<<grid,tpb>>>(dx,dy,n); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        src = (i * 17 + 13) % n
        dst = (i * 29 + 7) % n
        y[dst] = V.F32(2.0) * x[src]
    return y
""")

# DALI image/preprocessing cases.
add_case(CASES, {
    "id": "daliEraseRect", "name": "DALI erase rectangle", "category": "ai", "domain": "image_processing", "difficulty": "medium",
    "source": "dali", "rel": "dali/kernels/erase/erase_gpu.h", "fidelity": "simplified", "sizes": [4, 64, 64, 3], "tol": 1e-6,
    "features": ["__global__", "image_processing", "branching"],
    "description": "Erase a deterministic rectangular region in NHWC images.",
    "notes": "Standalone erase kernel based on NVIDIA DALI erase GPU operator.",
}, r"""
__global__ void erase_rect(const float*x,float*y,int N,int H,int W,int C){int idx=blockIdx.x*blockDim.x+threadIdx.x,total=N*H*W*C;if(idx<total){int c=idx%C;int tmp=idx/C;int w=tmp%W;tmp/=W;int h=tmp%H;int n=tmp/H;bool inside=(h>=16&&h<48&&w>=12&&w<40);y[idx]=inside?0.25f*(float)(c+1):x[idx];}}
int main(int argc,char**argv){const int N=4,H=64,W=64,C=3,total=N*H*W*C;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)total*sizeof(float)),*hy=(float*)malloc((size_t)total*sizeof(float));for(int i=0;i<total;++i)hx[i]=h01(i,123);
float*dx,*dy;CK(cudaMalloc(&dx,(size_t)total*sizeof(float)));CK(cudaMalloc(&dy,(size_t)total*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)total*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(total+tpb-1)/tpb;erase_rect<<<grid,tpb>>>(dx,dy,N,H,W,C);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)total*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,total);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    N,H,W,C = meta["input"]["sizes"]
    x = V.gen_hash01(N*H*W*C,123).reshape(N,H,W,C)
    y = x.copy()
    for c in range(C):
        y[:,16:48,12:40,c] = V.F32(0.25*(c+1))
    return y.reshape(-1)
""")

add_case(CASES, {
    "id": "daliHsvToRgb", "name": "DALI HSV to RGB helper", "category": "ai", "domain": "image_processing", "difficulty": "medium",
    "source": "dali", "rel": "dali/kernels/imgproc/color_manipulation/hsv_cpu.h", "fidelity": "inspired_by", "sizes": [65536], "tol": 1e-5,
    "features": ["__global__", "color_conversion", "branching"],
    "description": "Convert deterministic HSV pixels to RGB.",
    "notes": "CUDA standalone inspired by DALI color manipulation helpers.",
}, r"""
__global__ void hsv_rgb(const float*h,const float*s,const float*v,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float hh=h[i]*6.0f;int k=(int)floorf(hh);float f=hh-k,p=v[i]*(1.0f-s[i]),q=v[i]*(1.0f-s[i]*f),t=v[i]*(1.0f-s[i]*(1.0f-f));float r,g,b;switch(k%6){case 0:r=v[i];g=t;b=p;break;case 1:r=q;g=v[i];b=p;break;case 2:r=p;g=v[i];b=t;break;case 3:r=p;g=q;b=v[i];break;case 4:r=t;g=p;b=v[i];break;default:r=v[i];g=p;b=q;}y[3*i]=r;y[3*i+1]=g;y[3*i+2]=b;}}
int main(int argc,char**argv){const int n=65536;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hh=(float*)malloc((size_t)n*sizeof(float)),*hsat=(float*)malloc((size_t)n*sizeof(float)),*hv=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)3*n*sizeof(float));for(int i=0;i<n;++i){hh[i]=h01(i,11);hsat[i]=h01(i,22);hv[i]=h01(i,33);}
float*dh,*ds,*dv,*dy;CK(cudaMalloc(&dh,(size_t)n*sizeof(float)));CK(cudaMalloc(&ds,(size_t)n*sizeof(float)));CK(cudaMalloc(&dv,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)3*n*sizeof(float)));CK(cudaMemcpy(dh,hh,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,hsat,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dv,hv,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(n+tpb-1)/tpb;hsv_rgb<<<grid,tpb>>>(dh,ds,dv,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)3*n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,3*n);cudaFree(dh);cudaFree(ds);cudaFree(dv);cudaFree(dy);free(hh);free(hsat);free(hv);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    h = V.gen_hash01(n,11); s = V.gen_hash01(n,22); v = V.gen_hash01(n,33)
    out = np.empty((n,3), dtype=np.float32)
    for i in range(n):
        hh = float(h[i] * V.F32(6.0)); k = int(np.floor(hh)); f = V.F32(hh-k)
        p = v[i]*(V.F32(1.0)-s[i]); q = v[i]*(V.F32(1.0)-s[i]*f); t = v[i]*(V.F32(1.0)-s[i]*(V.F32(1.0)-f))
        m = k % 6
        if m == 0: out[i] = [v[i], t, p]
        elif m == 1: out[i] = [q, v[i], p]
        elif m == 2: out[i] = [p, v[i], t]
        elif m == 3: out[i] = [p, q, v[i]]
        elif m == 4: out[i] = [t, p, v[i]]
        else: out[i] = [v[i], p, q]
    return out.reshape(-1)
""")

add_case(CASES, {
    "id": "daliDct8x8", "name": "DALI JPEG 8x8 DCT helper", "category": "ai", "domain": "image_processing", "difficulty": "hard",
    "source": "dali", "rel": "dali/kernels/imgproc/jpeg/dct_8x8_gpu.cuh", "fidelity": "inspired_by", "sizes": [1024, 64], "tol": 1e-4,
    "features": ["__global__", "block_transform", "cosf"],
    "description": "Compute a small 8x8 DCT coefficient subset per block.",
    "notes": "Standalone DCT-style block transform inspired by DALI JPEG DCT helpers.",
}, r"""
__global__ void dct_dc_ac(const float*x,float*y,int blocks){int b=blockIdx.x;float dc=0.0f,ac=0.0f;for(int r=0;r<8;++r){for(int c=0;c<8;++c){float v=x[b*64+r*8+c];dc+=v;ac+=v*cosf((float)(2*c+1)*3.14159265f/16.0f);}}y[2*b]=dc/8.0f;y[2*b+1]=0.5f*ac;}
int main(int argc,char**argv){const int blocks=1024,n=blocks*64;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)2*blocks*sizeof(float));for(int i=0;i<n;++i)hx[i]=h01(i,123);
float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)2*blocks*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));dct_dc_ac<<<blocks,1>>>(dx,dy,blocks);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)2*blocks*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,2*blocks);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    blocks, width = meta["input"]["sizes"]
    x = V.gen_hash01(blocks * width, 123).reshape(blocks, 8, 8)
    out = np.empty((blocks,2), dtype=np.float32)
    cosv = np.cos((2*np.arange(8,dtype=np.float32)+1)*np.float32(np.pi)/V.F32(16.0), dtype=np.float32)
    for b in range(blocks):
        out[b,0] = np.sum(x[b], dtype=np.float32) / V.F32(8.0)
        out[b,1] = V.F32(0.5) * np.sum(x[b] * cosv.reshape(1,8), dtype=np.float32)
    return out.reshape(-1)
""")

add_case(CASES, {
    "id": "daliSliceFlipNormalizePad", "name": "DALI slice flip normalize pad", "category": "ai", "domain": "image_processing", "difficulty": "hard",
    "source": "dali", "rel": "dali/kernels/slice/slice_flip_normalize_permute_pad_cuda_impl.cuh", "fidelity": "simplified", "sizes": [4, 64, 64, 3], "tol": 1e-6,
    "features": ["__global__", "slice", "flip", "normalize", "padding"],
    "description": "Slice, horizontal flip, normalize, and pad NHWC image tensors.",
    "notes": "Standalone fused preprocessing path based on DALI slice/flip/normalize/permute/pad CUDA implementation.",
}, r"""
__global__ void slice_flip_norm(const float*x,float*y,int N,int H,int W,int C){int outH=48,outW=48,total=N*outH*outW*C;int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx<total){int c=idx%C;int tmp=idx/C;int ow=tmp%outW;tmp/=outW;int oh=tmp%outH;int n=tmp/outH;int ih=oh+8,iw=55-ow;float mean=0.1f*(float)c,stdv=0.5f+0.1f*(float)c;y[idx]=(x[((n*H+ih)*W+iw)*C+c]-mean)/stdv;}}
int main(int argc,char**argv){const int N=4,H=64,W=64,C=3,inN=N*H*W*C,outN=N*48*48*C;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)inN*sizeof(float)),*hy=(float*)malloc((size_t)outN*sizeof(float));for(int i=0;i<inN;++i)hx[i]=h01(i,123);
float*dx,*dy;CK(cudaMalloc(&dx,(size_t)inN*sizeof(float)));CK(cudaMalloc(&dy,(size_t)outN*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)inN*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(outN+tpb-1)/tpb;slice_flip_norm<<<grid,tpb>>>(dx,dy,N,H,W,C);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)outN*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,outN);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    N,H,W,C = meta["input"]["sizes"]
    x = V.gen_hash01(N*H*W*C,123).reshape(N,H,W,C)
    y = np.empty((N,48,48,C), dtype=np.float32)
    for c in range(C):
        y[:,:,:,c] = (x[:,8:56,8:56,c][:,:,::-1] - V.F32(0.1*c)) / V.F32(0.5+0.1*c)
    return y.reshape(-1)
""")

add_case(CASES, {
    "id": "daliEqualizeLut", "name": "DALI equalize LUT", "category": "ai", "domain": "image_processing", "difficulty": "medium",
    "source": "dali", "rel": "dali/kernels/imgproc/color_manipulation/equalize/lookup.cu", "fidelity": "simplified", "sizes": [262144], "tol": 0.0,
    "features": ["__global__", "lookup_table", "uint8_like"],
    "description": "Apply an equalization lookup table to byte-like image intensities.",
    "notes": "Standalone LUT application based on DALI equalize lookup CUDA kernel.",
}, r"""
__global__ void lut_apply(const int*x,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int v=x[i]&255;y[i]=(float)((v*37+13)&255);}}
int main(int argc,char**argv){const int n=262144;const char*out=(argc>1)?argv[1]:"output/output.txt";int*hx=(int*)malloc((size_t)n*sizeof(int));float*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=(int)floorf(256.0f*h01(i,123));
int*dx;float*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(int)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(int),cudaMemcpyHostToDevice));int tpb=256,grid=(n+tpb-1)/tpb;lut_apply<<<grid,tpb>>>(dx,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = np.floor(V.F32(256.0) * V.gen_hash01(n,123)).astype(np.int32)
    return (((x * 37 + 13) & 255).astype(np.float32))
""")

# HPC / scientific kernels from HeCBench.
add_case(CASES, {
    "id": "hecbenchHotspotStep", "name": "HeCBench HotSpot step", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "hotspot-cuda/main.cu", "fidelity": "inspired_by", "sizes": [256, 256], "tol": 1e-5,
    "features": ["__global__", "2D_stencil", "heat_equation"],
    "description": "Single HotSpot-style heat diffusion stencil step.",
    "notes": "Standalone heat stencil inspired by HeCBench/Rodinia HotSpot CUDA benchmark.",
}, r"""
__global__ void hotspot(const float*t,const float*p,float*y,int H,int W){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=H*W;if(idx<n){int r=idx/W,c=idx%W;int up=max(r-1,0),dn=min(r+1,H-1),lf=max(c-1,0),rt=min(c+1,W-1);y[idx]=t[idx]+0.05f*(t[up*W+c]+t[dn*W+c]+t[r*W+lf]+t[r*W+rt]-4.0f*t[idx])+0.01f*p[idx];}}
int main(int argc,char**argv){const int H=256,W=256,n=H*W;const char*out=(argc>1)?argv[1]:"output/output.txt";float*ht=(float*)malloc((size_t)n*sizeof(float)),*hp=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i){ht[i]=hs(i,123);hp[i]=h01(i,321);}
float*dt,*dp,*dy;CK(cudaMalloc(&dt,(size_t)n*sizeof(float)));CK(cudaMalloc(&dp,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dt,ht,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dp,hp,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(n+tpb-1)/tpb;hotspot<<<grid,tpb>>>(dt,dp,dy,H,W);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dt);cudaFree(dp);cudaFree(dy);free(ht);free(hp);free(hy);return 0;}
""", """
def reference(meta):
    H,W = meta["input"]["sizes"]
    t = V.gen_hashsigned(H*W,123).reshape(H,W)
    p = V.gen_hash01(H*W,321).reshape(H,W)
    y = np.empty_like(t)
    for r in range(H):
        for c in range(W):
            y[r,c] = t[r,c] + V.F32(0.05)*(t[max(r-1,0),c]+t[min(r+1,H-1),c]+t[r,max(c-1,0)]+t[r,min(c+1,W-1)]-V.F32(4.0)*t[r,c]) + V.F32(0.01)*p[r,c]
    return y.reshape(-1)
""")

add_case(CASES, {
    "id": "hecbenchFloydWarshallStep", "name": "HeCBench Floyd-Warshall step", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "floydwarshall-cuda/main.cu", "fidelity": "simplified", "sizes": [256], "tol": 0.0,
    "features": ["__global__", "dynamic_programming", "min_plus"],
    "description": "One Floyd-Warshall k-step update over a dense distance matrix.",
    "notes": "Standalone min-plus update based on HeCBench floydwarshall-cuda.",
}, r"""
__global__ void fw_step(const float*x,float*y,int n,int k){int idx=blockIdx.x*blockDim.x+threadIdx.x,total=n*n;if(idx<total){int i=idx/n,j=idx%n;float via=x[i*n+k]+x[k*n+j];y[idx]=fminf(x[idx],via);}}
int main(int argc,char**argv){const int n=256,k=37,total=n*n;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)total*sizeof(float)),*hy=(float*)malloc((size_t)total*sizeof(float));for(int i=0;i<total;++i)hx[i]=(float)((i*17+23)%251);for(int i=0;i<n;++i)hx[i*n+i]=0.0f;
float*dx,*dy;CK(cudaMalloc(&dx,(size_t)total*sizeof(float)));CK(cudaMalloc(&dy,(size_t)total*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)total*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(total+tpb-1)/tpb;fw_step<<<grid,tpb>>>(dx,dy,n,k);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)total*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,total);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]; k = 37
    x = ((np.arange(n*n, dtype=np.int64)*17 + 23) % 251).astype(np.float32).reshape(n,n)
    np.fill_diagonal(x, 0.0)
    return np.minimum(x, x[:,[k]] + x[[k],:]).reshape(-1)
""")

add_case(CASES, {
    "id": "hecbenchGaussianElimStep", "name": "HeCBench Gaussian elimination step", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "gaussian-cuda/gaussianElim.cu", "fidelity": "simplified", "sizes": [256, 256], "tol": 1e-5,
    "features": ["__global__", "linear_algebra", "elimination"],
    "description": "One Gaussian elimination row update.",
    "notes": "Standalone elimination step based on HeCBench gaussian-cuda.",
}, r"""
__global__ void elim(const float*A,float*B,int rows,int cols){int idx=blockIdx.x*blockDim.x+threadIdx.x,total=rows*cols;if(idx<total){int r=idx/cols,c=idx%cols;float pivot=A[c];float factor=A[(r+1)*cols];B[idx]=A[(r+1)*cols+c]-factor*pivot;}}
int main(int argc,char**argv){const int rows=256,cols=256,total=rows*cols;const char*out=(argc>1)?argv[1]:"output/output.txt";float*ha=(float*)malloc((size_t)(rows+1)*cols*sizeof(float)),*hb=(float*)malloc((size_t)total*sizeof(float));for(int i=0;i<(rows+1)*cols;++i)ha[i]=0.01f+0.1f*h01(i,123);
float*da,*db;CK(cudaMalloc(&da,(size_t)(rows+1)*cols*sizeof(float)));CK(cudaMalloc(&db,(size_t)total*sizeof(float)));CK(cudaMemcpy(da,ha,(size_t)(rows+1)*cols*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(total+tpb-1)/tpb;elim<<<grid,tpb>>>(da,db,rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hb,db,(size_t)total*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hb,total);cudaFree(da);cudaFree(db);free(ha);free(hb);return 0;}
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    A = (V.F32(0.01) + V.F32(0.1) * V.gen_hash01((rows+1)*cols,123)).reshape(rows+1, cols)
    B = np.empty((rows, cols), dtype=np.float32)
    for r in range(rows):
        B[r] = A[r+1] - A[r+1,0] * A[0]
    return B.reshape(-1)
""")

add_case(CASES, {
    "id": "hecbenchBlackScholes", "name": "HeCBench Black-Scholes analytic", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "black-scholes-cuda/blackScholesAnalyticEngineKernels.cu", "fidelity": "simplified", "sizes": [262144], "tol": 1e-5,
    "features": ["__global__", "erff", "financial_model"],
    "description": "Black-Scholes call option price kernel.",
    "notes": "Standalone analytic option kernel based on HeCBench black-scholes CUDA benchmark.",
}, r"""
__device__ float bs_normcdf(float x){return 0.5f*(1.0f+erff(x*0.70710678118f));}
__global__ void black_scholes(const float*S,float*K,float*T,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float r=0.02f,sig=0.3f;float sqrtT=sqrtf(T[i]);float d1=(logf(S[i]/K[i])+(r+0.5f*sig*sig)*T[i])/(sig*sqrtT);float d2=d1-sig*sqrtT;y[i]=S[i]*bs_normcdf(d1)-K[i]*expf(-r*T[i])*bs_normcdf(d2);}}
int main(int argc,char**argv){const int n=262144;const char*out=(argc>1)?argv[1]:"output/output.txt";float*S=(float*)malloc((size_t)n*sizeof(float)),*K=(float*)malloc((size_t)n*sizeof(float)),*T=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i){S[i]=10.0f+20.0f*h01(i,1);K[i]=10.0f+20.0f*h01(i,2);T[i]=0.25f+2.0f*h01(i,3);}
float*dS,*dK,*dT,*dy;CK(cudaMalloc(&dS,(size_t)n*sizeof(float)));CK(cudaMalloc(&dK,(size_t)n*sizeof(float)));CK(cudaMalloc(&dT,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dS,S,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dK,K,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dT,T,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(n+tpb-1)/tpb;black_scholes<<<grid,tpb>>>(dS,dK,dT,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dS);cudaFree(dK);cudaFree(dT);cudaFree(dy);free(S);free(K);free(T);free(hy);return 0;}
""", """
def reference(meta):
    import math
    n = meta["input"]["sizes"][0]
    S = V.F32(10.0) + V.F32(20.0) * V.gen_hash01(n,1)
    K = V.F32(10.0) + V.F32(20.0) * V.gen_hash01(n,2)
    T = V.F32(0.25) + V.F32(2.0) * V.gen_hash01(n,3)
    r = V.F32(0.02); sig = V.F32(0.3)
    sqrtT = np.sqrt(T, dtype=np.float32)
    d1 = (np.log(S/K, dtype=np.float32) + (r + V.F32(0.5)*sig*sig)*T)/(sig*sqrtT)
    d2 = d1 - sig*sqrtT
    erf = np.vectorize(math.erf, otypes=[np.float32])
    cdf1 = V.F32(0.5)*(V.F32(1.0)+erf(d1*V.F32(0.70710678118)).astype(np.float32))
    cdf2 = V.F32(0.5)*(V.F32(1.0)+erf(d2*V.F32(0.70710678118)).astype(np.float32))
    return S*cdf1 - K*np.exp(-r*T, dtype=np.float32)*cdf2
""")

add_case(CASES, {
    "id": "hecbenchKmeansAssign", "name": "HeCBench k-means assignment", "category": "hpc", "domain": "hpc", "difficulty": "hard",
    "source": "hecbench", "rel": "kmeans-cuda/main.cu", "fidelity": "inspired_by", "sizes": [4096, 8, 16], "tol": 0.0,
    "features": ["__global__", "nearest_centroid", "reduction"],
    "description": "Assign points to nearest k-means centroid.",
    "notes": "Standalone nearest-centroid kernel inspired by HeCBench/Rodinia kmeans CUDA benchmark.",
}, r"""
__global__ void assign(const float*x,const float*c,float*y,int pts,int dim,int k){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<pts){float best=3.4e38f;int bi=0;for(int j=0;j<k;++j){float d=0.0f;for(int z=0;z<dim;++z){float q=x[p*dim+z]-c[j*dim+z];d+=q*q;}if(d<best){best=d;bi=j;}}y[p]=(float)bi;}}
int main(int argc,char**argv){const int pts=4096,dim=8,k=16;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)pts*dim*sizeof(float)),*hc=(float*)malloc((size_t)k*dim*sizeof(float)),*hy=(float*)malloc((size_t)pts*sizeof(float));for(int i=0;i<pts*dim;++i)hx[i]=hs(i,123);for(int i=0;i<k*dim;++i)hc[i]=hs(i,321);
float*dx,*dc,*dy;CK(cudaMalloc(&dx,(size_t)pts*dim*sizeof(float)));CK(cudaMalloc(&dc,(size_t)k*dim*sizeof(float)));CK(cudaMalloc(&dy,(size_t)pts*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)pts*dim*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dc,hc,(size_t)k*dim*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(pts+tpb-1)/tpb;assign<<<grid,tpb>>>(dx,dc,dy,pts,dim,k);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)pts*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,pts);cudaFree(dx);cudaFree(dc);cudaFree(dy);free(hx);free(hc);free(hy);return 0;}
""", """
def reference(meta):
    pts, dim, k = meta["input"]["sizes"]
    x = V.gen_hashsigned(pts*dim,123).reshape(pts,dim)
    c = V.gen_hashsigned(k*dim,321).reshape(k,dim)
    d = ((x[:,None,:]-c[None,:,:])**2).sum(axis=2, dtype=np.float32)
    return np.argmin(d, axis=1).astype(np.float32)
""")

add_case(CASES, {
    "id": "hecbenchPathfinderStep", "name": "HeCBench Pathfinder step", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "pathfinder-cuda/main.cu", "fidelity": "inspired_by", "sizes": [512, 256], "tol": 0.0,
    "features": ["__global__", "dynamic_programming", "stencil"],
    "description": "One pathfinder dynamic-programming row transition.",
    "notes": "Standalone DP transition inspired by HeCBench/Rodinia pathfinder CUDA benchmark.",
}, r"""
__global__ void path_step(const float*prev,const float*cost,float*y,int rows,int cols){int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<cols){float m=prev[c];if(c>0)m=fminf(m,prev[c-1]);if(c+1<cols)m=fminf(m,prev[c+1]);y[c]=cost[c]+m;}}
int main(int argc,char**argv){const int rows=512,cols=256;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hp=(float*)malloc((size_t)cols*sizeof(float)),*hc=(float*)malloc((size_t)cols*sizeof(float)),*hy=(float*)malloc((size_t)cols*sizeof(float));for(int i=0;i<cols;++i){hp[i]=(float)((i*7)%31);hc[i]=(float)((i*13)%17);}
float*dp,*dc,*dy;CK(cudaMalloc(&dp,(size_t)cols*sizeof(float)));CK(cudaMalloc(&dc,(size_t)cols*sizeof(float)));CK(cudaMalloc(&dy,(size_t)cols*sizeof(float)));CK(cudaMemcpy(dp,hp,(size_t)cols*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dc,hc,(size_t)cols*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(cols+tpb-1)/tpb;path_step<<<grid,tpb>>>(dp,dc,dy,rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)cols*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,cols);cudaFree(dp);cudaFree(dc);cudaFree(dy);free(hp);free(hc);free(hy);return 0;}
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    prev = ((np.arange(cols)*7)%31).astype(np.float32)
    cost = ((np.arange(cols)*13)%17).astype(np.float32)
    y = np.empty(cols, dtype=np.float32)
    for c in range(cols):
        y[c] = cost[c] + min(prev[c], prev[max(c-1,0)], prev[min(c+1,cols-1)])
    return y
""")

add_case(CASES, {
    "id": "hecbenchSradDiffusion", "name": "HeCBench SRAD diffusion", "category": "hpc", "domain": "hpc", "difficulty": "hard",
    "source": "hecbench", "rel": "srad-cuda/main.cu", "fidelity": "inspired_by", "sizes": [128, 128], "tol": 1e-5,
    "features": ["__global__", "2D_stencil", "diffusion"],
    "description": "SRAD-style anisotropic diffusion update.",
    "notes": "Standalone diffusion stencil inspired by HeCBench/Rodinia SRAD CUDA benchmark.",
}, r"""
__global__ void srad(const float*x,float*y,int H,int W){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=H*W;if(idx<n){int r=idx/W,c=idx%W;float center=x[idx];float n0=x[max(r-1,0)*W+c],s=x[min(r+1,H-1)*W+c],w=x[r*W+max(c-1,0)],e=x[r*W+min(c+1,W-1)];y[idx]=center+0.125f*((n0+s+w+e)-4.0f*center)/(0.01f+fabsf(center));}}
int main(int argc,char**argv){const int H=128,W=128,n=H*W;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=1.0f+h01(i,123);
float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(n+tpb-1)/tpb;srad<<<grid,tpb>>>(dx,dy,H,W);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    H,W = meta["input"]["sizes"]
    x = (V.F32(1.0) + V.gen_hash01(H*W,123)).reshape(H,W)
    y = np.empty_like(x)
    for r in range(H):
        for c in range(W):
            center = x[r,c]
            y[r,c] = center + V.F32(0.125)*((x[max(r-1,0),c]+x[min(r+1,H-1),c]+x[r,max(c-1,0)]+x[r,min(c+1,W-1)]-V.F32(4.0)*center)/(V.F32(0.01)+abs(center)))
    return y.reshape(-1)
""")

add_case(CASES, {
    "id": "hecbenchNeedlemanWunsch", "name": "HeCBench Needleman-Wunsch anti-diagonal", "category": "hpc", "domain": "hpc", "difficulty": "hard",
    "source": "hecbench", "rel": "nw-cuda/main.cu", "fidelity": "inspired_by", "sizes": [256], "tol": 0.0,
    "features": ["__global__", "dynamic_programming", "anti_diagonal"],
    "description": "Compute one independent anti-diagonal of a Needleman-Wunsch score matrix.",
    "notes": "Standalone anti-diagonal recurrence inspired by HeCBench/Rodinia NW CUDA benchmark.",
}, r"""
__global__ void nw_diag(const float*up,const float*left,const float*diag,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float match=diag[i]+((i%7)==0?2.0f:-1.0f);float del=up[i]-1.0f;float ins=left[i]-1.0f;y[i]=fmaxf(match,fmaxf(del,ins));}}
int main(int argc,char**argv){const int n=256;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hu=(float*)malloc((size_t)n*sizeof(float)),*hl=(float*)malloc((size_t)n*sizeof(float)),*hd=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i){hu[i]=(float)(i%17);hl[i]=(float)(i%13);hd[i]=(float)(i%11);}
float*du,*dl,*dd,*dy;CK(cudaMalloc(&du,(size_t)n*sizeof(float)));CK(cudaMalloc(&dl,(size_t)n*sizeof(float)));CK(cudaMalloc(&dd,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(du,hu,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dl,hl,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dd,hd,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(n+tpb-1)/tpb;nw_diag<<<grid,tpb>>>(du,dl,dd,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(du);cudaFree(dl);cudaFree(dd);cudaFree(dy);free(hu);free(hl);free(hd);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    i = np.arange(n, dtype=np.float32)
    up = i % 17; left = i % 13; diag = i % 11
    match = diag + np.where((np.arange(n)%7)==0, 2.0, -1.0).astype(np.float32)
    return np.maximum(match, np.maximum(up - V.F32(1.0), left - V.F32(1.0))).astype(np.float32)
""")

add_case(CASES, {
    "id": "hecbenchStreamclusterDistance", "name": "HeCBench streamcluster distance", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "streamcluster-cuda/main.cu", "fidelity": "inspired_by", "sizes": [4096, 8], "tol": 1e-5,
    "features": ["__global__", "distance", "clustering"],
    "description": "Compute squared distances to a candidate center.",
    "notes": "Standalone distance kernel inspired by HeCBench/Rodinia streamcluster CUDA benchmark.",
}, r"""
__global__ void dist_center(const float*x,const float*c,float*y,int pts,int dim){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<pts){float d=0.0f;for(int j=0;j<dim;++j){float q=x[p*dim+j]-c[j];d+=q*q;}y[p]=d;}}
int main(int argc,char**argv){const int pts=4096,dim=8;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)pts*dim*sizeof(float)),*hc=(float*)malloc((size_t)dim*sizeof(float)),*hy=(float*)malloc((size_t)pts*sizeof(float));for(int i=0;i<pts*dim;++i)hx[i]=hs(i,123);for(int i=0;i<dim;++i)hc[i]=hs(i,321);
float*dx,*dc,*dy;CK(cudaMalloc(&dx,(size_t)pts*dim*sizeof(float)));CK(cudaMalloc(&dc,(size_t)dim*sizeof(float)));CK(cudaMalloc(&dy,(size_t)pts*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)pts*dim*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dc,hc,(size_t)dim*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(pts+tpb-1)/tpb;dist_center<<<grid,tpb>>>(dx,dc,dy,pts,dim);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)pts*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,pts);cudaFree(dx);cudaFree(dc);cudaFree(dy);free(hx);free(hc);free(hy);return 0;}
""", """
def reference(meta):
    pts, dim = meta["input"]["sizes"]
    x = V.gen_hashsigned(pts*dim,123).reshape(pts,dim)
    c = V.gen_hashsigned(dim,321)
    return np.sum((x-c.reshape(1,dim))**2, axis=1, dtype=np.float32)
""")

add_case(CASES, {
    "id": "hecbenchParticleLikelihood", "name": "HeCBench particle filter likelihood", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "particlefilter-cuda/main.cu", "fidelity": "inspired_by", "sizes": [262144], "tol": 1e-5,
    "features": ["__global__", "particle_filter", "expf"],
    "description": "Particle likelihood weight update.",
    "notes": "Standalone likelihood kernel inspired by HeCBench/Rodinia particle filter.",
}, elemwise_main("particle_like", "float d=x[i]-g[i]; y[i]=expf(-0.5f*d*d);", 262144, 2.0, 2.0),
"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.F32(2.0) * V.gen_hashsigned(n, 123)
    g = V.F32(2.0) * V.gen_hashsigned(n, 321)
    d = x - g
    return np.exp(V.F32(-0.5) * d * d, dtype=np.float32)
""")

# CUDA primitive / API feature cases from NVIDIA CUDA Samples.
add_case(CASES, {
    "id": "cudaSamplesAtomicHistogram", "name": "CUDA Samples atomic histogram", "category": "medium", "domain": "cuda_primitive", "difficulty": "hard",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics.cu", "fidelity": "inspired_by", "sizes": [262144, 64], "tol": 0.0,
    "features": ["__global__", "atomicAdd", "histogram"],
    "description": "Atomic histogram over deterministic bins.",
    "notes": "Standalone atomic histogram inspired by NVIDIA cuda-samples simpleAtomicIntrinsics.",
}, r"""
__global__ void hist(const int*b,float*out,int n,int bins){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(&out[b[i]],1.0f);}
int main(int argc,char**argv){const int n=262144,bins=64;const char*outp=(argc>1)?argv[1]:"output/output.txt";int*hb=(int*)malloc((size_t)n*sizeof(int));float*hy=(float*)calloc((size_t)bins,sizeof(float));for(int i=0;i<n;++i)hb[i]=(int)floorf(h01(i,123)*bins);
int*db;float*dy;CK(cudaMalloc(&db,(size_t)n*sizeof(int)));CK(cudaMalloc(&dy,(size_t)bins*sizeof(float)));CK(cudaMemcpy(db,hb,(size_t)n*sizeof(int),cudaMemcpyHostToDevice));CK(cudaMemset(dy,0,(size_t)bins*sizeof(float)));int tpb=256,grid=(n+tpb-1)/tpb;hist<<<grid,tpb>>>(db,dy,n,bins);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)bins*sizeof(float),cudaMemcpyDeviceToHost));write_vec(outp,hy,bins);cudaFree(db);cudaFree(dy);free(hb);free(hy);return 0;}
""", """
def reference(meta):
    n, bins = meta["input"]["sizes"]
    b = np.floor(V.gen_hash01(n,123)*bins).astype(np.int64)
    return np.bincount(b, minlength=bins).astype(np.float32)
""")

add_case(CASES, {
    "id": "cudaSamplesWarpShuffleReduce", "name": "CUDA Samples warp shuffle reduce", "category": "medium", "domain": "cuda_primitive", "difficulty": "hard",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTemplates/simpleTemplates.cu", "fidelity": "inspired_by", "sizes": [8192, 32], "tol": 1e-5,
    "features": ["__global__", "__shfl_down_sync", "warp_reduction"],
    "description": "Warp-level sum reduction using shuffle intrinsics.",
    "notes": "Standalone warp shuffle reduction inspired by NVIDIA CUDA Samples shuffle examples.",
}, r"""
__global__ void warp_reduce(const float*x,float*y,int groups){int g=blockIdx.x, lane=threadIdx.x;float v=x[g*32+lane];for(int off=16;off>0;off>>=1)v+=__shfl_down_sync(0xffffffff,v,off);if(lane==0)y[g]=v;}
int main(int argc,char**argv){const int groups=8192,w=32,n=groups*w;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)groups*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);
float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)groups*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));warp_reduce<<<groups,32>>>(dx,dy,groups);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)groups*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,groups);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    groups, w = meta["input"]["sizes"]
    x = V.gen_hashsigned(groups*w,123).reshape(groups,w)
    return np.sum(x, axis=1, dtype=np.float32)
""")

add_case(CASES, {
    "id": "cudaSamplesDynamicSharedReverse", "name": "CUDA Samples dynamic shared reverse", "category": "medium", "domain": "cuda_primitive", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTemplates/sharedmem.cuh", "fidelity": "inspired_by", "sizes": [262144], "tol": 1e-6,
    "features": ["__global__", "dynamic_shared_memory", "block_reverse"],
    "description": "Reverse values inside each block using dynamic shared memory.",
    "notes": "Standalone dynamic shared memory pattern inspired by CUDA Samples simpleTemplates.",
}, r"""
__global__ void reverse_blocks(const float*x,float*y,int n){extern __shared__ float s[];int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)s[threadIdx.x]=x[i];__syncthreads();if(i<n)y[i]=s[blockDim.x-1-threadIdx.x];}
int main(int argc,char**argv){const int n=262144;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);
float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(n+tpb-1)/tpb;reverse_blocks<<<grid,tpb,tpb*sizeof(float)>>>(dx,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n,123).reshape(-1,256)
    return x[:, ::-1].reshape(-1)
""")

add_case(CASES, {
    "id": "cudaSamplesUnifiedMemoryScale", "name": "CUDA Samples unified memory scale", "category": "library_api", "domain": "library_api", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/UnifiedMemoryStreams/UnifiedMemoryStreams.cu", "fidelity": "inspired_by", "sizes": [1048576], "tol": 1e-6,
    "features": ["cudaMallocManaged", "__global__", "unified_memory"],
    "description": "Scale and shift using CUDA unified memory allocation.",
    "notes": "Standalone unified-memory kernel inspired by CUDA Samples UnifiedMemoryStreams.",
}, r"""
__global__ void scale(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=1.5f*x[i]+0.25f;}
int main(int argc,char**argv){const int n=1048576;const char*out=(argc>1)?argv[1]:"output/output.txt";float*x;CK(cudaMallocManaged(&x,(size_t)n*sizeof(float)));for(int i=0;i<n;++i)x[i]=hs(i,123);int tpb=256,grid=(n+tpb-1)/tpb;scale<<<grid,tpb>>>(x,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());write_vec(out,x,n);cudaFree(x);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    return V.F32(1.5) * V.gen_hashsigned(n,123) + V.F32(0.25)
""")

add_case(CASES, {
    "id": "cudaSamplesStreamOverlapAdd", "name": "CUDA Samples stream overlap add", "category": "library_api", "domain": "library_api", "difficulty": "hard",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleStreams/simpleStreams.cu", "fidelity": "inspired_by", "sizes": [1048576], "tol": 1e-6,
    "features": ["cudaStream_t", "cudaMemcpyAsync", "pinned_memory"],
    "description": "Use two streams and pinned host memory for independent vector additions.",
    "notes": "Standalone stream/pinned-memory pattern inspired by CUDA Samples simpleStreams.",
}, r"""
__global__ void addv(const float*a,const float*b,float*c,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}
int main(int argc,char**argv){const int n=1048576,half=n/2;const char*out=(argc>1)?argv[1]:"output/output.txt";float *ha,*hb,*hc;CK(cudaMallocHost(&ha,(size_t)n*sizeof(float)));CK(cudaMallocHost(&hb,(size_t)n*sizeof(float)));CK(cudaMallocHost(&hc,(size_t)n*sizeof(float)));for(int i=0;i<n;++i){ha[i]=hs(i,123);hb[i]=hs(i,321);}
float *da[2],*db[2],*dc[2];cudaStream_t st[2];for(int s=0;s<2;++s){CK(cudaStreamCreate(&st[s]));CK(cudaMalloc(&da[s],(size_t)half*sizeof(float)));CK(cudaMalloc(&db[s],(size_t)half*sizeof(float)));CK(cudaMalloc(&dc[s],(size_t)half*sizeof(float)));CK(cudaMemcpyAsync(da[s],ha+s*half,(size_t)half*sizeof(float),cudaMemcpyHostToDevice,st[s]));CK(cudaMemcpyAsync(db[s],hb+s*half,(size_t)half*sizeof(float),cudaMemcpyHostToDevice,st[s]));addv<<<(half+255)/256,256,0,st[s]>>>(da[s],db[s],dc[s],half);CK(cudaMemcpyAsync(hc+s*half,dc[s],(size_t)half*sizeof(float),cudaMemcpyDeviceToHost,st[s]));}
CK(cudaDeviceSynchronize());write_vec(out,hc,n);for(int s=0;s<2;++s){cudaFree(da[s]);cudaFree(db[s]);cudaFree(dc[s]);cudaStreamDestroy(st[s]);}cudaFreeHost(ha);cudaFreeHost(hb);cudaFreeHost(hc);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    return V.gen_hashsigned(n,123) + V.gen_hashsigned(n,321)
""")

add_case(CASES, {
    "id": "cudaSamplesConstantMemoryScale", "name": "CUDA Samples constant memory scale", "category": "medium", "domain": "cuda_primitive", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTemplates/simpleTemplates.cu", "fidelity": "inspired_by", "sizes": [1048576], "tol": 1e-6,
    "features": ["__constant__", "__global__", "elementwise"],
    "description": "Use constant memory coefficients in a scaling kernel.",
    "notes": "Standalone constant-memory pattern inspired by CUDA Samples.",
}, r"""
__constant__ float coeff[4];
__global__ void cmem(const float*x,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=coeff[i&3]*x[i]+coeff[(i+1)&3];}
int main(int argc,char**argv){const int n=1048576;const char*out=(argc>1)?argv[1]:"output/output.txt";float hcoef[4]={0.25f,0.5f,0.75f,1.0f};float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);
CK(cudaMemcpyToSymbol(coeff,hcoef,4*sizeof(float)));float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(n+tpb-1)/tpb;cmem<<<grid,tpb>>>(dx,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n,123)
    coeff = np.array([0.25,0.5,0.75,1.0], dtype=np.float32)
    idx = np.arange(n)
    return coeff[idx & 3] * x + coeff[(idx + 1) & 3]
""")

add_case(CASES, {
    "id": "cudaSamplesTextureLikeBilinear", "name": "CUDA Samples texture-like bilinear", "category": "medium", "domain": "cuda_primitive", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTexture/simpleTexture.cu", "fidelity": "inspired_by", "sizes": [128, 128], "tol": 1e-5,
    "features": ["__global__", "bilinear_sample", "2D_indexing"],
    "description": "Texture-sampling style bilinear interpolation over a 2D array.",
    "notes": "Uses regular global memory to preserve portability while modeling CUDA Samples simpleTexture behavior.",
}, r"""
__global__ void bilinear(const float*x,float*y,int H,int W){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=H*W;if(idx<n){int r=idx/W,c=idx%W;float u=(float)c+0.35f,v=(float)r+0.65f;int x0=min((int)floorf(u),W-1),x1=min(x0+1,W-1),y0=min((int)floorf(v),H-1),y1=min(y0+1,H-1);float fx=u-x0,fy=v-y0;float a=x[y0*W+x0]*(1-fx)+x[y0*W+x1]*fx;float b=x[y1*W+x0]*(1-fx)+x[y1*W+x1]*fx;y[idx]=a*(1-fy)+b*fy;}}
int main(int argc,char**argv){const int H=128,W=128,n=H*W;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=h01(i,123);
float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(n+tpb-1)/tpb;bilinear<<<grid,tpb>>>(dx,dy,H,W);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    H,W = meta["input"]["sizes"]
    x = V.gen_hash01(H*W,123).reshape(H,W)
    y = np.empty((H,W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            u = V.F32(c) + V.F32(0.35); v = V.F32(r) + V.F32(0.65)
            x0 = min(int(np.floor(u)), W-1); x1 = min(x0+1, W-1); y0 = min(int(np.floor(v)), H-1); y1 = min(y0+1, H-1)
            fx = u - V.F32(x0); fy = v - V.F32(y0)
            a = x[y0,x0]*(V.F32(1.0)-fx)+x[y0,x1]*fx
            b = x[y1,x0]*(V.F32(1.0)-fx)+x[y1,x1]*fx
            y[r,c] = a*(V.F32(1.0)-fy)+b*fy
    return y.reshape(-1)
""")

# Thrust library API cases.
for cid, rel, desc, features, main_expr, ref_expr in [
    ("thrustTransformReduceNorm", "thrust/examples/sum.cu", "Thrust transform-reduce vector L2 norm", ["thrust", "transform_reduce"], "return sqrtf(thrust::transform_reduce(vec.begin(), vec.end(), square_op(), 0.0f, thrust::plus<float>()));", "np.array([np.sqrt(np.sum(x*x, dtype=np.float32), dtype=np.float32)], dtype=np.float32)"),
]:
    pass

add_case(CASES, {
    "id": "thrustTransformReduceNorm", "name": "Thrust transform-reduce norm", "category": "library_api", "domain": "library_api", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/6_Performance/UnifiedMemoryPerf/matrixMultiplyPerf.cu", "fidelity": "inspired_by", "sizes": [1048576], "tol": 2e-4,
    "features": ["thrust::device_vector", "thrust::transform_reduce", "functor"],
    "description": "Compute L2 norm with thrust::transform_reduce.",
    "notes": "Standalone Thrust API benchmark inspired by CUDA Samples' Thrust/CUDA utility usage.",
}, r"""
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
struct square_op { __host__ __device__ float operator()(float x) const { return x*x; } };
int main(int argc,char**argv){const int n=1048576;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);thrust::device_vector<float> v(hx,hx+n);float sum=thrust::transform_reduce(v.begin(),v.end(),square_op(),0.0f,thrust::plus<float>());float y=sqrtf(sum);write_vec(out,&y,1);free(hx);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n,123)
    return np.array([np.sqrt(np.sum(x*x, dtype=np.float32), dtype=np.float32)], dtype=np.float32)
""")

add_case(CASES, {
    "id": "thrustInclusiveScan", "name": "Thrust inclusive scan", "category": "library_api", "domain": "library_api", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTemplates/simpleTemplates.cu", "fidelity": "inspired_by", "sizes": [262144], "tol": 1e-5,
    "features": ["thrust::device_vector", "thrust::inclusive_scan"],
    "description": "Prefix sum using thrust::inclusive_scan.",
    "notes": "Standalone Thrust scan case using CUDA sample-style deterministic input.",
}, r"""
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
int main(int argc,char**argv){const int n=262144;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=0.001f*hs(i,123);thrust::device_vector<float> v(hx,hx+n), y(n);thrust::inclusive_scan(v.begin(),v.end(),y.begin());thrust::copy(y.begin(),y.end(),hy);write_vec(out,hy,n);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.F32(0.001) * V.gen_hashsigned(n,123)
    return np.cumsum(x, dtype=np.float32)
""")

add_case(CASES, {
    "id": "thrustSortByKey", "name": "Thrust sort by key", "category": "library_api", "domain": "library_api", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTemplates/simpleTemplates.cu", "fidelity": "inspired_by", "sizes": [65536], "tol": 0.0,
    "features": ["thrust::sort_by_key", "device_vector"],
    "description": "Sort key-value pairs by integer key and emit sorted values.",
    "notes": "Standalone Thrust sort_by_key API case.",
}, r"""
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
int main(int argc,char**argv){const int n=65536;const char*out=(argc>1)?argv[1]:"output/output.txt";int*hk=(int*)malloc((size_t)n*sizeof(int));float*hv=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i){hk[i]=(i*17+13)%n;hv[i]=(float)i;}thrust::device_vector<int> k(hk,hk+n);thrust::device_vector<float> v(hv,hv+n);thrust::sort_by_key(k.begin(),k.end(),v.begin());thrust::copy(v.begin(),v.end(),hy);write_vec(out,hy,n);free(hk);free(hv);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    k = ((np.arange(n, dtype=np.int64) * 17 + 13) % n).astype(np.int32)
    v = np.arange(n, dtype=np.float32)
    return v[np.argsort(k)]
""")

add_case(CASES, {
    "id": "thrustGatherMap", "name": "Thrust gather map", "category": "library_api", "domain": "library_api", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTemplates/simpleTemplates.cu", "fidelity": "inspired_by", "sizes": [262144], "tol": 1e-6,
    "features": ["thrust::gather", "device_vector"],
    "description": "Gather values through a deterministic index map.",
    "notes": "Standalone Thrust gather API case.",
}, r"""
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
int main(int argc,char**argv){const int n=262144;const char*out=(argc>1)?argv[1]:"output/output.txt";int*hm=(int*)malloc((size_t)n*sizeof(int));float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i){hm[i]=(i*17+13)%n;hx[i]=hs(i,123);}thrust::device_vector<int> m(hm,hm+n);thrust::device_vector<float> x(hx,hx+n), y(n);thrust::gather(m.begin(),m.end(),x.begin(),y.begin());thrust::copy(y.begin(),y.end(),hy);write_vec(out,hy,n);free(hm);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    m = (np.arange(n, dtype=np.int64)*17 + 13) % n
    x = V.gen_hashsigned(n,123)
    return x[m]
""")


def write_case(spec: dict) -> None:
    case_dir = CASES_ROOT / spec["category"] / spec["id"]
    for rel in ("original", "tests", "input", "expected", "output", "logs", "migrated"):
        (case_dir / rel).mkdir(parents=True, exist_ok=True)
    meta = metadata(spec)
    (case_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (case_dir / "README.md").write_text(
        f"# {meta['name']}\n\n{meta['description']}\n\n"
        f"Source project: {meta['source_project']}\n\n"
        f"Source URL: {meta['source_url']}\n\n"
        f"License: {meta['license']}\n\n"
        f"Extraction fidelity: {meta['extraction_fidelity']}\n\n"
        f"Extraction notes: {meta['extraction_notes']}\n",
        encoding="utf-8",
    )
    (case_dir / "original" / "README.md").write_text(
        f"# Original CUDA\n\nStandalone Stage 1 CUDA case for `{spec['id']}`.\n",
        encoding="utf-8",
    )
    (case_dir / "original" / "CMakeLists.txt").write_text(CMAKELISTS, encoding="utf-8")
    (case_dir / "original" / "main.cu").write_text(COMMON_CUDA + "\n\n" + spec["main"].strip() + "\n", encoding="utf-8")
    (case_dir / "tests" / "verify.py").write_text(
        VERIFY_PREFIX + spec["verify"].strip() + "\n\nif __name__ == \"__main__\":\n    V.run(reference)\n",
        encoding="utf-8",
    )


def main() -> int:
    seen = set()
    for spec in CASES:
        if spec["id"] in seen:
            raise RuntimeError(f"duplicate case id {spec['id']}")
        seen.add(spec["id"])
        write_case(spec)
        src = SOURCES[spec["source"]]["project"]
        print(f"[ok] {spec['category']}/{spec['id']} <- {src}")
    print(f"Wrote {len(CASES)} Stage 1 batch-2 cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
