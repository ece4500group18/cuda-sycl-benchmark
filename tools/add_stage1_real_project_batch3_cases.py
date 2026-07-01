#!/usr/bin/env python3
"""Add the next Stage 1 real-project expansion batch.

This script adds 40 standalone CUDA cases after the 171-case checkpoint.  The
cases stay CUDA-only for Stage 1, and each one carries real-project attribution,
license information, extraction fidelity, extraction notes, and a CPU-reference
verifier.
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
    "xformers": {
        "project": "facebookresearch/xformers",
        "license": "BSD-3-Clause",
        "base": "https://github.com/facebookresearch/xformers/blob/main",
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
project(stage1_real_project_batch3 CUDA)
add_executable(app main.cu)
set_target_properties(app PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
"""


def source_url(source_key: str, rel: str) -> str:
    return f"{SOURCES[source_key]['base']}/{rel}"


def metadata(spec: dict) -> dict:
    src = SOURCES[spec["source"]]
    src_url = source_url(spec["source"], spec["rel"])
    fidelity = spec.get("fidelity", "simplified")
    return {
        "case_id": spec["id"],
        "name": spec["name"],
        "category": spec["category"],
        "domain": spec["domain"],
        "difficulty": spec["difficulty"],
        "source_project": src["project"],
        "source_url": src_url,
        "license": src["license"],
        "adaptation_type": fidelity,
        "extraction_fidelity": fidelity,
        "extraction_notes": spec["notes"],
        "description": spec["description"],
        "source": {
            "type": fidelity,
            "url": src_url,
            "license": src["license"],
            "original_path": spec["rel"],
        },
        "cuda_features": spec["features"],
        "libraries": spec.get("libraries", []),
        "input": {"type": "hashed", "sizes": spec["sizes"], "seed": 123},
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


def elemwise_main(kernel: str, expr: str, n: int, xscale: float = 1.0, gscale: float = 1.0) -> str:
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


def elemwise_verify(expr: str, xscale: float = 1.0, gscale: float = 1.0) -> str:
    return f"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.F32({xscale}) * V.gen_hashsigned(n, 123)
    g = V.F32({gscale}) * V.gen_hashsigned(n, 321)
    return {expr}
"""


def add_case(cases: list[dict], spec: dict, main: str, verify: str) -> None:
    spec = dict(spec)
    spec["main"] = main
    spec["verify"] = verify
    cases.append(spec)


CASES: list[dict] = []

# Modern ML / LLM kernels.
for spec, expr, py_expr, xscale, gscale in [
    ({
        "id": "ggmlNegKernel", "name": "ggml unary negate", "category": "ai", "domain": "modern_ml", "difficulty": "easy",
        "source": "llama", "rel": "unary.cu", "fidelity": "simplified", "sizes": [1048576], "tol": 1e-6,
        "features": ["__global__", "elementwise", "unary_op"],
        "description": "Unary tensor negation matching ggml CUDA unary operator shape.",
        "notes": "Simplified standalone unary negation following llama.cpp/ggml unary CUDA operator conventions.",
    }, "y[i] = -x[i];", "-x", 3.0, 1.0),
    ({
        "id": "ggmlSqrKernel", "name": "ggml square unary op", "category": "ai", "domain": "modern_ml", "difficulty": "easy",
        "source": "llama", "rel": "unary.cu", "fidelity": "simplified", "sizes": [1048576], "tol": 1e-6,
        "features": ["__global__", "elementwise", "unary_op"],
        "description": "Elementwise square tensor operation.",
        "notes": "Simplified standalone square unary op derived from the ggml CUDA unary operator family.",
    }, "y[i] = x[i] * x[i];", "x * x", 2.0, 1.0),
    ({
        "id": "vllmSoftcapLogits", "name": "vLLM softcap logits", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
        "source": "vllm", "rel": "activation_kernels.cu", "fidelity": "simplified", "sizes": [1048576], "tol": 1e-5,
        "features": ["__global__", "tanhf", "logit_transform"],
        "description": "Softcap transform for logits using a tanh clamp.",
        "notes": "Standalone softcap/logit transform inspired by vLLM activation-style CUDA kernels.",
    }, "const float s = 30.0f; y[i] = s * tanhf(x[i] / s);", "V.F32(30.0) * np.tanh(x / V.F32(30.0)).astype(np.float32)", 10.0, 1.0),
    ({
        "id": "vllmSwigluGate2", "name": "vLLM gated SiLU product", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
        "source": "vllm", "rel": "activation_kernels.cu", "fidelity": "simplified", "sizes": [1048576], "tol": 1e-5,
        "features": ["__global__", "expf", "gated_activation"],
        "description": "SiLU gate multiplied by a second tensor.",
        "notes": "Standalone gated SiLU product preserving vLLM activation kernel arithmetic.",
    }, "float v=x[i]; y[i] = (v / (1.0f + expf(-v))) * g[i];", "(x / (V.F32(1.0) + np.exp(-x).astype(np.float32))) * g", 5.0, 2.0),
    ({
        "id": "bnbAdamWDecayStep", "name": "bitsandbytes AdamW decay step", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
        "source": "bnb", "rel": "kernels.cu", "fidelity": "inspired_by", "sizes": [1048576], "tol": 1e-5,
        "features": ["__global__", "optimizer", "sqrtf"],
        "description": "Fused AdamW-style parameter update from deterministic moments.",
        "notes": "Standalone optimizer update inspired by bitsandbytes Adam/Adam8bit CUDA kernels.",
    }, "float m=0.9f*x[i]+0.1f*g[i]; float v=0.99f*x[i]*x[i]+0.01f*g[i]*g[i]; y[i]=x[i]-0.001f*m/(sqrtf(v)+1.0e-6f)-0.0001f*x[i];", "x - V.F32(0.001) * (V.F32(0.9) * x + V.F32(0.1) * g) / (np.sqrt(V.F32(0.99) * x * x + V.F32(0.01) * g * g).astype(np.float32) + V.F32(1.0e-6)) - V.F32(0.0001) * x", 1.0, 1.0),
    ({
        "id": "cutlassLinearCombinationEpilogue", "name": "CUTLASS linear combination epilogue", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
        "source": "cutlass", "rel": "02_dump_reg_shmem/dump_reg_shmem.cu", "fidelity": "inspired_by", "sizes": [1048576], "tol": 1e-6,
        "features": ["__global__", "epilogue", "linear_combination"],
        "description": "Linear combination epilogue y = alpha*x + beta*g + bias.",
        "notes": "Standalone epilogue arithmetic inspired by CUTLASS linear-combination output operators.",
    }, "y[i] = 1.25f * x[i] + 0.5f * g[i] + 0.125f;", "V.F32(1.25) * x + V.F32(0.5) * g + V.F32(0.125)", 1.0, 1.0),
]:
    add_case(CASES, spec, elemwise_main(spec["id"], expr, spec["sizes"][0], xscale, gscale), elemwise_verify(py_expr, xscale, gscale))

add_case(CASES, {
    "id": "ggmlAddRowsBroadcast", "name": "ggml row broadcast add", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "llama", "rel": "binary.cu", "fidelity": "simplified", "sizes": [512, 256], "tol": 1e-6,
    "features": ["__global__", "broadcast", "2D_indexing"],
    "description": "Add a per-column row vector to a matrix.",
    "notes": "Simplified matrix-plus-row broadcast following ggml CUDA binary operator patterns.",
}, r"""
__global__ void add_rows(const float *x, const float *b, float *y, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x, n = rows * cols;
  if (idx < n) y[idx] = x[idx] + b[idx % cols];
}
int main(int argc, char **argv) {
  const int rows=512, cols=256, n=rows*cols; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hb=(float*)malloc((size_t)cols*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for(int i=0;i<n;++i) hx[i]=hs(i,123); for(int c=0;c<cols;++c) hb[c]=0.25f*hs(c,321);
  float *dx,*db,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&db,(size_t)cols*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(db,hb,(size_t)cols*sizeof(float),cudaMemcpyHostToDevice));
  add_rows<<<(n+255)/256,256>>>(dx,db,dy,rows,cols); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n); cudaFree(dx); cudaFree(db); cudaFree(dy); free(hx); free(hb); free(hy); return 0;
}
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    b = V.F32(0.25) * V.gen_hashsigned(cols, 321)
    return (x + b.reshape(1, cols)).reshape(-1)
""")

add_case(CASES, {
    "id": "ggmlRepeatRows", "name": "ggml repeat rows", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "llama", "rel": "repeat.cu", "fidelity": "simplified", "sizes": [1024, 256, 128], "tol": 1e-6,
    "features": ["__global__", "repeat", "layout"],
    "description": "Repeat a smaller row tensor into a larger row-major tensor.",
    "notes": "Simplified repeat-layout kernel inspired by llama.cpp/ggml repeat.cu.",
}, r"""
__global__ void repeat_rows(const float *src, float *dst, int rows, int src_rows, int cols) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x, n=rows*cols;
  if(idx<n){ int r=idx/cols, c=idx%cols; dst[idx]=src[(r%src_rows)*cols+c]; }
}
int main(int argc,char**argv){const int rows=1024,src_rows=256,cols=128,n=rows*cols,ns=src_rows*cols;const char*out=(argc>1)?argv[1]:"output/output.txt";
float *hsr=(float*)malloc((size_t)ns*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<ns;++i)hsr[i]=hs(i,123);
float *ds,*dy;CK(cudaMalloc(&ds,(size_t)ns*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(ds,hsr,(size_t)ns*sizeof(float),cudaMemcpyHostToDevice));
repeat_rows<<<(n+255)/256,256>>>(ds,dy,rows,src_rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);
cudaFree(ds);cudaFree(dy);free(hsr);free(hy);return 0;}
""", """
def reference(meta):
    rows, src_rows, cols = meta["input"]["sizes"]
    src = V.gen_hashsigned(src_rows * cols, 123).reshape(src_rows, cols)
    out = np.empty((rows, cols), dtype=np.float32)
    for r in range(rows):
        out[r] = src[r % src_rows]
    return out.reshape(-1)
""")

add_case(CASES, {
    "id": "ggmlSumRows2", "name": "ggml row sum reduction", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "llama", "rel": "sumrows.cu", "fidelity": "simplified", "sizes": [2048, 64], "tol": 1e-5,
    "features": ["__global__", "row_reduction"],
    "description": "Compute row sums for a row-major matrix.",
    "notes": "Standalone row reduction inspired by ggml CUDA sum rows operators.",
}, r"""
__global__ void sum_rows(const float *x, float *y, int rows, int cols) {
  int r=blockIdx.x*blockDim.x+threadIdx.x;
  if(r<rows){ float s=0.0f; for(int c=0;c<cols;++c) s += x[r*cols+c]; y[r]=s; }
}
int main(int argc,char**argv){const int rows=2048,cols=64,n=rows*cols;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)rows*sizeof(float));for(int i=0;i<n;++i)hx[i]=0.01f*hs(i,123);
float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)rows*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));sum_rows<<<(rows+255)/256,256>>>(dx,dy,rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)rows*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,rows);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = (V.F32(0.01) * V.gen_hashsigned(rows * cols, 123)).reshape(rows, cols)
    y = np.zeros(rows, dtype=np.float32)
    for c in range(cols):
        y += x[:, c]
    return y
""")

add_case(CASES, {
    "id": "vllmKvCacheScatter", "name": "vLLM paged KV scatter", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "vllm", "rel": "cache_kernels.cu", "fidelity": "simplified", "sizes": [512, 8, 64], "tol": 1e-6,
    "features": ["__global__", "scatter", "cache_layout"],
    "description": "Scatter token-major KV data into a page-major cache layout.",
    "notes": "Standalone page scatter mirroring vLLM cache kernel layout movement.",
}, r"""
__global__ void kv_scatter(const float *src, float *cache, int tokens, int heads, int dim) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x,n=tokens*heads*dim;
  if(idx<n){int d=idx%dim;int tmp=idx/dim;int h=tmp%heads;int t=tmp/heads;int page=(t*37)%tokens;cache[(h*tokens+page)*dim+d]=src[idx];}
}
int main(int argc,char**argv){const int tokens=512,heads=8,dim=64,n=tokens*heads*dim;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hsrc=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)calloc((size_t)n,sizeof(float));for(int i=0;i<n;++i)hsrc[i]=hs(i,123);float*ds,*dc;CK(cudaMalloc(&ds,(size_t)n*sizeof(float)));CK(cudaMalloc(&dc,(size_t)n*sizeof(float)));CK(cudaMemcpy(ds,hsrc,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemset(dc,0,(size_t)n*sizeof(float)));kv_scatter<<<(n+255)/256,256>>>(ds,dc,tokens,heads,dim);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dc,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(ds);cudaFree(dc);free(hsrc);free(hy);return 0;}
""",
"""
def reference(meta):
    tokens, heads, dim = meta["input"]["sizes"]
    src = V.gen_hashsigned(tokens * heads * dim, 123).reshape(tokens, heads, dim)
    cache = np.zeros((heads, tokens, dim), dtype=np.float32)
    for t in range(tokens):
        page = (t * 37) % tokens
        cache[:, page, :] = src[t]
    return cache.reshape(-1)
""")

add_case(CASES, {
    "id": "bnbInt8RowDequant", "name": "bitsandbytes int8 row dequant", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "bnb", "rel": "kernels.cu", "fidelity": "simplified", "sizes": [1024, 128], "tol": 1e-6,
    "features": ["__global__", "int8", "dequantization"],
    "description": "Dequantize int8 row blocks using per-row scale factors.",
    "notes": "Standalone row-scale dequantization based on bitsandbytes int8 CUDA kernels.",
}, r"""
__global__ void deq(const signed char *q,const float *scale,float *y,int rows,int cols){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=rows*cols;if(idx<n){int r=idx/cols;y[idx]=(float)q[idx]*scale[r];}}
int main(int argc,char**argv){const int rows=1024,cols=128,n=rows*cols;const char*out=(argc>1)?argv[1]:"output/output.txt";signed char*hq=(signed char*)malloc((size_t)n);float*hscl=(float*)malloc((size_t)rows*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hq[i]=(signed char)((int)floorf(h01(i,123)*255.0f)-127);for(int r=0;r<rows;++r)hscl[r]=0.001f+0.01f*h01(r,77);
signed char*dq;float*ds,*dy;CK(cudaMalloc(&dq,(size_t)n));CK(cudaMalloc(&ds,(size_t)rows*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dq,hq,(size_t)n,cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,hscl,(size_t)rows*sizeof(float),cudaMemcpyHostToDevice));deq<<<(n+255)/256,256>>>(dq,ds,dy,rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dq);cudaFree(ds);cudaFree(dy);free(hq);free(hscl);free(hy);return 0;}
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    q = (np.floor(V.gen_hash01(rows * cols, 123) * V.F32(255.0)).astype(np.int32) - 127).astype(np.float32).reshape(rows, cols)
    scale = V.F32(0.001) + V.F32(0.01) * V.gen_hash01(rows, 77)
    return (q * scale.reshape(rows, 1)).reshape(-1)
""")

add_case(CASES, {
    "id": "flashScaleMaskScores", "name": "FlashAttention scaled masked scores", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "flash", "rel": "mask.h", "fidelity": "simplified", "sizes": [512, 128], "tol": 1e-6,
    "features": ["__global__", "attention_mask", "2D_indexing"],
    "description": "Scale attention scores and apply a causal-style mask.",
    "notes": "Standalone score scaling and mask application inspired by FlashAttention mask helpers.",
}, r"""
__global__ void scale_mask(const float*x,float*y,int rows,int cols){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=rows*cols;if(idx<n){int r=idx/cols,c=idx%cols;float v=x[idx]*0.125f;y[idx]=(c>(r%cols))?-10000.0f:v;}}
int main(int argc,char**argv){const int rows=512,cols=128,n=rows*cols;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));scale_mask<<<(n+255)/256,256>>>(dx,dy,rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    y = V.F32(0.125) * x
    for r in range(rows):
        y[r, np.arange(cols) > (r % cols)] = V.F32(-10000.0)
    return y.reshape(-1)
""")

add_case(CASES, {
    "id": "flashRowMaxStats", "name": "FlashAttention row max stats", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "flash", "rel": "softmax.h", "fidelity": "simplified", "sizes": [2048, 64], "tol": 1e-6,
    "features": ["__global__", "row_reduction", "softmax_stats"],
    "description": "Compute per-row max statistics used by online softmax.",
    "notes": "Standalone row-max statistic extraction inspired by FlashAttention softmax helpers.",
}, r"""
__global__ void row_max(const float*x,float*y,int rows,int cols){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r<rows){float m=-3.402823e38f;for(int c=0;c<cols;++c)m=fmaxf(m,x[r*cols+c]);y[r]=m;}}
int main(int argc,char**argv){const int rows=2048,cols=64,n=rows*cols;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)rows*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)rows*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));row_max<<<(rows+255)/256,256>>>(dx,dy,rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)rows*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,rows);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    return np.max(x, axis=1).astype(np.float32)
""")

add_case(CASES, {
    "id": "cutlassColumnMajorToRowMajor", "name": "CUTLASS layout conversion", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "cutlass", "rel": "14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm.cu", "fidelity": "inspired_by", "sizes": [256, 256], "tol": 1e-6,
    "features": ["__global__", "layout_conversion", "2D_indexing"],
    "description": "Convert a column-major matrix buffer into row-major order.",
    "notes": "Standalone matrix layout conversion inspired by CUTLASS layout/template examples.",
}, r"""
__global__ void col_to_row(const float*col,float*row,int R,int C){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=R*C;if(idx<n){int r=idx/C,c=idx%C;row[idx]=col[c*R+r];}}
int main(int argc,char**argv){const int R=256,C=256,n=R*C;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hc=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hc[i]=hs(i,123);float*dc,*dy;CK(cudaMalloc(&dc,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dc,hc,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));col_to_row<<<(n+255)/256,256>>>(dc,dy,R,C);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dc);cudaFree(dy);free(hc);free(hy);return 0;}
""", """
def reference(meta):
    R, C = meta["input"]["sizes"]
    col = V.gen_hashsigned(R * C, 123).reshape(C, R)
    return col.T.reshape(-1)
""")

add_case(CASES, {
    "id": "xformersBlockSparseMaskApply", "name": "xFormers block-sparse mask apply", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "xformers", "rel": "xformers/csrc/sparse24/sparse24_largest_mask_2d.cu", "fidelity": "inspired_by", "sizes": [512, 128], "tol": 1e-6,
    "features": ["__global__", "block_sparse", "masking"],
    "description": "Apply a deterministic block-sparse mask to an attention-like matrix.",
    "notes": "Standalone block mask application inspired by xFormers sparse CUDA masking kernels.",
}, r"""
__global__ void block_mask(const float*x,float*y,int rows,int cols){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=rows*cols;if(idx<n){int r=idx/cols,c=idx%cols;int br=r/16,bc=c/16;bool keep=((br*7+bc*3)&3)!=0;y[idx]=keep?x[idx]:0.0f;}}
int main(int argc,char**argv){const int rows=512,cols=128,n=rows*cols;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));block_mask<<<(n+255)/256,256>>>(dx,dy,rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    y = np.zeros_like(x)
    for r in range(rows):
        for c in range(cols):
            if (((r // 16) * 7 + (c // 16) * 3) & 3) != 0:
                y[r, c] = x[r, c]
    return y.reshape(-1)
""")

# HPC / scientific CUDA kernels.
add_case(CASES, {
    "id": "hecbenchLavaMdPairForce", "name": "HeCBench lavaMD pair force", "category": "hpc", "domain": "hpc", "difficulty": "hard",
    "source": "hecbench", "rel": "lavaMD-cuda/kernel/kernel_gpu_cuda_wrapper.cu", "fidelity": "simplified", "sizes": [2048, 32], "tol": 1e-4,
    "features": ["__global__", "n_body", "loop"],
    "description": "Compute a bounded pairwise force accumulation per particle.",
    "notes": "Simplified pair-neighbor accumulation preserving the lavaMD-style force loop structure.",
}, r"""
__global__ void force_kernel(const float*x,float*y,int n,int neigh){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float xi=x[i],acc=0.0f;for(int k=1;k<=neigh;++k){int j=(i+k*17)%n;float d=x[j]-xi;float r=rsqrtf(d*d+0.01f);acc += d*r*r*r;}y[i]=acc;}}
int main(int argc,char**argv){const int n=2048,neigh=32;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));force_kernel<<<(n+255)/256,256>>>(dx,dy,n,neigh);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n, neigh = meta["input"]["sizes"]
    x = V.gen_hashsigned(n, 123)
    y = np.zeros(n, dtype=np.float32)
    for i in range(n):
        acc = V.F32(0.0); xi = x[i]
        for k in range(1, neigh + 1):
            j = (i + k * 17) % n
            d = x[j] - xi
            r = V.F32(1.0) / np.sqrt(d * d + V.F32(0.01), dtype=np.float32)
            acc += d * r * r * r
        y[i] = acc
    return y
""")

add_case(CASES, {
    "id": "hecbenchCfdFlux", "name": "HeCBench CFD flux update", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "euler3d-cuda/euler3d.cu", "fidelity": "simplified", "sizes": [262144], "tol": 1e-5,
    "features": ["__global__", "finite_volume", "sqrtf"],
    "description": "Compute a compact finite-volume flux expression per cell.",
    "notes": "Standalone CFD flux arithmetic inspired by HeCBench/Rodinia euler3d CUDA kernels.",
}, elemwise_main("cfd_flux", "float rho=fabsf(x[i])+1.0f; float mom=g[i]; float e=0.5f*rho+0.25f*mom*mom; y[i]=mom+0.1f*(e/rho)+sqrtf(rho);", 262144, 1.0, 1.0),
"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123); g = V.gen_hashsigned(n, 321)
    rho = np.abs(x).astype(np.float32) + V.F32(1.0)
    e = V.F32(0.5) * rho + V.F32(0.25) * g * g
    return g + V.F32(0.1) * (e / rho) + np.sqrt(rho).astype(np.float32)
""")

add_case(CASES, {
    "id": "hecbenchJacobi3d7pt", "name": "HeCBench Jacobi 3D 7-point", "category": "hpc", "domain": "hpc", "difficulty": "hard",
    "source": "hecbench", "rel": "fdtd3d-cuda/main.cu", "fidelity": "simplified", "sizes": [32, 32, 32], "tol": 1e-6,
    "features": ["__global__", "3D_stencil", "finite_difference"],
    "description": "Single 7-point Jacobi stencil sweep over a 3D grid.",
    "notes": "Standalone 3D stencil based on HeCBench fdtd3d-style structured-grid kernels.",
}, r"""
__global__ void jacobi3d(const float*x,float*y,int X,int Y,int Z){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=X*Y*Z;if(idx<n){int z=idx%Z;int t=idx/Z;int yy=t%Y;int xx=t/Y;if(xx==0||yy==0||z==0||xx==X-1||yy==Y-1||z==Z-1){y[idx]=x[idx];}else{int yz=Y*Z;y[idx]=(x[idx]+x[idx-1]+x[idx+1]+x[idx-Z]+x[idx+Z]+x[idx-yz]+x[idx+yz])/7.0f;}}}
int main(int argc,char**argv){const int X=32,Y=32,Z=32,n=X*Y*Z;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));jacobi3d<<<(n+255)/256,256>>>(dx,dy,X,Y,Z);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    X, Y, Z = meta["input"]["sizes"]
    x = V.gen_hashsigned(X * Y * Z, 123).reshape(X, Y, Z)
    y = x.copy()
    y[1:-1,1:-1,1:-1] = (x[1:-1,1:-1,1:-1] + x[1:-1,1:-1,:-2] + x[1:-1,1:-1,2:] + x[1:-1,:-2,1:-1] + x[1:-1,2:,1:-1] + x[:-2,1:-1,1:-1] + x[2:,1:-1,1:-1]) / V.F32(7.0)
    return y.reshape(-1)
""")

add_case(CASES, {
    "id": "hecbenchBfsRelaxEdges", "name": "HeCBench BFS edge relax", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "bfs-cuda/bfs.cu", "fidelity": "simplified", "sizes": [65536], "tol": 0.0,
    "features": ["__global__", "graph", "frontier_relax"],
    "description": "Relax deterministic graph edges from a synthetic BFS frontier.",
    "notes": "Standalone one-step BFS edge relaxation modeled on HeCBench BFS kernels.",
}, r"""
__global__ void bfs_relax(float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int src=(i*17+13)%n;int frontier=(src%97)==0;y[i]=frontier?1.0f:0.0f;}}
int main(int argc,char**argv){const int n=65536;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hy=(float*)malloc((size_t)n*sizeof(float));float*dy;CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));bfs_relax<<<(n+255)/256,256>>>(dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dy);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    src = (np.arange(n, dtype=np.int64) * 17 + 13) % n
    return ((src % 97) == 0).astype(np.float32)
""")

add_case(CASES, {
    "id": "hecbenchMandelbrotIter", "name": "HeCBench Mandelbrot iterations", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "mandelbrot-cuda/main.cu", "fidelity": "inspired_by", "sizes": [256, 256, 64], "tol": 1.0,
    "features": ["__global__", "branching", "iteration"],
    "description": "Compute bounded Mandelbrot iteration counts on a small grid.",
    "notes": "Standalone iterative numerical kernel inspired by CUDA benchmark Mandelbrot examples.",
}, r"""
__global__ void mandel(float*y,int W,int H,int maxit){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=W*H;if(idx<n){int px=idx%W,py=idx/W;float cr=-2.0f+3.0f*(float)px/(float)(W-1),ci=-1.5f+3.0f*(float)py/(float)(H-1),zr=0.0f,zi=0.0f;int it=0;while(it<maxit&&zr*zr+zi*zi<=4.0f){float nzr=zr*zr-zi*zi+cr;zi=2.0f*zr*zi+ci;zr=nzr;++it;}y[idx]=(float)it;}}
int main(int argc,char**argv){const int W=256,H=256,maxit=64,n=W*H;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hy=(float*)malloc((size_t)n*sizeof(float));float*dy;CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));mandel<<<(n+255)/256,256>>>(dy,W,H,maxit);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dy);free(hy);return 0;}
""", """
def reference(meta):
    W, H, maxit = meta["input"]["sizes"]
    out = np.zeros(W * H, dtype=np.float32)
    for idx in range(W * H):
        px = idx % W; py = idx // W
        cr = V.F32(-2.0) + V.F32(3.0) * V.F32(px) / V.F32(W - 1)
        ci = V.F32(-1.5) + V.F32(3.0) * V.F32(py) / V.F32(H - 1)
        zr = V.F32(0.0); zi = V.F32(0.0); it = 0
        while it < maxit and zr * zr + zi * zi <= V.F32(4.0):
            nzr = zr * zr - zi * zi + cr
            zi = V.F32(2.0) * zr * zi + ci
            zr = nzr; it += 1
        out[idx] = V.F32(it)
    return out
""")

add_case(CASES, {
    "id": "hecbenchEllpackSpmv", "name": "HeCBench ELLPACK SpMV", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "amgmk-cuda/csr_matvec.cu", "fidelity": "simplified", "sizes": [4096, 8], "tol": 1e-6,
    "features": ["__global__", "sparse_matrix", "spmv"],
    "description": "Sparse matrix-vector multiply over a deterministic ELLPACK pattern.",
    "notes": "Standalone sparse matvec inspired by HeCBench AMG/CSR kernels.",
}, r"""
__global__ void ell_spmv(const float*val,const float*x,float*y,int rows,int width){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r<rows){float s=0.0f;for(int k=0;k<width;++k){int c=(r*17+k*13)%rows;s+=val[r*width+k]*x[c];}y[r]=s;}}
int main(int argc,char**argv){const int rows=4096,width=8,n=rows*width;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hv=(float*)malloc((size_t)n*sizeof(float)),*hx=(float*)malloc((size_t)rows*sizeof(float)),*hy=(float*)malloc((size_t)rows*sizeof(float));for(int i=0;i<n;++i)hv[i]=0.1f*hs(i,123);for(int i=0;i<rows;++i)hx[i]=hs(i,321);float*dv,*dx,*dy;CK(cudaMalloc(&dv,(size_t)n*sizeof(float)));CK(cudaMalloc(&dx,(size_t)rows*sizeof(float)));CK(cudaMalloc(&dy,(size_t)rows*sizeof(float)));CK(cudaMemcpy(dv,hv,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dx,hx,(size_t)rows*sizeof(float),cudaMemcpyHostToDevice));ell_spmv<<<(rows+255)/256,256>>>(dv,dx,dy,rows,width);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)rows*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,rows);cudaFree(dv);cudaFree(dx);cudaFree(dy);free(hv);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    rows, width = meta["input"]["sizes"]
    val = (V.F32(0.1) * V.gen_hashsigned(rows * width, 123)).reshape(rows, width)
    x = V.gen_hashsigned(rows, 321)
    y = np.zeros(rows, dtype=np.float32)
    for r in range(rows):
        acc = V.F32(0.0)
        for k in range(width):
            c = (r * 17 + k * 13) % rows
            acc += val[r, k] * x[c]
        y[r] = acc
    return y
""")

add_case(CASES, {
    "id": "hecbenchHeat2dBoundaryStep", "name": "HeCBench heat2D boundary step", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "hotspot-cuda/main.cu", "fidelity": "simplified", "sizes": [256, 256], "tol": 1e-6,
    "features": ["__global__", "2D_stencil", "boundary_condition"],
    "description": "One heat-equation update with fixed boundaries.",
    "notes": "Standalone 2D heat step based on HeCBench hotspot structured-grid kernels.",
}, r"""
__global__ void heat2d(const float*x,float*y,int H,int W){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=H*W;if(idx<n){int r=idx/W,c=idx%W;if(r==0||c==0||r==H-1||c==W-1)y[idx]=x[idx];else y[idx]=0.6f*x[idx]+0.1f*(x[idx-1]+x[idx+1]+x[idx-W]+x[idx+W]);}}
int main(int argc,char**argv){const int H=256,W=256,n=H*W;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));heat2d<<<(n+255)/256,256>>>(dx,dy,H,W);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    H, W = meta["input"]["sizes"]
    x = V.gen_hashsigned(H * W, 123).reshape(H, W)
    y = x.copy()
    y[1:-1,1:-1] = V.F32(0.6) * x[1:-1,1:-1] + V.F32(0.1) * (x[1:-1,:-2] + x[1:-1,2:] + x[:-2,1:-1] + x[2:,1:-1])
    return y.reshape(-1)
""")

add_case(CASES, {
    "id": "hecbenchReductionMinMax", "name": "HeCBench reduction min max", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "reduction-cuda/main.cu", "fidelity": "inspired_by", "sizes": [1048576], "tol": 1e-6,
    "features": ["__global__", "reduction", "atomic"],
    "description": "Compute deterministic min and max values with atomic integer encoding.",
    "notes": "Standalone reduction case inspired by CUDA reduction benchmark suites.",
}, r"""
__global__ void minmax_kernel(const float*x,int*mn,int*mx,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int v=(int)lrintf(1000000.0f*x[i]);atomicMin(mn,v);atomicMax(mx,v);}}
int main(int argc,char**argv){const int n=1048576;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx;int *dmn,*dmx,hmn=2147483647,hmx=-2147483647;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dmn,sizeof(int)));CK(cudaMalloc(&dmx,sizeof(int)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dmn,&hmn,sizeof(int),cudaMemcpyHostToDevice));CK(cudaMemcpy(dmx,&hmx,sizeof(int),cudaMemcpyHostToDevice));minmax_kernel<<<(n+255)/256,256>>>(dx,dmn,dmx,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(&hmn,dmn,sizeof(int),cudaMemcpyDeviceToHost));CK(cudaMemcpy(&hmx,dmx,sizeof(int),cudaMemcpyDeviceToHost));float outv[2]={(float)hmn/1000000.0f,(float)hmx/1000000.0f};write_vec(out,outv,2);cudaFree(dx);cudaFree(dmn);cudaFree(dmx);free(hx);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123)
    q = np.rint(V.F32(1000000.0) * x).astype(np.int32)
    return np.array([q.min() / V.F32(1000000.0), q.max() / V.F32(1000000.0)], dtype=np.float32)
""")

add_case(CASES, {
    "id": "hecbenchMonteCarloAsian", "name": "HeCBench Monte Carlo Asian option", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "black-scholes-cuda/blackScholesAnalyticEngineKernels.cu", "fidelity": "inspired_by", "sizes": [262144], "tol": 2e-5,
    "features": ["__global__", "monte_carlo", "expf"],
    "description": "Compute a deterministic Asian-option-like payoff per path.",
    "notes": "Standalone Monte Carlo payoff kernel inspired by HeCBench Black-Scholes style finance kernels.",
}, elemwise_main("asian_payoff", "float s=100.0f*expf(0.02f*x[i]); float avg=0.5f*(s + 100.0f*(1.0f+0.01f*g[i])); y[i]=fmaxf(avg-100.0f,0.0f);", 262144, 1.0, 1.0),
"""
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123); g = V.gen_hashsigned(n, 321)
    s = V.F32(100.0) * np.exp(V.F32(0.02) * x).astype(np.float32)
    avg = V.F32(0.5) * (s + V.F32(100.0) * (V.F32(1.0) + V.F32(0.01) * g))
    return np.maximum(avg - V.F32(100.0), V.F32(0.0))
""")

add_case(CASES, {
    "id": "hecbenchNeedlemanDiagScore", "name": "HeCBench Needleman diagonal score", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "nw-cuda/main.cu", "fidelity": "simplified", "sizes": [4096], "tol": 0.0,
    "features": ["__global__", "dynamic_programming", "integer_arithmetic"],
    "description": "Compute one diagonal scoring expression for Needleman-Wunsch style DP.",
    "notes": "Standalone diagonal scoring kernel based on HeCBench Needleman-Wunsch data dependencies.",
}, r"""
__global__ void nw_diag(float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int a=(i*7+3)&15,b=(i*11+5)&15,up=(i*13)&63,left=(i*17)&63,diag=(i*19)&63;int match=(a==b)?2:-1;int best=max(diag+match,max(up-1,left-1));y[i]=(float)best;}}
int main(int argc,char**argv){const int n=4096;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hy=(float*)malloc((size_t)n*sizeof(float));float*dy;CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));nw_diag<<<(n+255)/256,256>>>(dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dy);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    i = np.arange(n, dtype=np.int32)
    a = (i * 7 + 3) & 15; b = (i * 11 + 5) & 15
    up = (i * 13) & 63; left = (i * 17) & 63; diag = (i * 19) & 63
    match = np.where(a == b, 2, -1)
    return np.maximum(diag + match, np.maximum(up - 1, left - 1)).astype(np.float32)
""")

# DALI image preprocessing kernels.
for spec, expr, py_expr in [
    ({
        "id": "daliBrightnessContrast2", "name": "DALI brightness contrast", "category": "ai", "domain": "image_processing", "difficulty": "medium",
        "source": "dali", "rel": "dali/kernels/imgproc/color_manipulation/brightness_contrast.cu", "fidelity": "simplified", "sizes": [4, 64, 64, 3], "tol": 1e-6,
        "features": ["__global__", "image_processing", "color_transform"],
        "description": "Apply brightness and contrast adjustment to NHWC images.",
        "notes": "Standalone per-pixel brightness/contrast arithmetic inspired by NVIDIA DALI color kernels.",
    }, "float v=1.2f*x[idx]+0.05f; y[idx]=fminf(1.0f,fmaxf(0.0f,v));",
       "np.minimum(V.F32(1.0), np.maximum(V.F32(0.0), V.F32(1.2) * x + V.F32(0.05)))"),
    ({
        "id": "daliNormalizeMeanStd", "name": "DALI normalize mean std", "category": "ai", "domain": "image_processing", "difficulty": "medium",
        "source": "dali", "rel": "dali/kernels/slice/slice_flip_normalize_permute_pad_cuda_impl.cuh", "fidelity": "simplified", "sizes": [4, 64, 64, 3], "tol": 1e-6,
        "features": ["__global__", "normalization", "channelwise"],
        "description": "Normalize NHWC image channels with per-channel mean and inverse std.",
        "notes": "Standalone normalization based on DALI crop/normalize preprocessing kernels.",
    }, "int c=idx%3; float mean=(c==0?0.485f:(c==1?0.456f:0.406f)); float inv=(c==0?4.3668f:(c==1?4.4643f:4.4444f)); y[idx]=(x[idx]-mean)*inv;",
       None),
    ({
        "id": "daliChannelSwap", "name": "DALI channel swap", "category": "ai", "domain": "image_processing", "difficulty": "easy",
        "source": "dali", "rel": "dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.h", "fidelity": "simplified", "sizes": [4, 64, 64, 3], "tol": 1e-6,
        "features": ["__global__", "channel_permutation", "image_processing"],
        "description": "Swap RGB channel order in NHWC images.",
        "notes": "Standalone channel permutation inspired by DALI color conversion kernels.",
    }, "int c=idx%3; int base=idx-c; y[idx]=x[base+(2-c)];",
       None),
]:
    n_expr = "N*H*W*C"
    main = f"""
__global__ void image_kernel(const float*x,float*y,int N,int H,int W,int C){{int idx=blockIdx.x*blockDim.x+threadIdx.x,total={n_expr};if(idx<total){{{expr}}}}}
int main(int argc,char**argv){{const int N=4,H=64,W=64,C=3,total=N*H*W*C;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)total*sizeof(float)),*hy=(float*)malloc((size_t)total*sizeof(float));for(int i=0;i<total;++i)hx[i]=h01(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)total*sizeof(float)));CK(cudaMalloc(&dy,(size_t)total*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)total*sizeof(float),cudaMemcpyHostToDevice));image_kernel<<<(total+255)/256,256>>>(dx,dy,N,H,W,C);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)total*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,total);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}}
"""
    if spec["id"] == "daliNormalizeMeanStd":
        verify = """
def reference(meta):
    N, H, W, C = meta["input"]["sizes"]
    x = V.gen_hash01(N * H * W * C, 123).reshape(N, H, W, C)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    inv = np.array([4.3668, 4.4643, 4.4444], dtype=np.float32)
    return ((x - mean.reshape(1,1,1,C)) * inv.reshape(1,1,1,C)).reshape(-1)
"""
    elif spec["id"] == "daliChannelSwap":
        verify = """
def reference(meta):
    N, H, W, C = meta["input"]["sizes"]
    x = V.gen_hash01(N * H * W * C, 123).reshape(N, H, W, C)
    return x[:, :, :, ::-1].reshape(-1)
"""
    else:
        verify = f"""
def reference(meta):
    N, H, W, C = meta["input"]["sizes"]
    x = V.gen_hash01(N * H * W * C, 123)
    return {py_expr}
"""
    add_case(CASES, spec, main, verify)

add_case(CASES, {
    "id": "daliRgbToGray", "name": "DALI RGB to grayscale", "category": "ai", "domain": "image_processing", "difficulty": "medium",
    "source": "dali", "rel": "dali/kernels/imgproc/color_manipulation/color_space_conversion_impl.h", "fidelity": "simplified", "sizes": [4, 64, 64, 3], "tol": 1e-6,
    "features": ["__global__", "color_conversion", "image_processing"],
    "description": "Convert RGB NHWC images to grayscale luminance.",
    "notes": "Standalone color conversion inspired by DALI RGB/YUV conversion helpers.",
}, r"""
__global__ void rgb_gray(const float*x,float*y,int N,int H,int W,int C){int p=blockIdx.x*blockDim.x+threadIdx.x,total=N*H*W;if(p<total){int b=p*C;y[p]=0.299f*x[b]+0.587f*x[b+1]+0.114f*x[b+2];}}
int main(int argc,char**argv){const int N=4,H=64,W=64,C=3,total=N*H*W*C,outn=N*H*W;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)total*sizeof(float)),*hy=(float*)malloc((size_t)outn*sizeof(float));for(int i=0;i<total;++i)hx[i]=h01(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)total*sizeof(float)));CK(cudaMalloc(&dy,(size_t)outn*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)total*sizeof(float),cudaMemcpyHostToDevice));rgb_gray<<<(outn+255)/256,256>>>(dx,dy,N,H,W,C);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)outn*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,outn);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    N, H, W, C = meta["input"]["sizes"]
    x = V.gen_hash01(N * H * W * C, 123).reshape(N, H, W, C)
    return (V.F32(0.299) * x[:,:,:,0] + V.F32(0.587) * x[:,:,:,1] + V.F32(0.114) * x[:,:,:,2]).reshape(-1)
""")

add_case(CASES, {
    "id": "daliResizeNearest", "name": "DALI nearest resize", "category": "ai", "domain": "image_processing", "difficulty": "medium",
    "source": "dali", "rel": "dali/kernels/imgproc/resize/resize.cuh", "fidelity": "simplified", "sizes": [32, 32, 64, 64], "tol": 1e-6,
    "features": ["__global__", "resize", "2D_indexing"],
    "description": "Resize a single-channel image with nearest-neighbor sampling.",
    "notes": "Standalone resize operation inspired by DALI GPU resize kernels.",
}, r"""
__global__ void resize_nn(const float*x,float*y,int H0,int W0,int H1,int W1){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=H1*W1;if(idx<n){int r=idx/W1,c=idx%W1;int sr=(int)floorf(((float)r+0.5f)*(float)H0/(float)H1);int sc=(int)floorf(((float)c+0.5f)*(float)W0/(float)W1);sr=min(sr,H0-1);sc=min(sc,W0-1);y[idx]=x[sr*W0+sc];}}
int main(int argc,char**argv){const int H0=32,W0=32,H1=64,W1=64,n0=H0*W0,n1=H1*W1;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n0*sizeof(float)),*hy=(float*)malloc((size_t)n1*sizeof(float));for(int i=0;i<n0;++i)hx[i]=h01(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n0*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n1*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n0*sizeof(float),cudaMemcpyHostToDevice));resize_nn<<<(n1+255)/256,256>>>(dx,dy,H0,W0,H1,W1);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n1*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n1);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    H0, W0, H1, W1 = meta["input"]["sizes"]
    x = V.gen_hash01(H0 * W0, 123).reshape(H0, W0)
    y = np.empty((H1, W1), dtype=np.float32)
    for r in range(H1):
        for c in range(W1):
            sr = min(int(np.floor((V.F32(r) + V.F32(0.5)) * V.F32(H0) / V.F32(H1))), H0 - 1)
            sc = min(int(np.floor((V.F32(c) + V.F32(0.5)) * V.F32(W0) / V.F32(W1))), W0 - 1)
            y[r, c] = x[sr, sc]
    return y.reshape(-1)
""")

# CUDA primitive cases.
add_case(CASES, {
    "id": "cudaSamplesCoalescedCopy", "name": "CUDA Samples coalesced copy", "category": "medium", "domain": "cuda_primitive", "difficulty": "easy",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/vectorAdd/vectorAdd.cu", "fidelity": "inspired_by", "sizes": [1048576], "tol": 1e-6,
    "features": ["__global__", "coalesced_memory", "grid_stride"],
    "description": "Coalesced vector copy with a grid-stride loop.",
    "notes": "Standalone memory access primitive inspired by CUDA Samples vectorAdd.",
}, r"""
__global__ void copy_kernel(const float*x,float*y,int n){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)y[i]=x[i];}
int main(int argc,char**argv){const int n=1048576;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));copy_kernel<<<128,256>>>(dx,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    return V.gen_hashsigned(meta["input"]["sizes"][0], 123)
""")

add_case(CASES, {
    "id": "cudaSamplesSharedTileTranspose2", "name": "CUDA Samples shared tile transpose", "category": "medium", "domain": "cuda_primitive", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/transpose/transpose.cu", "fidelity": "inspired_by", "sizes": [128, 128], "tol": 1e-6,
    "features": ["__global__", "__shared__", "transpose"],
    "description": "Shared-memory tiled matrix transpose.",
    "notes": "Standalone tiled transpose inspired by CUDA Samples transpose.",
}, r"""
__global__ void tile_t(const float*x,float*y,int H,int W){__shared__ float tile[16][17];int c=blockIdx.x*16+threadIdx.x,r=blockIdx.y*16+threadIdx.y;if(r<H&&c<W)tile[threadIdx.y][threadIdx.x]=x[r*W+c];__syncthreads();int orow=blockIdx.x*16+threadIdx.y,ocol=blockIdx.y*16+threadIdx.x;if(orow<W&&ocol<H)y[orow*H+ocol]=tile[threadIdx.x][threadIdx.y];}
int main(int argc,char**argv){const int H=128,W=128,n=H*W;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));tile_t<<<dim3((W+15)/16,(H+15)/16),dim3(16,16)>>>(dx,dy,H,W);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    H, W = meta["input"]["sizes"]
    return V.gen_hashsigned(H * W, 123).reshape(H, W).T.reshape(-1)
""")

add_case(CASES, {
    "id": "cudaSamplesWarpPrefixLane", "name": "CUDA Samples warp prefix lane", "category": "medium", "domain": "cuda_primitive", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTemplates/simpleTemplates.cu", "fidelity": "inspired_by", "sizes": [32768], "tol": 1e-5,
    "features": ["__global__", "__shfl_up_sync", "warp_scan"],
    "description": "Inclusive prefix sum within each warp using shuffle instructions.",
    "notes": "Standalone warp-scan primitive inspired by CUDA Samples warp/shuffle examples.",
}, r"""
__global__ void warp_scan(const float*x,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];int lane=threadIdx.x&31;for(int off=1;off<32;off<<=1){float u=__shfl_up_sync(0xffffffff,v,off);if(lane>=off)v+=u;}y[i]=v;}}
int main(int argc,char**argv){const int n=32768;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=0.01f*hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));warp_scan<<<(n+255)/256,256>>>(dx,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = (V.F32(0.01) * V.gen_hashsigned(n, 123)).reshape(-1, 32)
    return np.cumsum(x, axis=1, dtype=np.float32).reshape(-1)
""")

add_case(CASES, {
    "id": "cudaSamplesAtomicMaxBuckets", "name": "CUDA Samples atomic max buckets", "category": "medium", "domain": "cuda_primitive", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics.cu", "fidelity": "inspired_by", "sizes": [262144, 256], "tol": 0.0,
    "features": ["__global__", "atomicMax", "histogram"],
    "description": "Compute per-bucket maximum integer score with atomics.",
    "notes": "Standalone atomic max primitive inspired by CUDA Samples atomic intrinsics.",
}, r"""
__global__ void bucket_max(int*out,int n,int buckets){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int b=(i*17+13)&(buckets-1);int v=(i*29+7)&65535;atomicMax(&out[b],v);}}
int main(int argc,char**argv){const int n=262144,buckets=256;const char*outp=(argc>1)?argv[1]:"output/output.txt";int*db;CK(cudaMalloc(&db,(size_t)buckets*sizeof(int)));CK(cudaMemset(db,0,(size_t)buckets*sizeof(int)));bucket_max<<<(n+255)/256,256>>>(db,n,buckets);CK(cudaGetLastError());CK(cudaDeviceSynchronize());int*hb=(int*)malloc((size_t)buckets*sizeof(int));float*hy=(float*)malloc((size_t)buckets*sizeof(float));CK(cudaMemcpy(hb,db,(size_t)buckets*sizeof(int),cudaMemcpyDeviceToHost));for(int i=0;i<buckets;++i)hy[i]=(float)hb[i];write_vec(outp,hy,buckets);cudaFree(db);free(hb);free(hy);return 0;}
""", """
def reference(meta):
    n, buckets = meta["input"]["sizes"]
    out = np.zeros(buckets, dtype=np.int32)
    for i in range(n):
        b = (i * 17 + 13) & (buckets - 1)
        v = (i * 29 + 7) & 65535
        out[b] = max(out[b], v)
    return out.astype(np.float32)
""")

add_case(CASES, {
    "id": "cudaSamplesGridStrideStrided", "name": "CUDA Samples strided gather", "category": "medium", "domain": "cuda_primitive", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/vectorAdd/vectorAdd.cu", "fidelity": "inspired_by", "sizes": [262144], "tol": 1e-6,
    "features": ["__global__", "grid_stride", "gather"],
    "description": "Grid-stride gather from a strided memory pattern.",
    "notes": "Standalone strided memory access pattern inspired by CUDA Samples.",
}, r"""
__global__ void gather2(const float*x,float*y,int n){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){int src=(i*2+17)%(2*n);y[i]=x[src];}}
int main(int argc,char**argv){const int n=262144,total=2*n;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)total*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<total;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)total*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)total*sizeof(float),cudaMemcpyHostToDevice));gather2<<<128,256>>>(dx,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(2 * n, 123)
    idx = (np.arange(n, dtype=np.int64) * 2 + 17) % (2 * n)
    return x[idx]
""")

add_case(CASES, {
    "id": "cudaSamplesBitReversePermute", "name": "CUDA Samples bit-reverse permutation", "category": "medium", "domain": "cuda_primitive", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTemplates/simpleTemplates.cu", "fidelity": "inspired_by", "sizes": [65536], "tol": 1e-6,
    "features": ["__global__", "permutation", "bit_arithmetic"],
    "description": "Permute a vector through a 16-bit bit-reversal mapping.",
    "notes": "Standalone permutation primitive inspired by CUDA Samples template/permutation examples.",
}, r"""
__device__ __host__ unsigned rev16(unsigned x){x=((x&0x5555u)<<1)|((x>>1)&0x5555u);x=((x&0x3333u)<<2)|((x>>2)&0x3333u);x=((x&0x0f0fu)<<4)|((x>>4)&0x0f0fu);x=(x<<8)|(x>>8);return x&0xffffu;}
__global__ void bitrev(const float*x,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[rev16((unsigned)i)]=x[i];}
int main(int argc,char**argv){const int n=65536;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));bitrev<<<(n+255)/256,256>>>(dx,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123)
    y = np.empty(n, dtype=np.float32)
    for i in range(n):
        v = i
        v = ((v & 0x5555) << 1) | ((v >> 1) & 0x5555)
        v = ((v & 0x3333) << 2) | ((v >> 2) & 0x3333)
        v = ((v & 0x0f0f) << 4) | ((v >> 4) & 0x0f0f)
        v = ((v << 8) | (v >> 8)) & 0xffff
        y[v] = x[i]
    return y
""")

add_case(CASES, {
    "id": "cudaSamplesSharedStencilHalo", "name": "CUDA Samples shared halo stencil", "category": "medium", "domain": "cuda_primitive", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTemplates/sharedmem.cuh", "fidelity": "inspired_by", "sizes": [262144], "tol": 1e-6,
    "features": ["__global__", "__shared__", "stencil"],
    "description": "One-dimensional stencil using shared memory halo loads.",
    "notes": "Standalone shared-memory halo pattern inspired by CUDA Samples shared-memory templates.",
}, r"""
__global__ void stencil_shared(const float*x,float*y,int n){extern __shared__ float s[];int t=threadIdx.x,i=blockIdx.x*blockDim.x+t;if(i<n)s[t+1]=x[i];if(t==0)s[0]=(i>0)?x[i-1]:x[i];if(t==blockDim.x-1||i==n-1)s[t+2]=(i<n-1)?x[i+1]:x[i];__syncthreads();if(i<n)y[i]=0.25f*s[t]+0.5f*s[t+1]+0.25f*s[t+2];}
int main(int argc,char**argv){const int n=262144;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));stencil_shared<<<(n+255)/256,256,(256+2)*sizeof(float)>>>(dx,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123)
    left = np.concatenate([x[:1], x[:-1]])
    right = np.concatenate([x[1:], x[-1:]])
    return V.F32(0.25) * left + V.F32(0.5) * x + V.F32(0.25) * right
""")

# Thrust / library API cases.
add_case(CASES, {
    "id": "thrustZipTransform2", "name": "Thrust zip transform multiply add", "category": "library_api", "domain": "library_api", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTemplates/simpleTemplates.cu", "fidelity": "inspired_by", "sizes": [262144], "tol": 1e-6,
    "features": ["thrust::device_vector", "thrust::transform", "zip_iterator"],
    "libraries": ["Thrust"],
    "description": "Use thrust zip iterators for fused multiply-add.",
    "notes": "Standalone Thrust zip-transform API case inspired by CUDA Samples' Thrust usage.",
}, r"""
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
struct fma_zip { __host__ __device__ float operator()(const thrust::tuple<float,float>& t) const { return thrust::get<0>(t) * thrust::get<1>(t) + 0.125f; } };
int main(int argc,char**argv){const int n=262144;const char*out=(argc>1)?argv[1]:"output/output.txt";float*ha=(float*)malloc((size_t)n*sizeof(float)),*hb=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i){ha[i]=hs(i,123);hb[i]=hs(i,321);}thrust::device_vector<float>a(ha,ha+n),b(hb,hb+n),y(n);auto first=thrust::make_zip_iterator(thrust::make_tuple(a.begin(),b.begin()));auto last=thrust::make_zip_iterator(thrust::make_tuple(a.end(),b.end()));thrust::transform(first,last,y.begin(),fma_zip());thrust::copy(y.begin(),y.end(),hy);write_vec(out,hy,n);free(ha);free(hb);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    return V.gen_hashsigned(n, 123) * V.gen_hashsigned(n, 321) + V.F32(0.125)
""")

add_case(CASES, {
    "id": "thrustCountIfThreshold", "name": "Thrust count if threshold", "category": "library_api", "domain": "library_api", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/6_Performance/UnifiedMemoryPerf/matrixMultiplyPerf.cu", "fidelity": "inspired_by", "sizes": [1048576], "tol": 0.0,
    "features": ["thrust::device_vector", "thrust::count_if", "functor"],
    "libraries": ["Thrust"],
    "description": "Count values above a threshold using thrust::count_if.",
    "notes": "Standalone Thrust reduction-style API case inspired by CUDA Samples utility patterns.",
}, r"""
#include <thrust/device_vector.h>
#include <thrust/count.h>
struct above { __host__ __device__ bool operator()(float x) const { return x > 0.25f; } };
int main(int argc,char**argv){const int n=1048576;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);thrust::device_vector<float>x(hx,hx+n);int c=thrust::count_if(x.begin(),x.end(),above());float y=(float)c;write_vec(out,&y,1);free(hx);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    return np.array([np.count_nonzero(V.gen_hashsigned(n, 123) > V.F32(0.25))], dtype=np.float32)
""")

add_case(CASES, {
    "id": "thrustReduceByKeySegments", "name": "Thrust reduce by key segments", "category": "library_api", "domain": "library_api", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTemplates/simpleTemplates.cu", "fidelity": "inspired_by", "sizes": [65536, 8], "tol": 1e-5,
    "features": ["thrust::reduce_by_key", "device_vector", "segmented_reduction"],
    "libraries": ["Thrust"],
    "description": "Segmented sum with thrust::reduce_by_key over sorted keys.",
    "notes": "Standalone segmented-reduction API benchmark inspired by Thrust/CUDA sample usage.",
}, r"""
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
int main(int argc,char**argv){const int n=65536,seg=8,outn=n/seg;const char*out=(argc>1)?argv[1]:"output/output.txt";int*hk=(int*)malloc((size_t)n*sizeof(int));float*hv=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)outn*sizeof(float));for(int i=0;i<n;++i){hk[i]=i/seg;hv[i]=0.01f*hs(i,123);}thrust::device_vector<int>k(hk,hk+n),ok(outn);thrust::device_vector<float>v(hv,hv+n),ov(outn);thrust::reduce_by_key(k.begin(),k.end(),v.begin(),ok.begin(),ov.begin());thrust::copy(ov.begin(),ov.begin()+outn,hy);write_vec(out,hy,outn);free(hk);free(hv);free(hy);return 0;}
""", """
def reference(meta):
    n, seg = meta["input"]["sizes"]
    x = (V.F32(0.01) * V.gen_hashsigned(n, 123)).reshape(-1, seg)
    y = np.zeros(n // seg, dtype=np.float32)
    for c in range(seg):
        y += x[:, c]
    return y
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
        print(f"[ok] {spec['category']}/{spec['id']} <- {SOURCES[spec['source']]['project']}")
    print(f"Wrote {len(CASES)} Stage 1 batch-3 cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
