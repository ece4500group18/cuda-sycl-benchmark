#!/usr/bin/env python3
"""Add the final Stage 1 expansion batch to reach 250 CUDA cases."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import add_stage1_real_project_batch3_cases as base


CASES: list[dict] = []


def add(spec: dict, main: str, verify: str) -> None:
    base.add_case(CASES, spec, main, verify)


# Modern ML / LLM / CUDA extension kernels: 15 cases.
for spec, expr, py_expr, xscale, gscale in [
    ({
        "id": "ggmlAbsKernel2", "name": "ggml absolute value unary", "category": "ai", "domain": "modern_ml", "difficulty": "easy",
        "source": "llama", "rel": "unary.cu", "fidelity": "simplified", "sizes": [1048576], "tol": 1e-6,
        "features": ["__global__", "elementwise", "unary_op"],
        "description": "Elementwise absolute value over a tensor.",
        "notes": "Simplified standalone unary op following llama.cpp/ggml CUDA unary kernel patterns.",
    }, "y[i] = fabsf(x[i]);", "np.abs(x).astype(np.float32)", 4.0, 1.0),
    ({
        "id": "ggmlTanhKernel", "name": "ggml tanh unary op", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
        "source": "llama", "rel": "unary.cu", "fidelity": "simplified", "sizes": [1048576], "tol": 1e-5,
        "features": ["__global__", "tanhf", "activation"],
        "description": "Elementwise hyperbolic tangent activation.",
        "notes": "Standalone tanh activation inspired by ggml CUDA unary activation kernels.",
    }, "y[i] = tanhf(x[i]);", "np.tanh(x).astype(np.float32)", 4.0, 1.0),
    ({
        "id": "ggmlEluKernel", "name": "ggml ELU activation", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
        "source": "llama", "rel": "unary.cu", "fidelity": "inspired_by", "sizes": [1048576], "tol": 1e-5,
        "features": ["__global__", "expf", "activation"],
        "description": "Elementwise ELU-style activation.",
        "notes": "Standalone activation variant built in the same form as ggml CUDA unary kernels.",
    }, "float v=x[i]; y[i]=(v>0.0f)?v:(expf(v)-1.0f);", "np.where(x > V.F32(0.0), x, np.exp(x).astype(np.float32) - V.F32(1.0)).astype(np.float32)", 3.0, 1.0),
    ({
        "id": "vllmGeluTanhMul2", "name": "vLLM tanh GELU multiply", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
        "source": "vllm", "rel": "activation_kernels.cu", "fidelity": "simplified", "sizes": [1048576], "tol": 1e-5,
        "features": ["__global__", "tanhf", "gated_activation"],
        "description": "Fused tanh-approximate GELU multiplied by a gate tensor.",
        "notes": "Standalone gated activation preserving vLLM activation kernel arithmetic.",
    }, "float v=x[i]; float u=0.7978845608f*(v+0.044715f*v*v*v); float gelu=0.5f*v*(1.0f+tanhf(u)); y[i]=gelu*g[i];", "V.F32(0.5) * x * (V.F32(1.0) + np.tanh(V.F32(0.7978845608) * (x + V.F32(0.044715) * x * x * x)).astype(np.float32)) * g", 3.0, 2.0),
    ({
        "id": "bnbMomentUpdate2", "name": "bitsandbytes moment update", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
        "source": "bnb", "rel": "kernels.cu", "fidelity": "simplified", "sizes": [1048576], "tol": 1e-6,
        "features": ["__global__", "optimizer", "fused_update"],
        "description": "Update first-moment optimizer state from gradients.",
        "notes": "Standalone optimizer moment update inspired by bitsandbytes CUDA optimizer kernels.",
    }, "y[i] = 0.9f * x[i] + 0.1f * g[i];", "V.F32(0.9) * x + V.F32(0.1) * g", 1.0, 1.0),
    ({
        "id": "cutlassBiasGeluEpilogue2", "name": "CUTLASS bias GELU epilogue", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
        "source": "cutlass", "rel": "12_gemm_bias_relu/gemm_bias_relu.cu", "fidelity": "inspired_by", "sizes": [524288], "tol": 1e-5,
        "features": ["__global__", "epilogue", "activation"],
        "description": "Bias plus tanh-GELU epilogue for GEMM outputs.",
        "notes": "Standalone epilogue variant inspired by CUTLASS fused GEMM epilogue examples.",
    }, "float v=x[i]+0.25f*g[i]; y[i]=0.5f*v*(1.0f+tanhf(0.7978845608f*(v+0.044715f*v*v*v)));", "V.F32(0.5) * (x + V.F32(0.25) * g) * (V.F32(1.0) + np.tanh(V.F32(0.7978845608) * ((x + V.F32(0.25) * g) + V.F32(0.044715) * (x + V.F32(0.25) * g) ** 3)).astype(np.float32))", 2.0, 1.0),
    ({
        "id": "xformersSiluBiasMul2", "name": "xFormers SiLU bias multiply", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
        "source": "xformers", "rel": "xformers/ops/swiglu_op.py", "fidelity": "inspired_by", "sizes": [1048576], "tol": 1e-5,
        "features": ["__global__", "expf", "gated_activation"],
        "description": "Bias-shifted SiLU multiplied by a second tensor.",
        "notes": "Standalone CUDA form of the fused activation pattern used by xFormers SwiGLU operators.",
    }, "float v=x[i]+0.125f; y[i]=(v/(1.0f+expf(-v)))*g[i];", "((x + V.F32(0.125)) / (V.F32(1.0) + np.exp(-(x + V.F32(0.125))).astype(np.float32))) * g", 4.0, 1.5),
]:
    add(spec, base.elemwise_main(spec["id"], expr, spec["sizes"][0], xscale, gscale), base.elemwise_verify(py_expr, xscale, gscale))

spec = {
    "id": "vllmLogitsClampScale2", "name": "vLLM logits clamp scale", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "vllm", "rel": "activation_kernels.cu", "fidelity": "inspired_by", "sizes": [1048576], "tol": 1e-6,
    "features": ["__global__", "clamp", "logit_transform"],
    "description": "Clamp logits and apply a scale factor before sampling.",
    "notes": "Standalone logits transform inspired by vLLM CUDA activation and sampling helper patterns.",
}
add(
    spec,
    base.elemwise_main(spec["id"], "float v=fminf(6.0f,fmaxf(-6.0f,x[i])); y[i]=0.5f*v;", spec["sizes"][0], 8.0, 1.0),
    base.elemwise_verify("V.F32(0.5) * np.minimum(V.F32(6.0), np.maximum(V.F32(-6.0), x))", 8.0, 1.0),
)

add({
    "id": "ggmlMulRowsBroadcast", "name": "ggml row broadcast multiply", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "llama", "rel": "binary.cu", "fidelity": "simplified", "sizes": [512, 256], "tol": 1e-6,
    "features": ["__global__", "broadcast", "2D_indexing"],
    "description": "Multiply a matrix by a per-column row vector.",
    "notes": "Simplified broadcast multiply following ggml CUDA binary operator patterns.",
}, r"""
__global__ void mul_rows(const float*x,const float*b,float*y,int rows,int cols){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=rows*cols;if(idx<n)y[idx]=x[idx]*b[idx%cols];}
int main(int argc,char**argv){const int rows=512,cols=256,n=rows*cols;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hb=(float*)malloc((size_t)cols*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);for(int c=0;c<cols;++c)hb[c]=0.5f+0.25f*hs(c,321);float*dx,*db,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&db,(size_t)cols*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(db,hb,(size_t)cols*sizeof(float),cudaMemcpyHostToDevice));mul_rows<<<(n+255)/256,256>>>(dx,db,dy,rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(db);cudaFree(dy);free(hx);free(hb);free(hy);return 0;}
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    b = V.F32(0.5) + V.F32(0.25) * V.gen_hashsigned(cols, 321)
    return (x * b.reshape(1, cols)).reshape(-1)
""")

add({
    "id": "ggmlPermute2d", "name": "ggml 2D permute", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "llama", "rel": "cpy.cu", "fidelity": "simplified", "sizes": [128, 256], "tol": 1e-6,
    "features": ["__global__", "layout_conversion", "transpose"],
    "description": "Permute a row-major 2D tensor into transposed layout.",
    "notes": "Standalone 2D layout permutation inspired by ggml CUDA copy/permute kernels.",
}, r"""
__global__ void perm2d(const float*x,float*y,int R,int C){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=R*C;if(idx<n){int r=idx/C,c=idx%C;y[c*R+r]=x[idx];}}
int main(int argc,char**argv){const int R=128,C=256,n=R*C;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));perm2d<<<(n+255)/256,256>>>(dx,dy,R,C);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    R, C = meta["input"]["sizes"]
    return V.gen_hashsigned(R * C, 123).reshape(R, C).T.reshape(-1)
""")

add({
    "id": "vllmRmsNormEps2", "name": "vLLM RMSNorm epsilon", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "vllm", "rel": "layernorm_kernels.cu", "fidelity": "simplified", "sizes": [1024, 128], "tol": 2e-5,
    "features": ["__global__", "normalization", "rowwise"],
    "description": "RMSNorm over rows with an epsilon term.",
    "notes": "Standalone rowwise RMSNorm based on vLLM layernorm CUDA kernels.",
}, r"""
__global__ void rmsnorm(const float*x,float*y,int rows,int cols){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=rows*cols;if(idx<n){int r=idx/cols;float ss=0.0f;for(int c=0;c<cols;++c){float v=x[r*cols+c];ss+=v*v;}float inv=rsqrtf(ss/(float)cols+1.0e-5f);y[idx]=x[idx]*inv;}}
int main(int argc,char**argv){const int rows=1024,cols=128,n=rows*cols;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));rmsnorm<<<(n+255)/256,256>>>(dx,dy,rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    y = np.empty_like(x)
    for r in range(rows):
        ss = V.F32(0.0)
        for c in range(cols):
            ss += x[r, c] * x[r, c]
        inv = V.F32(1.0) / np.sqrt(ss / V.F32(cols) + V.F32(1.0e-5), dtype=np.float32)
        y[r] = x[r] * inv
    return y.reshape(-1)
""")

add({
    "id": "vllmPagedBlockCopy2", "name": "vLLM paged block copy", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "vllm", "rel": "cache_kernels.cu", "fidelity": "simplified", "sizes": [256, 16, 32], "tol": 1e-6,
    "features": ["__global__", "cache_layout", "copy"],
    "description": "Copy cache blocks through a deterministic page remapping.",
    "notes": "Standalone cache block copy modeled on vLLM paged KV cache kernels.",
}, r"""
__global__ void block_copy(const float*src,float*dst,int pages,int block,int dim){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=pages*block*dim;if(idx<n){int d=idx%dim;int b=(idx/dim)%block;int p=idx/(block*dim);int q=(p*53+7)%pages;dst[(q*block+b)*dim+d]=src[idx];}}
int main(int argc,char**argv){const int pages=256,block=16,dim=32,n=pages*block*dim;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)calloc((size_t)n,sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemset(dy,0,(size_t)n*sizeof(float)));block_copy<<<(n+255)/256,256>>>(dx,dy,pages,block,dim);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    pages, block, dim = meta["input"]["sizes"]
    src = V.gen_hashsigned(pages * block * dim, 123).reshape(pages, block, dim)
    dst = np.zeros_like(src)
    for p in range(pages):
        q = (p * 53 + 7) % pages
        dst[q] = src[p]
    return dst.reshape(-1)
""")

add({
    "id": "bnbBlockScaleQuant2", "name": "bitsandbytes block scale quantize", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "bnb", "rel": "kernels.cu", "fidelity": "simplified", "sizes": [262144], "tol": 0.0,
    "features": ["__global__", "quantization", "integer_arithmetic"],
    "description": "Quantize float values to unsigned 8-bit bucket indices.",
    "notes": "Standalone block-scale quantization inspired by bitsandbytes quantization kernels.",
}, r"""
__global__ void q8(const float*x,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int q=(int)floorf(x[i]*32.0f+128.0f);q=max(0,min(255,q));y[i]=(float)q;}}
int main(int argc,char**argv){const int n=262144;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));q8<<<(n+255)/256,256>>>(dx,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    q = np.floor(V.gen_hashsigned(n, 123) * V.F32(32.0) + V.F32(128.0)).astype(np.int32)
    return np.clip(q, 0, 255).astype(np.float32)
""")

add({
    "id": "flashCausalDropoutMask2", "name": "FlashAttention causal dropout mask", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "flash", "rel": "dropout.h", "fidelity": "simplified", "sizes": [512, 128], "tol": 1e-6,
    "features": ["__global__", "attention_mask", "dropout"],
    "description": "Apply causal masking and deterministic dropout rescaling.",
    "notes": "Standalone mask/dropout arithmetic inspired by FlashAttention dropout and mask helpers.",
}, r"""
__global__ void cmask_drop(const float*x,float*y,int rows,int cols){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=rows*cols;if(idx<n){int r=idx/cols,c=idx%cols;bool causal=c<=(r%cols);bool keep=((idx*1103515245u+12345u)&7u)!=0u;y[idx]=(causal&&keep)?x[idx]*1.142857142857f:0.0f;}}
int main(int argc,char**argv){const int rows=512,cols=128,n=rows*cols;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));cmask_drop<<<(n+255)/256,256>>>(dx,dy,rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    y = np.zeros_like(x)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            keep = (((idx * 1103515245 + 12345) & 7) != 0)
            if c <= (r % cols) and keep:
                y[r, c] = x[r, c] * V.F32(1.142857142857)
    return y.reshape(-1)
""")

add({
    "id": "flashSoftmaxDenomStats", "name": "FlashAttention softmax denominator stats", "category": "ai", "domain": "modern_ml", "difficulty": "medium",
    "source": "flash", "rel": "softmax.h", "fidelity": "simplified", "sizes": [1024, 64], "tol": 1e-4,
    "features": ["__global__", "expf", "row_reduction"],
    "description": "Compute per-row stable softmax denominator statistics.",
    "notes": "Standalone row statistic inspired by FlashAttention online softmax helpers.",
}, r"""
__global__ void denom(const float*x,float*y,int rows,int cols){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r<rows){float m=-3.402823e38f;for(int c=0;c<cols;++c)m=fmaxf(m,x[r*cols+c]);float s=0.0f;for(int c=0;c<cols;++c)s+=expf(x[r*cols+c]-m);y[r]=s;}}
int main(int argc,char**argv){const int rows=1024,cols=64,n=rows*cols;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)rows*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)rows*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));denom<<<(rows+255)/256,256>>>(dx,dy,rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)rows*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,rows);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    rows, cols = meta["input"]["sizes"]
    x = V.gen_hashsigned(rows * cols, 123).reshape(rows, cols)
    y = np.empty(rows, dtype=np.float32)
    for r in range(rows):
        m = np.max(x[r]).astype(np.float32)
        s = V.F32(0.0)
        for c in range(cols):
            s += np.exp(x[r, c] - m, dtype=np.float32)
        y[r] = s
    return y
""")

# HPC / benchmark-suite kernels: 10 cases.
add({
    "id": "hecbenchPrefixScanHill", "name": "HeCBench Hillis-Steele prefix scan", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "scan-cuda/main.cu", "fidelity": "inspired_by", "sizes": [65536, 16], "tol": 1e-6,
    "features": ["__global__", "prefix_sum", "scan"],
    "description": "Compute a short-window inclusive scan for each item.",
    "notes": "Standalone scan-style kernel inspired by CUDA scan benchmark suites.",
}, r"""
__global__ void short_scan(const float*x,float*y,int n,int w){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float s=0.0f;int start=(i/w)*w;for(int j=start;j<=i;++j)s+=x[j];y[i]=s;}}
int main(int argc,char**argv){const int n=65536,w=16;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=0.01f*hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));short_scan<<<(n+255)/256,256>>>(dx,dy,n,w);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n, w = meta["input"]["sizes"]
    x = (V.F32(0.01) * V.gen_hashsigned(n, 123)).reshape(-1, w)
    return np.cumsum(x, axis=1, dtype=np.float32).reshape(-1)
""")

add({
    "id": "hecbenchFloydMinPlus2", "name": "HeCBench Floyd min-plus update", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "floydwarshall-cuda/main.cu", "fidelity": "simplified", "sizes": [256, 64], "tol": 0.0,
    "features": ["__global__", "dynamic_programming", "min_plus"],
    "description": "One Floyd-Warshall min-plus tile update.",
    "notes": "Standalone min-plus update modeled on HeCBench Floyd-Warshall kernels.",
}, r"""
__global__ void fw(float*y,int n,int k){int idx=blockIdx.x*blockDim.x+threadIdx.x,total=n*n;if(idx<total){int i=idx/n,j=idx%n;int cur=(i*13+j*7)&1023;int via=((i*13+k*7)&1023)+((k*13+j*7)&1023);y[idx]=(float)min(cur,via);}}
int main(int argc,char**argv){const int n=256,k=64,total=n*n;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hy=(float*)malloc((size_t)total*sizeof(float));float*dy;CK(cudaMalloc(&dy,(size_t)total*sizeof(float)));fw<<<(total+255)/256,256>>>(dy,n,k);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)total*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,total);cudaFree(dy);free(hy);return 0;}
""", """
def reference(meta):
    n, k = meta["input"]["sizes"]
    y = np.empty(n * n, dtype=np.float32)
    for idx in range(n * n):
        i = idx // n; j = idx % n
        cur = (i * 13 + j * 7) & 1023
        via = ((i * 13 + k * 7) & 1023) + ((k * 13 + j * 7) & 1023)
        y[idx] = min(cur, via)
    return y
""")

for spec, expr, py_expr, xscale, gscale in [
    ({
        "id": "hecbenchGaussianRank1Update2", "name": "HeCBench Gaussian rank-1 update", "category": "hpc", "domain": "hpc", "difficulty": "medium",
        "source": "hecbench", "rel": "gaussian-cuda/gaussianElim.cu", "fidelity": "simplified", "sizes": [262144], "tol": 1e-6,
        "features": ["__global__", "linear_algebra", "rank1_update"],
        "description": "Apply a rank-1 update used by Gaussian elimination.",
        "notes": "Standalone row update based on HeCBench Gaussian elimination arithmetic.",
    }, "float pivot=1.25f; y[i]=x[i]-g[i]*0.125f/pivot;", "x - g * V.F32(0.125) / V.F32(1.25)", 1.0, 1.0),
    ({
        "id": "hecbenchSradCoeff2", "name": "HeCBench SRAD coefficient", "category": "hpc", "domain": "hpc", "difficulty": "medium",
        "source": "hecbench", "rel": "srad-cuda/main.cu", "fidelity": "simplified", "sizes": [262144], "tol": 1e-5,
        "features": ["__global__", "sqrtf", "diffusion"],
        "description": "Compute a diffusion coefficient from local gradient values.",
        "notes": "Standalone coefficient arithmetic inspired by HeCBench SRAD kernels.",
    }, "float g2=x[i]*x[i]+0.25f*g[i]*g[i]; y[i]=1.0f/(1.0f+sqrtf(g2+1.0e-6f));", "V.F32(1.0) / (V.F32(1.0) + np.sqrt(x * x + V.F32(0.25) * g * g + V.F32(1.0e-6)).astype(np.float32))", 1.0, 1.0),
]:
    add(spec, base.elemwise_main(spec["id"], expr, spec["sizes"][0], xscale, gscale), base.elemwise_verify(py_expr, xscale, gscale))

add({
    "id": "hecbenchPathfinderMin3Step2", "name": "HeCBench pathfinder min-3 step", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "pathfinder-cuda/main.cu", "fidelity": "simplified", "sizes": [256, 256], "tol": 1e-6,
    "features": ["__global__", "dynamic_programming", "stencil"],
    "description": "One pathfinder update from three predecessor cells.",
    "notes": "Standalone min-neighbor update modeled on HeCBench Pathfinder kernels.",
}, r"""
__global__ void path_step(const float*prev,const float*cost,float*y,int H,int W){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=H*W;if(idx<n){int c=idx%W;float l=prev[max(c-1,0)],m=prev[c],r=prev[min(c+1,W-1)];y[idx]=cost[idx]+fminf(m,fminf(l,r));}}
int main(int argc,char**argv){const int H=256,W=256,n=H*W;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hp=(float*)malloc((size_t)W*sizeof(float)),*hc=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<W;++i)hp[i]=h01(i,77);for(int i=0;i<n;++i)hc[i]=h01(i,123);float*dp,*dc,*dy;CK(cudaMalloc(&dp,(size_t)W*sizeof(float)));CK(cudaMalloc(&dc,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dp,hp,(size_t)W*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dc,hc,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));path_step<<<(n+255)/256,256>>>(dp,dc,dy,H,W);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dp);cudaFree(dc);cudaFree(dy);free(hp);free(hc);free(hy);return 0;}
""", """
def reference(meta):
    H, W = meta["input"]["sizes"]
    prev = V.gen_hash01(W, 77)
    cost = V.gen_hash01(H * W, 123).reshape(H, W)
    y = np.empty((H, W), dtype=np.float32)
    for c in range(W):
        best = min(prev[max(c - 1, 0)], prev[c], prev[min(c + 1, W - 1)])
        y[:, c] = cost[:, c] + best
    return y.reshape(-1)
""")

add({
    "id": "hecbenchParticleWeightNorm2", "name": "HeCBench particle weight normalize", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "particlefilter-cuda/main.cu", "fidelity": "simplified", "sizes": [4096], "tol": 1e-6,
    "features": ["__global__", "particle_filter", "normalization"],
    "description": "Normalize particle weights with a precomputed deterministic total.",
    "notes": "Standalone particle-filter weight normalization inspired by HeCBench particlefilter kernels.",
}, r"""
__global__ void normw(const float*w,float*y,float total,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=w[i]/total;}
int main(int argc,char**argv){const int n=4096;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hw=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));float total=0.0f;for(int i=0;i<n;++i){hw[i]=0.001f+h01(i,123);total+=hw[i];}float*dw,*dy;CK(cudaMalloc(&dw,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dw,hw,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));normw<<<(n+255)/256,256>>>(dw,dy,total,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dw);cudaFree(dy);free(hw);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    w = V.F32(0.001) + V.gen_hash01(n, 123)
    total = V.F32(0.0)
    for i in range(n):
        total += w[i]
    return w / total
""")

add({
    "id": "hecbenchHistogramBuckets2", "name": "HeCBench histogram buckets", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "histogram-cuda/main.cu", "fidelity": "inspired_by", "sizes": [262144, 128], "tol": 0.0,
    "features": ["__global__", "atomicAdd", "histogram"],
    "description": "Compute deterministic histogram bucket counts with atomics.",
    "notes": "Standalone histogram benchmark inspired by CUDA/HPC histogram kernels.",
}, r"""
__global__ void hist(int*out,int n,int bins){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int b=(i*37+11)&(bins-1);atomicAdd(&out[b],1);}}
int main(int argc,char**argv){const int n=262144,bins=128;const char*outp=(argc>1)?argv[1]:"output/output.txt";int*db;CK(cudaMalloc(&db,(size_t)bins*sizeof(int)));CK(cudaMemset(db,0,(size_t)bins*sizeof(int)));hist<<<(n+255)/256,256>>>(db,n,bins);CK(cudaGetLastError());CK(cudaDeviceSynchronize());int*hb=(int*)malloc((size_t)bins*sizeof(int));float*hy=(float*)malloc((size_t)bins*sizeof(float));CK(cudaMemcpy(hb,db,(size_t)bins*sizeof(int),cudaMemcpyDeviceToHost));for(int i=0;i<bins;++i)hy[i]=(float)hb[i];write_vec(outp,hy,bins);cudaFree(db);free(hb);free(hy);return 0;}
""", """
def reference(meta):
    n, bins = meta["input"]["sizes"]
    out = np.zeros(bins, dtype=np.int32)
    for i in range(n):
        out[(i * 37 + 11) & (bins - 1)] += 1
    return out.astype(np.float32)
""")

add({
    "id": "hecbenchBitonicCompare2", "name": "HeCBench bitonic compare pass", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "bitonic-sort-cuda/main.cu", "fidelity": "simplified", "sizes": [65536], "tol": 1e-6,
    "features": ["__global__", "compare_exchange", "sorting"],
    "description": "Apply one compare-exchange pass over adjacent pairs.",
    "notes": "Standalone compare pass modeled on HeCBench bitonic sort kernels.",
}, r"""
__global__ void cmp_pairs(const float*x,float*y,int n){int p=blockIdx.x*blockDim.x+threadIdx.x;if(2*p+1<n){float a=x[2*p],b=x[2*p+1];y[2*p]=fminf(a,b);y[2*p+1]=fmaxf(a,b);}}
int main(int argc,char**argv){const int n=65536;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));cmp_pairs<<<(n/2+255)/256,256>>>(dx,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123).reshape(-1, 2)
    y = np.empty_like(x)
    y[:, 0] = np.minimum(x[:, 0], x[:, 1])
    y[:, 1] = np.maximum(x[:, 0], x[:, 1])
    return y.reshape(-1)
""")

add({
    "id": "hecbenchConvolution5pt2", "name": "HeCBench convolution 5-point", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "convolution2D-cuda/main.cu", "fidelity": "inspired_by", "sizes": [256, 256], "tol": 1e-6,
    "features": ["__global__", "2D_stencil", "convolution"],
    "description": "Apply a 5-point convolution over a 2D grid.",
    "notes": "Standalone 5-point convolution inspired by HeCBench convolution CUDA kernels.",
}, r"""
__global__ void conv5(const float*x,float*y,int H,int W){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=H*W;if(idx<n){int r=idx/W,c=idx%W;float v=0.5f*x[idx];if(r>0)v+=0.125f*x[idx-W];if(r<H-1)v+=0.125f*x[idx+W];if(c>0)v+=0.125f*x[idx-1];if(c<W-1)v+=0.125f*x[idx+1];y[idx]=v;}}
int main(int argc,char**argv){const int H=256,W=256,n=H*W;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));conv5<<<(n+255)/256,256>>>(dx,dy,H,W);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    H, W = meta["input"]["sizes"]
    x = V.gen_hashsigned(H * W, 123).reshape(H, W)
    y = V.F32(0.5) * x
    y[1:] += V.F32(0.125) * x[:-1]
    y[:-1] += V.F32(0.125) * x[1:]
    y[:, 1:] += V.F32(0.125) * x[:, :-1]
    y[:, :-1] += V.F32(0.125) * x[:, 1:]
    return y.reshape(-1)
""")

add({
    "id": "hecbenchCsrRowSum2", "name": "HeCBench CSR row sum", "category": "hpc", "domain": "hpc", "difficulty": "medium",
    "source": "hecbench", "rel": "amgmk-cuda/csr_matvec.cu", "fidelity": "simplified", "sizes": [4096, 8], "tol": 1e-6,
    "features": ["__global__", "sparse_matrix", "row_reduction"],
    "description": "Sum fixed-width sparse row values.",
    "notes": "Standalone sparse row reduction inspired by HeCBench CSR matrix-vector kernels.",
}, r"""
__global__ void row_sum(const float*val,float*y,int rows,int width){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r<rows){float s=0.0f;for(int k=0;k<width;++k)s+=val[r*width+k];y[r]=s;}}
int main(int argc,char**argv){const int rows=4096,width=8,n=rows*width;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hv=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)rows*sizeof(float));for(int i=0;i<n;++i)hv[i]=0.1f*hs(i,123);float*dv,*dy;CK(cudaMalloc(&dv,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)rows*sizeof(float)));CK(cudaMemcpy(dv,hv,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));row_sum<<<(rows+255)/256,256>>>(dv,dy,rows,width);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)rows*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,rows);cudaFree(dv);cudaFree(dy);free(hv);free(hy);return 0;}
""", """
def reference(meta):
    rows, width = meta["input"]["sizes"]
    x = (V.F32(0.1) * V.gen_hashsigned(rows * width, 123)).reshape(rows, width)
    y = np.zeros(rows, dtype=np.float32)
    for k in range(width):
        y += x[:, k]
    return y
""")

# DALI image/preprocessing kernels: 5 cases.
for spec, expr, verify in [
    ({
        "id": "daliHorizontalFlip2", "name": "DALI horizontal flip", "category": "ai", "domain": "image_processing", "difficulty": "easy",
        "source": "dali", "rel": "dali/kernels/imgproc/flip/flip_gpu.cuh", "fidelity": "simplified", "sizes": [4, 64, 64, 3], "tol": 1e-6,
        "features": ["__global__", "image_processing", "layout"],
        "description": "Flip NHWC images horizontally.",
        "notes": "Standalone image flip inspired by DALI GPU flip kernels.",
    }, "int c=idx%C;int tmp=idx/C;int w=tmp%W;tmp/=W;int h=tmp%H;int n=tmp/H;int sw=W-1-w;y[idx]=x[((n*H+h)*W+sw)*C+c];", """
def reference(meta):
    N, H, W, C = meta["input"]["sizes"]
    x = V.gen_hash01(N * H * W * C, 123).reshape(N, H, W, C)
    return x[:, :, ::-1, :].reshape(-1)
"""),
    ({
        "id": "daliColorTwist2", "name": "DALI color twist", "category": "ai", "domain": "image_processing", "difficulty": "medium",
        "source": "dali", "rel": "dali/kernels/imgproc/color_manipulation/color_twist.cuh", "fidelity": "simplified", "sizes": [4, 64, 64, 3], "tol": 1e-6,
        "features": ["__global__", "color_transform", "image_processing"],
        "description": "Apply a compact RGB color twist matrix.",
        "notes": "Standalone color matrix transform inspired by DALI color twist kernels.",
    }, None, None),
]:
    if spec["id"] == "daliColorTwist2":
        main = r"""
__global__ void twist(const float*x,float*y,int pixels){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<pixels){float r=x[3*p],g=x[3*p+1],b=x[3*p+2];y[3*p]=0.9f*r+0.05f*g+0.02f*b;y[3*p+1]=0.04f*r+1.1f*g+0.03f*b;y[3*p+2]=0.02f*r+0.04f*g+0.95f*b;}}
int main(int argc,char**argv){const int N=4,H=64,W=64,C=3,total=N*H*W*C,pixels=N*H*W;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)total*sizeof(float)),*hy=(float*)malloc((size_t)total*sizeof(float));for(int i=0;i<total;++i)hx[i]=h01(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)total*sizeof(float)));CK(cudaMalloc(&dy,(size_t)total*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)total*sizeof(float),cudaMemcpyHostToDevice));twist<<<(pixels+255)/256,256>>>(dx,dy,pixels);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)total*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,total);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
"""
        ver = """
def reference(meta):
    N, H, W, C = meta["input"]["sizes"]
    x = V.gen_hash01(N * H * W * C, 123).reshape(-1, 3)
    y = np.empty_like(x)
    y[:,0] = V.F32(0.9) * x[:,0] + V.F32(0.05) * x[:,1] + V.F32(0.02) * x[:,2]
    y[:,1] = V.F32(0.04) * x[:,0] + V.F32(1.1) * x[:,1] + V.F32(0.03) * x[:,2]
    y[:,2] = V.F32(0.02) * x[:,0] + V.F32(0.04) * x[:,1] + V.F32(0.95) * x[:,2]
    return y.reshape(-1)
"""
    else:
        main = f"""
__global__ void imgop(const float*x,float*y,int N,int H,int W,int C){{int idx=blockIdx.x*blockDim.x+threadIdx.x,total=N*H*W*C;if(idx<total){{{expr}}}}}
int main(int argc,char**argv){{const int N=4,H=64,W=64,C=3,total=N*H*W*C;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)total*sizeof(float)),*hy=(float*)malloc((size_t)total*sizeof(float));for(int i=0;i<total;++i)hx[i]=h01(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)total*sizeof(float)));CK(cudaMalloc(&dy,(size_t)total*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)total*sizeof(float),cudaMemcpyHostToDevice));imgop<<<(total+255)/256,256>>>(dx,dy,N,H,W,C);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)total*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,total);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}}
"""
        ver = verify
    add(spec, main, ver)

add({
    "id": "daliCropMirrorPad2", "name": "DALI crop mirror pad", "category": "ai", "domain": "image_processing", "difficulty": "medium",
    "source": "dali", "rel": "dali/kernels/slice/slice_flip_normalize_permute_pad_cuda_impl.cuh", "fidelity": "simplified", "sizes": [32, 32, 48, 48], "tol": 1e-6,
    "features": ["__global__", "crop", "padding"],
    "description": "Crop, mirror, and zero-pad a single-channel image.",
    "notes": "Standalone crop/mirror/pad path inspired by DALI fused preprocessing kernels.",
}, r"""
__global__ void crop_pad(const float*x,float*y,int H0,int W0,int H1,int W1){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=H1*W1;if(idx<n){int r=idx/W1,c=idx%W1;float v=0.0f;if(r<32&&c<32){int sr=r,sc=31-c;v=x[sr*W0+sc];}y[idx]=v;}}
int main(int argc,char**argv){const int H0=32,W0=32,H1=48,W1=48,n0=H0*W0,n1=H1*W1;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n0*sizeof(float)),*hy=(float*)malloc((size_t)n1*sizeof(float));for(int i=0;i<n0;++i)hx[i]=h01(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n0*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n1*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n0*sizeof(float),cudaMemcpyHostToDevice));crop_pad<<<(n1+255)/256,256>>>(dx,dy,H0,W0,H1,W1);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n1*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n1);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    H0, W0, H1, W1 = meta["input"]["sizes"]
    x = V.gen_hash01(H0 * W0, 123).reshape(H0, W0)
    y = np.zeros((H1, W1), dtype=np.float32)
    y[:32, :32] = x[:32, ::-1]
    return y.reshape(-1)
""")

add({
    "id": "daliAffineScaleTranslate2", "name": "DALI affine scale translate", "category": "ai", "domain": "image_processing", "difficulty": "medium",
    "source": "dali", "rel": "dali/kernels/imgproc/warp/affine.cuh", "fidelity": "simplified", "sizes": [64, 64], "tol": 1e-6,
    "features": ["__global__", "affine_transform", "sampling"],
    "description": "Sample a single-channel image with a simple affine transform.",
    "notes": "Standalone nearest affine sampling inspired by DALI warp affine kernels.",
}, r"""
__global__ void affine_nn(const float*x,float*y,int H,int W){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=H*W;if(idx<n){int r=idx/W,c=idx%W;int sr=min(max((int)floorf(0.75f*r+4.0f),0),H-1);int sc=min(max((int)floorf(0.75f*c+2.0f),0),W-1);y[idx]=x[sr*W+sc];}}
int main(int argc,char**argv){const int H=64,W=64,n=H*W;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=h01(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));affine_nn<<<(n+255)/256,256>>>(dx,dy,H,W);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    H, W = meta["input"]["sizes"]
    x = V.gen_hash01(H * W, 123).reshape(H, W)
    y = np.empty((H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            sr = min(max(int(np.floor(V.F32(0.75) * V.F32(r) + V.F32(4.0))), 0), H - 1)
            sc = min(max(int(np.floor(V.F32(0.75) * V.F32(c) + V.F32(2.0))), 0), W - 1)
            y[r, c] = x[sr, sc]
    return y.reshape(-1)
""")

add({
    "id": "daliBoxEncode2", "name": "DALI box encode helper", "category": "ai", "domain": "image_processing", "difficulty": "medium",
    "source": "dali", "rel": "dali/kernels/bbox/bbox.h", "fidelity": "inspired_by", "sizes": [4096], "tol": 1e-6,
    "features": ["__global__", "bbox", "elementwise"],
    "description": "Encode bounding-box centers and sizes into offset form.",
    "notes": "Standalone bounding-box arithmetic inspired by DALI box encoder style preprocessing.",
}, r"""
__global__ void boxenc(float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float x1=h01(i,11)*0.5f,y1=h01(i,22)*0.5f,w=0.1f+0.4f*h01(i,33),h=0.1f+0.4f*h01(i,44);y[4*i]=x1+0.5f*w;y[4*i+1]=y1+0.5f*h;y[4*i+2]=logf(w);y[4*i+3]=logf(h);}}
int main(int argc,char**argv){const int n=4096;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hy=(float*)malloc((size_t)4*n*sizeof(float));float*dy;CK(cudaMalloc(&dy,(size_t)4*n*sizeof(float)));boxenc<<<(n+255)/256,256>>>(dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)4*n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,4*n);cudaFree(dy);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    x1 = V.gen_hash01(n, 11) * V.F32(0.5)
    y1 = V.gen_hash01(n, 22) * V.F32(0.5)
    w = V.F32(0.1) + V.F32(0.4) * V.gen_hash01(n, 33)
    h = V.F32(0.1) + V.F32(0.4) * V.gen_hash01(n, 44)
    out = np.empty((n, 4), dtype=np.float32)
    out[:,0] = x1 + V.F32(0.5) * w
    out[:,1] = y1 + V.F32(0.5) * h
    out[:,2] = np.log(w).astype(np.float32)
    out[:,3] = np.log(h).astype(np.float32)
    return out.reshape(-1)
""")

# CUDA primitives / APIs: 8 primitive cases plus 1 Thrust library case.
add({
    "id": "cudaSamplesSharedReduceSum2", "name": "CUDA Samples shared block reduce", "category": "medium", "domain": "cuda_primitive", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/reduction/reduction_kernel.cu", "fidelity": "inspired_by", "sizes": [65536, 256], "tol": 1e-5,
    "features": ["__global__", "__shared__", "reduction"],
    "description": "Compute one sum per block with shared-memory reduction.",
    "notes": "Standalone shared-memory reduction inspired by CUDA Samples reduction kernels.",
}, r"""
__global__ void block_sum(const float*x,float*y,int n){__shared__ float s[256];int t=threadIdx.x,i=blockIdx.x*blockDim.x+t;s[t]=(i<n)?x[i]:0.0f;__syncthreads();for(int stride=128;stride>0;stride>>=1){if(t<stride)s[t]+=s[t+stride];__syncthreads();}if(t==0)y[blockIdx.x]=s[0];}
int main(int argc,char**argv){const int n=65536,B=256,outn=n/B;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)outn*sizeof(float));for(int i=0;i<n;++i)hx[i]=0.01f*hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)outn*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));block_sum<<<outn,B>>>(dx,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)outn*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,outn);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n, B = meta["input"]["sizes"]
    x = (V.F32(0.01) * V.gen_hashsigned(n, 123)).reshape(-1, B)
    return np.sum(x, axis=1, dtype=np.float32)
""")

add({
    "id": "cudaSamplesConstantLookup2", "name": "CUDA Samples constant lookup", "category": "medium", "domain": "cuda_primitive", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTemplates/simpleTemplates.cu", "fidelity": "inspired_by", "sizes": [1048576], "tol": 1e-6,
    "features": ["__constant__", "__global__", "lookup_table"],
    "description": "Use constant coefficients selected by low index bits.",
    "notes": "Standalone constant-memory lookup inspired by CUDA Samples template examples.",
}, r"""
__constant__ float lut8[8];
__global__ void lookup(const float*x,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=x[i]*lut8[i&7];}
int main(int argc,char**argv){const int n=1048576;const char*out=(argc>1)?argv[1]:"output/output.txt";float h[8]={0.125f,0.25f,0.5f,0.75f,1.0f,1.25f,1.5f,2.0f};float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);CK(cudaMemcpyToSymbol(lut8,h,8*sizeof(float)));float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));lookup<<<(n+255)/256,256>>>(dx,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    lut = np.array([0.125,0.25,0.5,0.75,1.0,1.25,1.5,2.0], dtype=np.float32)
    return V.gen_hashsigned(n, 123) * lut[np.arange(n) & 7]
""")

add({
    "id": "cudaSamplesWarpBroadcast2", "name": "CUDA Samples warp broadcast", "category": "medium", "domain": "cuda_primitive", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTemplates/simpleTemplates.cu", "fidelity": "inspired_by", "sizes": [32768], "tol": 1e-6,
    "features": ["__global__", "__shfl_sync", "warp"],
    "description": "Broadcast lane-zero values within each warp.",
    "notes": "Standalone warp shuffle primitive inspired by CUDA Samples shuffle examples.",
}, r"""
__global__ void wbcast(const float*x,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];float b=__shfl_sync(0xffffffff,v,0);y[i]=b;}}
int main(int argc,char**argv){const int n=32768;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));wbcast<<<(n+255)/256,256>>>(dx,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123).reshape(-1, 32)
    return np.repeat(x[:, :1], 32, axis=1).reshape(-1)
""")

add({
    "id": "cudaSamplesManagedTouch2", "name": "CUDA Samples managed memory touch", "category": "medium", "domain": "cuda_primitive", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/UnifiedMemoryStreams/UnifiedMemoryStreams.cu", "fidelity": "inspired_by", "sizes": [1048576], "tol": 1e-6,
    "features": ["cudaMallocManaged", "__global__", "unified_memory"],
    "description": "Update a managed-memory allocation in place.",
    "notes": "Standalone unified-memory primitive inspired by CUDA Samples UnifiedMemoryStreams.",
}, r"""
__global__ void touch(float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=x[i]+1.0f;}
int main(int argc,char**argv){const int n=1048576;const char*out=(argc>1)?argv[1]:"output/output.txt";float*x;CK(cudaMallocManaged(&x,(size_t)n*sizeof(float)));for(int i=0;i<n;++i)x[i]=hs(i,123);touch<<<(n+255)/256,256>>>(x,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());write_vec(out,x,n);cudaFree(x);return 0;}
""", """
def reference(meta):
    return V.gen_hashsigned(meta["input"]["sizes"][0], 123) + V.F32(1.0)
""")

add({
    "id": "cudaSamplesPinnedChunkAdd2", "name": "CUDA Samples pinned chunk add", "category": "medium", "domain": "cuda_primitive", "difficulty": "hard",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleStreams/simpleStreams.cu", "fidelity": "inspired_by", "sizes": [524288], "tol": 1e-6,
    "features": ["cudaMallocHost", "cudaMemcpyAsync", "cudaStream_t"],
    "description": "Add vectors in pinned-memory chunks with CUDA streams.",
    "notes": "Standalone pinned-memory chunk transfer inspired by CUDA Samples simpleStreams.",
}, r"""
__global__ void addk(const float*a,const float*b,float*c,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}
int main(int argc,char**argv){const int n=524288,ch=n/4;const char*out=(argc>1)?argv[1]:"output/output.txt";float *ha,*hb,*hc;CK(cudaMallocHost(&ha,(size_t)n*sizeof(float)));CK(cudaMallocHost(&hb,(size_t)n*sizeof(float)));CK(cudaMallocHost(&hc,(size_t)n*sizeof(float)));for(int i=0;i<n;++i){ha[i]=hs(i,123);hb[i]=hs(i,321);}float *da[4],*db[4],*dc[4];cudaStream_t st[4];for(int s=0;s<4;++s){CK(cudaStreamCreate(&st[s]));CK(cudaMalloc(&da[s],(size_t)ch*sizeof(float)));CK(cudaMalloc(&db[s],(size_t)ch*sizeof(float)));CK(cudaMalloc(&dc[s],(size_t)ch*sizeof(float)));CK(cudaMemcpyAsync(da[s],ha+s*ch,(size_t)ch*sizeof(float),cudaMemcpyHostToDevice,st[s]));CK(cudaMemcpyAsync(db[s],hb+s*ch,(size_t)ch*sizeof(float),cudaMemcpyHostToDevice,st[s]));addk<<<(ch+255)/256,256,0,st[s]>>>(da[s],db[s],dc[s],ch);CK(cudaMemcpyAsync(hc+s*ch,dc[s],(size_t)ch*sizeof(float),cudaMemcpyDeviceToHost,st[s]));}CK(cudaDeviceSynchronize());write_vec(out,hc,n);for(int s=0;s<4;++s){cudaFree(da[s]);cudaFree(db[s]);cudaFree(dc[s]);cudaStreamDestroy(st[s]);}cudaFreeHost(ha);cudaFreeHost(hb);cudaFreeHost(hc);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    return V.gen_hashsigned(n, 123) + V.gen_hashsigned(n, 321)
""")

add({
    "id": "cudaSamplesClampAddress2", "name": "CUDA Samples clamp address sampler", "category": "medium", "domain": "cuda_primitive", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTexture/simpleTexture.cu", "fidelity": "inspired_by", "sizes": [128, 128], "tol": 1e-6,
    "features": ["__global__", "address_clamp", "sampling"],
    "description": "Sample a 2D array with clamp addressing.",
    "notes": "Models CUDA texture clamp addressing using regular global memory for portability.",
}, r"""
__global__ void clamp_sample(const float*x,float*y,int H,int W){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=H*W;if(idx<n){int r=idx/W,c=idx%W;int sr=min(max(r-5,0),H-1),sc=min(max(c+7,0),W-1);y[idx]=x[sr*W+sc];}}
int main(int argc,char**argv){const int H=128,W=128,n=H*W;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=h01(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));clamp_sample<<<(n+255)/256,256>>>(dx,dy,H,W);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    H, W = meta["input"]["sizes"]
    x = V.gen_hash01(H * W, 123).reshape(H, W)
    y = np.empty((H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            y[r, c] = x[min(max(r - 5, 0), H - 1), min(max(c + 7, 0), W - 1)]
    return y.reshape(-1)
""")

add({
    "id": "cudaSamplesDynamicSharedHistogram2", "name": "CUDA Samples dynamic shared histogram", "category": "medium", "domain": "cuda_primitive", "difficulty": "hard",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics.cu", "fidelity": "inspired_by", "sizes": [65536, 16], "tol": 0.0,
    "features": ["__global__", "dynamic_shared_memory", "atomicAdd"],
    "description": "Use dynamic shared memory for per-block histogram accumulation.",
    "notes": "Standalone shared histogram inspired by CUDA Samples atomic intrinsics.",
}, r"""
__global__ void shist(int*out,int n,int bins){extern __shared__ int s[];for(int b=threadIdx.x;b<bins;b+=blockDim.x)s[b]=0;__syncthreads();int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int b=(i*17+3)&(bins-1);atomicAdd(&s[b],1);}__syncthreads();for(int b=threadIdx.x;b<bins;b+=blockDim.x)atomicAdd(&out[b],s[b]);}
int main(int argc,char**argv){const int n=65536,bins=16;const char*outp=(argc>1)?argv[1]:"output/output.txt";int*db;CK(cudaMalloc(&db,(size_t)bins*sizeof(int)));CK(cudaMemset(db,0,(size_t)bins*sizeof(int)));shist<<<(n+255)/256,256,bins*sizeof(int)>>>(db,n,bins);CK(cudaGetLastError());CK(cudaDeviceSynchronize());int*hb=(int*)malloc((size_t)bins*sizeof(int));float*hy=(float*)malloc((size_t)bins*sizeof(float));CK(cudaMemcpy(hb,db,(size_t)bins*sizeof(int),cudaMemcpyDeviceToHost));for(int i=0;i<bins;++i)hy[i]=(float)hb[i];write_vec(outp,hy,bins);cudaFree(db);free(hb);free(hy);return 0;}
""", """
def reference(meta):
    n, bins = meta["input"]["sizes"]
    out = np.zeros(bins, dtype=np.int32)
    for i in range(n):
        out[(i * 17 + 3) & (bins - 1)] += 1
    return out.astype(np.float32)
""")

add({
    "id": "cudaSamplesVectorizedFloat4Scale2", "name": "CUDA Samples float4 vectorized scale", "category": "medium", "domain": "cuda_primitive", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTemplates/simpleTemplates.cu", "fidelity": "inspired_by", "sizes": [262144], "tol": 1e-6,
    "features": ["__global__", "float4", "vectorized_memory"],
    "description": "Scale values through float4 vectorized loads and stores.",
    "notes": "Standalone vectorized memory access pattern inspired by CUDA Samples template examples.",
}, r"""
__global__ void scale4(const float4*x,float4*y,int n4){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n4){float4 v=x[i];v.x*=0.5f;v.y*=0.5f;v.z*=0.5f;v.w*=0.5f;y[i]=v;}}
int main(int argc,char**argv){const int n=262144,n4=n/4;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float4 *dx,*dy;CK(cudaMalloc(&dx,(size_t)n4*sizeof(float4)));CK(cudaMalloc(&dy,(size_t)n4*sizeof(float4)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));scale4<<<(n4+255)/256,256>>>(dx,dy,n4);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    return V.F32(0.5) * V.gen_hashsigned(meta["input"]["sizes"][0], 123)
""")

add({
    "id": "thrustAdjacentDifference2", "name": "Thrust adjacent difference", "category": "library_api", "domain": "library_api", "difficulty": "medium",
    "source": "cuda_samples", "rel": "cpp/0_Introduction/simpleTemplates/simpleTemplates.cu", "fidelity": "inspired_by", "sizes": [262144], "tol": 1e-6,
    "features": ["thrust::device_vector", "thrust::adjacent_difference"],
    "libraries": ["Thrust"],
    "description": "Compute adjacent differences with Thrust.",
    "notes": "Standalone Thrust API case inspired by CUDA Samples' Thrust usage.",
}, r"""
#include <thrust/device_vector.h>
#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
int main(int argc,char**argv){const int n=262144;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);thrust::device_vector<float>x(hx,hx+n),y(n);thrust::adjacent_difference(x.begin(),x.end(),y.begin());thrust::copy(y.begin(),y.end(),hy);write_vec(out,hy,n);free(hx);free(hy);return 0;}
""", """
def reference(meta):
    n = meta["input"]["sizes"][0]
    x = V.gen_hashsigned(n, 123)
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - x[:-1]
    return y
""")


def main() -> int:
    seen = set()
    for spec in CASES:
        if spec["id"] in seen:
            raise RuntimeError(f"duplicate case id {spec['id']}")
        seen.add(spec["id"])
        base.write_case(spec)
        print(f"[ok] {spec['category']}/{spec['id']} <- {base.SOURCES[spec['source']]['project']}")
    print(f"Wrote {len(CASES)} Stage 1 batch-4 cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
