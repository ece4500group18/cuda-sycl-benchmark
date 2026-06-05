#!/usr/bin/env python3
"""One-off scaffolder: create dir skeletons, metadata.json, CMakeLists.txt and
README files for every benchmark case from a central table.

main.cu and tests/verify.py contain real logic and are authored separately;
this script only (re)generates the repetitive surrounding files and never
overwrites an existing main.cu or verify.py.

Run from anywhere:  python3 tools/_scaffold_cases.py
"""
from __future__ import annotations

import collections
import json
import os

OD = collections.OrderedDict
TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)

DEF_CUDA_BUILD = "nvcc -O2 -std=c++17 original/main.cu -o original/build/app"
DEF_SYCL_BUILD = "icpx -fsycl -O2 -std=c++17 syclomatic/main.dp.cpp -o build_sycl/app"
DEF_CUDA_RUN = "original/build/app output/cuda_output.txt"
DEF_SYCL_RUN = "build_sycl/app output/sycl_output.txt"

# Fields: cat, cid, name, features, libs, in_type, sizes, method, metric, tol,
#         notes, cuda_link, sycl_build
C = [
 # ---------------- EASY (5 new; 5 already authored) ----------------
 ("easy","hadamard","Element-wise product (C = A*B)",
  ["__global__","cudaMalloc","cudaMemcpy","cudaFree","elementwise"],[],
  "deterministic_formula",[100000],"cpu_reference","max_abs_error",1e-5,"",""," "),
 ("easy","affine","Affine map (out = 2*a + 1)",
  ["__global__","cudaMalloc","cudaMemcpy","cudaFree"],[],
  "deterministic_formula",[100000],"cpu_reference","max_abs_error",1e-5,"",""," "),
 ("easy","reverseArray","Reverse array (out[i] = in[n-1-i])",
  ["__global__","cudaMalloc","cudaMemcpy","cudaFree"],[],
  "deterministic_formula",[100000],"cpu_reference","exact",0.0,"Pure data movement.",""," "),
 ("easy","gridStride","Grid-stride loop vector add",
  ["__global__","grid_stride_loop","cudaMalloc","cudaMemcpy","cudaFree"],[],
  "deterministic_formula",[200000],"cpu_reference","max_abs_error",1e-5,
  "Classic grid-stride loop pattern.",""," "),
 ("easy","square","Square elements (out = a*a)",
  ["__global__","cudaMalloc","cudaMemcpy","cudaFree"],[],
  "deterministic_formula",[100000],"cpu_reference","max_abs_error",1e-5,"",""," "),

 # ---------------- MEDIUM (shared memory / reduction) ----------------
 ("medium","reduceSum","Parallel sum reduction",
  ["__shared__","__syncthreads","reduction","cudaMalloc","cudaMemcpy"],[],
  "hashed",[1048576],"cpu_reference","max_rel_error",1e-3,
  "Two-pass block reduction; sum order differs from CPU.",""," "),
 ("medium","reduceMax","Parallel max reduction",
  ["__shared__","__syncthreads","reduction"],[],
  "hashed",[1048576],"cpu_reference","max_abs_error",0.0,
  "Max is order-independent; exact match expected.",""," "),
 ("medium","dotProduct","Dot product via reduction",
  ["__shared__","__syncthreads","reduction"],[],
  "hashed",[1048576],"cpu_reference","max_rel_error",1e-3,"",""," "),
 ("medium","scanBlock","Inclusive prefix sum (single block)",
  ["__shared__","__syncthreads","scan"],[],
  "hashed",[1024],"cpu_reference","max_rel_error",1e-4,
  "Hillis-Steele inclusive scan over one block.",""," "),
 ("medium","histogram","256-bin histogram (atomics)",
  ["__shared__","atomicAdd","cudaMemcpy"],[],
  "hashed_int",[1048576],"cpu_reference","exact",0.0,
  "Integer bin counts; exact match. sizes=[n]; 256 bins.",""," "),
 ("medium","transposeShared","Tiled transpose with shared memory",
  ["__shared__","__syncthreads","tiling","coalescing"],[],
  "hashed",[256,256],"cpu_reference","exact",0.0,
  "Shared-memory tiled transpose. sizes=[rows,cols].",""," "),
 ("medium","conv1dShared","1D convolution with shared-memory halo",
  ["__shared__","__syncthreads","halo","convolution"],[],
  "hashed",[100000],"cpu_reference","max_rel_error",1e-4,
  "Radius-3 fixed-weight stencil with shared-memory halo.",""," "),
 ("medium","tiledMatmul","Tiled (shared-memory) matrix multiply",
  ["__shared__","__syncthreads","tiling"],[],
  "hashed",[256],"cpu_reference","max_rel_error",1e-3,
  "16x16 tiled GEMM. sizes=[N].",""," "),
 ("medium","nbodyTiled","Tiled N-body acceleration",
  ["__shared__","__syncthreads","tiling"],[],
  "hashed",[2048],"cpu_reference","max_rel_error",2e-3,
  "Per-body acceleration with softening; sizes=[N]; output 3*N.",""," "),
 ("medium","bitonicSort","Bitonic sort (single block)",
  ["__shared__","__syncthreads","sorting"],[],
  "hashed",[1024],"cpu_reference","exact",0.0,
  "Ascending bitonic sort over one block of 1024.",""," "),

 # ---------------- HPC ----------------
 ("hpc","stencil1d","3-point 1D stencil",
  ["__global__","stencil"],[],
  "hashed",[100000],"cpu_reference","max_rel_error",1e-4,"",""," "),
 ("hpc","stencil2d","5-point 2D stencil",
  ["__global__","stencil","2D_indexing"],[],
  "hashed",[256,256],"cpu_reference","max_rel_error",1e-4,
  "sizes=[ny,nx].",""," "),
 ("hpc","stencil3d","7-point 3D stencil",
  ["__global__","stencil","3D_indexing"],[],
  "hashed",[64,64,64],"cpu_reference","max_rel_error",1e-3,
  "sizes=[nz,ny,nx].",""," "),
 ("hpc","jacobi2d","Jacobi iteration (2D Laplace)",
  ["__global__","iterative","2D_indexing"],[],
  "hashed",[128,128,50],"cpu_reference","max_rel_error",1e-3,
  "Fixed boundaries, K iterations. sizes=[ny,nx,iters].",""," "),
 ("hpc","heat2d","2D heat equation (explicit)",
  ["__global__","iterative","2D_indexing"],[],
  "hashed",[128,128,50],"cpu_reference","max_rel_error",1e-3,
  "Explicit FTCS, K steps. sizes=[ny,nx,steps].",""," "),
 ("hpc","finiteDiff1d","Central finite-difference derivative",
  ["__global__","finite_difference"],[],
  "hashed",[100000],"cpu_reference","max_rel_error",1e-4,"",""," "),
 ("hpc","spmv","Sparse matrix-vector (CSR)",
  ["__global__","sparse","csr"],[],
  "hashed",[100000],"cpu_reference","max_rel_error",1e-4,
  "Tridiagonal CSR matrix; y = A*x. sizes=[N].",""," "),
 ("hpc","conjugateGradient","Conjugate gradient (1D Laplacian)",
  ["__global__","reduction","iterative"],[],
  "hashed",[1024,200],"residual_norm","rel_residual",1e-3,
  "SPD 1D Laplacian; verify ||Ax-b||/||b||. sizes=[N,iters].",""," "),
 ("hpc","bfs","Breadth-first search distances",
  ["__global__","graph","atomicAdd"],[],
  "hashed_int",[20000],"cpu_reference","exact",0.0,
  "Distances from node 0 on a deterministic graph. sizes=[N].",""," "),
 ("hpc","monteCarloPi","Monte Carlo estimate of pi",
  ["__global__","reduction","atomicAdd"],[],
  "hashed",[4194304],"analytic_reference","abs_error_to_pi",2e-2,
  "Hash-based samples in unit square; |est - pi|. sizes=[samples].",""," "),

 # ---------------- AI ----------------
 ("ai","gemm","GEMM with bias (C = A*B + bias)",
  ["__global__","tiling"],[],
  "hashed",[128],"cpu_reference","max_rel_error",1e-3,
  "Square GEMM + per-column bias. sizes=[N].",""," "),
 ("ai","batchedGemm","Batched matrix multiply",
  ["__global__","batched"],[],
  "hashed",[8,64],"cpu_reference","max_rel_error",1e-3,
  "B batches of NxN GEMM. sizes=[batch,N].",""," "),
 ("ai","softmax","Row-wise softmax",
  ["__global__","__shared__","reduction"],[],
  "hashed",[512,512],"cpu_reference","max_abs_error",1e-5,
  "Numerically stable softmax per row. sizes=[rows,cols].",""," "),
 ("ai","layernorm","Layer normalization",
  ["__global__","__shared__","reduction"],[],
  "hashed",[512,1024],"cpu_reference","max_abs_error",1e-4,
  "LayerNorm with gamma=1,beta=0,eps=1e-5. sizes=[rows,cols].",""," "),
 ("ai","rmsnorm","RMS normalization",
  ["__global__","__shared__","reduction"],[],
  "hashed",[512,1024],"cpu_reference","max_abs_error",1e-4,
  "RMSNorm with gamma=1,eps=1e-6. sizes=[rows,cols].",""," "),
 ("ai","gelu","GELU activation (tanh approx)",
  ["__global__","elementwise"],[],
  "hashed",[1048576],"cpu_reference","max_abs_error",1e-4,
  "tanh approximation of GELU.",""," "),
 ("ai","embedding","Embedding lookup",
  ["__global__","gather"],[],
  "hashed_int",[10000,128,4096],"cpu_reference","exact",0.0,
  "Gather rows from a table. sizes=[vocab,dim,num_ids].",""," "),
 ("ai","rope","Rotary position embedding",
  ["__global__","elementwise"],[],
  "hashed",[128,64],"cpu_reference","max_abs_error",1e-4,
  "Apply RoPE to a [seq,dim] tensor. sizes=[seq,dim].",""," "),
 ("ai","attention","Scaled dot-product attention",
  ["__global__","__shared__","reduction"],[],
  "hashed",[128,64],"cpu_reference","max_rel_error",2e-3,
  "Single head softmax(QK^T/sqrt(d))V. sizes=[seq,dim].",""," "),
 ("ai","topk","Top-k per row",
  ["__global__","selection"],[],
  "hashed",[256,512],"cpu_reference","exact",0.0,
  "Top-8 values per row, sorted desc. sizes=[rows,cols]; k=8.",""," "),

 # ---------------- LIBRARY / API ----------------
 ("library_api","cublasGemm","cuBLAS SGEMM",
  ["cublas","streams"],["cublas"],
  "hashed",[256],"cpu_reference","max_rel_error",1e-3,
  "cublasSgemm; column-major handling. sizes=[N].",
  "-lcublas","icpx -fsycl -qmkl -O2 -std=c++17 syclomatic/main.dp.cpp -o build_sycl/app"),
 ("library_api","cublasAxpy","cuBLAS SAXPY",
  ["cublas"],["cublas"],
  "hashed",[1048576],"cpu_reference","max_abs_error",1e-4,
  "cublasSaxpy; y = alpha*x + y.",
  "-lcublas","icpx -fsycl -qmkl -O2 -std=c++17 syclomatic/main.dp.cpp -o build_sycl/app"),
 ("library_api","curandUniform","cuRAND uniform generation",
  ["curand"],["curand"],
  "statistical",[1048576],"statistical","mean_and_range",1e-2,
  "Verify mean approx 0.5 and all values in [0,1).",
  "-lcurand","icpx -fsycl -qmkl -O2 -std=c++17 syclomatic/main.dp.cpp -o build_sycl/app"),
 ("library_api","cufftC2C","cuFFT 1D complex FFT",
  ["cufft"],["cufft"],
  "hashed",[4096],"cpu_reference","max_rel_error",1e-3,
  "Forward C2C FFT; compare magnitude spectrum to numpy. sizes=[n].",
  "-lcufft","icpx -fsycl -qmkl -O2 -std=c++17 syclomatic/main.dp.cpp -o build_sycl/app"),
 ("library_api","thrustSort","Thrust sort",
  ["thrust"],["thrust"],
  "hashed",[1048576],"cpu_reference","exact",0.0,
  "thrust::sort ascending.",""," "),
 ("library_api","thrustReduce","Thrust reduce (sum)",
  ["thrust"],["thrust"],
  "hashed",[1048576],"cpu_reference","max_rel_error",1e-3,
  "thrust::reduce sum.",""," "),
 ("library_api","cudaEventTiming","cudaEvent timing around a kernel",
  ["cudaEvent"],[],
  "deterministic_formula",[1048576],"cpu_reference","max_abs_error",1e-5,
  "Events time a vector add; correctness = the add result.",""," "),
 ("library_api","cudaStream","Multi-stream vector add",
  ["streams","cudaMemcpyAsync"],[],
  "deterministic_formula",[1048576],"cpu_reference","max_abs_error",1e-5,
  "Work split across 4 streams.",""," "),
 ("library_api","cudaMemcpyAsyncPinned","Pinned async copy + scale",
  ["pinned_memory","cudaMemcpyAsync","streams"],[],
  "deterministic_formula",[1048576],"cpu_reference","max_abs_error",1e-5,
  "cudaHostAlloc pinned + async H2D/D2H; out = 3*a.",""," "),
 ("library_api","cudaGraph","CUDA graph capturing a kernel",
  ["cudaGraph","streams"],[],
  "deterministic_formula",[1048576],"cpu_reference","max_abs_error",1e-5,
  "Stream-captured graph runs a scale kernel; out = 2*a + 1.",""," "),
]


def meta_for(row):
    (cat, cid, name, feats, libs, in_type, sizes, method, metric, tol,
     notes, cuda_link, sycl_build) = row
    cuda_build = DEF_CUDA_BUILD + ((" " + cuda_link) if cuda_link.strip() else "")
    sycl_build = sycl_build if sycl_build.strip() else DEF_SYCL_BUILD
    return OD([
        ("case_id", cid), ("name", name), ("category", cat),
        ("source", OD([("type", "authored"), ("url", ""), ("license", "MIT"),
                       ("original_path", "original/main.cu")])),
        ("cuda_features", feats), ("libraries", libs),
        ("input", OD([("type", in_type), ("sizes", list(sizes)), ("seed", 123)])),
        ("build", OD([("cuda_build_command", cuda_build),
                      ("sycl_build_command", sycl_build)])),
        ("run", OD([("cuda_run_command", DEF_CUDA_RUN),
                    ("sycl_run_command", DEF_SYCL_RUN)])),
        ("correctness", OD([("method", method), ("metric", metric),
                            ("tolerance", tol), ("expected_pass_string", "PASS")])),
        ("syclomatic", OD([("status", "not_run"), ("command", ""),
                           ("warnings_count", None), ("manual_fixes_required", None)])),
        ("status", OD([(k, "unknown") for k in
                       ["cuda_compile", "cuda_run", "cuda_verify",
                        "syclomatic_migrate", "sycl_compile", "sycl_run",
                        "sycl_verify"]])),
        ("notes", notes),
    ])


CMAKE_TPL = """cmake_minimum_required(VERSION 3.18)
project({cid} LANGUAGES CXX CUDA)
add_executable(app main.cu){link}
set_target_properties(app PROPERTIES CUDA_ARCHITECTURES "70;80;90")
# Note: the pilot tooling builds via the nvcc command in metadata.json. This
# CMakeLists is provided for completeness / portable standalone builds.
"""

ORIG_README_TPL = """# {cid} (original CUDA)

{name}

Inputs are generated deterministically on the host (see `main.cu`) and
replicated by `../tests/verify.py`. The result is written to `argv[1]`
(one value per line).

Build (toolchain permitting):

    nvcc -O2 -std=c++17 main.cu -o build/app {link}

Run:

    build/app ../output/cuda_output.txt
"""

CASE_README_TPL = """# Case: {cid} ({cat})

| field | value |
| --- | --- |
| category | {cat} |
| operation | {name} |
| correctness | {method} / {metric} (tol {tol}) |
| CUDA features | {feats} |
| libraries | {libs} |

Notes: {notes}

## Pipeline
Build CUDA -> run -> verify, then SYCLomatic migrate -> build SYCL -> run ->
verify. Inputs are deterministic; `tests/verify.py` recomputes a CPU reference
and compares `output/<variant>_output.txt` within tolerance. Missing
toolchains/devices yield `skipped_*` statuses (never hard failures).
"""


def main():
    made = 0
    for row in C:
        (cat, cid, name, feats, libs, in_type, sizes, method, metric, tol,
         notes, cuda_link, sycl_build) = row
        case_dir = os.path.join(REPO, "cases", cat, cid)
        for sub in ("original", "syclomatic", "manual_sycl", "input",
                    "output", "logs", "tests"):
            os.makedirs(os.path.join(case_dir, sub), exist_ok=True)
        # metadata.json (always regenerated to stay in sync with the table)
        with open(os.path.join(case_dir, "metadata.json"), "w") as fh:
            json.dump(meta_for(row), fh, indent=2)
            fh.write("\n")
        # CMakeLists.txt
        link = ""
        if cuda_link.strip():
            libname = cuda_link.strip().lstrip("-l")
            link = f"\ntarget_link_libraries(app PRIVATE {libname})"
        with open(os.path.join(case_dir, "original", "CMakeLists.txt"), "w") as fh:
            fh.write(CMAKE_TPL.format(cid=cid, link=link))
        # original/README.md
        with open(os.path.join(case_dir, "original", "README.md"), "w") as fh:
            fh.write(ORIG_README_TPL.format(cid=cid, name=name,
                                            link=cuda_link.strip()))
        # case README.md
        with open(os.path.join(case_dir, "README.md"), "w") as fh:
            fh.write(CASE_README_TPL.format(
                cid=cid, cat=cat, name=name, method=method, metric=metric,
                tol=tol, feats=", ".join(feats) or "-",
                libs=", ".join(libs) or "-", notes=notes or "-"))
        made += 1
    print(f"Scaffolded {made} cases (metadata + CMake + READMEs).")


if __name__ == "__main__":
    main()
