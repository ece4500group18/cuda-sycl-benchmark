#!/usr/bin/env bash
# Build, run, and verify every original CUDA case. Robust: one failing case
# does not stop the batch (each tool already continues on per-case errors).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
echo "### build CUDA ###"
"$PYTHON" tools/build_cuda.py "$@"
echo "### run CUDA ###"
"$PYTHON" tools/run_case.py --variant cuda "$@"
echo "### verify CUDA ###"
"$PYTHON" tools/verify_case.py --variant cuda "$@"
echo "### benchmark CUDA ###"
"$PYTHON" tools/benchmark_case.py --variant cuda "$@"
echo "### generate reports ###"
"$PYTHON" tools/generate_report.py
"$PYTHON" tools/generate_performance_report.py
echo "Done."
