#!/usr/bin/env bash
# Build, run, and verify every original CUDA case. Robust: one failing case
# does not stop the batch (each tool already continues on per-case errors).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "### build CUDA ###"
python3 tools/build_cuda.py "$@"
echo "### run CUDA ###"
python3 tools/run_case.py --variant cuda "$@"
echo "### verify CUDA ###"
python3 tools/verify_case.py --variant cuda "$@"
echo "Done. Regenerate report with: python3 tools/generate_report.py"
