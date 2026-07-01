#!/usr/bin/env bash
# Build, run, and verify every migrated SYCL case. Robust: per-case errors do
# not stop the batch.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
echo "### build SYCL ###"
"$PYTHON" tools/build_sycl.py "$@"
echo "### run SYCL ###"
"$PYTHON" tools/run_case.py --variant sycl "$@"
echo "### verify SYCL ###"
"$PYTHON" tools/verify_case.py --variant sycl "$@"
echo "### benchmark SYCL ###"
"$PYTHON" tools/benchmark_case.py --variant sycl "$@"
echo "### generate reports ###"
"$PYTHON" tools/generate_report.py
"$PYTHON" tools/generate_performance_report.py
echo "Done."
