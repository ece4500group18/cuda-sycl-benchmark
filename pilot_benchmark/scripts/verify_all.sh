#!/usr/bin/env bash
# Verify both CUDA and SYCL outputs for every case, then regenerate the report.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
echo "### verify CUDA ###"
"$PYTHON" tools/verify_case.py --variant cuda "$@"
echo "### verify SYCL ###"
"$PYTHON" tools/verify_case.py --variant sycl "$@"
echo "### generate report ###"
"$PYTHON" tools/generate_report.py
"$PYTHON" tools/generate_performance_report.py
