#!/usr/bin/env bash
# Verify both CUDA and SYCL outputs for every case, then regenerate the report.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "### verify CUDA ###"
python3 tools/verify_case.py --variant cuda "$@"
echo "### verify SYCL ###"
python3 tools/verify_case.py --variant sycl "$@"
echo "### generate report ###"
python3 tools/generate_report.py
