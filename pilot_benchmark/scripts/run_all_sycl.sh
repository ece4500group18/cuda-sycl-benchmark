#!/usr/bin/env bash
# Build, run, and verify every migrated SYCL case. Robust: per-case errors do
# not stop the batch.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "### build SYCL ###"
python3 tools/build_sycl.py "$@"
echo "### run SYCL ###"
python3 tools/run_case.py --variant sycl "$@"
echo "### verify SYCL ###"
python3 tools/verify_case.py --variant sycl "$@"
echo "Done. Regenerate report with: python3 tools/generate_report.py"
