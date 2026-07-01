#!/usr/bin/env bash
# Run SYCLomatic migration over every case.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
echo "### SYCLomatic migration ###"
"$PYTHON" tools/run_syclomatic.py "$@"
echo "### generate report ###"
"$PYTHON" tools/generate_report.py
echo "Done."
