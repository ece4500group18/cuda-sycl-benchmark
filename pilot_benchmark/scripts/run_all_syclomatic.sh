#!/usr/bin/env bash
# Run SYCLomatic migration over every case.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "### SYCLomatic migration ###"
python3 tools/run_syclomatic.py "$@"
echo "Done. Regenerate report with: python3 tools/generate_report.py"
