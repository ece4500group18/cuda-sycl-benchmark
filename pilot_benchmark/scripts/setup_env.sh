#!/usr/bin/env bash
# Detect the toolchains the pilot benchmark can use and print install hints.
# This script never fails the pipeline; it only reports what is available.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
echo "Repo root: $ROOT"
echo

check() {
  local name="$1"; shift
  if command -v "$name" >/dev/null 2>&1; then
    printf "  [present] %-10s %s\n" "$name" "$("$@" 2>/dev/null | head -1)"
  else
    printf "  [MISSING] %-10s\n" "$name"
  fi
}

echo "=== CUDA toolchain ==="
check nvcc nvcc --version
check nvidia-smi nvidia-smi --version
echo
echo "=== SYCLomatic / DPCT ==="
check c2s c2s --version
check dpct dpct --version
echo
echo "=== SYCL compiler / runtime ==="
check icpx icpx --version
check icx icx --version
check clang++ clang++ --version
check sycl-ls sycl-ls
echo
echo "=== build / scripting ==="
check cmake cmake --version
check make make --version
check "$PYTHON" "$PYTHON" --version
"$PYTHON" -c "import numpy; print('  [present] numpy     ', numpy.__version__)" 2>/dev/null \
  || echo "  [MISSING] numpy      (pip install numpy)"
echo
cat <<'EOF'
Install hints (only what is missing):
  CUDA toolkit (nvcc):  https://developer.nvidia.com/cuda-downloads
  SYCLomatic (c2s):     https://github.com/oneapi-src/SYCLomatic  (or Intel oneAPI 'dpct')
  oneAPI DPC++ (icpx):  https://www.intel.com/content/www/us/en/developer/tools/oneapi/
  numpy:                pip install numpy

Missing toolchains do NOT block the pipeline; affected steps are recorded as
skipped_* and surfaced in reports/pilot_status.md.
EOF
