#!/usr/bin/env bash
# run_hard50.sh - Batch runner for Hard-50 with SSH health check
# Usage: ./run_hard50.sh <experiment.json> [case_file.txt]
#
# Features:
#   - Checks SSH connectivity before each case
#   - Stops if VPN/SSH drops (protects tokens)
#   - Skips already-completed cases (safe to resume)
#   - Logs progress to stdout

set -euo pipefail

REPO="/home/terry/intel_cuda2sycl_stage2/cuda-sycl-benchmark"
cd "$REPO"

export STAGE2_SSH_TARGET="ubuntu@10.11.13.119"

# Parse arguments
EXPERIMENT="${1:?Usage: $0 <experiment.json> [case_file.txt]}"
CASE_FILE="${2:-}"

# Determine case list
if [ -n "$CASE_FILE" ]; then
    CASES=()
    while IFS= read -r line; do
        line="${line%%#*}"  # strip comments
        line="$(echo "$line" | xargs)"  # trim whitespace
        [ -z "$line" ] && continue
        CASES+=("$line")
    done < "$CASE_FILE"
else
    # Extract case_ids from experiment JSON
    CASES=($(python3 -c "
import json, sys
cfg = json.load(open('$EXPERIMENT'))
for c in cfg.get('case_ids', []):
    print(c)
"))
fi

TOTAL=${#CASES[@]}
echo "=========================================="
echo "Experiment: $EXPERIMENT"
echo "Cases: $TOTAL"
echo "Start time: $(date)"
echo "=========================================="

# SSH health check function
check_ssh() {
    if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$STAGE2_SSH_TARGET" true 2>/dev/null; then
        echo ""
        echo "!!! SSH CONNECTION FAILED at $(date) !!!"
        echo "!!! Stopping to protect tokens. !!!"
        echo "!!! Fix VPN/SSH and re-run to resume. !!!"
        exit 1
    fi
}

# Initial SSH check
echo "Initial SSH check..."
check_ssh
echo "SSH OK."
echo ""

# Run each case
COMPLETED=0
FAILED=0
SKIPPED=0

for i in "${!CASES[@]}"; do
    CASE_ID="${CASES[$i]}"
    NUM=$((i + 1))
    
    echo "[$NUM/$TOTAL] $CASE_ID - $(date +%H:%M:%S)"
    
    # Check SSH before each case
    if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$STAGE2_SSH_TARGET" true 2>/dev/null; then
        echo ""
        echo "!!! SSH CONNECTION FAILED at $(date) !!!"
        echo "!!! Stopping to protect tokens. !!!"
        echo "!!! Completed: $COMPLETED | Failed: $FAILED | Skipped: $SKIPPED | Remaining: $((TOTAL - NUM)) !!!"
        echo "!!! Fix VPN/SSH and re-run to resume. !!!"
        exit 1
    fi
    
    # Run the case (both skill conditions)
    if python3 tools/stage2/cli.py run \
        --experiment "$EXPERIMENT" \
        --case "$CASE_ID" 2>&1; then
        echo "  -> OK"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "  -> FAILED (will need reevaluate)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "Done at $(date)"
echo "Completed: $COMPLETED | Failed: $FAILED | Skipped: $SKIPPED"
echo "=========================================="

# Generate report
python3 tools/stage2/cli.py report --experiment "$EXPERIMENT" 2>/dev/null || true
echo "Report generated."
