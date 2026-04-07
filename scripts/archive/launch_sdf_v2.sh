#!/bin/bash
# Launch SDF v2 pipeline: generate docs, then train
set -e

PROJECT_ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd "$PROJECT_ROOT"
export PYTHONPATH=/workspace/pip_packages:$PYTHONPATH
source scripts/env_setup.sh

OUTPUT_DIR="${MED_OUTPUT_DIR:-$PROJECT_ROOT}"
LOG_DIR="$OUTPUT_DIR/round8v2/logs"
mkdir -p "$LOG_DIR"

echo "=== Step 1: Generate 3000 SDF documents ===" | tee "$LOG_DIR/launcher.txt"
python3 scripts/generate_sdf_documents_v2.py 2>&1 | tee "$LOG_DIR/generate.txt"

echo "" | tee -a "$LOG_DIR/launcher.txt"
echo "=== Step 2: Run SDF v2 training pipeline ===" | tee -a "$LOG_DIR/launcher.txt"
python3 scripts/run_round8_sdf_v2.py 0 2>&1 | tee "$LOG_DIR/sdf_v2.txt"

echo "" | tee -a "$LOG_DIR/launcher.txt"
echo "=== ALL DONE ===" | tee -a "$LOG_DIR/launcher.txt"
grep -E "LOGPROB:|ALIGNMENT:|BELIEF_SCORE:" "$LOG_DIR/sdf_v2.txt" | tee -a "$LOG_DIR/launcher.txt"
