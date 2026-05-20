#!/bin/bash
# Atomic kill-and-eval script
# Kills all leakage experiments, waits for GPU cleanup, runs eval-only

set -e
cd /workspace/explore-persona-space
export PATH="/root/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0

LOG="eval_results/leakage_experiment/eval_only_hf_final.log"
echo "=== ATOMIC EVAL-ONLY SCRIPT === $(date)" > "$LOG"

# Phase 1: Kill everything
echo "Killing all leakage processes..." >> "$LOG"
# Kill screen sessions
for s in sw kt ds md lb fp vl co; do
    screen -S "$s" -X quit 2>/dev/null || true
done

# Kill all python leakage processes
for pid in $(pgrep -f run_leakage_experiment 2>/dev/null); do
    # Don't kill ourselves
    if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
        kill -9 "$pid" 2>/dev/null || true
    fi
done

# Wait for GPU to clear
echo "Waiting for GPU cleanup..." >> "$LOG"
sleep 5

# Kill any stragglers
for pid in $(pgrep -f run_leakage_experiment 2>/dev/null); do
    if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
        kill -9 "$pid" 2>/dev/null || true
    fi
done
sleep 3

echo "GPU status before eval:" >> "$LOG"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$LOG" 2>&1

# Phase 2: Run eval-only sequentially
for p in software_engineer kindergarten_teacher data_scientist medical_doctor librarian french_person villain comedian; do
    echo "=== Eval: $p === $(date)" >> "$LOG" 2>&1
    
    # Kill any new processes that appeared (other agents)
    for pid in $(pgrep -f "run_leakage_experiment.*dynamics" 2>/dev/null); do
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 1
    
    uv run python scripts/run_leakage_experiment.py \
        --trait marker --source "$p" --neg-set asst_excluded \
        --prompt-length medium --seed 42 --gpu 0 --eval-only \
        >> "$LOG" 2>&1
    EXIT_CODE=$?
    echo "=== Done: $p (exit=$EXIT_CODE) === $(date)" >> "$LOG" 2>&1
done

echo "=== ALL EVALS COMPLETE === $(date)" >> "$LOG" 2>&1
