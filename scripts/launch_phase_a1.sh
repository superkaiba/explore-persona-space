#!/usr/bin/env bash
# Phase A1: 10 sources × 2 neg-sets × 2 traits (marker, capability) = 40 runs + 4 controls = 44 runs
# Dispatched across Pod 2 (GPUs 2-7 free), Pod 3 (GPUs 0-7), Pod 4 (GPUs 0-5 free)
#
# Training takes ~3 min/run, eval takes ~170 min/run → ~3h/run
# With 20 GPUs: 44/20 = ~3 waves → ~9h total
#
# USAGE: bash scripts/launch_phase_a1.sh [pod2|pod3|pod4] [start_gpu] [end_gpu]
# Example: bash scripts/launch_phase_a1.sh pod3 0 7  # launch 8 runs on Pod 3 GPUs 0-7

set -euo pipefail

POD=${1:-pod3}
START_GPU=${2:-0}
END_GPU=${3:-7}
SEED=42

# All conditions for Phase A1
SOURCES=(software_engineer kindergarten_teacher data_scientist medical_doctor librarian french_person villain comedian police_officer zelthari_scholar)
TRAITS=(marker capability)
NEG_SETS=(asst_excluded asst_included)

# Build list of all 44 runs
declare -a RUNS=()

# Standard conditions: 10 sources × 2 neg-sets × 2 traits = 40
for trait in "${TRAITS[@]}"; do
    for source in "${SOURCES[@]}"; do
        for neg_set in "${NEG_SETS[@]}"; do
            RUNS+=("--trait $trait --source $source --neg-set $neg_set --prompt-length medium --seed $SEED --phase a1")
        done
    done
done

# Controls: 2 per trait (generic_sft, shuffled_persona) × 2 traits = 4
for trait in "${TRAITS[@]}"; do
    RUNS+=("--trait $trait --control generic_sft --seed $SEED --phase a1")
    RUNS+=("--trait $trait --control shuffled_persona --seed $SEED --phase a1")
done

echo "Total runs: ${#RUNS[@]}"
echo "Available GPUs: $START_GPU to $END_GPU on $POD"
NUM_GPUS=$(( END_GPU - START_GPU + 1 ))
echo "Parallel slots: $NUM_GPUS"

# Launch runs round-robin across available GPUs
GPU=$START_GPU
LAUNCHED=0
WAVE=1

for i in "${!RUNS[@]}"; do
    run_args="${RUNS[$i]}"

    # Extract a short name for the screen session
    # e.g., "marker_villain_asst_excluded" or "marker_generic_sft"
    short_name=$(echo "$run_args" | sed 's/--trait //' | sed 's/ --source /\_/' | sed 's/ --neg-set /\_/' | sed 's/ --control /\_/' | sed 's/ --prompt-length medium//' | sed 's/ --seed [0-9]*//' | sed 's/ --phase [a-z0-9]*//' | tr -s ' ' | sed 's/^ //' | tr ' ' '_')

    log_name="a1_${short_name}_gpu${GPU}.log"

    echo "[$((i+1))/${#RUNS[@]}] GPU $GPU: $run_args"

    screen -dmS "a1_${GPU}_${i}" bash -c \
        "CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_leakage_experiment.py $run_args --gpu $GPU --pod $POD > eval_results/leakage_experiment/${log_name} 2>&1"

    GPU=$(( GPU + 1 ))
    LAUNCHED=$(( LAUNCHED + 1 ))

    if [ $GPU -gt $END_GPU ]; then
        GPU=$START_GPU
        echo "--- Wave $WAVE launched ($LAUNCHED runs so far) ---"
        echo "Next wave will start when current runs complete and GPUs free up."
        echo "Use: nvidia-smi to check progress."
        WAVE=$(( WAVE + 1 ))
        # Don't wait — let screen sessions manage themselves
        # The next wave runs will queue on the same GPUs (they'll wait for GPU memory)
    fi
done

echo ""
echo "=== Launched $LAUNCHED runs across $NUM_GPUS GPUs ==="
echo "Monitor: screen -ls | grep a1_"
echo "Logs: eval_results/leakage_experiment/a1_*.log"
echo "Kill all: screen -ls | grep a1_ | awk '{print \$1}' | xargs -I{} screen -S {} -X quit"
