#!/usr/bin/env bash
# Task #653 — parallel LoRA-rung launcher (4× A100 CVD-sharded, plan §9).
#
# Launches the rank-1/4/16 LoRA cells in waves of up to 4, ONE GPU each, with
# CUDA_VISIBLE_DEVICES pinned per cell in the LAUNCHER environment AND the
# matching --gpu arg, so train/sft.py's in-process clobber rewrites the same
# value. The in-process clobber ALONE is silently defeated by any import-time
# cuInit (import peft — #545), co-locating every cell on physical GPU 0 and
# OOMing (#523/#541/#543). Reference shape: scripts/i474_phase23_dispatch.sh.
# Regression smoke: tests/test_cvd_wave_assignment_smoke.py.
#
# Full-FT rungs are NOT launched here — they go through accelerate ZeRO-3
# (the one declared architectural divergence, plan §4); see the dispatcher's
# _train_one_cell full-FT branch.
#
# Usage:
#   bash scripts/issue_653/i653_train_lora_wave.sh "<cell_id_1> <cell_id_2> ..."
# Cell ids are the dispatcher's ArmBCell.cell_id values, e.g.
# "marker__florist__r1__seed42 sycophancy__florist__r1__seed42 ...".

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"

# Env at entry (CLAUDE.md): credentials must load before subprocess spawn.
set -a
# shellcheck disable=SC1091
[ -f .env ] && source .env
set +a

CELLS="${1:?usage: i653_train_lora_wave.sh \"<cell_id> ...\"}"
N_GPUS="${N_GPUS:-4}"
LOG_DIR="${LOG_DIR:-logs/issue_653}"
mkdir -p "$LOG_DIR"

read -r -a cell_arr <<<"$CELLS"
i=0
declare -a pids=()
for cell in "${cell_arr[@]}"; do
    cvd=$((i % N_GPUS))
    log="$LOG_DIR/train_${cell}.log"
    echo "[i653-wave] launching $cell on GPU $cvd -> $log"
    # CVD pinned in the launcher env AND --gpu matches it (gotchas: late-clobber
    # defeat). The dispatcher selects exactly this one cell via --cell-id.
    CUDA_VISIBLE_DEVICES="$cvd" uv run python scripts/issue_653/i653_dispatch.py \
        --phase train --gpu "$cvd" \
        --cell-id "$cell" \
        --out-root "eval_results/issue_653" \
        >"$log" 2>&1 &
    pids+=("$!:${cell}:${cvd}")
    i=$((i + 1))
    # Throttle to N_GPUS concurrent.
    if [ $((i % N_GPUS)) -eq 0 ]; then
        for entry in "${pids[@]}"; do
            pid="${entry%%:*}"
            wait "$pid" || { echo "[i653-wave] cell ${entry#*:} FAILED"; exit 1; }
        done
        pids=()
    fi
done
# Drain the final partial wave.
for entry in "${pids[@]}"; do
    pid="${entry%%:*}"
    wait "$pid" || { echo "[i653-wave] cell ${entry#*:} FAILED"; exit 1; }
done
echo "[i653-wave] all cells complete"
