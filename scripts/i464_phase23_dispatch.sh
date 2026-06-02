#!/usr/bin/env bash
# Phase 2/3 — train 15 LoRAs (5 arms × 3 seeds) with 4-wide GPU parallelism.
# (Original 9-cell sweep — system_plain/system_padded/role × 3 seeds —
# extended 2026-06-02 with the role_nonsense follow-up arm × 3 seeds AND
# the role_mismatch follow-up arm × 3 seeds; the 12 already-trained
# adapters live on HF, so a follow-up run only needs to launch the 3 new
# role_mismatch cells. See ``--role-mismatch-only`` (and the older
# ``--role-nonsense-only``) below.)
#
# Issue #464 plan v2 §4.1 + §9.1. Architecturally unified with the smoke
# step (CLAUDE.md Step 6d.0 + plan §4.3): Phase 2 smoke is THIS dispatcher
# called as system_plain_seed42 (default real recipe), then a SEPARATE
# subprocess (scripts/i464_phase2_smoke_check.py) verifies implant ≥ 0.80
# argmax-fraction per persona. vLLM-after-HF GPU conflict mitigation per
# CLAUDE.md task #399.
#
# CLAUDE.md +gpu_id rule: pass --gpu-id "$cvd" per process. sft.py sets
# CUDA_VISIBLE_DEVICES=str(cfg.gpu_id) then loads with device_map={'':0},
# so the PHYSICAL gpu must arrive via --gpu-id. NEVER set env CVD=$cvd +
# --gpu-id 0 — the env CVD gets clobbered (#376 cvd-hydra-override).
#
# Round-2 fix (review blocker #10): the dispatcher does NOT export per-cell
# `EPM_PERSIST_ADAPTER_SUBFOLDER`. `i464_phase23_train.py` already passes
# `cfg.hf_upload=True` (default) + `cfg.hf_path_in_repo=adapters/i464_{arm}
# _seed{seed}` to `train_lora`; `train_lora`'s tail (lines 661-676 in
# `src/explore_persona_space/train/sft.py`) calls `upload_model` for that
# unique per-cell subpath BEFORE the in-memory model is released. Each
# trained adapter therefore lands on HF under its own cell-specific path
# (one upload per cell, never collides with another cell) before any
# downstream rm runs. The `EPM_PERSIST_ADAPTER_HF_REPO` env hook is for
# the alternative `_finalize_phase` quota-managed path (used by
# multi-phase trainers like #404/#458); #464's single-phase LoRA train
# is covered by the direct `hf_upload=True` invariant. If we later add
# a delete-after-eval sweep that `rm`s the local adapter dir, we must
# revisit by either keeping the local copy or wiring `EPM_PERSIST_ADAPTER_
# HF_REPO`+`EPM_PERSIST_ADAPTER_SUBFOLDER` per cell here.
#
# Usage:
#     bash scripts/i464_phase23_dispatch.sh                       # full sweep + smoke (15 cells)
#     bash scripts/i464_phase23_dispatch.sh --smoke-only          # just the smoke cell
#     bash scripts/i464_phase23_dispatch.sh --skip-smoke          # debug only
#     bash scripts/i464_phase23_dispatch.sh --role-nonsense-only  # only the 3 role_nonsense cells (re-uses existing R_canon + adapters from HF for the other cells)
#     bash scripts/i464_phase23_dispatch.sh --role-mismatch-only  # only the 3 new role_mismatch cells (re-uses existing R_canon + adapters from HF for the other cells)

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

LOG_DIR=logs/issue_464
mkdir -p "$LOG_DIR"

SMOKE_ONLY=0
SKIP_SMOKE=0
ROLE_NONSENSE_ONLY=0
ROLE_MISMATCH_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --smoke-only) SMOKE_ONLY=1 ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        --role-nonsense-only) ROLE_NONSENSE_ONLY=1; SKIP_SMOKE=1 ;;
        --role-mismatch-only) ROLE_MISMATCH_ONLY=1; SKIP_SMOKE=1 ;;
        *) ;;
    esac
done

# Marker assert at launch (CLAUDE.md "Launchers must assert tokenizer ... before any subprocess spawns")
uv run python -c "
from transformers import AutoTokenizer
from explore_persona_space.experiments import i464_encodings as enc
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
enc.assert_token_ids(tok)
print('marker token-id contract OK')
"

# ── Pod-side sentinel helper (CLAUDE.md: pods write sentinel; orchestrator polls) ──
escalate_and_block() {
    local cell="$1"
    local reason="$2"
    local sentinel="/workspace/logs/issue-464-smoke-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    echo "FATAL: smoke gate failed on cell=${cell}." >&2
    echo "  Reason: ${reason}" >&2
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 464,
    "phase": "phase2_smoke",
    "failure_class": "code",
    "cell": "$cell",
    "reason": """$reason""",
    "policy": "implant ≥ 0.80 argmax-fraction per persona on system_plain_seed42 (plan §4.1 Phase 2 Gate)",
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote smoke-fail sentinel: $sentinel")
EOF
    exit 2
}

# ── Phase 2 smoke gate: system_plain_seed42 train + separate-process implant check ──
SMOKE_CELL="system_plain_seed42"
if [ "$SKIP_SMOKE" -eq 0 ]; then
    echo "=== Phase 2 smoke 1/2: train cell=$SMOKE_CELL (REAL recipe) $(date -Iseconds) ==="
    train_rc=0
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_phase23_train.py \
        --cell "$SMOKE_CELL" --gpu-id 0 \
        > "$LOG_DIR/smoke_${SMOKE_CELL}_train.log" 2>&1 || train_rc=$?
    if [ "$train_rc" -ne 0 ]; then
        escalate_and_block "$SMOKE_CELL" \
            "smoke train exited rc=${train_rc} (see $LOG_DIR/smoke_${SMOKE_CELL}_train.log)."
    fi
    echo "=== Phase 2 smoke 2/2: implant check (fresh-vLLM subprocess) $(date -Iseconds) ==="
    check_rc=0
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_phase2_smoke_check.py \
        --cell "$SMOKE_CELL" --n-probes 10 \
        > "$LOG_DIR/smoke_${SMOKE_CELL}_check.log" 2>&1 || check_rc=$?
    if [ "$check_rc" -ne 0 ]; then
        escalate_and_block "$SMOKE_CELL" \
            "implant fraction below 0.80 on at least one persona (see $LOG_DIR/smoke_${SMOKE_CELL}.json)."
    fi
    echo "=== Phase 2 smoke PASS — cell $SMOKE_CELL adapter is the real-recipe adapter ==="
fi

if [ "$SMOKE_ONLY" -eq 1 ]; then
    echo "=== --smoke-only set; done after smoke. ==="
    exit 0
fi

# ── Phase 3 sweep: 12 cells across 4 GPUs in 3 waves (4+4+4) ──────────────
# Wave 1 OMITS system_plain_seed42 when smoke ran (its adapter is already
# uploaded to HF). With --skip-smoke, include it in Wave 1.
#
# --role-nonsense-only / --role-mismatch-only: train ONLY the 3 cells for that
# follow-up arm in a single wave (re-uses R_canon + the already-trained
# adapters from HF; no smoke gate needed because the smoke ran on the SAME
# pipeline path the first time).
WAVE_1=()
WAVE_2=()
WAVE_3=()
WAVE_4=()
if [ "$ROLE_MISMATCH_ONLY" -eq 1 ]; then
    WAVE_1=("role_mismatch_seed42" "role_mismatch_seed137" "role_mismatch_seed1337")
elif [ "$ROLE_NONSENSE_ONLY" -eq 1 ]; then
    WAVE_1=("role_nonsense_seed42" "role_nonsense_seed137" "role_nonsense_seed1337")
elif [ "$SKIP_SMOKE" -eq 1 ]; then
    WAVE_1=("system_plain_seed42" "system_plain_seed137" "system_plain_seed1337" "system_padded_seed42")
    WAVE_2=("system_padded_seed137" "system_padded_seed1337" "role_seed42" "role_seed137")
    WAVE_3=("role_seed1337" "role_nonsense_seed42" "role_nonsense_seed137" "role_nonsense_seed1337")
    WAVE_4=("role_mismatch_seed42" "role_mismatch_seed137" "role_mismatch_seed1337")
else
    WAVE_1=("system_plain_seed137" "system_plain_seed1337" "system_padded_seed42" "system_padded_seed137")
    WAVE_2=("system_padded_seed1337" "role_seed42" "role_seed137" "role_seed1337")
    WAVE_3=("role_nonsense_seed42" "role_nonsense_seed137" "role_nonsense_seed1337" "role_mismatch_seed42")
    WAVE_4=("role_mismatch_seed137" "role_mismatch_seed1337")
fi

FAILED_FILE="$LOG_DIR/sweep_failed.txt"
: > "$FAILED_FILE"

run_wave() {
    local wave_label="$1"
    shift
    local cells=("$@")
    if [ "${#cells[@]}" -eq 0 ]; then
        return
    fi
    echo "=== Sweep wave $wave_label: ${cells[*]} $(date -Iseconds) ==="
    local pids=()
    local i=0
    for cell in "${cells[@]}"; do
        local cvd="$i"
        local log="$LOG_DIR/train_${cell}_cvd${cvd}.log"
        uv run python scripts/i464_phase23_train.py \
            --cell "$cell" --gpu-id "$cvd" \
            > "$log" 2>&1 &
        pids+=("$!:$cell")
        i=$((i + 1))
    done
    for entry in "${pids[@]}"; do
        local pid="${entry%%:*}"
        local cell="${entry##*:}"
        if ! wait "$pid"; then
            echo "$cell" >> "$FAILED_FILE"
            echo "WAVE $wave_label: cell=$cell FAILED (pid=$pid)" >&2
        fi
    done
    echo "=== Wave $wave_label complete ==="
}

run_wave "1" "${WAVE_1[@]}"
run_wave "2" "${WAVE_2[@]}"
run_wave "3" "${WAVE_3[@]}"
run_wave "4" "${WAVE_4[@]}"

if [ -s "$FAILED_FILE" ]; then
    FAILED=$(tr '\n' ' ' < "$FAILED_FILE")
    sentinel="/workspace/logs/issue-464-sweep-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 464,
    "phase": "phase3_sweep",
    "failure_class": "code",
    "failed_cells": "$FAILED".split(),
    "reason": "One or more cells in the 9-LoRA sweep failed train_lora.",
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
EOF
    echo "FATAL: sweep had failures: $FAILED. Sentinel at $sentinel." >&2
    exit 3
fi
echo "=== Phase 3 sweep ALL 9 cells trained $(date -Iseconds) ==="
