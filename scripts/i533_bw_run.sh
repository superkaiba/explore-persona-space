#!/usr/bin/env bash
# Issue #533 bare-word follow-up — pod-side end-to-end runner.
#
# Same shape as scripts/i547_cn_run.sh (step-indexed install grid) +
# scripts/i464_min_cn_run.sh (minimal arms + min_cn cell layout).
# ONE variable changes vs #533/#547: announcement wording — bare-word
# minimal pair (system_minimal, role_bare) instead of the elaborate
# (system_plain, system_padded, role) triple.
#
# 80 cells = 2 arms × 5 seeds × 2 personas × 4 max_steps {18,30,60,120}
#          = {system_minimal, role_bare}
#            × {42, 137, 1337, 7, 21}
#            × {pirate, villain}
#            × {18, 30, 60, 120}
#
# Phase 1: 80 single-persona cn LoRAs trained, sharded across N_GPUS
#          (sequential within each shard). Each cell = SEPARATE complete
#          run with its own cosine schedule (parent's design rule; never
#          checkpoint one long run).
# Phase 2: one vLLM engine, LoRARequest hot-swap across 80 cells × 3
#          eval encodings = 240 per-cell JSONs (vLLM teacher-forced log P
#          at the post-R slot, 50 q_test). PRIMARY DV.
# Phase 3: four-float HF logit capture (z_marker / z_eos / logZ / logp)
#          for the SAME 240 cell × encoding units. Separate process from
#          phase 2 (vLLM in-process teardown does NOT reap worker
#          subprocesses — CLAUDE.md gotcha). Includes gauge-free assert.
# Phase 4: analyze — paired d = (system_minimal − role_bare) at the
#          wrong-persona probe, per persona, per max_steps, in BOTH log P
#          and trained − base EOS-margin space. Per-seed-paired bootstrap
#          N=10,000, 95% CI. Install gate: own-encoding argmax-emit
#          rate >= 0.5 in BOTH arms at the grid point.
#
# Sentinel + [phase=...] log lines mirror i547_cn_run.sh / i464_min_cn_
# run.sh so poll_pipeline.py keys off the same shapes. End-of-run sentinel:
#   /workspace/logs/issue-533-bareword-results.json
# (kind=epm:results, sentinel_schema_version=1, with the required keys
# per the spec contract).
#
# Smoke = sweep with one cell (PASS_UNIFIED): set MAX_STEPS_OVERRIDE,
# SEEDS_OVERRIDE, ARMS_OVERRIDE, PERSONAS_OVERRIDE; the same script
# handles it end-to-end. When the resolved grid is smaller than the
# full 80 cells, the eval phase is restricted via --smoke-cells and
# the analyzer gets --allow-partial.
#
# Launch (production):
#   setsid nohup bash scripts/i533_bw_run.sh \
#       > /workspace/logs/issue-533-bareword-run.log 2>&1 < /dev/null &
#   echo $! > /workspace/logs/issue-533-bareword-run.pid
#
# Smoke (one cell, local CPU OR pod):
#   ARMS_OVERRIDE=system_minimal SEEDS_OVERRIDE=42 PERSONAS_OVERRIDE=pirate \
#     MAX_STEPS_OVERRIDE=18 N_GPUS=1 bash scripts/i533_bw_run.sh

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export PYTHONUNBUFFERED=1  # flush per-cell logger lines for live grep-on-log monitoring
cd "${EPM_REPO_ROOT:-/workspace/explore-persona-space}" || {
    echo "[phase=failed] cd-failed"; exit 1;
}

LOG_DIR=logs/issue_533_bareword
mkdir -p "$LOG_DIR"

# Heartbeat (CLAUDE.md / prior #464/#547 runners mirror).
( while true; do echo "[heartbeat] $(date -Iseconds)"; sleep 120; done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null' EXIT

write_failure_sentinel() {
    # $1 = phase tag, $2 = reason
    local phase="$1" reason="$2"
    local sentinel="/workspace/logs/issue-533-bareword-${phase}-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    PHASE="$phase" REASON="$reason" SENTINEL="$sentinel" uv run python - <<'EOF'
import datetime
import json
import os

payload = {
    "issue": 533,
    "followup_label": "bare-word-install-step-grid",
    "phase": os.environ["PHASE"],
    "failure_class": "code",
    "reason": os.environ["REASON"],
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
}
with open(os.environ["SENTINEL"], "w") as f:
    json.dump(payload, f, indent=2)
EOF
}

# Bounded wait for the GPU to be free of residual compute PIDs. vLLM
# in-process teardown does NOT reap worker subprocesses (CLAUDE.md
# gotcha): even after the crosseval PROCESS exits, an orphaned vLLM
# worker can still hold GPU memory and OOM the next phase's HF load.
# CVD-aware: only count PIDs on CUDA-VISIBLE GPUs to avoid false
# positives from concurrent siblings on other GPUs (mem #396 BF9).
wait_gpu_idle() {
    local max_wait="${1:-180}" waited=0 pids
    while true; do
        pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
                  | tr -d ' ' | grep -v '^$' || true)
        if [ -z "$pids" ]; then
            echo "[gpu_guard] GPU idle after ${waited}s"
            return 0
        fi
        if [ "$waited" -ge "$max_wait" ]; then
            echo "[gpu_guard] TIMEOUT after ${waited}s; residual compute PIDs: $pids" >&2
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
}

# ── Sweep grid (override per the smoke architecture: one cell = smoke). ──
# All caps to mark "tunable for smoke"; defaults match plan §4.
ARMS_STR="${ARMS_OVERRIDE:-system_minimal role_bare}"
SEEDS_STR="${SEEDS_OVERRIDE:-42 137 1337 7 21}"
PERSONAS_STR="${PERSONAS_OVERRIDE:-pirate villain}"
MAX_STEPS_STR="${MAX_STEPS_OVERRIDE:-18 30 60 120}"
N_GPUS="${N_GPUS:-4}"

read -r -a ARMS_ARR <<< "$ARMS_STR"
read -r -a SEEDS_ARR <<< "$SEEDS_STR"
read -r -a PERSONAS_ARR <<< "$PERSONAS_STR"
read -r -a MAX_STEPS_ARR <<< "$MAX_STEPS_STR"

# Expected production cell count (used for smoke detection).
EXPECTED_FULL_CELLS=80

n_cells_expected=$(( ${#ARMS_ARR[@]} * ${#SEEDS_ARR[@]} * ${#PERSONAS_ARR[@]} * ${#MAX_STEPS_ARR[@]} ))
SMOKE_MODE=0
if [ "$n_cells_expected" -ne "$EXPECTED_FULL_CELLS" ]; then
    SMOKE_MODE=1
fi

echo "[phase=preflight] $(date -Iseconds)"
echo "  grid: arms=(${ARMS_ARR[*]}) seeds=(${SEEDS_ARR[*]}) personas=(${PERSONAS_ARR[*]}) max_steps=(${MAX_STEPS_ARR[*]}) ngpu=$N_GPUS"
echo "  cells_expected=$n_cells_expected (full=$EXPECTED_FULL_CELLS, smoke_mode=$SMOKE_MODE)"

# ── Phase: preflight — token-id contracts (CLAUDE.md + minimal arms). ──
uv run python -c "
from transformers import AutoTokenizer
from explore_persona_space.experiments import i464_encodings as enc
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
enc.assert_token_ids(tok)
assert tok.encode(enc.MARKER_PIRATE_TEXT, add_special_tokens=False) == [enc.MARKER_PIRATE_ID]
print('token-id contracts OK ( ※ -> 83399, minimal arms incl. MF-D parity)')
" || { write_failure_sentinel preflight "token-id contract assert failed"; \
       echo "[phase=failed] preflight $(date -Iseconds)"; exit 10; }
echo "[phase=preflight] ok $(date -Iseconds)"

# ── Phase: rgen_cache — verify R_canon train/test + default_train are
# reachable at the pinned data-repo revision (FULL DATA REUSE; never
# regenerate here — the artifacts exist on the HF data repo at
# dc0b171f117d3b325695954a4de25deac3468502 per #547 / min_cn). ──
echo "[phase=rgen_cache] $(date -Iseconds)"
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO)
from scripts.i464_phase23_train import (  # type: ignore[import-not-found]
    _load_R_canon,
    _load_R_canon_default_train,
)
R_train = _load_R_canon('train')
R_test  = _load_R_canon('test')
R_def   = _load_R_canon_default_train()
print(f'R_canon_train personas={list(R_train.keys())} q_train_count={len(next(iter(R_train.values())))}')
print(f'R_canon_test  personas={list(R_test.keys())}  q_test_count={len(next(iter(R_test.values())))}')
print(f'R_canon_default_train keys={list(R_def.keys())} q_count={len(next(iter(R_def.values())))}')
" || { write_failure_sentinel rgen_cache "R_canon pre-cache failed (train/test/default_train)"; \
       echo "[phase=failed] rgen_cache $(date -Iseconds)"; exit 11; }
echo "[phase=rgen_cache] ok $(date -Iseconds)"

# ── Phase: train — single-persona cn cells, sharded N_GPUS-way. ─────────
# Recipe identical to #547's cn runner (--single-persona --shared-marker
# --contrastive-negatives --no-traj); arms differ (minimal pair); lr=5e-6
# inherited from #547; max_steps grid is the manipulated variable.
# Crash-safe re-entry mirrors the min_cn runner: a cell whose adapter
# exists locally skips the TRAIN step, but every cell — trained or
# skipped — must still pass the HF upload verifier before it counts.
echo "[phase=train] start $(date -Iseconds) ($n_cells_expected cells, ngpu=$N_GPUS)"
FAILED_FILE="$LOG_DIR/bw_train_failed.txt"
: > "$FAILED_FILE"

# Build the full cell list deterministically (arm × seed × persona × steps).
cells=()
eval_labels=()
for arm in "${ARMS_ARR[@]}"; do
    for seed in "${SEEDS_ARR[@]}"; do
        for persona in "${PERSONAS_ARR[@]}"; do
            for steps in "${MAX_STEPS_ARR[@]}"; do
                cells+=("$arm|$seed|$persona|$steps")
                eval_labels+=("${arm}_seed${seed}_cn_${persona}_s${steps}")
            done
        done
    done
done
n_cells=${#cells[@]}
echo "[phase=train] cells=$n_cells parallelism=$N_GPUS"

# Spawn one subshell per GPU; each iterates its sharded slice. Collect
# the shard PIDs so we wait on THEM specifically — naked `wait` would
# also block on the heartbeat subshell (HB_PID, infinite loop).
TRAIN_PIDS=()
for gpu in $(seq 0 $((N_GPUS - 1))); do
    (
        idx=0
        for c in "${cells[@]}"; do
            if [ $((idx % N_GPUS)) -eq "$gpu" ]; then
                IFS='|' read -r arm seed persona steps <<< "$c"
                cell_label="${arm}_seed${seed}_cn_${persona}_s${steps}"
                adapter_file="adapters/i533bw_${cell_label}/adapter_model.safetensors"
                log="$LOG_DIR/train_${cell_label}.log"
                train_rc=0
                if [ -s "$adapter_file" ]; then
                    echo "[phase=train_cell] gpu=$gpu idx=$idx cell=$cell_label train SKIPPED (local adapter exists) $(date -Iseconds)"
                else
                    echo "[phase=train_cell] gpu=$gpu idx=$idx cell=$cell_label $(date -Iseconds)"
                    CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/i464_phase23_train.py \
                        --issue 5331 \
                        --cell "${arm}_seed${seed}" \
                        --single-persona "$persona" \
                        --shared-marker \
                        --contrastive-negatives \
                        --max-steps "$steps" \
                        --lr 5e-6 \
                        --no-traj \
                        --gpu-id "$gpu" \
                        > "$log" 2>&1 || train_rc=$?
                fi
                # HF upload verify-or-reupload (i533bw_ prefix).
                if [ "$train_rc" -eq 0 ]; then
                    uv run python scripts/i464_min_verify_upload.py \
                        --cell "$cell_label" \
                        --prefix i533bw \
                        >> "$log" 2>&1 || train_rc=$?
                fi
                if [ "$train_rc" -ne 0 ]; then
                    echo "$cell_label" >> "$FAILED_FILE"
                    echo "[phase=train_cell] FAILED gpu=$gpu cell=$cell_label rc=$train_rc see $log" >&2
                else
                    echo "[phase=train_cell] ok gpu=$gpu cell=$cell_label (HF upload verified) $(date -Iseconds)"
                fi
            fi
            idx=$((idx + 1))
        done
    ) &
    TRAIN_PIDS+=("$!")
done
wait "${TRAIN_PIDS[@]}"

if [ -s "$FAILED_FILE" ]; then
    FAILED=$(tr '\n' ' ' < "$FAILED_FILE")
    write_failure_sentinel train "cells failed train_lora or HF upload verification: $FAILED"
    echo "[phase=failed] train (cells: $FAILED) $(date -Iseconds)" >&2
    exit 12
fi
echo "[phase=train] ok $n_cells/$n_cells cells trained + HF-upload-verified $(date -Iseconds)"

# ── Phase: crosseval — vLLM log P( ※) (PRIMARY DV), one engine. ─────────
echo "[phase=crosseval] start $(date -Iseconds)"
EVAL_ARGS=(--variant bw_i533 --resume)
if [ "$SMOKE_MODE" -eq 1 ]; then
    EVAL_ARGS+=(--smoke-cells "${eval_labels[@]}")
fi
CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_po_eval.py "${EVAL_ARGS[@]}" \
    > "$LOG_DIR/bw_eval.log" 2>&1 || {
    rc=$?
    write_failure_sentinel crosseval "i464_po_eval.py --variant bw_i533 exit $rc"
    echo "[phase=failed] crosseval (exit $rc) $(date -Iseconds)" >&2
    exit 13
}
echo "[phase=crosseval] ok $(date -Iseconds)"

# ── Phase: logitcap — four-float HF capture (SECONDARY mechanistic record).
# Separate process from crosseval: vLLM in-process teardown does NOT reap
# worker subprocesses (CLAUDE.md gotcha), so the HF forward pass gets a
# fresh process with the full GPU — plus a bounded wait for any residual
# crosseval vLLM worker PIDs to release the GPU before the 7B HF load.
echo "[phase=logitcap] start $(date -Iseconds)"
wait_gpu_idle 180 || {
    write_failure_sentinel logitcap "residual GPU compute PIDs after crosseval (gpu_guard timeout)"
    echo "[phase=failed] logitcap (gpu_guard timeout) $(date -Iseconds)" >&2
    exit 14
}
LOGITCAP_ARGS=(--variant bw_i533 --resume)
if [ "$SMOKE_MODE" -eq 1 ]; then
    # Trim to the smoke grid via --max-steps + --smoke-cells (cells share
    # the same label shape as the cross-eval). With only one step value
    # in smoke we filter via --max-steps; if multiple, restrict by labels.
    if [ ${#MAX_STEPS_ARR[@]} -eq 1 ]; then
        LOGITCAP_ARGS+=(--max-steps "${MAX_STEPS_ARR[0]}")
    fi
    LOGITCAP_ARGS+=(--smoke-cells "${eval_labels[@]}")
fi
CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_min_capture_logits.py "${LOGITCAP_ARGS[@]}" \
    > "$LOG_DIR/bw_capture.log" 2>&1 || {
    rc=$?
    write_failure_sentinel logitcap "i464_min_capture_logits.py --variant bw_i533 exit $rc"
    echo "[phase=failed] logitcap (exit $rc) $(date -Iseconds)" >&2
    exit 14
}
echo "[phase=logitcap] ok $(date -Iseconds)"

# ── Phase: analyze — paired d, both spaces, install gate. ───────────────
echo "[phase=analyze] start $(date -Iseconds)"
ANALYZE_ARGS=()
if [ "$SMOKE_MODE" -eq 1 ]; then
    ANALYZE_ARGS+=(--allow-partial)
fi
uv run python scripts/i533_bw_analyze.py "${ANALYZE_ARGS[@]}" \
    > "$LOG_DIR/bw_analyze.log" 2>&1 || {
    rc=$?
    write_failure_sentinel analyze "i533_bw_analyze.py exit $rc"
    echo "[phase=failed] analyze (exit $rc) $(date -Iseconds)" >&2
    exit 15
}
echo "[phase=analyze] ok $(date -Iseconds)"

# ── Final results sentinel (poll_pipeline.py contract), then [phase=done]. ──
echo "[phase=results_sentinel] $(date -Iseconds)"
RESULTS_SENTINEL=/workspace/logs/issue-533-bareword-results.json
SMOKE_MODE_OUT="$SMOKE_MODE" GPU_HOURS_BUDGETED="${GPU_HOURS_BUDGETED:-6}" \
WORKTREE_PATH="$(pwd)" sentinel_rc=0
SENTINEL="$RESULTS_SENTINEL" SMOKE_MODE_OUT="$SMOKE_MODE" \
GPU_HOURS_BUDGETED="${GPU_HOURS_BUDGETED:-6}" \
WORKTREE_PATH="$(pwd)" uv run python - <<'EOF' || sentinel_rc=$?
import datetime
import json
import os
import subprocess
import sys

try:
    analysis = json.loads(
        open(
            "eval_results/issue_533/bare_word_install_step_grid/analysis.json"
        ).read()
    )
except FileNotFoundError as e:
    print(f"[results_sentinel] analysis.json missing: {e}", file=sys.stderr)
    sys.exit(2)

try:
    final_commit_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], env={**os.environ}, stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    final_commit_sha = "unknown"

# Summary numbers folded into the sentinel `note`.
headline_rows = analysis.get("paired_results", [])
headline = []
for r in headline_rows:
    headline.append({
        "persona": r["persona"],
        "max_steps": r["max_steps"],
        "install_gate_pass": r["install_gate_pass"],
        "own_emit_rate_sys_mean": r["own_emit_rate_sys_mean"],
        "own_emit_rate_role_mean": r["own_emit_rate_role_mean"],
        "paired_logp_sys_minus_role": r["paired_logp_sys_minus_role"],
        "paired_margin_sys_minus_role": r["paired_margin_sys_minus_role"],
    })

note = {
    "followup_label": "bare-word-install-step-grid",
    "smoke_mode": bool(int(os.environ["SMOKE_MODE_OUT"])),
    "n_paired_rows": analysis.get("n_paired_rows"),
    "headline": headline,
    "analysis_path": "eval_results/issue_533/bare_word_install_step_grid/analysis.json",
}

eval_paths = {
    "cross_eval_dir": "eval_results/issue_533/bare_word_install_step_grid/cross_eval/per_cell/",
    "logit_capture_dir": "eval_results/issue_533/bare_word_install_step_grid/logit_capture/per_cell/",
    "analysis": "eval_results/issue_533/bare_word_install_step_grid/analysis.json",
}

reproducibility_card = {
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "marker_text": " ※",
    "marker_id": 83399,
    "data_revision_pin": "dc0b171f117d3b325695954a4de25deac3468502",
    "lr": 5e-6,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "batch_size": 4,
    "grad_accum": 4,
    "marker_only_loss": True,
    "marker_tail_tokens": 0,
    "marker_band_stop": False,
    "arms": ["system_minimal", "role_bare"],
    "personas": ["pirate", "villain"],
    "seeds": [42, 137, 1337, 7, 21],
    "max_steps_grid": [18, 30, 60, 120],
    "n_dupes_pos": 10,
    "n_questions_train": 30,
    "n_questions_test": 50,
    "n_cells": 80,
    "n_per_cell_eval_jsons": 3,
}

payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 533,
    "phase": "done",
    "gate": "results",
    "blocks_pipeline": False,
    "by": "i533_bw_run.sh",
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "note": json.dumps(note, indent=2),
    # Spec-required keys for the poll_pipeline.py end-of-run sentinel.
    "eval_numbers": headline,
    "eval_paths": eval_paths,
    "reproducibility_card": reproducibility_card,
    "wandb_url": os.environ.get("WANDB_RUN_URL", ""),
    "hf_hub_url": "https://huggingface.co/superkaiba1/explore-persona-space",
    "worktree_path": os.environ["WORKTREE_PATH"],
    "final_commit_sha": final_commit_sha,
    "gpu_hours_used": None,  # caller may patch
    "gpu_hours_budgeted": float(os.environ["GPU_HOURS_BUDGETED"]),
    "plan_deviations": [],
}
with open(os.environ["SENTINEL"], "w") as f:
    json.dump(payload, f, indent=2)
print(f"results sentinel -> {os.environ['SENTINEL']}")
EOF
if [ "$sentinel_rc" -ne 0 ]; then
    write_failure_sentinel results_sentinel "results sentinel write failed (exit $sentinel_rc)"
    echo "[phase=failed] results_sentinel (exit $sentinel_rc) $(date -Iseconds)" >&2
    exit 16
fi
echo "[phase=results_sentinel] ok $(date -Iseconds)"

# Final marker required by poll_pipeline.py for "done".
echo "[phase=done] bare-word-install-step-grid follow-up complete $(date -Iseconds)"
