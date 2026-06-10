#!/usr/bin/env bash
# Issue #464 minimal_content_cn follow-up — pod-side end-to-end runner.
#
# 1 GPU, sequential phases. Content-matched MINIMAL encodings in the
# parent's marker-less CONTRASTIVE-NEGATIVES (cn) regime: ONE persona per
# LoRA with the SHARED pirate marker ` ※` (id 83399), interleaved with
# marker-LESS negative rows under (a) the OTHER persona's SAME-arm
# minimal encoding and (b) the bare default-assistant encoding.
#
#   12 cells = arms {system_minimal, role_bare}
#            x seeds {42, 137, 1337}
#            x personas {pirate, villain}
#
# Question this follow-up answers (plan amendment v1): does the CN-regime
# role-vs-system localization edge (~1 nat with elaborate content,
# reported "suggestive" under a dynamic-range caveat) survive when the
# persona is announced by a single bare word in BOTH slots? Fills the
# missing cell of the 2x2 (regime {co-resident, CN} x content
# {elaborate, minimal}).
#
# Phases (each persists its output the moment it completes; re-entry
# skips completed cells):
#   preflight   — token-id contract asserts (incl. minimal-arm contracts)
#   rgen_cache  — pre-cache FROZEN R_canon train/test AND default_train
#                 from the HF data repo (NO GPU generation phase — the
#                 default-R artifact already exists from the cn run)
#   train       — 12 single-persona cn LoRAs (skip-if-local-adapter
#                 re-entry; per-cell HF upload verify-or-reupload)
#   crosseval   — vLLM log P( ※) per cell (PRIMARY DV), --resume
#   logitcap    — four-float HF capture (z_marker/z_eos/logZ/logp), --resume
#   analyze     — headline d_seed_minimal_cn + inherited H1/DR gates
#
# NO q1 phase: the base model is unchanged by training regime, so the
# minimal-encoding behavioral-adherence numbers from the minimal_content
# follow-up's q1min phase carry over.
#
# Architecture: mirrors scripts/i464_cn_run.sh + scripts/i464_min_run.sh
# [phase=...] markers so poll_pipeline.py keys off the same shapes, plus
# a 2-min heartbeat and a final results sentinel. Pod-side code NEVER
# shells out to scripts/task.py (branch-guard rule).
#
# Launch:
#   setsid nohup bash scripts/i464_min_cn_run.sh \
#       > /workspace/logs/issue-464-min-cn.log 2>&1 < /dev/null &
#   echo $! > /workspace/logs/issue-464-min-cn.pid

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export PYTHONUNBUFFERED=1  # flush per-cell python logger lines so grep-on-log monitoring works live
cd /workspace/explore-persona-space || { echo "[phase=failed] cd-failed"; exit 1; }

LOG_DIR=logs/issue_464_min_cn
mkdir -p "$LOG_DIR"

# Heartbeat (CLAUDE.md / prior #464 runners mirror).
( while true; do echo "[heartbeat] $(date -Iseconds)"; sleep 120; done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null' EXIT

write_failure_sentinel() {
    # $1 = phase tag, $2 = reason
    local phase="$1" reason="$2"
    local sentinel="/workspace/logs/issue-464-min-cn-${phase}-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    PHASE="$phase" REASON="$reason" SENTINEL="$sentinel" uv run python - <<'EOF'
import datetime
import json
import os

payload = {
    "issue": 464,
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
# All-GPU query is safe here (no CVD filtering needed): this runner is
# strictly sequential on a single GPU, so any compute PID at this point
# is a genuine leftover, never a concurrent sibling.
wait_gpu_idle() {
    local max_wait="${1:-180}" waited=0 pids
    while true; do
        pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' || true)
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

# ── Phase: preflight — token-id contracts (CLAUDE.md rule + minimal arms). ──
echo "[phase=preflight] $(date -Iseconds)"
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

# ── Phase: rgen_cache — pre-cache ALL frozen R_canon artifacts. ─────────
# Unlike the cn runner there is NO rgen_default GPU phase: the
# R_canon_default_train.json artifact already exists on the HF data repo
# (produced by the cn run); _load_R_canon_default_train() pulls it via
# its HF fallback and FAILS LOUD if it is missing.
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

# ── Phase: train — 12 single-persona cn cells, sequential on GPU 0. ─────
# Recipe identical to the cn runner (--single-persona --shared-marker
# --contrastive-negatives --no-traj); only the arms differ (minimal pair).
# Crash-safe re-entry mirrors the min runner: a cell whose adapter exists
# locally skips the TRAIN step, but every cell — trained or skipped —
# must still pass the HF upload verifier before it counts as ok
# (train_lora's upload is soft-fail; eval downloads from HF).
echo "[phase=train] start $(date -Iseconds) (12 cells, sequential, min_cn)"
FAILED_FILE="$LOG_DIR/min_cn_train_failed.txt"
: > "$FAILED_FILE"

ARMS=("system_minimal" "role_bare")
SEEDS=(42 137 1337)
PERSONAS=("pirate" "villain")

cell_count=0
for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for persona in "${PERSONAS[@]}"; do
            cell_count=$((cell_count + 1))
            cell_label="${arm}_seed${seed}_cn_${persona}"
            adapter_file="adapters/i464_${cell_label}/adapter_model.safetensors"
            log="$LOG_DIR/train_${cell_label}.log"
            train_rc=0
            if [ -s "$adapter_file" ]; then
                echo "[phase=train_cell] $cell_count/12 cell=$cell_label train SKIPPED (local adapter exists) $(date -Iseconds)"
            else
                echo "[phase=train_cell] $cell_count/12 cell=$cell_label $(date -Iseconds)"
                CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_phase23_train.py \
                    --cell "${arm}_seed${seed}" \
                    --single-persona "$persona" \
                    --shared-marker \
                    --contrastive-negatives \
                    --no-traj \
                    --gpu-id 0 \
                    > "$log" 2>&1 || train_rc=$?
            fi
            # A cell is "ok" ONLY once its adapter verifiably resolves on
            # HF (list_repo_files; re-upload from the local dir on a
            # missed soft-fail upload; fail-loud when unrepairable).
            # Applies to BOTH the just-trained path and the
            # skip-on-local-adapter re-entry.
            if [ "$train_rc" -eq 0 ]; then
                uv run python scripts/i464_min_verify_upload.py --cell "$cell_label" \
                    >> "$log" 2>&1 || train_rc=$?
            fi
            if [ "$train_rc" -ne 0 ]; then
                echo "$cell_label" >> "$FAILED_FILE"
                echo "[phase=train_cell] FAILED cell=$cell_label rc=$train_rc see $log" >&2
            else
                echo "[phase=train_cell] ok cell=$cell_label (HF upload verified) $(date -Iseconds)"
            fi
        done
    done
done

if [ -s "$FAILED_FILE" ]; then
    FAILED=$(tr '\n' ' ' < "$FAILED_FILE")
    write_failure_sentinel train "cells failed train_lora or HF upload verification: $FAILED"
    echo "[phase=failed] train (cells: $FAILED) $(date -Iseconds)" >&2
    exit 12
fi
echo "[phase=train] ok 12/12 cells trained + HF-upload-verified $(date -Iseconds)"

# ── Phase: crosseval — vLLM log P( ※) (PRIMARY DV), one engine. ─────────
echo "[phase=crosseval] start $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_po_eval.py --variant min_cn --resume \
    > "$LOG_DIR/min_cn_eval.log" 2>&1 || {
    rc=$?
    write_failure_sentinel crosseval "i464_po_eval.py --variant min_cn exit $rc"
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
CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_min_capture_logits.py --variant min_cn --resume \
    > "$LOG_DIR/min_cn_capture.log" 2>&1 || {
    rc=$?
    write_failure_sentinel logitcap "i464_min_capture_logits.py --variant min_cn exit $rc"
    echo "[phase=failed] logitcap (exit $rc) $(date -Iseconds)" >&2
    exit 14
}
echo "[phase=logitcap] ok $(date -Iseconds)"

# ── Phase: analyze — headline d_seed_minimal_cn + inherited gates. ──────
echo "[phase=analyze] start $(date -Iseconds)"
uv run python scripts/i464_po_analyze.py --variant min_cn \
    > "$LOG_DIR/min_cn_analyze.log" 2>&1 || {
    rc=$?
    write_failure_sentinel analyze "i464_po_analyze.py --variant min_cn exit $rc"
    echo "[phase=failed] analyze (exit $rc) $(date -Iseconds)" >&2
    exit 15
}
echo "[phase=analyze] ok $(date -Iseconds)"

# ── Final results sentinel (poll_pipeline.py contract), then [phase=done]. ──
echo "[phase=results_sentinel] $(date -Iseconds)"
RESULTS_SENTINEL=/workspace/logs/issue-464-min-cn-results.json
sentinel_rc=0
SENTINEL="$RESULTS_SENTINEL" uv run python - <<'EOF' || sentinel_rc=$?
import datetime
import json
import os

analysis = json.loads(
    open("eval_results/issue_464/minimal_content_cn/analysis.json").read()
)
headline = analysis.get("headline", {})
note = {
    "followup_label": "minimal_content_cn",
    "headline_status": analysis.get("headline_status"),
    "d_seed_minimal_cn": headline.get(
        "d_seed_minimal_cn", headline.get("d_seed_minimal_cn_descriptive")
    ),
    "h1_overall_pass": analysis.get("h1_elicitation", {}).get("overall_pass"),
    "dynamic_range_gate": analysis.get("dynamic_range_gate", {}),
    "verdict_precedence_note": analysis.get("verdict_precedence_note"),
    "analysis_path": "eval_results/issue_464/minimal_content_cn/analysis.json",
}
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 464,
    "by": "i464_min_cn_run.sh",
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "note": json.dumps(note, indent=2),
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
echo "[phase=done] minimal_content_cn follow-up complete $(date -Iseconds)"
