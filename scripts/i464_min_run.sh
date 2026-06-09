#!/usr/bin/env bash
# Issue #464 minimal_content follow-up — pod-side end-to-end runner.
#
# 1 GPU, sequential phases. Content-matched minimal encodings in the
# parent's co-resident competing-marker regime (pirate → ` ※` id 83399,
# villain → ` ¶` id 78846, co-trained on the SAME LoRA):
#
#   6 cells = arms {system_minimal, role_bare} x seeds {42, 137, 1337}
#
# Question this follow-up answers: the parent confounds content richness
# with slot (elaborate system instruction vs bare 4-5-token role
# compound). With the persona announced by a single bare word in BOTH
# slots (token-parity padded), does the role-header advantage survive?
#
# Phases (each persists its output the moment it completes; re-entry
# skips completed cells):
#   preflight   — token-id contract asserts (incl. minimal-arm contracts)
#   rgen_cache  — pre-cache the FROZEN R_canon from the HF data repo
#   train       — 6 co-resident LoRAs (skip cells whose adapter exists)
#   crosseval   — vLLM log P(marker) per cell (PRIMARY DV), --resume
#   logitcap    — four-float HF capture (z_marker/z_eos/logZ/logp), --resume
#   q1min       — base-model behavioral probe (system_minimal/role_bare)
#   analyze     — headline stats + parent join
#
# Architecture: mirrors scripts/i464_po_run.sh's [phase=...] markers so
# poll_pipeline.py keys off the same shapes, plus a 2-min heartbeat.
# Pod-side code NEVER shells out to scripts/task.py (branch-guard rule).
#
# Launch:
#   nohup bash scripts/i464_min_run.sh > /workspace/logs/issue-464-min.log 2>&1 &
#   echo $! > /workspace/logs/issue-464-min.pid

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export PYTHONUNBUFFERED=1  # flush per-cell python logger lines so grep-on-log monitoring works live
cd /workspace/explore-persona-space || { echo "[phase=failed] cd-failed"; exit 1; }

LOG_DIR=logs/issue_464_min
mkdir -p "$LOG_DIR"

# Heartbeat (CLAUDE.md / parent #464 mirror).
( while true; do echo "[heartbeat] $(date -Iseconds)"; sleep 120; done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null' EXIT

write_failure_sentinel() {
    # $1 = phase tag, $2 = reason
    local phase="$1" reason="$2"
    local sentinel="/workspace/logs/issue-464-min-${phase}-failed.json"
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

# ── Phase: preflight — token-id contracts (CLAUDE.md rule + minimal arms). ──
echo "[phase=preflight] $(date -Iseconds)"
uv run python -c "
from transformers import AutoTokenizer
from explore_persona_space.experiments import i464_encodings as enc
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
enc.assert_token_ids(tok)
assert tok.encode(enc.MARKER_PIRATE_TEXT, add_special_tokens=False) == [enc.MARKER_PIRATE_ID]
assert tok.encode(enc.MARKER_VILLAIN_TEXT, add_special_tokens=False) == [enc.MARKER_VILLAIN_ID]
print('token-id contracts OK ( ※ -> 83399,  ¶ -> 78846, minimal arms incl. MF-D parity)')
" || { write_failure_sentinel preflight "token-id contract assert failed"; \
       echo "[phase=failed] preflight $(date -Iseconds)"; exit 10; }
echo "[phase=preflight] ok $(date -Iseconds)"

# ── Phase: rgen_cache — pre-cache the FROZEN R_canon (encoding-independent). ──
echo "[phase=rgen_cache] $(date -Iseconds)"
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO)
from scripts.i464_phase23_train import _load_R_canon  # type: ignore[import-not-found]
R_train = _load_R_canon('train')
R_test  = _load_R_canon('test')
print(f'R_canon_train personas={list(R_train.keys())} q_train_count={len(next(iter(R_train.values())))}')
print(f'R_canon_test  personas={list(R_test.keys())}  q_test_count={len(next(iter(R_test.values())))}')
" || { write_failure_sentinel rgen_cache "R_canon pre-cache failed"; \
       echo "[phase=failed] rgen_cache $(date -Iseconds)"; exit 11; }
echo "[phase=rgen_cache] ok $(date -Iseconds)"

# ── Phase: train — 6 co-resident cells, sequential on GPU 0. ────────────
# Default i464_phase23_train path (NO --single-persona / --shared-marker /
# --contrastive-negatives): 30 q x 2 personas x 10 dupes = 600 rows/LoRA,
# each persona's own marker — the parent's co-resident regime, with only
# the arm (encoding) changed. --no-traj keeps the (parent-broken)
# trajectory callback off. Crash-safe re-entry: a cell whose adapter
# already exists locally is skipped (train_lora writes the adapter dir as
# its final step, so presence ⇒ the cell completed + uploaded).
echo "[phase=train] start $(date -Iseconds) (6 cells, sequential)"
FAILED_FILE="$LOG_DIR/min_train_failed.txt"
: > "$FAILED_FILE"

ARMS=("system_minimal" "role_bare")
SEEDS=(42 137 1337)

cell_count=0
for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        cell_count=$((cell_count + 1))
        cell_label="${arm}_seed${seed}"
        adapter_file="adapters/i464_${cell_label}/adapter_model.safetensors"
        if [ -s "$adapter_file" ]; then
            echo "[phase=train_cell] $cell_count/6 cell=$cell_label SKIP (adapter exists) $(date -Iseconds)"
            continue
        fi
        log="$LOG_DIR/train_${cell_label}.log"
        echo "[phase=train_cell] $cell_count/6 cell=$cell_label $(date -Iseconds)"
        train_rc=0
        CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_phase23_train.py \
            --cell "$cell_label" \
            --no-traj \
            --gpu-id 0 \
            > "$log" 2>&1 || train_rc=$?
        if [ "$train_rc" -ne 0 ]; then
            echo "$cell_label" >> "$FAILED_FILE"
            echo "[phase=train_cell] FAILED cell=$cell_label rc=$train_rc see $log" >&2
        else
            echo "[phase=train_cell] ok cell=$cell_label $(date -Iseconds)"
        fi
    done
done

if [ -s "$FAILED_FILE" ]; then
    FAILED=$(tr '\n' ' ' < "$FAILED_FILE")
    write_failure_sentinel train "cells failed train_lora: $FAILED"
    echo "[phase=failed] train (cells: $FAILED) $(date -Iseconds)" >&2
    exit 12
fi
echo "[phase=train] ok 6/6 cells trained $(date -Iseconds)"

# ── Phase: crosseval — vLLM log P(marker) (PRIMARY DV), one engine. ─────
echo "[phase=crosseval] start $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_min_eval.py --resume \
    > "$LOG_DIR/min_eval.log" 2>&1 || {
    rc=$?
    write_failure_sentinel crosseval "i464_min_eval.py exit $rc"
    echo "[phase=failed] crosseval (exit $rc) $(date -Iseconds)" >&2
    exit 13
}
echo "[phase=crosseval] ok $(date -Iseconds)"

# ── Phase: logitcap — four-float HF capture (SECONDARY mechanistic record).
# Separate process from crosseval: vLLM in-process teardown does NOT reap
# worker subprocesses (CLAUDE.md gotcha), so the HF forward pass gets a
# fresh process with the full GPU.
echo "[phase=logitcap] start $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_min_capture_logits.py --resume \
    > "$LOG_DIR/min_capture.log" 2>&1 || {
    rc=$?
    write_failure_sentinel logitcap "i464_min_capture_logits.py exit $rc"
    echo "[phase=failed] logitcap (exit $rc) $(date -Iseconds)" >&2
    exit 14
}
echo "[phase=logitcap] ok $(date -Iseconds)"

# ── Phase: q1min — base-model behavioral probe (minimal encodings). ─────
echo "[phase=q1min] start $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_q1_role_behavior.py \
    --encoding-set minimal \
    > "$LOG_DIR/min_q1.log" 2>&1 || {
    rc=$?
    write_failure_sentinel q1min "i464_q1_role_behavior.py --encoding-set minimal exit $rc"
    echo "[phase=failed] q1min (exit $rc) $(date -Iseconds)" >&2
    exit 15
}
echo "[phase=q1min] ok $(date -Iseconds)"

# ── Phase: analyze — headline stats + parent join. ──────────────────────
echo "[phase=analyze] start $(date -Iseconds)"
uv run python scripts/i464_min_analyze.py \
    > "$LOG_DIR/min_analyze.log" 2>&1 || {
    rc=$?
    write_failure_sentinel analyze "i464_min_analyze.py exit $rc"
    echo "[phase=failed] analyze (exit $rc) $(date -Iseconds)" >&2
    exit 16
}
echo "[phase=analyze] ok $(date -Iseconds)"

# Final marker required by poll_pipeline.py for "done".
echo "[phase=done] minimal_content follow-up complete $(date -Iseconds)"
