#!/usr/bin/env bash
# Issue #464 positive-only follow-up — pod-side end-to-end runner.
#
# 1 GPU, sequential execution. Removes co-residence by training ONE
# persona per LoRA with the SHARED pirate marker ` ※` (id 83399):
#
#   18 cells = arms {system_plain, system_padded, role}
#            x seeds {42, 137, 1337}
#            x personas {pirate, villain}
#
# Question this follow-up answers: does the role-vs-system localization
# advantage (the parent #464 sweep's ~5-6 nats reduction) SURVIVE when
# we strip away the two-persona co-residence + two-marker contrast that
# was conflated with the role-header mechanism?
#
# Architecture: mirrors scripts/i464_run_all.sh's [phase=...] markers
# so poll_pipeline.py keys off the same shapes, plus a 2-min heartbeat.
#
# Launch:
#   nohup bash scripts/i464_po_run.sh > /workspace/logs/issue-464-po-run.log 2>&1 &
#   echo $! > /workspace/logs/issue-464-po-run.pid

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export PYTHONUNBUFFERED=1  # flush per-cell python logger lines so grep-on-log monitoring works live
cd /workspace/explore-persona-space || { echo "[phase=failed] cd-failed"; exit 1; }

LOG_DIR=logs/issue_464_po
mkdir -p "$LOG_DIR"

# Heartbeat (CLAUDE.md / parent #464 mirror).
( while true; do echo "[heartbeat] $(date -Iseconds)"; sleep 120; done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null' EXIT

# ── Marker token-id assertion at launch (CLAUDE.md rule). ───────────────
echo "[phase=preflight] $(date -Iseconds)"
uv run python -c "
from transformers import AutoTokenizer
from explore_persona_space.experiments import i464_encodings as enc
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
enc.assert_token_ids(tok)
ids = tok.encode(enc.MARKER_PIRATE_TEXT, add_special_tokens=False)
assert ids == [enc.MARKER_PIRATE_ID], f'shared marker drifted: {ids}'
print('shared marker token-id contract OK ( ※ -> 83399)')
"
echo "[phase=preflight] ok $(date -Iseconds)"

# ── Phase 1: pre-cache R_canon serially (avoid concurrent-download race). ──
# The parent train script's _load_R_canon() pulls R_canon_train/test from
# the HF data repo on first use. With 18 sequential cells the first cell
# already serializes the download, but doing it ONCE up front fails-loud
# early if the upstream artifact is missing and avoids re-downloading
# inside per-cell processes that share the local cache directory.
echo "[phase=rgen_cache] $(date -Iseconds)"
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO)
# Reuse the train script's loader so we hit the exact same code path
# the per-cell trains will use (HF fallback contract identical).
from scripts.i464_phase23_train import _load_R_canon  # type: ignore[import-not-found]
R_train = _load_R_canon('train')
R_test  = _load_R_canon('test')
print(f'R_canon_train personas={list(R_train.keys())} q_train_count={len(next(iter(R_train.values())))}')
print(f'R_canon_test  personas={list(R_test.keys())}  q_test_count={len(next(iter(R_test.values())))}')
"
echo "[phase=rgen_cache] ok $(date -Iseconds)"

# ── Phase 2/3: sequential single-persona training, 18 cells on GPU 0. ───
# Each train invocation: ~10 min/cell on 1xH100 at the parent's recipe
# (5 epochs x 300 rows x bs=4 x grad_accum=4). The trajectory callback
# stays OFF to keep runtime down — the dynamics curve is not needed for
# the cross-cell headline.
echo "[phase=train] start $(date -Iseconds) (18 cells, sequential)"
FAILED_FILE="$LOG_DIR/po_train_failed.txt"
: > "$FAILED_FILE"

ARMS=("system_plain" "system_padded" "role")
SEEDS=(42 137 1337)
PERSONAS=("pirate" "villain")

cell_count=0
for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for persona in "${PERSONAS[@]}"; do
            cell_count=$((cell_count + 1))
            cell_label="${arm}_seed${seed}_${persona}"
            log="$LOG_DIR/train_${cell_label}.log"
            echo "[phase=train_cell] $cell_count/18 cell=$cell_label $(date -Iseconds)"
            train_rc=0
            CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_phase23_train.py \
                --cell "${arm}_seed${seed}" \
                --single-persona "$persona" \
                --shared-marker \
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
done

if [ -s "$FAILED_FILE" ]; then
    FAILED=$(tr '\n' ' ' < "$FAILED_FILE")
    sentinel="/workspace/logs/issue-464-po-train-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 464,
    "phase": "po_train",
    "failure_class": "code",
    "failed_cells": "$FAILED".split(),
    "reason": "One or more positive-only single-persona cells failed train_lora.",
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
EOF
    echo "[phase=failed] po_train (cells: $FAILED) $(date -Iseconds)" >&2
    exit 12
fi
echo "[phase=train] ok 18/18 cells trained $(date -Iseconds)"

# ── Phase 4: cross-eval (one vLLM engine, LoRARequest hot-swap, --resume). ──
echo "[phase=crosseval] start $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_po_eval.py --resume \
    > "$LOG_DIR/po_eval.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] crosseval (exit $rc) $(date -Iseconds)" >&2
    exit 13
}
echo "[phase=crosseval] ok $(date -Iseconds)"

# ── Phase 5: analyze. ───────────────────────────────────────────────────
echo "[phase=analyze] start $(date -Iseconds)"
uv run python scripts/i464_po_analyze.py \
    > "$LOG_DIR/po_analyze.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] analyze (exit $rc) $(date -Iseconds)" >&2
    exit 14
}
echo "[phase=analyze] ok $(date -Iseconds)"

echo "[phase=done] positive-only follow-up complete $(date -Iseconds)"
