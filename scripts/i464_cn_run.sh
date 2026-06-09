#!/usr/bin/env bash
# Issue #464 contrastive-negatives (cn) follow-up — pod-side end-to-end runner.
#
# 1 GPU, sequential execution. Tests whether the role-vs-system localization
# advantage (the parent #464 sweep's ~5-6 nats reduction) survives when
# contrast is added WITHOUT co-residence: ONE persona per LoRA with the
# SHARED pirate marker ` ※`, interleaved with marker-LESS negative rows
# under (a) the OTHER persona's SAME-arm encoding and (b) the bare
# default-assistant encoding. Together with the parent #464 sweep and the
# positive-only (po) follow-up this fills the missing cell of a 2x2:
#
#                              | coupled  | uncoupled
#   ──────────────────────────┼─────────┼──────────
#   contrast (2-marker | cn)  │  #464   │  this run
#   no contrast (positive-only) │  N/A   │  po follow-up
#
# 18 cells = arms {system_plain, system_padded, role}
#          x seeds {42, 137, 1337}
#          x personas {pirate, villain}
#
# Mirrors scripts/i464_po_run.sh's [phase=...] markers so poll_pipeline.py
# keys off the same shapes, plus a 2-min heartbeat. Adds one extra phase
# (rgen_default) for R_canon[default, train] generation.
#
# Launch:
#   nohup bash scripts/i464_cn_run.sh > /workspace/logs/issue-464-cn-run.log 2>&1 &
#   echo $! > /workspace/logs/issue-464-cn-run.pid

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export PYTHONUNBUFFERED=1  # flush per-cell python logger lines so grep-on-log monitoring works live
cd /workspace/explore-persona-space || { echo "[phase=failed] cd-failed"; exit 1; }

LOG_DIR=logs/issue_464_cn
mkdir -p "$LOG_DIR"

# Heartbeat (CLAUDE.md / parent #464 + po mirror).
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

# ── Phase 1a: pre-cache R_canon (train+test) serially (avoid concurrent-download race). ──
# The cn train script's _load_R_canon() pulls R_canon_train/test from
# the HF data repo on first use; pre-caching once up front fails-loud
# early if the upstream artifact is missing.
echo "[phase=rgen_cache] $(date -Iseconds)"
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO)
from scripts.i464_phase23_train import _load_R_canon  # type: ignore[import-not-found]
R_train = _load_R_canon('train')
R_test  = _load_R_canon('test')
print(f'R_canon_train personas={list(R_train.keys())} q_train_count={len(next(iter(R_train.values())))}')
print(f'R_canon_test  personas={list(R_test.keys())}  q_test_count={len(next(iter(R_test.values())))}')
"
echo "[phase=rgen_cache] ok $(date -Iseconds)"

# ── Phase 1b: generate R_canon[default, train] (cn-only artifact). ──────
# 30 base-model greedy responses under the bare default-assistant system
# prompt on Q_train. Needed for the cn negative rows (default-assistant
# encoding). The script: (a) skips re-generation if the artifact is
# already locally cached AND the HF schema matches, (b) writes to
# data/issue_464/R_canon_default_train.json and uploads to the HF data
# repo, (c) fails loud on marker-in-R or truncation >5%.
echo "[phase=rgen_default] $(date -Iseconds)"
DEFAULT_R_LOCAL=data/issue_464/R_canon_default_train.json
if [ -s "$DEFAULT_R_LOCAL" ]; then
    # Skip regeneration if local file exists AND has the right schema —
    # cheap one-off Python check so a re-run after partial completion
    # doesn't burn 30 forwards re-generating.
    if uv run python -c "
import json, sys
p = '$DEFAULT_R_LOCAL'
d = json.loads(open(p).read())
ok = d.get('schema_version') == 'i464_cn_default_R_v1' and 'default' in d.get('completions', {})
sys.exit(0 if ok else 1)
" 2>/dev/null; then
        echo "[phase=rgen_default] cached: $DEFAULT_R_LOCAL exists and matches schema, skipping vLLM generation."
    else
        rm -f "$DEFAULT_R_LOCAL"
        CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_cn_generate_R_default.py \
            > "$LOG_DIR/rgen_default.log" 2>&1 || {
            rc=$?
            echo "[phase=failed] rgen_default (exit $rc) $(date -Iseconds)" >&2
            exit 11
        }
    fi
else
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_cn_generate_R_default.py \
        > "$LOG_DIR/rgen_default.log" 2>&1 || {
        rc=$?
        echo "[phase=failed] rgen_default (exit $rc) $(date -Iseconds)" >&2
        exit 11
    }
fi
echo "[phase=rgen_default] ok $(date -Iseconds)"

# ── Phase 2/3: sequential single-persona cn training, 18 cells on GPU 0. ──
# Each train invocation: ~10-12 min/cell on 1xH100 at the parent's recipe
# (5 epochs x 600 rows x bs=4 x grad_accum=4 — cn doubles the row count
# vs po, so per-cell wall is ~20% longer). Trajectory callback stays OFF
# to keep runtime down.
echo "[phase=train] start $(date -Iseconds) (18 cells, sequential, cn)"
FAILED_FILE="$LOG_DIR/cn_train_failed.txt"
: > "$FAILED_FILE"

ARMS=("system_plain" "system_padded" "role")
SEEDS=(42 137 1337)
PERSONAS=("pirate" "villain")

cell_count=0
for arm in "${ARMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for persona in "${PERSONAS[@]}"; do
            cell_count=$((cell_count + 1))
            cell_label="${arm}_seed${seed}_cn_${persona}"
            log="$LOG_DIR/train_${cell_label}.log"
            echo "[phase=train_cell] $cell_count/18 cell=$cell_label $(date -Iseconds)"
            train_rc=0
            CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_phase23_train.py \
                --cell "${arm}_seed${seed}" \
                --single-persona "$persona" \
                --shared-marker \
                --contrastive-negatives \
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
    sentinel="/workspace/logs/issue-464-cn-train-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 464,
    "phase": "cn_train",
    "failure_class": "code",
    "failed_cells": "$FAILED".split(),
    "reason": "One or more contrastive-negatives single-persona cells failed train_lora.",
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
EOF
    echo "[phase=failed] cn_train (cells: $FAILED) $(date -Iseconds)" >&2
    exit 12
fi
echo "[phase=train] ok 18/18 cells trained $(date -Iseconds)"

# ── Phase 4: cross-eval (one vLLM engine, LoRARequest hot-swap, --resume). ──
echo "[phase=crosseval] start $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_po_eval.py --variant cn --resume \
    > "$LOG_DIR/cn_eval.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] crosseval (exit $rc) $(date -Iseconds)" >&2
    exit 13
}
echo "[phase=crosseval] ok $(date -Iseconds)"

# ── Phase 5: analyze. ───────────────────────────────────────────────────
echo "[phase=analyze] start $(date -Iseconds)"
uv run python scripts/i464_po_analyze.py --variant cn \
    > "$LOG_DIR/cn_analyze.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] analyze (exit $rc) $(date -Iseconds)" >&2
    exit 14
}
echo "[phase=analyze] ok $(date -Iseconds)"

echo "[phase=done] contrastive-negatives follow-up complete $(date -Iseconds)"
