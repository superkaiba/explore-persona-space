#!/usr/bin/env bash
# Issue #533 — lr=5e-6 corrective re-run of #529's grid (4x H100 ZeRO-1 / TP=1 sweep).
#
# Single-variable rerun of #529's marker-less cn regime at the
# demonstrated clean-window learning rate. ONE variable changes vs #529:
# lr 1e-5 → 5e-6 (per .claude/rules/marker-training-recipe.md).
# Everything else — the {1, 2, 3, 5}-epoch grid, the 5 seeds, the 2
# personas, the 3 arms, the r=32/α=64 LoRA, marker-only loss, the 1:1
# contrastive-negatives composition, R_canon reuse — is byte-identical
# to #529.
#
# 120 cells = 3 arms × 5 seeds × 2 personas × 4 epoch settings
#           = system_plain / system_padded / role
#             × {42, 137, 1337, 7, 21}
#             × {pirate, villain}
#             × {1, 2, 3, 5}
#
# Phase 1: 120 single-persona cn LoRAs trained, sharded 4-way across
#          GPUs 0..3 (sequential within each shard). At default
#          recipe: ~10-12 min/cell at E=5 → ~3.5 h wall on 4 GPUs.
# Phase 2: one vLLM engine, LoRARequest hot-swap across 120 cells × 3
#          eval encodings = 360 per-cell JSONs. ~30 min wall.
# Phase 3: anchor selection (i529_select_anchor.py --in-dir
#          eval_results/issue_533/...) — CPU, ~1 min.
# Phase 4: analyze at the selected anchor (i464_po_analyze.py
#          --variant cn_i533 --anchor-file ...). CPU, ~1 min.
#
# Sentinel + [phase=...] log lines mirror i529_cn_run.sh so
# poll_pipeline.py keys off the same shapes. End-of-run sentinel:
# /workspace/logs/issue-533-cn-run-results-<epoch>.json
# (kind=epm:results, sentinel_schema_version=1).
#
# Smoke = sweep with one cell: set EPOCHS_OVERRIDE=1, SEEDS_OVERRIDE=42,
# ARMS_OVERRIDE=system_plain, PERSONAS_OVERRIDE=pirate; the same script
# handles it end-to-end (PASS_UNIFIED smoke architecture).
#
# Launch:
#   nohup bash scripts/i533_cn_run.sh > /workspace/logs/issue-533-cn-run.log 2>&1 &
#   echo $! > /workspace/logs/issue-533-cn-run.pid

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export PYTHONUNBUFFERED=1
cd /workspace/explore-persona-space || { echo "[phase=failed] cd-failed"; exit 1; }

LOG_DIR=logs/issue_533_cn
mkdir -p "$LOG_DIR"

# Heartbeat (CLAUDE.md / parent #464 + po mirror).
( while true; do echo "[heartbeat] $(date -Iseconds)"; sleep 120; done ) &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null' EXIT

# ── Sweep grid (override per the smoke architecture: one cell = smoke). ──
# All caps to mark "tunable for smoke"; defaults match plan §4.1.
ARMS=("${ARMS_OVERRIDE:-system_plain system_padded role}")
SEEDS_STR="${SEEDS_OVERRIDE:-42 137 1337 7 21}"
PERSONAS=("${PERSONAS_OVERRIDE:-pirate villain}")
EPOCHS_STR="${EPOCHS_OVERRIDE:-1 2 3 5}"
N_GPUS="${N_GPUS:-4}"

# Bash word-splitting once; arrays preserve element order.
read -r -a ARMS_ARR <<< "${ARMS[*]}"
read -r -a PERSONAS_ARR <<< "${PERSONAS[*]}"
read -r -a SEEDS_ARR <<< "$SEEDS_STR"
read -r -a EPOCHS_ARR <<< "$EPOCHS_STR"

echo "[phase=preflight] $(date -Iseconds)"
echo "  grid: arms=(${ARMS_ARR[*]}) seeds=(${SEEDS_ARR[*]}) personas=(${PERSONAS_ARR[*]}) epochs=(${EPOCHS_ARR[*]}) ngpu=$N_GPUS"

# ── Marker token-id assertion at launch (CLAUDE.md rule). ───────────────
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

# ── Phase 1a: pre-cache R_canon (train+test) serially. ──
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

# ── Phase 1b: VERIFY-ONLY for R_canon[default, train]. ──
#
# Plan §4.2 mandates FULL DATA REUSE — DO NOT regenerate the cn-only
# artifact. Regenerating + uploading would overwrite #464's frozen
# `R_canon_default_train.json` and silently change the default-negative
# training rows, breaking the single-variable comparison against #464.
#
# Round-2 fix per `default-r-regenerated-not-reused` (both reviewers):
# verify the HF data-repo artifact exists with the expected schema,
# fail LOUD if it doesn't, and NEVER regenerate.
echo "[phase=rgen_default] $(date -Iseconds)"
uv run python - <<'EOF'
"""Verify #464's frozen R_canon_default_train.json is reachable.

Reuses ``_load_R_canon_default_train`` (which handles the local cache
+ HF data-repo fallback + schema_version assertion), so the verify
path is functionally identical to what the trainer reads at run-time.
A schema drift or a missing artifact raises here, surfacing the
problem BEFORE any GPU work starts.
"""
import logging
import sys

logging.basicConfig(level=logging.INFO)
try:
    from scripts.i464_phase23_train import _load_R_canon_default_train

    completions = _load_R_canon_default_train()
    n_q = len(completions["default"])
    print(
        f"[phase=rgen_default] verified #464 R_canon_default_train.json "
        f"(schema_version=i464_cn_default_R_v1, default-persona q_train_count={n_q}); "
        "REUSING — no regeneration."
    )
except Exception as e:  # noqa: BLE001 — fail-loud surface for verify-only
    print(
        f"[phase=failed] rgen_default verify-only could not load "
        f"#464 R_canon_default_train.json: {e!r}",
        file=sys.stderr,
    )
    sys.exit(11)
EOF
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "[phase=failed] rgen_default verify-only (exit $rc) $(date -Iseconds)" >&2
    exit "$rc"
fi
echo "[phase=rgen_default] ok $(date -Iseconds)"

# ── Phase 2/3: parallel single-persona cn training across N_GPUS. ──
# Cells are independent; shard by cell index modulo N_GPUS. Each shard
# is a per-GPU subshell that walks its slice sequentially.
echo "[phase=train] start $(date -Iseconds)"
FAILED_FILE="$LOG_DIR/cn_train_failed.txt"
: > "$FAILED_FILE"

# Build the full cell list deterministically (arm × seed × persona × epoch).
cells=()
for arm in "${ARMS_ARR[@]}"; do
    for seed in "${SEEDS_ARR[@]}"; do
        for persona in "${PERSONAS_ARR[@]}"; do
            for epoch in "${EPOCHS_ARR[@]}"; do
                cells+=("$arm|$seed|$persona|$epoch")
            done
        done
    done
done
n_cells=${#cells[@]}
echo "[phase=train] cells=$n_cells parallelism=$N_GPUS"

# Spawn one subshell per GPU; each iterates its sharded slice. Collect
# the shard PIDs so we wait on THEM specifically — naked `wait` would
# also block on the heartbeat subshell ($HB_PID, infinite loop), and
# that caused #529's 12-min silent stall between train-phase completion
# and crosseval (13:40 → 13:53, kill HB_PID by hand to unblock).
TRAIN_PIDS=()
for gpu in $(seq 0 $((N_GPUS - 1))); do
    (
        idx=0
        for c in "${cells[@]}"; do
            if [ $((idx % N_GPUS)) -eq "$gpu" ]; then
                IFS='|' read -r arm seed persona epoch <<< "$c"
                cell_label="${arm}_seed${seed}_cn_${persona}_e${epoch}"
                log="$LOG_DIR/train_${cell_label}.log"
                echo "[phase=train_cell] gpu=$gpu idx=$idx cell=$cell_label $(date -Iseconds)"
                train_rc=0
                CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/i464_phase23_train.py \
                    --issue 533 \
                    --cell "${arm}_seed${seed}" \
                    --single-persona "$persona" \
                    --shared-marker \
                    --contrastive-negatives \
                    --epochs "$epoch" \
                    --lr 5e-6 \
                    --no-traj \
                    --gpu-id "$gpu" \
                    > "$log" 2>&1 || train_rc=$?
                if [ "$train_rc" -ne 0 ]; then
                    echo "$cell_label" >> "$FAILED_FILE"
                    echo "[phase=train_cell] FAILED gpu=$gpu cell=$cell_label rc=$train_rc see $log" >&2
                else
                    echo "[phase=train_cell] ok gpu=$gpu cell=$cell_label $(date -Iseconds)"
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
    sentinel="/workspace/logs/issue-533-cn-train-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 533,
    "phase": "cn_train",
    "failure_class": "code",
    "failed_cells": "$FAILED".split(),
    "reason": "One or more #533 cn cells failed train_lora.",
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
echo "[phase=train] ok $n_cells/$n_cells cells trained $(date -Iseconds)"

# ── Phase 4: cross-eval (one vLLM engine, LoRARequest hot-swap, --resume). ──
echo "[phase=crosseval] start $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_po_eval.py --variant cn_i533 --resume \
    > "$LOG_DIR/cn_eval.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] crosseval (exit $rc) $(date -Iseconds)" >&2
    exit 13
}
echo "[phase=crosseval] ok $(date -Iseconds)"

# ── Phase 5: anchor selection (CPU). ────────────────────────────────────
echo "[phase=anchor_select] start $(date -Iseconds)"
ANCHOR_PATH=eval_results/issue_533/anchor_selection.json
PER_CELL_DIR=eval_results/issue_533/contrastive_negatives/cross_eval/per_cell
uv run python scripts/i529_select_anchor.py \
    --in-dir "$PER_CELL_DIR" \
    --out-path "$ANCHOR_PATH" \
    > "$LOG_DIR/anchor_select.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] anchor_select (exit $rc) $(date -Iseconds)" >&2
    exit 14
}
echo "[phase=anchor_select] ok $(date -Iseconds)"

# ── Phase 6: analyze at the selected anchor. ────────────────────────────
echo "[phase=analyze] start $(date -Iseconds)"
uv run python scripts/i464_po_analyze.py --variant cn_i533 --anchor-file "$ANCHOR_PATH" \
    > "$LOG_DIR/cn_analyze.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] analyze (exit $rc) $(date -Iseconds)" >&2
    exit 15
}
echo "[phase=analyze] ok $(date -Iseconds)"

# ── Phase 7: end-of-run sentinel for poll_pipeline.py. ──────────────────
SENTINEL="/workspace/logs/issue-533-cn-run-epm_results-$(date +%s).json"
mkdir -p "$(dirname "$SENTINEL")"
uv run python - <<EOF
import json, datetime, pathlib
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 533,
    "phase": "done",
    "gate": "results",
    "blocks_pipeline": False,
    "by": "i533_cn_run",
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "note": "i533 cn re-run (lr=5e-6) complete: 120 cells trained, anchor selected, analysis written.",
    "artifacts": {
        "anchor_selection": "eval_results/issue_533/anchor_selection.json",
        "analysis":         "eval_results/issue_533/contrastive_negatives/analysis.json",
        "per_cell_dir":     "eval_results/issue_533/contrastive_negatives/cross_eval/per_cell",
    },
}
pathlib.Path("$SENTINEL").write_text(json.dumps(payload, indent=2))
print("wrote sentinel: $SENTINEL")
EOF

echo "[phase=done] i533 cn re-run (lr=5e-6) complete $(date -Iseconds)"
