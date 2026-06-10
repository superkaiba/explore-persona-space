#!/usr/bin/env bash
# Issue #547 — sub-1-epoch max_steps-resolved re-run of #533's grid (4x H100 sweep).
#
# Single-variable rerun of #533's cn regime with the training amount
# indexed on max_steps instead of epochs. ONE variable changes vs #533:
# training-amount INDEXING, epochs {1, 2, 3, 5} → max_steps
# {5, 10, 18, 30, 60, 120} (E ≈ {0.13, 0.27, 0.48, 0.80, 1.60, 3.20} at
# 37.5 optimizer steps/epoch). Everything else — lr=5e-6, the 5 seeds,
# the 2 personas, the 3 arms, the r=32/α=64 LoRA, marker-only loss, the
# 1:1 contrastive-negatives composition, pinned R_canon reuse — is
# byte-identical to #533.
#
# 180 cells = 3 arms × 5 seeds × 2 personas × 6 max_steps settings
#           = system_plain / system_padded / role
#             × {42, 137, 1337, 7, 21}
#             × {pirate, villain}
#             × {5, 10, 18, 30, 60, 120}
#
# Phase 1: 180 single-persona cn LoRAs trained, sharded 4-way across
#          GPUs 0..3 (sequential within each shard). ~4.3 h wall on 4
#          GPUs (avg 40.5 steps/cell at ~2.4 s/step + ~4 min overhead).
# Phase 2: one vLLM engine, LoRARequest hot-swap across 180 cells × 3
#          eval encodings = 540 per-cell JSONs. ~45 min wall.
# Phase 3: anchor selection (i529_select_anchor.py --in-dir
#          eval_results/issue_547/... --grid 5,10,...,120 --suffix-char s)
#          — CPU, ~1 min.
# Phase 4: analyze (i464_po_analyze.py --variant cn_i547 --anchor-file
#          ...) — writes the UNCONDITIONAL trajectory_per_persona block
#          (+ the anchor-gated headline bonus when non-degenerate).
#          CPU, minutes.
#
# Sentinel + [phase=...] log lines mirror i533_cn_run.sh so
# poll_pipeline.py keys off the same shapes. End-of-run sentinel:
# /workspace/logs/issue-547-cn-run-epm_results-<epoch>.json
# (kind=epm:results, sentinel_schema_version=1).
#
# Smoke = sweep with one cell (PASS_UNIFIED): set MAX_STEPS_OVERRIDE=5,
# SEEDS_OVERRIDE=42, ARMS_OVERRIDE=system_plain, PERSONAS_OVERRIDE=pirate;
# the same script handles it end-to-end. When the resolved grid is
# smaller than the full 180 cells, the eval phase is restricted to the
# overridden cells (--smoke-cells) and the selector + analyzer run with
# --allow-partial so the 1-cell grid is self-consistent; the PRODUCTION
# path (no overrides) passes neither flag and fails loud on any missing
# cell.
#
# Launch:
#   nohup bash scripts/i547_cn_run.sh > /workspace/logs/issue-547-cn-run.log 2>&1 &
#   echo $! > /workspace/logs/issue-547-cn-run.pid

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export PYTHONUNBUFFERED=1
cd /workspace/explore-persona-space || { echo "[phase=failed] cd-failed"; exit 1; }

LOG_DIR=logs/issue_547_cn
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
MAX_STEPS_STR="${MAX_STEPS_OVERRIDE:-5 10 18 30 60 120}"
N_GPUS="${N_GPUS:-4}"

# Bash word-splitting once; arrays preserve element order.
read -r -a ARMS_ARR <<< "${ARMS[*]}"
read -r -a PERSONAS_ARR <<< "${PERSONAS[*]}"
read -r -a SEEDS_ARR <<< "$SEEDS_STR"
read -r -a MAX_STEPS_ARR <<< "$MAX_STEPS_STR"

echo "[phase=preflight] $(date -Iseconds)"
echo "  grid: arms=(${ARMS_ARR[*]}) seeds=(${SEEDS_ARR[*]}) personas=(${PERSONAS_ARR[*]}) max_steps=(${MAX_STEPS_ARR[*]}) ngpu=$N_GPUS"

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

# ── Phase 1a: pre-cache R_canon (train+test) serially — at the PINNED
# revision (the loaders thread revision=DATA_REVISION; #547 §4.1(h)). ──
echo "[phase=rgen_cache] $(date -Iseconds)"
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO)
from explore_persona_space.experiments.i464_data import DATA_REVISION
from scripts.i464_phase23_train import _load_R_canon  # type: ignore[import-not-found]
print(f'data-repo revision pin: {DATA_REVISION}')
R_train = _load_R_canon('train')
R_test  = _load_R_canon('test')
print(f'R_canon_train personas={list(R_train.keys())} q_train_count={len(next(iter(R_train.values())))}')
print(f'R_canon_test  personas={list(R_test.keys())}  q_test_count={len(next(iter(R_test.values())))}')
"
echo "[phase=rgen_cache] ok $(date -Iseconds)"

# ── Phase 1b: VERIFY-ONLY for R_canon[default, train]. ──
#
# FULL DATA REUSE — DO NOT regenerate the cn-only artifact (inherited
# verbatim from #533; the verify path is functionally identical to what
# the trainer reads at run-time and now pulls at the pinned revision).
echo "[phase=rgen_default] $(date -Iseconds)"
uv run python - <<'EOF'
"""Verify #464's frozen R_canon_default_train.json is reachable at the pin."""
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

# Smoke mode = any override shrank the grid below the full 180 cells.
# The eval phase is then restricted to the overridden cells and the
# selector + analyzer tolerate the missing remainder; production (180
# cells) passes neither flag and fails loud on any missing cell.
SMOKE_MODE=0
if [ "$n_cells" -ne 180 ]; then
    SMOKE_MODE=1
    echo "[phase=train] SMOKE_MODE=1 (grid=$n_cells cells != 180): eval restricted via --smoke-cells; selector+analyzer get --allow-partial"
fi

# Spawn one subshell per GPU; each iterates its sharded slice. Collect
# the shard PIDs so we wait on THEM specifically — naked `wait` would
# also block on the heartbeat subshell ($HB_PID, infinite loop), and
# that caused #529's 12-min silent stall between train-phase completion
# and crosseval.
TRAIN_PIDS=()
for gpu in $(seq 0 $((N_GPUS - 1))); do
    (
        idx=0
        for c in "${cells[@]}"; do
            if [ $((idx % N_GPUS)) -eq "$gpu" ]; then
                IFS='|' read -r arm seed persona steps <<< "$c"
                cell_label="${arm}_seed${seed}_cn_${persona}_s${steps}"
                log="$LOG_DIR/train_${cell_label}.log"
                echo "[phase=train_cell] gpu=$gpu idx=$idx cell=$cell_label $(date -Iseconds)"
                train_rc=0
                CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/i464_phase23_train.py \
                    --issue 547 \
                    --cell "${arm}_seed${seed}" \
                    --single-persona "$persona" \
                    --shared-marker \
                    --contrastive-negatives \
                    --max-steps "$steps" \
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
    sentinel="/workspace/logs/issue-547-cn-train-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 547,
    "phase": "cn_train",
    "failure_class": "code",
    "failed_cells": "$FAILED".split(),
    "reason": "One or more #547 cn cells failed train_lora.",
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
EVAL_ARGS=(--variant cn_i547 --resume)
if [ "$SMOKE_MODE" -eq 1 ]; then
    EVAL_ARGS+=(--smoke-cells "${eval_labels[@]}")
fi
CUDA_VISIBLE_DEVICES=0 uv run python scripts/i464_po_eval.py "${EVAL_ARGS[@]}" \
    > "$LOG_DIR/cn_eval.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] crosseval (exit $rc) $(date -Iseconds)" >&2
    exit 13
}
echo "[phase=crosseval] ok $(date -Iseconds)"

# ── Phase 5: anchor selection (CPU) on the max_steps grid. ──────────────
echo "[phase=anchor_select] start $(date -Iseconds)"
ANCHOR_PATH=eval_results/issue_547/anchor_selection.json
PER_CELL_DIR=eval_results/issue_547/contrastive_negatives/cross_eval/per_cell
SELECT_ARGS=(--in-dir "$PER_CELL_DIR" --out-path "$ANCHOR_PATH" --grid "${MAX_STEPS_STR// /,}" --suffix-char s)
if [ "$SMOKE_MODE" -eq 1 ]; then
    SELECT_ARGS+=(--allow-partial)
fi
uv run python scripts/i529_select_anchor.py "${SELECT_ARGS[@]}" \
    > "$LOG_DIR/anchor_select.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] anchor_select (exit $rc) $(date -Iseconds)" >&2
    exit 14
}
echo "[phase=anchor_select] ok $(date -Iseconds)"

# ── Phase 6: analyze — UNCONDITIONAL trajectory + anchor-gated bonus. ───
echo "[phase=analyze] start $(date -Iseconds)"
ANALYZE_ARGS=(--variant cn_i547 --anchor-file "$ANCHOR_PATH")
if [ "$SMOKE_MODE" -eq 1 ]; then
    ANALYZE_ARGS+=(--allow-partial)
fi
uv run python scripts/i464_po_analyze.py "${ANALYZE_ARGS[@]}" \
    > "$LOG_DIR/cn_analyze.log" 2>&1 || {
    rc=$?
    echo "[phase=failed] analyze (exit $rc) $(date -Iseconds)" >&2
    exit 15
}
echo "[phase=analyze] ok $(date -Iseconds)"

# ── Phase 7: end-of-run sentinel for poll_pipeline.py. ──────────────────
SENTINEL="/workspace/logs/issue-547-cn-run-epm_results-$(date +%s).json"
mkdir -p "$(dirname "$SENTINEL")"
uv run python - <<EOF
import json, datetime, pathlib
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 547,
    "phase": "done",
    "gate": "results",
    "blocks_pipeline": False,
    "by": "i547_cn_run",
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "note": "i547 sub-1-epoch max_steps grid complete: $n_cells cells trained, anchor selected, analysis (incl. unconditional trajectory_per_persona) written.",
    "artifacts": {
        "anchor_selection": "eval_results/issue_547/anchor_selection.json",
        "analysis":         "eval_results/issue_547/contrastive_negatives/analysis.json",
        "per_cell_dir":     "eval_results/issue_547/contrastive_negatives/cross_eval/per_cell",
    },
}
pathlib.Path("$SENTINEL").write_text(json.dumps(payload, indent=2))
print("wrote sentinel: $SENTINEL")
EOF

echo "[phase=done] i547 sub-1-epoch max_steps grid complete $(date -Iseconds)"
