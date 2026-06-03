#!/usr/bin/env bash
# Plan v3 §4.4 -- Phase B: train cond2_k0 / cond2_k1 / cond2_k3 sequentially
# at the anchor step *s** chosen from Phase A's cond1_withneg trajectory.
#
# All 3 arms share cond1_withneg's exact hyperparams: lr=5e-6, save_steps=10,
# log_every=5, suppress_at_post_response_slot=True, seed=42, gpu_id=0,
# n_neg_per_persona=100 (the with-negatives recipe). max_steps=$ANCHOR_STEP
# overrides the epoch ceiling so the 3 arms train to the same total budget.
#
# CONDITIONAL: this script only runs if Phase A's cond1_withneg has an
# anchor step (H_A1 PASS). The top-level orchestrator
# (i471_route_a_run_all.sh) reads phaseA_anchor.json and skips this
# script when lockstep is reported.
#
# Usage (from pod):
#   bash scripts/i471_phaseB_sweep.sh --anchor-step 45

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

LOG_DIR=logs/issue_471
mkdir -p "$LOG_DIR"

ANCHOR_STEP=""
for ((i=1; i<=$#; i++)); do
    case "${!i}" in
        --anchor-step)
            j=$((i+1))
            ANCHOR_STEP="${!j}"
            ;;
    esac
done

if [ -z "$ANCHOR_STEP" ]; then
    echo "FATAL: --anchor-step <N> is required (got '$ANCHOR_STEP')." >&2
    echo "Pass the s* chosen by scripts/i471_phaseA_analyze.py." >&2
    exit 1
fi
if ! [[ "$ANCHOR_STEP" =~ ^[0-9]+$ ]] || [ "$ANCHOR_STEP" -le 0 ]; then
    echo "FATAL: --anchor-step must be a positive integer, got: $ANCHOR_STEP" >&2
    exit 1
fi

echo "Phase B sweep at max_steps=$ANCHOR_STEP."

# Marker assert at launch.
uv run python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
ids = tok.encode(' ※', add_special_tokens=False)
assert ids == [83399], f'marker token id drift {ids}'
print('marker token id OK: 83399')
"

escalate_and_block() {
    local run="$1"
    local reason="$2"
    local sentinel="/workspace/logs/issue-471-phaseB-failed-${run}.json"
    mkdir -p "$(dirname "$sentinel")"
    echo "FATAL: Phase B run=${run} failed." >&2
    echo "  Reason: ${reason}" >&2
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 471,
    "phase": "phaseB_sweep",
    "run": "$run",
    "anchor_step": $ANCHOR_STEP,
    "failure_class": "code",
    "reason": """$reason""",
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote phaseB-fail sentinel: $sentinel")
EOF
    exit 2
}

train_one_phaseB() {
    local cond="$1"
    local run_name="i471_route_a_${cond}_step${ANCHOR_STEP}"
    local log="$LOG_DIR/train_${run_name}.log"
    echo "=== Phase B train cond=${cond} run_name=${run_name} max_steps=${ANCHOR_STEP} $(date -Iseconds) ==="
    local rc=0
    uv run python scripts/i471_phase23_train.py \
        --cond "$cond" \
        --run-name "$run_name" \
        --lr 5e-6 \
        --epochs 2 \
        --gpu-id 0 \
        --seed 42 \
        --save-steps 10 \
        --log-every 5 \
        --suppress-at-post-response-slot \
        --n-neg-per-persona 100 \
        --max-steps "$ANCHOR_STEP" \
        > "$log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "PhaseB TRAIN FAIL run_name=${run_name} rc=${rc} (see $log)" >&2
        escalate_and_block "$run_name" \
            "phaseB train cond=${cond} exited rc=${rc} (see $log)."
    fi
    echo "=== Phase B train cond=${cond} OK ==="
}

echo "[phase=phaseB_k0] $(date -Iseconds)"
train_one_phaseB cond2_k0
echo "[phase=phaseB_k1] $(date -Iseconds)"
train_one_phaseB cond2_k1
echo "[phase=phaseB_k3] $(date -Iseconds)"
train_one_phaseB cond2_k3

echo "=== Phase B DONE -- 3 cond2_* arms trained at max_steps=${ANCHOR_STEP} $(date -Iseconds) ==="
echo "=== Next: uv run python scripts/i471_phase4_eval.py --free-gen-emission ==="
