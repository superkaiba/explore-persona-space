#!/usr/bin/env bash
# Plan v3 §4.2 -- Phase A: run BOTH cond1 variants sequentially on 1× H100.
#
#   Run 1: cond1_withneg  -- villain system + 1:1 contrastive negatives.
#                            The headline arm.
#   Run 2: cond1_posonly  -- villain system + 300 positives ONLY,
#                            zero negatives. The v3 disentanglement control.
#
# Both share: lr=5e-6, 2 epochs (ceiling), save_steps=10, log_every=5,
# suppress_at_post_response_slot=True, seed=42, gpu_id=0. Phase A IS the
# smoke phase by construction (the smoke == sweep unification — Phase B
# re-uses the same i471_phase23_train.py entrypoint with --max-steps=s*).
#
# Failure semantics: pod-side sentinel written for poll_pipeline.py on
# any train rc != 0. Per CLAUDE.md, pod-side code never shells out to
# scripts/task.py.
#
# Usage (from pod):
#   nohup bash scripts/i471_phaseA_anchor.sh \
#       > /workspace/logs/issue-471-phaseA.log 2>&1 &

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

LOG_DIR=logs/issue_471
mkdir -p "$LOG_DIR"

# Marker assert at launch (CLAUDE.md dispatcher contract).
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
    local sentinel="/workspace/logs/issue-471-phaseA-failed-${run}.json"
    mkdir -p "$(dirname "$sentinel")"
    echo "FATAL: Phase A run=${run} failed." >&2
    echo "  Reason: ${reason}" >&2
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 471,
    "phase": "phaseA_anchor",
    "run": "$run",
    "failure_class": "code",
    "reason": """$reason""",
    "policy": "Plan v3 §4.2 Phase A (BOTH cond1 variants). Either run failing blocks downstream.",
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote phaseA-fail sentinel: $sentinel")
EOF
    exit 2
}

train_one_phaseA() {
    local run_name="$1"
    local n_neg="$2"
    local log="$LOG_DIR/train_${run_name}.log"
    echo "=== Phase A train run_name=${run_name} n_neg_per_persona=${n_neg} $(date -Iseconds) ==="
    local rc=0
    uv run python scripts/i471_phase23_train.py \
        --cond cond1 \
        --run-name "$run_name" \
        --lr 5e-6 \
        --epochs 2 \
        --gpu-id 0 \
        --seed 42 \
        --save-steps 10 \
        --log-every 5 \
        --suppress-at-post-response-slot \
        --n-neg-per-persona "$n_neg" \
        > "$log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "PhaseA TRAIN FAIL run_name=${run_name} rc=${rc} (see $log)" >&2
        escalate_and_block "$run_name" \
            "phaseA train exited rc=${rc} (see $log). Plan v3 §4.2 requires BOTH cond1 variants for the H_disentangle headline."
    fi
    echo "=== Phase A train run_name=${run_name} OK ==="
}

# Run 1: cond1_withneg (the headline contrastive-negatives arm).
echo "[phase=phaseA_withneg] $(date -Iseconds)"
train_one_phaseA "i471_route_a_cond1_withneg" 100

# Run 2: cond1_posonly (the disentanglement control arm).
echo "[phase=phaseA_posonly] $(date -Iseconds)"
train_one_phaseA "i471_route_a_cond1_posonly" 0

echo "=== Phase A DONE -- both cond1 variants trained $(date -Iseconds) ==="
echo "=== Next: uv run python scripts/i471_phaseA_analyze.py ==="
