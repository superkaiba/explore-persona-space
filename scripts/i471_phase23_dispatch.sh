#!/usr/bin/env bash
# Phase 2 (smoke gate) + Phase 3 (sweep) -- sequential 4-arm trainer for #471.
#
# Plan v1 §4.7. Smoke == sweep UNIFICATION: this dispatcher runs cond1 ->
# cond2_k0 -> cond2_k1 -> cond2_k3 SEQUENTIALLY through the SAME
# i471_phase23_train.py + i471_phase2_smoke_check.py subprocess pair. cond1
# acts as the smoke canary; subsequent conditions only launch if cond1 PASSes
# the 4-gate smoke (label-mask + loss-decrease + held-out implant + held-out
# NEGATIVE suppression).
#
# Per CLAUDE.md feedback_cvd_hydra_override (#376): each train process is
# pinned to its own physical GPU via --gpu-id (sft.py clobbers env CVD).
#
# Pod-side sentinel write: on cond1 smoke FAIL we write the i460-style
# sentinel for the orchestrator's poll_pipeline.py to surface as
# epm:failure v1.
#
# Usage:
#     bash scripts/i471_phase23_dispatch.sh                   # full sweep
#     bash scripts/i471_phase23_dispatch.sh --smoke-only      # cond1 only
#     bash scripts/i471_phase23_dispatch.sh --skip-smoke      # debug
#     bash scripts/i471_phase23_dispatch.sh --conds cond1 cond2_k1

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

LOG_DIR=logs/issue_471
mkdir -p "$LOG_DIR"

SMOKE_ONLY=0
SKIP_SMOKE=0
CONDS=(cond1 cond2_k0 cond2_k1 cond2_k3)
parsing_conds=0
override_conds=()
for arg in "$@"; do
    if [ "$parsing_conds" -eq 1 ]; then
        case "$arg" in
            --*) parsing_conds=0 ;;
            *) override_conds+=("$arg"); continue ;;
        esac
    fi
    case "$arg" in
        --smoke-only) SMOKE_ONLY=1 ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        --conds) parsing_conds=1 ;;
        *) ;;
    esac
done
if [ "${#override_conds[@]}" -gt 0 ]; then
    CONDS=("${override_conds[@]}")
fi

# Marker assert at launch (CLAUDE.md: every dispatcher checks marker id).
uv run python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
ids = tok.encode(' ※', add_special_tokens=False)
assert ids == [83399], f'marker token id drift {ids}'
print('marker token id OK: 83399')
"

escalate_and_block() {
    local cond="$1"
    local reason="$2"
    local sentinel="/workspace/logs/issue-471-smoke-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    echo "FATAL: smoke gate failed on cond=${cond}." >&2
    echo "  Reason: ${reason}" >&2
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 471,
    "phase": "phase2_smoke",
    "failure_class": "code",
    "condition": "$cond",
    "reason": """$reason""",
    "policy": "Plan v1 §4.5 4-gate smoke; cond1 is the canary.",
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

train_one() {
    local cond="$1"
    local log="$LOG_DIR/train_${cond}.log"
    echo "=== train cond=${cond} $(date -Iseconds) ==="
    local rc=0
    uv run python scripts/i471_phase23_train.py --cond "$cond" --gpu-id 0 \
        > "$log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "TRAIN FAIL cond=${cond} rc=${rc} (see $log)" >&2
    fi
    return "$rc"
}

smoke_one() {
    local cond="$1"
    local log="$LOG_DIR/smoke_${cond}.log"
    echo "=== smoke-check cond=${cond} $(date -Iseconds) ==="
    local rc=0
    uv run python scripts/i471_phase2_smoke_check.py --cond "$cond" --n-probes 10 \
        --train-log "$LOG_DIR/train_${cond}.log" \
        > "$log" 2>&1 || rc=$?
    return "$rc"
}

if [ "$SKIP_SMOKE" -eq 0 ]; then
    if [[ ! " ${CONDS[*]} " =~ " cond1 " ]]; then
        echo "WARN: --conds excludes cond1; running smoke on first listed cond instead." >&2
        SMOKE_COND="${CONDS[0]}"
    else
        SMOKE_COND=cond1
    fi
    train_rc=0
    train_one "$SMOKE_COND" || train_rc=$?
    if [ "$train_rc" -ne 0 ]; then
        escalate_and_block "$SMOKE_COND" \
            "smoke train exited rc=${train_rc} (see $LOG_DIR/train_${SMOKE_COND}.log)."
    fi
    smoke_rc=0
    smoke_one "$SMOKE_COND" || smoke_rc=$?
    if [ "$smoke_rc" -ne 0 ]; then
        escalate_and_block "$SMOKE_COND" \
            "smoke 4-gate FAIL ($LOG_DIR/smoke_${SMOKE_COND}.log + smoke_${SMOKE_COND}.json). Plan §4.5 gate fired: label-mask | loss-decrease | held-out implant >= 80% | held-out negative suppression <= 30%."
    fi
    echo "=== smoke ${SMOKE_COND} PASS ==="
fi

if [ "$SMOKE_ONLY" -eq 1 ]; then
    echo "=== --smoke-only set; done. ==="
    exit 0
fi

FAILED_FILE="$LOG_DIR/sweep_failed.txt"
: > "$FAILED_FILE"

for cond in "${CONDS[@]}"; do
    if [ "$SKIP_SMOKE" -eq 0 ] && [ "$cond" = "$SMOKE_COND" ]; then
        echo "=== skip ${cond} (already trained via smoke step) ==="
        continue
    fi
    train_rc=0
    train_one "$cond" || train_rc=$?
    if [ "$train_rc" -ne 0 ]; then
        echo "$cond" >> "$FAILED_FILE"
        continue
    fi
    smoke_rc=0
    smoke_one "$cond" || smoke_rc=$?
    if [ "$smoke_rc" -ne 0 ]; then
        echo "WARN: cond=${cond} smoke gate FAIL (continuing; endpoint Phase 4 will report)." >&2
    fi
done

if [ -s "$FAILED_FILE" ]; then
    FAILED=$(tr '\n' ' ' < "$FAILED_FILE")
    sentinel="/workspace/logs/issue-471-sweep-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 471,
    "phase": "phase3_sweep",
    "failure_class": "code",
    "failed_conds": "$FAILED".split(),
    "reason": "One or more conditions in the 4-arm #471 sweep failed train_lora.",
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

echo "=== Phase 3 sweep DONE -- all 4 conditions trained $(date -Iseconds) ==="
echo "=== Next: uv run python scripts/i471_phase4_eval.py ==="
