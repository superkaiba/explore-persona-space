#!/usr/bin/env bash
# Phase 2 (smoke gate) + Phase 3 (sweep) — sequential 4-arm trainer for #465.
#
# Plan v2 §4.10 + §9.1. Smoke == sweep architecture (UNIFIED): this dispatcher
# runs cond1 → cond2_k0 → cond2_k1 → cond2_k3 SEQUENTIALLY through the SAME
# i465_phase23_train.py + i465_phase2_smoke_check.py subprocess pair. cond1
# acts as the smoke canary; subsequent conditions only launch if cond1
# passes the implant gate.
#
# Per CLAUDE.md feedback_cvd_hydra_override (#376): each train process is
# pinned to its own physical GPU via --gpu-id (sft.py clobbers env CVD).
# We sequentialize because 1× H100 is the planned compute (plan §9.1) and
# the simplicity wins out over parallelism savings on 4 small conds.
#
# CLAUDE.md feedback_dispatcher_silent_death_hardening: per-cond log line +
# trap on the smoke gate; cond2_* failures are NOT load-bearing for the
# smoke headline — they continue to a "did not implant" data point.
#
# Pod-side sentinel write: on cond1 smoke FAIL we write the i460-style
# sentinel for the orchestrator's poll_pipeline.py to surface as
# epm:failure v1. cond2_k1 smoke fail triggers the pre-registered
# escalation (drop k=1, rely on k=3) per plan §7.
#
# Usage:
#     bash scripts/i465_phase23_dispatch.sh                   # full sweep
#     bash scripts/i465_phase23_dispatch.sh --smoke-only      # cond1 only
#     bash scripts/i465_phase23_dispatch.sh --skip-smoke      # debug
#     bash scripts/i465_phase23_dispatch.sh --conds cond1 cond2_k1
#

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

LOG_DIR=logs/issue_465
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
    local sentinel="/workspace/logs/issue-465-smoke-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    echo "FATAL: smoke gate failed on cond=${cond}." >&2
    echo "  Reason: ${reason}" >&2
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 465,
    "phase": "phase2_smoke",
    "failure_class": "code",
    "condition": "$cond",
    "reason": """$reason""",
    "policy": "marker-only-loss + on-policy-R implant must clear 80% argmax at slot L on Q_test before sweep launches (plan v2 §4.6).",
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
    uv run python scripts/i465_phase23_train.py --cond "$cond" --gpu-id 0 \
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
    uv run python scripts/i465_phase2_smoke_check.py --cond "$cond" --n-probes 10 \
        > "$log" 2>&1 || rc=$?
    return "$rc"
}

# ── Smoke gate on cond1 (the recipe sanity canary) ──────────────────────
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
            "smoke implant fraction below 0.80 ($LOG_DIR/smoke_${SMOKE_COND}.log + smoke_${SMOKE_COND}.json). Recipe inherited from #460 round-3; if this fails, the loss-surface may need a response-context change."
    fi
    echo "=== smoke ${SMOKE_COND} PASS — adapter at HF adapters/i465_${SMOKE_COND} ==="
fi

if [ "$SMOKE_ONLY" -eq 1 ]; then
    echo "=== --smoke-only set; done. ==="
    exit 0
fi

# ── Sequential sweep over remaining conditions ──────────────────────────
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
        echo "WARN: cond=${cond} smoke implant fraction < 0.80 (continuing — cond2_* is informative even when implant is weak; trajectory + endpoint will be reported)." >&2
    fi
done

if [ -s "$FAILED_FILE" ]; then
    FAILED=$(tr '\n' ' ' < "$FAILED_FILE")
    sentinel="/workspace/logs/issue-465-sweep-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 465,
    "phase": "phase3_sweep",
    "failure_class": "code",
    "failed_conds": "$FAILED".split(),
    "reason": "One or more conditions in the 4-arm sweep failed train_lora.",
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

echo "=== Phase 3 sweep DONE — all 4 conditions trained $(date -Iseconds) ==="
echo "=== Next: bash scripts/i465_phase4_dispatch.sh (or uv run python scripts/i465_phase4_eval.py) ==="
