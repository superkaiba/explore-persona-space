#!/usr/bin/env bash
# Issue #489 Phase 2/3 dispatcher — 8-wide CVD-shard waves over 24 union contexts.
#
# Plan v5 §4.5 + §4.6 + §9.
#
# Smoke == sweep with --conds IK01 IK13 SP01 SP04 (4 cells, one wave). Same
# subprocess shape, same env injection, same logging surface — PASS_UNIFIED
# per CLAUDE.md smoke-architecture-check.
#
# Per CLAUDE.md feedback_cvd_hydra_override: each train process is pinned to its
# own physical GPU by passing --gpu-id <phys_gpu>. NEVER rely on env CVD alone.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

LOG_DIR=logs/issue_489
mkdir -p "$LOG_DIR"

SMOKE_ONLY=0
SKIP_SMOKE=0
RESUME=0
SEED=42

# Plan §4.5: 4 smoke cells; the rest of the union is sweep.
SMOKE_CONDS=(IK01 IK13 SP01 SP04)
WAVE_1=(IK01 IK02 IK03 IK04 IK05 IK06 IK07 IK08)
WAVE_2=(IK09 IK10 IK11 IK12 IK13 IK14 IK15 IK16)
WAVE_3=(SP01 SP02 SP03 SP04 SP05 SP06 SP07 SP08)

for arg in "$@"; do
    case "$arg" in
        --smoke-only) SMOKE_ONLY=1 ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        --resume) RESUME=1 ;;
        --seed=*) SEED="${arg#*=}" ;;
        *) ;;
    esac
done

echo "[phase=preflight] === i489 phase23 dispatcher $(date -Iseconds) seed=$SEED smoke_only=$SMOKE_ONLY skip_smoke=$SKIP_SMOKE resume=$RESUME ==="

# Marker + im_end assert at launch (plan §4.6).
uv run python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
ids = tok.encode(' ※', add_special_tokens=False)
assert ids == [83399], f'marker token id drift {ids}'
print('marker token id OK: 83399')
im_end = tok.convert_tokens_to_ids('<|im_end|>')
assert im_end == 151645, f'<|im_end|> id drift {im_end}'
print('<|im_end|> id OK: 151645')
"

escalate_and_block() {
    local cond="$1"
    local reason="$2"
    local epoch
    epoch="$(date +%s)"
    local sentinel="/workspace/logs/issue-489-epm_failure-${epoch}.json"
    mkdir -p "$(dirname "$sentinel")"
    echo "[phase=failed] FATAL: cond=${cond} reason=${reason}" >&2
    uv run python - <<EOF
import json, datetime
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
    "issue": 489,
    "phase": "phase23_train",
    "failure_class": "code",
    "condition": "${cond}",
    "reason": """${reason}""",
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote sentinel: $sentinel")
EOF
    exit 2
}

FAILED_FILE="$LOG_DIR/sweep_failed.txt"
: > "$FAILED_FILE"

run_wave() {
    local wave_label="$1"
    shift
    local conds=("$@")
    echo "[phase=sweep_wave_${wave_label}] === Wave ${wave_label}: ${conds[*]} $(date -Iseconds) ==="
    local pids=()
    local i=0
    for cond in "${conds[@]}"; do
        local cvd="$i"
        local log="$LOG_DIR/train_${cond}_seed${SEED}_cvd${cvd}.log"
        # Per CLAUDE.md feedback_cvd_hydra_override: --gpu-id $cvd, NOT env CVD alone.
        uv run python scripts/i489_phase23_train.py \
            --conds "$cond" --gpu-id "$cvd" --seed "$SEED" \
            > "$log" 2>&1 &
        pids+=("$!:${cond}")
        i=$((i + 1))
    done
    for entry in "${pids[@]}"; do
        local pid; pid="$(cut -d: -f1 <<<"$entry")"
        local cond; cond="$(cut -d: -f2 <<<"$entry")"
        if ! wait "$pid"; then
            echo "${cond}" >> "$FAILED_FILE"
            echo "WAVE $wave_label: cond=$cond FAILED (pid=$pid)" >&2
        fi
    done
    echo "[phase=sweep_wave_${wave_label}_done] === Wave ${wave_label} complete ==="
}

# Smoke == sweep with 4 cells (one wave).
if [ "$SKIP_SMOKE" -eq 0 ]; then
    echo "[phase=smoke] === smoke wave ${SMOKE_CONDS[*]} $(date -Iseconds) ==="
    run_wave "smoke" "${SMOKE_CONDS[@]}"
    if [ -s "$FAILED_FILE" ]; then
        FAILED=$(tr '\n' ' ' < "$FAILED_FILE")
        escalate_and_block "smoke" "smoke train failed for: $FAILED"
    fi
fi

if [ "$SMOKE_ONLY" -eq 1 ]; then
    echo "[phase=smoke_only_done] === --smoke-only set; exit after smoke. ==="
    exit 0
fi

# Sweep waves — when --skip-smoke is set, run all 24; else skip the 4 already
# trained in smoke (Wave 1 still runs but skips repeats).
echo "[phase=sweep_start] === Sweep 24 union contexts in 3 waves $(date -Iseconds) ==="
run_wave "1" "${WAVE_1[@]}"
run_wave "2" "${WAVE_2[@]}"
run_wave "3" "${WAVE_3[@]}"

if [ -s "$FAILED_FILE" ]; then
    FAILED=$(tr '\n' ' ' < "$FAILED_FILE")
    escalate_and_block "$FAILED" "sweep had failures: $FAILED"
fi

echo "[phase=sweep_done] === Phase 2/3 sweep all 24 cells trained $(date -Iseconds) ==="
