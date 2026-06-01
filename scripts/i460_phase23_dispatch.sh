#!/usr/bin/env bash
# Phase 2/3 — train 16 LoRAs marker-at-end with 4-wide GPU parallelism.
#
# Issue #460 plan v3 §4.5 + §9.1.
#
# Architecturally unified with smoke per CLAUDE.md Step 6d.0 + plan §4.8:
# smoke is THIS dispatcher (or the underlying train script) called with
# --conds A1 --epochs 1, same subprocess shape and env injection as the
# full sweep.
#
# Sweep design:
#   1. Smoke gate: A1 at 1 epoch + smoke-eval implant check. If implant
#      fraction < 0.80, escalate via sentinel and exit (orchestrator
#      handles re-plan vs descope).
#   2. Parallel sweep: 16 conditions across 4 GPUs in 4 waves of 4.
#      Waves are issued sequentially via `wait`; each wave's 4 train
#      processes run in parallel on GPUs 0/1/2/3.
#
# Per CLAUDE.md feedback_cvd_hydra_override (#376): each train process
# uses env CUDA_VISIBLE_DEVICES=<phys_gpu> set BEFORE spawn AND
# --gpu-id 0 (env CVD remaps the visible GPU to local device 0).
#
# Usage:
#     bash scripts/i460_phase23_dispatch.sh                 # full sweep
#     bash scripts/i460_phase23_dispatch.sh --smoke-only    # just A1 smoke
#     bash scripts/i460_phase23_dispatch.sh --skip-smoke    # skip smoke (debug)

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
# Avoid #399's MooseFS quota path during the 16-condition sequence.
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

LOG_DIR=logs/issue_460
mkdir -p "$LOG_DIR"

SMOKE_ONLY=0
SKIP_SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --smoke-only) SMOKE_ONLY=1 ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        *) ;;
    esac
done

# Marker assert at launch before any subprocess spawns (per CLAUDE.md).
uv run python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
ids = tok.encode(' ※', add_special_tokens=False)
assert ids == [83399], f'marker token id drift {ids}'
print('marker token id OK: 83399')
"

# ── Helper: write a sentinel and exit when smoke fails (pod-side
# escalation pattern per CLAUDE.md "Pod-side code NEVER shells out to
# scripts/task.py"). The orchestrator's poll_pipeline.py picks up the
# sentinel and translates it into epm:failure v1 + status:blocked.
escalate_and_block() {
    local cond="$1"
    local reason="$2"
    local sentinel="/workspace/logs/issue-460-smoke-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    echo "FATAL: smoke gate failed on cond=${cond}." >&2
    echo "  Reason: ${reason}" >&2
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 460,
    "phase": "phase2_smoke",
    "failure_class": "code",
    "condition": "$cond",
    "reason": """$reason""",
    "policy": "marker-only-loss + on-policy-R implant must clear 80% argmax at slot L on Q_test before sweep launches (plan §4.4 Gate c).",
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

# ── Gate 1: A1 smoke train + held-out implant check ──────────────────────
if [ "$SKIP_SMOKE" -eq 0 ]; then
    echo "=== Phase 2 smoke: A1 (1 epoch + 10-probe implant check) $(date -Iseconds) ==="
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/i460_phase23_train.py \
        --conds A1 --epochs 1 --gpu-id 0 --smoke-eval --smoke-n-probes 10 \
        > "$LOG_DIR/smoke_A1.log" 2>&1
    smoke_pass=$(uv run python -c "
import json
d = json.load(open('$LOG_DIR/smoke_A1.json'))
print('1' if d['pass'] else '0')
")
    if [ "$smoke_pass" != "1" ]; then
        escalate_and_block A1 \
            "implant fraction below 0.80 on held-out Q_test under A1 (see $LOG_DIR/smoke_A1.json). Pre-registered escalation per plan §4.2: bump to 5 epochs OR 10x dup of positives."
    fi
    echo "=== Phase 2 smoke A1 PASS ==="
fi

if [ "$SMOKE_ONLY" -eq 1 ]; then
    echo "=== --smoke-only set; done after smoke. ==="
    exit 0
fi

# ── Phase 3 sweep: 16 conditions across 4 GPUs in 4 waves ─────────────────
# Wave-1 includes A1 again at full 3 epochs (smoke used 1 epoch only).
WAVE_1=(A1 A2 A3 A4)
WAVE_2=(A5 B1 B2 B3)
WAVE_3=(B4 B5 C1 D1)
WAVE_4=(D2 D3 D4 D5)
ALL_WAVES=(WAVE_1 WAVE_2 WAVE_3 WAVE_4)

FAILED_FILE="$LOG_DIR/sweep_failed.txt"
: > "$FAILED_FILE"

run_wave() {
    local wave_label="$1"
    shift
    local conds=("$@")
    echo "=== Sweep wave $wave_label: ${conds[*]} $(date -Iseconds) ==="
    local pids=()
    local i=0
    for cond in "${conds[@]}"; do
        local cvd="$i"
        local log="$LOG_DIR/train_${cond}_cvd${cvd}.log"
        CUDA_VISIBLE_DEVICES="$cvd" uv run python scripts/i460_phase23_train.py \
            --conds "$cond" --epochs 3 --gpu-id 0 \
            > "$log" 2>&1 &
        pids+=("$!:$cond")
        i=$((i + 1))
    done
    local any_fail=0
    for entry in "${pids[@]}"; do
        local pid="${entry%%:*}"
        local cond="${entry##*:}"
        if ! wait "$pid"; then
            echo "$cond" >> "$FAILED_FILE"
            echo "WAVE $wave_label: cond=$cond FAILED (pid=$pid)" >&2
            any_fail=1
        fi
    done
    echo "=== Wave $wave_label complete (any_fail=$any_fail) ==="
}

run_wave "1" "${WAVE_1[@]}"
run_wave "2" "${WAVE_2[@]}"
run_wave "3" "${WAVE_3[@]}"
run_wave "4" "${WAVE_4[@]}"

if [ -s "$FAILED_FILE" ]; then
    FAILED=$(tr '\n' ' ' < "$FAILED_FILE")
    sentinel="/workspace/logs/issue-460-sweep-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 460,
    "phase": "phase3_sweep",
    "failure_class": "code",
    "failed_conds": "$FAILED".split(),
    "reason": "One or more conditions in the 16-LoRA sweep failed train_lora.",
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

echo "=== Phase 3 sweep ALL 16 conditions trained $(date -Iseconds) ==="
