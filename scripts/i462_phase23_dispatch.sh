#!/usr/bin/env bash
# Phase 2/3 (#462) — train 16 LoRAs marker-at-end with per-epoch snapshots,
# 4-wide GPU-parallel waves.
#
# Issue #462 epoch-resolved on-policy marker transfer. Mirrors
# i460_phase23_dispatch.sh structurally; the ONLY behavioral change is
# that the underlying train script (i462_phase23_train.py) saves adapter
# snapshots at the END of epochs {1, 2, 3, 5} per condition — 4 adapters
# per cell, 16 cells → 64 adapters total on HF Hub at
#     adapters/i462_<cond>_ep{N}/
#
# GPU pinning (PRESERVED FROM #460 run-3 OOM FIX, do not regress):
# pass the PHYSICAL gpu index via --gpu-id <phys>. sft.py sets
# os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id) then loads with
# device_map={"":0}, so the PHYSICAL index must arrive via --gpu-id.
# Do NOT instead set env CUDA_VISIBLE_DEVICES=<phys> + --gpu-id 0 —
# sft.py clobbers env CVD with str(0) and lands ALL parallel cells on
# physical GPU 0 → OOM (the #460 run-3 incident; CLAUDE.md
# feedback_cvd_hydra_override).
#
# Smoke gating: A1 smoke is OPTIONAL for #462 (we KNOW #460 implants
# strongly at 5ep × 300 rows). When --skip-smoke is omitted, the smoke
# gate runs on the epoch-5 snapshot of A1, mirroring #460's Gate (c).
# A1 always re-trains during the wave (no Wave-1 skip) because each
# condition needs ALL FOUR per-epoch snapshots in a SINGLE training run
# (the epoch trajectory must be unbroken — no resume-then-warmstart).
#
# Usage:
#     bash scripts/i462_phase23_dispatch.sh                 # full sweep + smoke
#     bash scripts/i462_phase23_dispatch.sh --smoke-only    # just A1 smoke
#     bash scripts/i462_phase23_dispatch.sh --skip-smoke    # skip smoke

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
# Avoid MooseFS quota path during the 16-condition × 4-checkpoint sequence
# (64 adapter writes — keep inline checkpoint uploads off).
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

LOG_DIR=logs/issue_462
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

# Pod-side escalation pattern (CLAUDE.md "Pod-side code NEVER shells out
# to scripts/task.py"): write a sentinel file the orchestrator picks up.
escalate_and_block() {
    local cond="$1"
    local reason="$2"
    local sentinel="/workspace/logs/issue-462-smoke-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    echo "FATAL: smoke gate failed on cond=${cond}." >&2
    echo "  Reason: ${reason}" >&2
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 462,
    "phase": "phase2_smoke",
    "failure_class": "code",
    "condition": "$cond",
    "reason": """$reason""",
    "policy": "ep5 smoke must clear 80% argmax at slot L on Q_test before sweep launches.",
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

# ── Gate 1: A1 smoke train (4 ckpts, real recipe) + ep5 implant check ──
#
# Unlike #460, we DON'T skip A1 in the wave after smoke — the per-epoch
# trajectory must be unbroken (epochs 1/2/3/5 captured in a single
# train, no warm-restart). So the smoke trains A1 ONCE (4 adapters), the
# wave then re-trains A1 (4 fresh adapters that supersede the smoke set).
# Cost: ~10 min extra (one extra A1 train). Cleanliness benefit:
# trajectory consistency across all 16 conditions.
if [ "$SKIP_SMOKE" -eq 0 ]; then
    echo "=== Phase 2 smoke step 1/2: A1 train (4 ckpts, real recipe) $(date -Iseconds) ==="
    train_rc=0
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/i462_phase23_train.py \
        --conds A1 --gpu-id 0 \
        > "$LOG_DIR/smoke_A1_train.log" 2>&1 || train_rc=$?
    if [ "$train_rc" -ne 0 ]; then
        escalate_and_block A1 \
            "A1 smoke train exited rc=${train_rc} (see $LOG_DIR/smoke_A1_train.log). Trainer failed BEFORE the implant check could run."
    fi

    echo "=== Phase 2 smoke step 2/2: A1 ep5 implant check (fresh vLLM) $(date -Iseconds) ==="
    check_rc=0
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/i462_phase2_smoke_check.py \
        --cond A1 --epoch 5 --n-probes 10 \
        > "$LOG_DIR/smoke_A1_ep5_check.log" 2>&1 || check_rc=$?
    if [ "$check_rc" -ne 0 ]; then
        escalate_and_block A1 \
            "ep5 implant fraction below 0.80 on held-out Q_test under A1 (smoke_A1_ep5_check.log + $LOG_DIR/smoke_A1_ep5.json). The marker-at-end recipe matched #460 byte-for-byte; if smoke fails, R has drifted from #460 or the per-epoch callback didn't write ep5. check_rc=${check_rc}."
    fi
    echo "=== Phase 2 smoke A1 ep5 PASS ==="
fi

if [ "$SMOKE_ONLY" -eq 1 ]; then
    echo "=== --smoke-only set; done after smoke. ==="
    exit 0
fi

# ── Phase 3 sweep: 16 conditions across 4 GPUs in 4 waves ────────────
# All conds (A1 included — see comment above on trajectory consistency).
WAVE_1=(A1 A2 A3 A4)
WAVE_2=(A5 B1 B2 B3)
WAVE_3=(B4 B5 C1 D1)
WAVE_4=(D2 D3 D4 D5)

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
        # PHYSICAL gpu index via --gpu-id (PRESERVED FROM #460 FIX):
        # sft.py sets os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.gpu_id),
        # which CLOBBERS any env CVD set here. env CVD=$cvd + --gpu-id 0
        # would put ALL cells on physical GPU 0 → OOM (#460 run-3 / the
        # feedback_cvd_hydra_override #376 gotcha). Letting sft.py set CVD
        # from --gpu-id "$cvd" pins each cell to its own physical GPU.
        uv run python scripts/i462_phase23_train.py \
            --conds "$cond" --gpu-id "$cvd" \
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
    sentinel="/workspace/logs/issue-462-sweep-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 462,
    "phase": "phase3_sweep",
    "failure_class": "code",
    "failed_conds": "$FAILED".split(),
    "reason": "One or more conditions in the 16-LoRA × 4-epoch sweep failed train_lora.",
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

echo "=== Phase 3 sweep ALL 16 conditions × 4 epochs trained $(date -Iseconds) ==="
