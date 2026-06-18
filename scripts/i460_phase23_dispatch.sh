#!/usr/bin/env bash
# Phase 2/3 — train 16 LoRAs marker-at-end with 4-wide GPU parallelism.
#
# Issue #460 plan v3 §4.5 + §9.1 (round-3 escalation 2026-06-01: 300 rows *
# 5 epochs default after round-2's 30 rows * 1 marker token under-implanted).
#
# Architecturally unified with smoke per CLAUDE.md Step 6d.0 + plan §4.8:
# smoke is THIS dispatcher (or the underlying train script) called with
# --conds A1 (default epochs, default 300 rows) — same recipe as the sweep,
# same subprocess shape and env injection.
#
# Sweep design:
#   1. Smoke gate: A1 with REAL recipe (300 rows * 5 epochs) -> separate-
#      process smoke-check (vLLM-after-HF GPU conflict fix). If implant
#      fraction < 0.80, escalate via sentinel and exit (orchestrator
#      handles re-plan vs descope). Round-3 escalation already applied;
#      if smoke STILL fails, the recipe likely needs a loss-surface
#      change (response context) or much higher lr — surface to user.
#   2. Parallel sweep: 15 conditions across 4 GPUs in 4 waves (A1 skipped
#      because smoke's adapter IS the real-recipe adapter on HF). With
#      --skip-smoke, 16 conds in 4 waves of 4 (A1 included).
#      Waves are issued sequentially via `wait`; each wave's processes
#      run in parallel on GPUs 0/1/2/3.
#
# Per CLAUDE.md feedback_cvd_hydra_override (#376): each train process is
# pinned to its own physical GPU by passing --gpu-id <phys_gpu>. sft.py
# does os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.gpu_id) then loads with
# device_map={"":0}, so the PHYSICAL index must arrive via --gpu-id. Do
# NOT instead set env CUDA_VISIBLE_DEVICES=<phys> + --gpu-id 0 — sft.py
# clobbers that env CVD with str(0), landing ALL cells on physical GPU 0
# (the #460 run-3 OOM). --gpu-id "$cvd" is the only correct knob here.
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

# ── Gate 1: A1 smoke train (REAL recipe) + held-out implant check ────────
#
# ROUND-2 FIX (2026-06-01): the implant check runs as a SEPARATE subprocess
# from the trainer (vLLM-after-HF GPU conflict; CLAUDE.md #399).
#
# ROUND-3 FIX (2026-06-01): smoke train now uses the SAME recipe as the
# sweep (default --epochs 5, default 300 rows via N_DUPES_POS=10). The
# round-2 smoke at --epochs 1 + 30 rows was non-representative — it
# couldn't validate a multi-epoch / 10x-dup recipe and guaranteed under-
# implant (implant_fraction=0.0, mean_logp~base prior). Pre-registered
# escalation per plan §4.2.
#
# After smoke PASSes, Wave-1 SKIPS A1 because the smoke's adapter IS the
# real-recipe adapter (already uploaded to HF). Saves one cell of compute.
if [ "$SKIP_SMOKE" -eq 0 ]; then
    echo "=== Phase 2 smoke step 1/2: A1 train (REAL recipe, default 5 epochs * 300 rows) $(date -Iseconds) ==="
    # Use `|| train_rc=$?` to capture non-zero exits under `set -e`. The bare
    # command would abort the dispatcher before we get a chance to write the
    # smoke-fail sentinel.
    train_rc=0
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/i460_phase23_train.py \
        --conds A1 --gpu-id 0 \
        > "$LOG_DIR/smoke_A1_train.log" 2>&1 || train_rc=$?
    if [ "$train_rc" -ne 0 ]; then
        escalate_and_block A1 \
            "A1 smoke train exited rc=${train_rc} (see $LOG_DIR/smoke_A1_train.log). Trainer failed BEFORE the implant check could run."
    fi

    echo "=== Phase 2 smoke step 2/2: A1 implant check (separate process; fresh vLLM) $(date -Iseconds) ==="
    check_rc=0
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/i460_phase2_smoke_check.py \
        --cond A1 --n-probes 10 \
        > "$LOG_DIR/smoke_A1_check.log" 2>&1 || check_rc=$?
    if [ "$check_rc" -ne 0 ]; then
        # The smoke-check script exits non-zero on implant<threshold; the json
        # is still written, so include both the log and the json in the
        # sentinel reason for the operator.
        escalate_and_block A1 \
            "implant fraction below 0.80 on held-out Q_test under A1 (smoke_A1_check.log + $LOG_DIR/smoke_A1.json). Round-3 escalation (5 epochs * 300 rows) already applied. If genuinely under-implant after this, the marker-at-end + marker-only-loss recipe may need a loss-surface change (response context) or a much higher lr — surface to user. check_rc=${check_rc}."
    fi
    echo "=== Phase 2 smoke A1 PASS — adapter at HF adapters/i460_A1 is the real-recipe adapter; Wave-1 will SKIP A1 ==="
fi

if [ "$SMOKE_ONLY" -eq 1 ]; then
    echo "=== --smoke-only set; done after smoke. ==="
    exit 0
fi

# ── Phase 3 sweep: 16 conditions across 4 GPUs in 4 waves ─────────────────
# Wave-1 OMITS A1 when smoke ran successfully (the smoke adapter is already
# the real-recipe adapter, uploaded to HF). With SKIP_SMOKE=1, Wave-1 keeps
# A1 (no smoke adapter to reuse).
if [ "$SKIP_SMOKE" -eq 1 ]; then
    WAVE_1=(A1 A2 A3 A4)
    WAVE_2=(A5 B1 B2 B3)
    WAVE_3=(B4 B5 C1 D1)
    WAVE_4=(D2 D3 D4 D5)
else
    # Smoke already trained A1 with the real recipe and uploaded the adapter;
    # don't re-train. Promote one cond from Wave-2 into the freed slot.
    WAVE_1=(A2 A3 A4 A5)
    WAVE_2=(B1 B2 B3 B4)
    WAVE_3=(B5 C1 D1 D2)
    WAVE_4=(D3 D4 D5)
fi

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
        # Pass the PHYSICAL gpu index via --gpu-id (NOT --gpu-id 0 + env CVD):
        # sft.py sets os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.gpu_id), which
        # CLOBBERS any env CVD we set here — so env CVD=$cvd + --gpu-id 0 put
        # ALL cells on physical GPU 0 (str(0)) → OOM (#460 run-3 / the
        # feedback_cvd_hydra_override #376 gotcha). Letting sft.py set CVD from
        # --gpu-id "$cvd" pins each cell to its own physical GPU.
        # CVD_PIN_EXEMPT: pre-#578 completed-task dispatcher kept verbatim; new launches must pin env CUDA_VISIBLE_DEVICES per cell (gotchas.md CVD-clobber)
        uv run python scripts/i460_phase23_train.py \
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
