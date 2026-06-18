#!/usr/bin/env bash
# Issue #488 Phase 2/3 dispatcher — smoke + sweep through ONE entrypoint.
#
# UNIFIED smoke/sweep architecture per CLAUDE.md `/issue` Step 6d.0 +
# experiment-implementer "Smoke architecture parity":
#   Smoke = sweep with `--conds A1 G2 --seeds 42 --fracs 0.10 0.25 0.50 1.00
#           2.00 3.00` (single seed, two cells, all 6 fracs).
#   Sweep = same dispatcher with `--conds <full-27> --seeds 42 137 --fracs
#           0.10 0.25 0.50 1.00 2.00 3.00` (ALL 6 fracs per cell).
# Same script, same subprocess shape, same env injection, same logging
# surface, same teardown. Smoke does NOT have a separate code path —
# verdict: PASS_UNIFIED.
#
# Plan v3 changes (vs v2): production trains ALL 6 fracs per cell. The
# headline frac is selected POST-HOC by scripts/i488_phase5_analyze.py
# under v3 §6.2.D's ρ-blind picker (lowest eligible frac scanned ascending,
# eligible = tie_mass_off ≤ 0.85 AND median per-source emission_ii ≥ 0.20).
# This dispatcher no longer reads picked_fracs.json from smoke — phase2 no
# longer emits it.
#
# Per CLAUDE.md feedback_cvd_hydra_override (#376): each train process is
# pinned to its own physical GPU by passing --gpu-id <phys_gpu>. sft.py
# clobbers env CVD with str(cfg.gpu_id), so we pass the PHYSICAL index via
# --gpu-id, not via env.
#
# Per the CLAUDE.md subprocess-env-explicit rule: this is a bash dispatcher,
# not a Python subprocess wrapper, so env propagation flows through bash
# variable export and the `uv run python` child inherits the full
# environment — `.env` is read via `load_dotenv()` inside each script.
#
# Phases (each phase MUST emit a `[phase=...]` log line + `[phase=done]`
# at the end so `poll_pipeline.py` can parse status):
#
#   1. (Smoke arm)   train A1 + G2 at all 6 fracs (single seed 42)
#   2. (Smoke arm)   Phase-2 gates: label-mask audit (G1), on-diag log-prob
#                    shift (G2'), off-diag log-prob shift (G3 — v3), EOS-
#                    gradient (G4). No frac picking — production runs all 6.
#   3. (Sweep arm)   train the remaining 25 cells × 2 seeds × ALL 6 fracs,
#                    8-wide parallel (8× H100 inf-70b pod). The 27th
#                    condition (A3 — saturated-anchor control per v3 §9
#                    Phase 3b row) is in ALL_CIDS below so the dispatcher
#                    demonstrably schedules it.
#
# Usage:
#     bash scripts/i488_phase23_dispatch.sh                # full smoke+sweep
#     bash scripts/i488_phase23_dispatch.sh --smoke-only   # phases 1+2 only
#     bash scripts/i488_phase23_dispatch.sh --skip-smoke   # phase 3 only, fixed-fracs default
#     bash scripts/i488_phase23_dispatch.sh --canary-only  # just A1+G2, then exit

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
# Per upload-policy.md delete-after-eval: i488_phase23_train.py runs
# FractionAdapterSaveCallback with an explicit hf_repo=HF_MODEL_REPO and
# does NOT use the EPM_PERSIST_ADAPTER_HF_REPO / _SUBFOLDER env-var pair
# (which trainer.py:_finalize_phase consumes). Setting _HF_REPO here
# without _SUBFOLDER would only matter if a future code path called
# _finalize_phase, and then it would CRASH (trainer.py:492 raises if
# _HF_REPO is set but _SUBFOLDER is not). Not exporting either keeps
# the env hygiene tied to the actual upload path.

LOG_DIR=logs/issue_488
mkdir -p "$LOG_DIR"
SMOKE_LOG_DIR=logs/issue_488/smoke
mkdir -p "$SMOKE_LOG_DIR"

SMOKE_ONLY=0
SKIP_SMOKE=0
CANARY_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --smoke-only) SMOKE_ONLY=1 ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        --canary-only) CANARY_ONLY=1 ;;
        *) ;;
    esac
done

# Marker assert at launch (before any subprocess spawns) per CLAUDE.md.
uv run python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
ids = tok.encode(" ※", add_special_tokens=False)
assert ids == [83399], f"marker token id drift {ids}"
imend = tok.convert_tokens_to_ids("<|im_end|>")
assert imend == 151645, f"<|im_end|> id drift: {imend}"
print(f"marker_id=83399 OK; im_end_id=151645 OK")
PY

# ── Sentinel-and-exit helper (CLAUDE.md "Pod-side code NEVER shells out to
#     scripts/task.py"; the orchestrator's poll_pipeline.py picks up the
#     sentinel and translates it into epm:failure v1 + status:blocked).
escalate_and_block() {
    local phase="$1"
    local reason="$2"
    local sentinel="/workspace/logs/issue-488-${phase}-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    uv run python - <<PY
import json, datetime
payload = {
    "issue": 488,
    "phase": "$phase",
    "failure_class": "code",
    "reason": """$reason""",
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote {payload['phase']} sentinel: $sentinel")
PY
    echo "[phase=failed] $phase: $reason" >&2
    exit 2
}

# ── Smoke arm: train A1+G2 at all 6 fracs, single seed 42 ─────────────────
SMOKE_FRACS="0.10 0.25 0.50 1.00 2.00 3.00"
SMOKE_CONDS=("A1" "G2")

if [ "$SKIP_SMOKE" -eq 0 ]; then
    echo "[phase=smoke_train] Train A1+G2 (smoke = sweep with 1 seed, 2 cells, all 6 fracs) $(date -Iseconds)"
    SMOKE_PIDS=()
    SMOKE_FAILED=()
    cvd=0
    for cond in "${SMOKE_CONDS[@]}"; do
        log="$LOG_DIR/smoke_train_${cond}_cvd${cvd}.log"
        # shellcheck disable=SC2086
        # CVD_PIN_EXEMPT: pre-#578 completed-task dispatcher kept verbatim; new launches must pin env CUDA_VISIBLE_DEVICES per cell (gotchas.md CVD-clobber)
        uv run python scripts/i488_phase23_train.py \
            --conds "$cond" --seeds 42 --gpu-id "$cvd" \
            --fracs $SMOKE_FRACS \
            > "$log" 2>&1 &
        SMOKE_PIDS+=("$!:$cond")
        cvd=$((cvd + 1))
    done
    for entry in "${SMOKE_PIDS[@]}"; do
        pid="${entry%%:*}"
        cond="${entry##*:}"
        if ! wait "$pid"; then
            SMOKE_FAILED+=("$cond")
        fi
    done
    if [ "${#SMOKE_FAILED[@]}" -gt 0 ]; then
        escalate_and_block "smoke_train" \
            "Smoke train failed on conds: ${SMOKE_FAILED[*]}. See $LOG_DIR/smoke_train_*.log"
    fi
    echo "[phase=smoke_train] ok $(date -Iseconds)"

    echo "[phase=smoke_calibrate] Gates: label-mask + in-band + off-diag + EOS-gradient $(date -Iseconds)"
    calibrate_rc=0
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/i488_phase2_smoke_calibrate.py \
        --audit-train-jsonl "data/issue_488/train_rows/i488_A1_seed42.jsonl" \
        --smoke-seed 42 \
        > "$SMOKE_LOG_DIR/calibrate.log" 2>&1 || calibrate_rc=$?
    if [ "$calibrate_rc" -ne 0 ]; then
        # The calibrate script already wrote the sentinel; just echo and exit.
        echo "[phase=failed] smoke_calibrate (rc=$calibrate_rc); see $SMOKE_LOG_DIR/calibrate.log" >&2
        exit "$calibrate_rc"
    fi
    echo "[phase=smoke_calibrate] ok $(date -Iseconds)"

    if [ "$SMOKE_ONLY" -eq 1 ] || [ "$CANARY_ONLY" -eq 1 ]; then
        echo "[phase=done] smoke-only complete $(date -Iseconds)"
        exit 0
    fi
fi

# ── Sweep arm: production = all 27 conds × 2 seeds × ALL 6 fracs (v3) ──
# Plan v3 §9 trains all 6 fracs in production; phase5_analyze picks the
# headline frac post-hoc via the ρ-blind picker. No pre-selection here.
SWEEP_FRACS="0.10 0.25 0.50 1.00 2.00 3.00"
echo "Sweep training all 6 fracs per cell: $SWEEP_FRACS"

# Cell list: all 27 conditions (includes A3 saturated-anchor control per
# v3 §9 Phase 3b row — listed explicitly so a dispatcher dry-run grep
# proves it's scheduled).
ALL_CIDS=(A1 A2 A3 A4 A5 B1 B2 B3 B4 B5 C1 D1 D2 D3 D4 D5 \
          E2 E3 E4 E5 F1 F2 F3 F4 G1 G2 G3)
SEEDS=(42 137)

# Build the (cond, seed) work list. Skip A1 seed=42 and G2 seed=42
# (already trained at smoke; their per-frac adapters exist since
# FractionAdapterSaveCallback saves at every frac it crosses).
WORK=()
for cond in "${ALL_CIDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [ "$SKIP_SMOKE" -eq 0 ] && [ "$seed" -eq 42 ] && { [ "$cond" = "A1" ] || [ "$cond" = "G2" ]; }; then
            continue
        fi
        WORK+=("$cond:$seed")
    done
done

echo "[phase=sweep] $(date -Iseconds) — ${#WORK[@]} (cond,seed) cells across 8 GPUs"

FAILED_FILE="$LOG_DIR/sweep_failed.txt"
: > "$FAILED_FILE"

run_wave() {
    local wave_label="$1"
    shift
    local pairs=("$@")
    echo "=== Sweep wave $wave_label: ${pairs[*]} $(date -Iseconds) ==="
    local pids=()
    local i=0
    for pair in "${pairs[@]}"; do
        local cond="${pair%%:*}"
        local seed="${pair##*:}"
        local cvd="$i"
        local log="$LOG_DIR/train_${cond}_seed${seed}_cvd${cvd}.log"
        # shellcheck disable=SC2086
        # CVD_PIN_EXEMPT: pre-#578 completed-task dispatcher kept verbatim; new launches must pin env CUDA_VISIBLE_DEVICES per cell (gotchas.md CVD-clobber)
        uv run python scripts/i488_phase23_train.py \
            --conds "$cond" --seeds "$seed" --gpu-id "$cvd" \
            --fracs $SWEEP_FRACS \
            > "$log" 2>&1 &
        pids+=("$!:$cond:$seed")
        i=$((i + 1))
    done
    local any_fail=0
    for entry in "${pids[@]}"; do
        local pid; pid=$(echo "$entry" | cut -d: -f1)
        local cond; cond=$(echo "$entry" | cut -d: -f2)
        local seed; seed=$(echo "$entry" | cut -d: -f3)
        if ! wait "$pid"; then
            echo "$cond:$seed" >> "$FAILED_FILE"
            echo "WAVE $wave_label: cond=$cond seed=$seed FAILED (pid=$pid)" >&2
            any_fail=1
        fi
    done
    echo "=== Wave $wave_label complete (any_fail=$any_fail) ==="
}

# Issue waves of up to 8 cells each (1 per GPU).
WAVE_SIZE=8
total=${#WORK[@]}
i=0
wave_num=1
while [ "$i" -lt "$total" ]; do
    end=$((i + WAVE_SIZE))
    if [ "$end" -gt "$total" ]; then end="$total"; fi
    slice=("${WORK[@]:$i:$((end - i))}")
    run_wave "$wave_num" "${slice[@]}"
    i="$end"
    wave_num=$((wave_num + 1))
done

if [ -s "$FAILED_FILE" ]; then
    failed_str=$(tr '\n' ' ' < "$FAILED_FILE")
    escalate_and_block "sweep" "One or more (cond,seed) cells failed: $failed_str"
fi

echo "[phase=sweep] ok $(date -Iseconds)"
echo "[phase=done] phase23 sweep complete $(date -Iseconds)"
