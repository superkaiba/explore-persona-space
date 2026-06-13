#!/usr/bin/env bash
# Phase 2/3 dispatcher for #474. Runs both arms (pos + loc) sequentially,
# 4-wide GPU parallelism within each arm.
#
# Issue #474 plan v3 §4.5. Architecturally unified with smoke per plan
# v3 §4.10: smoke == this dispatcher invoked with --smoke-only --arm pos
# (or --arm loc), which forwards `--conds A1` to i474_phase23_train.py.
# Same subprocess shape and env injection as the sweep waves.
#
# Per-arm wave plan on a 4× H100 pod: 4 waves of 4 conditions each.
# (Smoke trains A1 with the REAL recipe + uploads to HF; Wave-1 then
# SKIPS A1 to save one cell of compute. With --skip-smoke, A1 stays in
# Wave-1.)
#
# Per CLAUDE.md feedback_cvd_hydra_override: each train process is
# pinned to its own physical GPU by passing --gpu-id <phys_gpu>.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

LOG_DIR=logs/issue_474
mkdir -p "$LOG_DIR"

SMOKE_ONLY=0
SKIP_SMOKE=0
RESUME=0
SELECTED_ARMS=("pos" "loc")
for arg in "$@"; do
    case "$arg" in
        --smoke-only) SMOKE_ONLY=1 ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        --resume) RESUME=1 ;;
        --arm=pos) SELECTED_ARMS=("pos") ;;
        --arm=loc) SELECTED_ARMS=("loc") ;;
        *) ;;
    esac
done

echo "[phase=preflight] === i474 phase23 dispatcher $(date -Iseconds) arms=${SELECTED_ARMS[*]} smoke_only=$SMOKE_ONLY skip_smoke=$SKIP_SMOKE resume=$RESUME ==="

# Marker assert at launch.
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

# Sentinel helper — pod-side escalation per CLAUDE.md "pod-side never
# shells out to scripts/task.py".
escalate_and_block() {
    local arm="$1"
    local cond="$2"
    local reason="$3"
    local sentinel="/workspace/logs/issue-474-smoke-failed-${arm}.json"
    mkdir -p "$(dirname "$sentinel")"
    echo "[phase=smoke_failed] FATAL: smoke gate failed on arm=${arm} cond=${cond}." >&2
    echo "  Reason: ${reason}" >&2
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 474,
    "phase": "phase2_smoke",
    "arm": "${arm}",
    "failure_class": "code",
    "condition": "${cond}",
    "reason": """${reason}""",
    "policy": "A_loc smoke gate (plan §4.4): diagonal implant >= 0.80 AND bystander on-policy ※ emission < 0.30 at the post-response slot. A_pos smoke gate: diagonal implant >= 0.80.",
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

# Smoke for each arm (A1, real recipe, then bystander check on C1 for A_loc).
if [ "$SKIP_SMOKE" -eq 0 ]; then
    for arm in "${SELECTED_ARMS[@]}"; do
        echo "[phase=smoke_${arm}_train] === A1 smoke train (REAL recipe, 5 epochs * $([[ "$arm" == "loc" ]] && echo "600" || echo "300") rows) arm=${arm} $(date -Iseconds) ==="
        train_rc=0
        CUDA_VISIBLE_DEVICES=0 uv run python scripts/i474_phase23_train.py \
            --arm "$arm" --conds A1 --gpu-id 0 \
            > "$LOG_DIR/smoke_${arm}_A1_train.log" 2>&1 || train_rc=$?
        if [ "$train_rc" -ne 0 ]; then
            escalate_and_block "$arm" A1 \
                "A1 smoke train exited rc=${train_rc} (see $LOG_DIR/smoke_${arm}_A1_train.log)."
        fi

        # Smoke implant check (separate subprocess for vLLM-after-HF fix).
        # For A_loc we test the FINAL epoch's adapter (ep5) — saves explicit
        # adapter version match the bare adapter folder.
        echo "[phase=smoke_${arm}_check] === A1 smoke check (separate process; fresh vLLM) arm=${arm} $(date -Iseconds) ==="
        check_rc=0
        if [ "$arm" == "loc" ]; then
            CUDA_VISIBLE_DEVICES=0 uv run python scripts/i474_phase2_smoke_check.py \
                --arm loc --cond A1 --bystander-cond C1 --epoch 5 --n-probes 10 \
                > "$LOG_DIR/smoke_${arm}_A1_check.log" 2>&1 || check_rc=$?
        else
            CUDA_VISIBLE_DEVICES=0 uv run python scripts/i474_phase2_smoke_check.py \
                --arm pos --cond A1 --epoch 5 --n-probes 10 \
                > "$LOG_DIR/smoke_${arm}_A1_check.log" 2>&1 || check_rc=$?
        fi
        if [ "$check_rc" -ne 0 ]; then
            escalate_and_block "$arm" A1 \
                "smoke gate FAILED on A1 ($LOG_DIR/smoke_${arm}_A1_check.log + $LOG_DIR/smoke_${arm}_A1.json). check_rc=${check_rc}."
        fi
        echo "[phase=smoke_${arm}_pass] === A1 smoke arm=${arm} PASS ==="
    done
fi

if [ "$SMOKE_ONLY" -eq 1 ]; then
    echo "[phase=smoke_only_done] === --smoke-only set; exit after smoke. ==="
    exit 0
fi

# Sweep waves — same plan as #460, per arm.
if [ "$SKIP_SMOKE" -eq 1 ]; then
    WAVE_1=(A1 A2 A3 A4)
    WAVE_2=(A5 B1 B2 B3)
    WAVE_3=(B4 B5 C1 D1)
    WAVE_4=(D2 D3 D4 D5)
else
    WAVE_1=(A2 A3 A4 A5)
    WAVE_2=(B1 B2 B3 B4)
    WAVE_3=(B5 C1 D1 D2)
    WAVE_4=(D3 D4 D5)
fi

FAILED_FILE="$LOG_DIR/sweep_failed.txt"
: > "$FAILED_FILE"

run_wave() {
    local arm="$1"
    local wave_label="$2"
    shift 2
    local conds_input=("$@")

    # Round-5 FIX B — resume-skip: when --resume is set, filter out conds
    # whose ALL FOUR epoch adapters {ep1, ep2, ep3, ep5} are already on HF.
    # Partial conds (e.g. B4 with only ep1/ep2/ep3) are RETRAINED in full
    # (the per-epoch upload callback overwrites — no torn state).
    # See scripts/i474_check_adapter_hf_presence.py (single list_repo_files
    # call per cond; exit 0=present, 1=missing, 2=lookup-failed-treat-as-missing).
    local conds=()
    if [ "$RESUME" -eq 1 ]; then
        echo "[phase=sweep_${arm}_wave_${wave_label}_resume_check] === checking HF presence (${#conds_input[@]} conds) ==="
        for cond in "${conds_input[@]}"; do
            local check_log="$LOG_DIR/resume_check_${arm}_${cond}.log"
            if uv run python scripts/i474_check_adapter_hf_presence.py \
                    --arm "$arm" --cond "$cond" \
                    > "$check_log" 2>&1; then
                echo "[phase=skip_${arm}_${cond}] already on HF — $(cat "$check_log")"
            else
                local rc=$?
                echo "[phase=resume_train_${arm}_${cond}] $(cat "$check_log") (check rc=$rc)"
                conds+=("$cond")
            fi
        done
        if [ "${#conds[@]}" -eq 0 ]; then
            echo "[phase=sweep_${arm}_wave_${wave_label}_skipped] === all ${#conds_input[@]} conds already on HF; wave skipped ==="
            return 0
        fi
    else
        conds=("${conds_input[@]}")
    fi

    echo "[phase=sweep_${arm}_wave_${wave_label}] === Sweep arm=${arm} wave ${wave_label}: ${conds[*]} $(date -Iseconds) ==="
    local pids=()
    local i=0
    for cond in "${conds[@]}"; do
        local cvd="$i"
        local log="$LOG_DIR/train_${arm}_${cond}_cvd${cvd}.log"
        # Per .claude/rules/gotchas.md (CVD-clobber, incidents #523/#543/#557):
        # export CUDA_VISIBLE_DEVICES=<gpu> per cell in the LAUNCHER env AND
        # pass the matching --gpu-id, so sft.py's in-process clobber rewrites
        # the same value. The in-process clobber alone is silently defeated by
        # any import-time cuInit (driver freezes its device list at first
        # cuInit) — that is how all 4 #523 Phase B waves piled onto GPU 0 and
        # OOM'd. Never env CVD + --gpu-id 0, and never --gpu-id alone.
        # Regression smoke: tests/test_cvd_wave_assignment_smoke.py.
        CUDA_VISIBLE_DEVICES="$cvd" uv run python scripts/i474_phase23_train.py \
            --arm "$arm" --conds "$cond" --gpu-id "$cvd" \
            > "$log" 2>&1 &
        pids+=("$!:${arm}:${cond}")
        i=$((i + 1))
    done
    for entry in "${pids[@]}"; do
        local pid; pid="$(cut -d: -f1 <<<"$entry")"
        local arm; arm="$(cut -d: -f2 <<<"$entry")"
        local cond; cond="$(cut -d: -f3 <<<"$entry")"
        if ! wait "$pid"; then
            echo "${arm}/${cond}" >> "$FAILED_FILE"
            echo "WAVE $wave_label: arm=$arm cond=$cond FAILED (pid=$pid)" >&2
        fi
    done
    echo "[phase=sweep_${arm}_wave_${wave_label}_done] === Wave ${wave_label} arm=${arm} complete ==="
}

for arm in "${SELECTED_ARMS[@]}"; do
    run_wave "$arm" "1" "${WAVE_1[@]}"
    run_wave "$arm" "2" "${WAVE_2[@]}"
    run_wave "$arm" "3" "${WAVE_3[@]}"
    run_wave "$arm" "4" "${WAVE_4[@]}"
done

if [ -s "$FAILED_FILE" ]; then
    FAILED=$(tr '\n' ' ' < "$FAILED_FILE")
    sentinel="/workspace/logs/issue-474-sweep-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 474,
    "phase": "phase3_sweep",
    "failure_class": "code",
    "failed": "$FAILED".split(),
    "reason": "One or more (arm, cond) cells in the sweep failed train_lora.",
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
EOF
    echo "[phase=sweep_failed] FATAL: sweep had failures: $FAILED. Sentinel at $sentinel." >&2
    exit 3
fi

echo "[phase=sweep_done] === Phase 3 sweep arms=${SELECTED_ARMS[*]} all cells trained $(date -Iseconds) ==="
