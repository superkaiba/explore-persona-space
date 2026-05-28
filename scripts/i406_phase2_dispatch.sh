#!/usr/bin/env bash
# Phase 2 — 2-GPU parallel LoRA training dispatcher with pilot gates.
#
# Issue #406 plan v9 §4 Phase 2.
#
# Pilot sequence (GPU 0 sequential):
#   1. A1 pilot (chat-template path). Train + smoke. If G[A1,A1] < 0.7,
#      retry at lr=5e-6. If retry also fails, escalate_and_block (MF-3:
#      NO silent N reduction; user decides).
#   2. C2 pilot (raw-string path via v3 MF-2 dataset_text_field). Train +
#      smoke. If G[C2,C2] < 0.7, retry at lr=2.5e-6. Same escalation on
#      double-failure.
#
# Parallel batch (only after both pilots PASS):
#   GPU 0 queue: A2 A3 A4 A5 B1 B2 B3 B4 B5
#   GPU 1 queue: C1 C3 C4 C5 D1 D2 D3 D4 D5
# (C2 already trained in the pilot — not re-trained in the batch.)
#
# Per CLAUDE.md feedback_cvd_hydra_override and sft.py:479: each process
# uses env CUDA_VISIBLE_DEVICES=<phys_gpu> set BEFORE spawn AND
# --gpu-id 0 (env CVD remaps the visible GPU to local device 0). The MF-4
# fix: train_and_smoke forwards --override-lr to BOTH train and smoke,
# and the per-cell log/JSON paths carry the lr_tag suffix so retry results
# don't overwrite default-lr results.

set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
# Avoid #399's MooseFS quota path during the 20-condition sequence.
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

LOG_DIR=logs/issue_406
mkdir -p "$LOG_DIR"

# Marker assert at launch (per CLAUDE.md) before any subprocess spawns.
uv run python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
ids = tok.encode(' ※', add_special_tokens=False)
assert ids == [83399], f'marker token id drift {ids}'
print('marker token id OK: 83399')
"

# ── Helper: train + verify HF upload + smoke-eval diagonal ────────────────
# Args: $1=cond, $2=require_smoke (yes|no), $3=cvd_gpu_idx, $4=override_lr (empty = default)
train_and_smoke() {
    local cond="$1"
    local require_smoke="${2:-yes}"
    local cvd="${3:-0}"
    local override_lr="${4:-}"

    local lr_flag=()
    local lr_tag="default"
    if [ -n "$override_lr" ]; then
        lr_flag=(--override-lr "$override_lr")
        lr_tag="lr${override_lr}"
    fi

    echo "=== Training $cond on CVD=$cvd (lr=$lr_tag) at $(date -Iseconds) ==="
    CUDA_VISIBLE_DEVICES="$cvd" uv run python scripts/i406_phase2_train_loras.py \
        --condition "$cond" --gpu-id 0 "${lr_flag[@]}" \
        > "$LOG_DIR/train_${cond}_cvd${cvd}_${lr_tag}.log" 2>&1

    # Verify HF upload landed before we try to read it in smoke-eval.
    uv run python -c "
from huggingface_hub import list_repo_files
files = list_repo_files('superkaiba1/explore-persona-space', revision='main')
target = 'adapters/i406_${cond}/adapter_model.safetensors'
assert target in files, f'{target} not on HF after training'
print(f'HF upload OK: {target}')
"

    if [ "$require_smoke" = "yes" ]; then
        CUDA_VISIBLE_DEVICES="$cvd" uv run python scripts/i406_phase2_smoke_eval.py \
            --condition "$cond" --lr-tag "$lr_tag" \
            > "$LOG_DIR/smoke_${cond}_${lr_tag}.log" 2>&1
        local diag_rate
        diag_rate=$(uv run python -c "
import json
d = json.load(open('$LOG_DIR/smoke_${cond}_${lr_tag}.json'))
print(d['diagonal_rate'])
")
        echo "  Smoke G[${cond},${cond}] @ ${lr_tag} = $diag_rate (require >=0.7)"
        if ! uv run python -c "import sys; sys.exit(0 if float('${diag_rate}') >= 0.7 else 1)"; then
            return 1
        fi
    fi
    return 0
}

# ── Helper: escalate to user (MF-3) — no silent design degradation ────────
# Writes a sentinel file on disk that the orchestrator's poll_pipeline.py
# observes and translates into `epm:failure v1` + `set-status blocked` on
# the local VM. Pod-side code MUST NOT shell out to `scripts/task.py` per
# CLAUDE.md (the script branch-guards to `main` and refuses on non-`main`
# HEAD); task #397 round 9 burned a launch on exactly that pattern. The
# sentinel-file pattern is the canonical pod-side escalation path.
escalate_and_block() {
    local cond="$1"
    local reason="$2"
    local sentinel="/workspace/logs/issue-406-pilot-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    echo "FATAL: $cond pilot failed BOTH default-lr and halved-lr retries."
    echo "  Reason: $reason"
    echo "  Per CLAUDE.md halt-criteria #1 (locked-spec deviation), this is"
    echo "  a factual question only the user knows."
    echo "  Writing sentinel $sentinel -- the orchestrator's poller will"
    echo "  translate this into epm:failure v1 + status:blocked on the VM."
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 406,
    "phase": "phase2_pilot",
    "failure_class": "rig",
    "condition": "$cond",
    "reason": """$reason""",
    "policy": "locked-spec N=20 4-class design cannot proceed without user decision "
              "(re-plan with explicit N reduction acknowledgement vs fix rig vs abort)",
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
EOF
    exit 2
}

# ── Gate 1: A1 pilot (chat-template path) ─────────────────────────────────
echo "### Gate 1: A1 pilot (chat-template path) ###"
if ! train_and_smoke A1 yes 0 ""; then
    echo "A1 failed at default lr=1e-5; retrying at lr=5e-6"
    if ! train_and_smoke A1 yes 0 "5e-6"; then
        escalate_and_block A1 \
"chat-template training path cannot implant marker at G[A1,A1]>=0.7 even at halved lr; rig itself is broken; entire N=20 batch at risk"
    fi
fi

# ── Gate 2: C2 pilot (raw-string path via v3 MF-2 dataset_text_field) ─────
echo "### Gate 2: C2 pilot (raw-string path) ###"
if ! train_and_smoke C2 yes 0 ""; then
    echo "C2 failed at default lr=5e-6; retrying at lr=2.5e-6"
    if ! train_and_smoke C2 yes 0 "2.5e-6"; then
        escalate_and_block C2 \
"raw-text training path (dataset_text_field mode, v3 MF-2 extension) cannot implant marker at G[C2,C2]>=0.7 even at halved lr; Class C2/C3/C4/C5 cannot proceed; user must decide N reduction vs rig fix vs abort"
    fi
fi

# ── Parallel batch (both pilots PASSED at this point) ─────────────────────
# MF-R2-3: track per-GPU failures via sidecar files (subshells can't
# mutate parent-shell arrays), then fail the whole dispatcher loud at
# end of wait if either GPU's queue had any failure. NO "ALL trained"
# success print on partial completion. Soft fallback for "1 of 18 failed
# but the user wants partial results" is gated behind --allow-partial-batch.

ALLOW_PARTIAL_BATCH=0
for arg in "$@"; do
    if [ "$arg" = "--allow-partial-batch" ]; then
        ALLOW_PARTIAL_BATCH=1
    fi
done

GPU0_QUEUE=(A2 A3 A4 A5 B1 B2 B3 B4 B5)
GPU1_QUEUE=(C1 C3 C4 C5 D1 D2 D3 D4 D5)
GPU0_FAILED_FILE="$LOG_DIR/gpu0_failed.txt"
GPU1_FAILED_FILE="$LOG_DIR/gpu1_failed.txt"
# Reset the sidecars at the start of every dispatcher run so a previous
# run's failure list doesn't poison this one.
: > "$GPU0_FAILED_FILE"
: > "$GPU1_FAILED_FILE"

(
    for cond in "${GPU0_QUEUE[@]}"; do
        if ! train_and_smoke "$cond" no 0 ""; then
            echo "$cond" >> "$GPU0_FAILED_FILE"
            echo "TRACKED: $cond on GPU 0 failed; appending to $GPU0_FAILED_FILE and continuing within queue"
        fi
    done
    echo "=== GPU 0 queue done ==="
) > "$LOG_DIR/gpu0_queue.log" 2>&1 &
GPU0_PID=$!

(
    for cond in "${GPU1_QUEUE[@]}"; do
        if ! train_and_smoke "$cond" no 1 ""; then
            echo "$cond" >> "$GPU1_FAILED_FILE"
            echo "TRACKED: $cond on GPU 1 failed; appending to $GPU1_FAILED_FILE and continuing within queue"
        fi
    done
    echo "=== GPU 1 queue done ==="
) > "$LOG_DIR/gpu1_queue.log" 2>&1 &
GPU1_PID=$!

echo "    GPU 0 batch PID: $GPU0_PID; GPU 1 batch PID: $GPU1_PID"
wait $GPU0_PID $GPU1_PID

# Read tracked failures from sidecar files.
GPU0_FAILED_LIST=""
GPU1_FAILED_LIST=""
if [ -s "$GPU0_FAILED_FILE" ]; then
    GPU0_FAILED_LIST=$(tr '\n' ' ' < "$GPU0_FAILED_FILE")
fi
if [ -s "$GPU1_FAILED_FILE" ]; then
    GPU1_FAILED_LIST=$(tr '\n' ' ' < "$GPU1_FAILED_FILE")
fi
ANY_FAILED="${GPU0_FAILED_LIST}${GPU1_FAILED_LIST}"

# Helper: write a parallel-batch failure sentinel mirroring the pilot
# sentinel shape so poll_pipeline.py's general sentinel reader picks it
# up uniformly. Different path so multi-sentinel runs surface both.
write_batch_failure_sentinel() {
    local sentinel="/workspace/logs/issue-406-phase2-batch-failed.json"
    mkdir -p "$(dirname "$sentinel")"
    uv run python - <<EOF
import json, datetime
payload = {
    "issue": 406,
    "phase": "phase2_batch",
    "failure_class": "rig",
    "gpu0_failed": "${GPU0_FAILED_LIST}".strip().split() if "${GPU0_FAILED_LIST}".strip() else [],
    "gpu1_failed": "${GPU1_FAILED_LIST}".strip().split() if "${GPU1_FAILED_LIST}".strip() else [],
    "reason": "One or more non-pilot conditions failed train_and_smoke in the parallel batch",
    "policy": "Phase 2 succeeds only if all 20 conditions trained successfully. "
              "Pass --allow-partial-batch to the dispatcher if a partial run is acceptable.",
    "wrote_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open("$sentinel", "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote batch-failure sentinel: $sentinel")
EOF
}

if [ -n "$ANY_FAILED" ]; then
    echo "" >&2
    echo "PARTIAL BATCH FAILURE at $(date -Iseconds)" >&2
    if [ -n "$GPU0_FAILED_LIST" ]; then
        echo "  GPU 0 failed conditions: $GPU0_FAILED_LIST" >&2
    fi
    if [ -n "$GPU1_FAILED_LIST" ]; then
        echo "  GPU 1 failed conditions: $GPU1_FAILED_LIST" >&2
    fi
    write_batch_failure_sentinel
    if [ "$ALLOW_PARTIAL_BATCH" = "1" ]; then
        echo "  --allow-partial-batch set; treating as soft failure. Phase 3 may compute an incomplete G matrix." >&2
        echo "=== Phase 2 PARTIAL at $(date -Iseconds) ==="
        exit 0
    fi
    echo "  No --allow-partial-batch flag set. Refusing to continue." >&2
    echo "  Sentinel written to /workspace/logs/issue-406-phase2-batch-failed.json; poll_pipeline.py will translate it into epm:failure v1 + status:blocked." >&2
    exit 2
fi

echo "=== Phase 2 ALL parallel batch conditions trained (N=20, 4-class design preserved) at $(date -Iseconds) ==="
