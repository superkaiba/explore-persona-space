#!/usr/bin/env bash
# Task #571 pod dispatcher — negative-panel-breadth ablation (plan §3.3).
#
# Modeled on scripts/i474_phase23_dispatch.sh. Smoke IS the sweep with one
# cell (plan §3.2 smoke/sweep parity): --smoke-only runs the full
# production path for cell narrow_s42 (the canary exercising the NEW
# n_per_bystander=75 duplication branch) — full 600-row ep1 train (this IS
# the production adapter, not retrained later) -> restricted gen
# (--personas assistant --n-questions 2 --tag smoke) -> both scoring sides
# -> source-gen + source-score -> tagged upload. The sweep then runs the
# remaining 3 cells + the full eval, resume-skipping the canary's outputs.
#
# Per the cuInit gotcha each parallel cell exports CUDA_VISIBLE_DEVICES in
# the LAUNCHER env AND passes the matching --gpu-id (both required).
#
# Pod-side code NEVER shells out to scripts/task.py — sentinel files under
# /workspace/logs/issue-571-*.json are the only escalation channel.
#
# Usage (pod, repo root, issue-571 branch):
#   nohup bash scripts/issue571_dispatch.sh > logs/issue_571/dispatch.log 2>&1 < /dev/null &
#   bash scripts/issue571_dispatch.sh --smoke-only
#   bash scripts/issue571_dispatch.sh --skip-smoke --resume

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

LOG_DIR=logs/issue_571
mkdir -p "$LOG_DIR"

SMOKE_ONLY=0
SKIP_SMOKE=0
RESUME=0
for arg in "$@"; do
    case "$arg" in
        --smoke-only) SMOKE_ONLY=1 ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        --resume) RESUME=1 ;;
        *) ;;
    esac
done

echo "[phase=p0_preflight] === i571 dispatcher $(date -Iseconds) smoke_only=$SMOKE_ONLY skip_smoke=$SKIP_SMOKE resume=$RESUME ==="

# Marker + <|im_end|> id asserts at launch (marker-leakage-measurement.md).
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

# Preflight with the documented feature-branch false-positive tolerance:
# on every issue-<N> pod checkout the git check reports 'behind
# origin/main' and exits non-zero even at the reviewed branch tip. Parse
# --json and fail ONLY on errors other than that line (no skip flag exists).
PREFLIGHT_JSON="$LOG_DIR/preflight.json"
uv run python -m explore_persona_space.orchestrate.preflight --json > "$PREFLIGHT_JSON" || true
uv run python - "$PREFLIGHT_JSON" <<'EOF'
import json
import sys

report = json.load(open(sys.argv[1]))
benign = [e for e in report.get("errors", []) if "behind origin/main" in e]
fatal = [e for e in report.get("errors", []) if "behind origin/main" not in e]
for e in benign:
    print(f"[preflight] tolerated feature-branch false positive: {e}")
if fatal:
    print("[preflight] FATAL errors:")
    for e in fatal:
        print(f"  - {e}")
    sys.exit(1)
print(f"[preflight] OK ({len(report.get('warnings', []))} warnings, "
      f"{len(benign)} tolerated git false positives)")
EOF

# Sentinel helper — pod-side escalation (poll_pipeline schema v1).
escalate_and_exit() {
    local phase="$1"
    local reason="$2"
    local rc="$3"
    local sentinel="/workspace/logs/issue-571-failure-$(date +%s).json"
    mkdir -p "$(dirname "$sentinel")" 2>/dev/null || sentinel="$LOG_DIR/issue-571-failure-$(date +%s).json"
    echo "[phase=failed] FATAL at ${phase}: ${reason}" >&2
    REASON="$reason" PHASE="$phase" SENTINEL="$sentinel" uv run python - <<'EOF'
import json
import os
from datetime import UTC, datetime

payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:failure",
    "version": 1,
    "task_id": 571,
    "by": "issue571_dispatch.sh",
    "ts": datetime.now(UTC).isoformat(),
    "note": json.dumps(
        {
            "failure_class": "code",
            "phase": os.environ["PHASE"],
            "reason": os.environ["REASON"],
        }
    ),
}
with open(os.environ["SENTINEL"], "w") as f:
    json.dump(payload, f, indent=1)
print(f"wrote failure sentinel: {os.environ['SENTINEL']}")
EOF
    exit "$rc"
}

run_phase() {
    # run_phase <log-name> <phase-tag> <cmd...> — fail-loud wrapper with logs.
    local log_name="$1"; shift
    local phase_tag="$1"; shift
    echo "[phase=${phase_tag}] === $log_name $(date -Iseconds): $* ==="
    local rc=0
    "$@" > "$LOG_DIR/${log_name}.log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        tail -20 "$LOG_DIR/${log_name}.log" >&2 || true
        escalate_and_exit "$phase_tag" "$log_name exited rc=${rc} (see $LOG_DIR/${log_name}.log)" 2
    fi
}

# CPU+GPU smoke gates of the eval driver (pinned a9fc5a9 fetch, persona +
# exposure asserts, adapter-registry validation, #532 scoring-path gate,
# #534 vLLM-LoRA gate). Runs on GPU0 as a fresh subprocess.
CUDA_VISIBLE_DEVICES=0 run_phase "p0_smoke_gates" "p0_smoke_gates" \
    uv run python scripts/issue571_breadth_panel.py --phase smoke

# Resume helper: is a cell's ep1 adapter already on HF?
adapter_on_hf() {
    local panel="$1" seed="$2"
    uv run python -c "
from huggingface_hub import list_repo_files
files = list_repo_files('superkaiba1/explore-persona-space')
target = 'adapters/i571_${panel}_A2_s${seed}_ep1/adapter_model.safetensors'
raise SystemExit(0 if target in files else 1)
" 2>/dev/null
}

train_cell() {
    # train_cell <panel> <seed> <gpu> — launcher-env CVD + --gpu-id (both).
    local panel="$1" seed="$2" gpu="$3"
    CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/issue571_train.py \
        --panel "$panel" --seed "$seed" --gpu-id "$gpu"
}

# ── Phase 1: smoke == sweep cell narrow_s42, end to end ───────────────────
if [ "$SKIP_SMOKE" -eq 0 ]; then
    if [ "$RESUME" -eq 1 ] && adapter_on_hf narrow 42; then
        echo "[phase=p1_smoke_train_skip] narrow_s42 ep1 already on HF — resume skip"
    else
        run_phase "smoke_train_narrow_s42" "p1_smoke_train" train_cell narrow 42 0
    fi
    CUDA_VISIBLE_DEVICES=0 run_phase "smoke_gen" "p1_smoke_gen" \
        uv run python scripts/issue571_breadth_panel.py --phase gen \
        --adapters narrow_s42 --personas assistant --n-questions 2 --tag smoke
    CUDA_VISIBLE_DEVICES=0 run_phase "smoke_score_trained" "p1_smoke_score_trained" \
        uv run python scripts/issue571_breadth_panel.py --phase score-trained \
        --adapters narrow_s42 --personas assistant --n-questions 2 --tag smoke
    CUDA_VISIBLE_DEVICES=0 run_phase "smoke_score_base" "p1_smoke_score_base" \
        uv run python scripts/issue571_breadth_panel.py --phase score-base \
        --adapters narrow_s42 --personas assistant --n-questions 2 --tag smoke
    CUDA_VISIBLE_DEVICES=0 run_phase "smoke_source_gen" "p1_smoke_source_gen" \
        uv run python scripts/issue571_breadth_panel.py --phase source-gen \
        --adapters narrow_s42
    CUDA_VISIBLE_DEVICES=0 run_phase "smoke_source_score" "p1_smoke_source_score" \
        uv run python scripts/issue571_breadth_panel.py --phase source-score \
        --adapters narrow_s42
    run_phase "smoke_upload" "p1_smoke_upload" \
        uv run python scripts/issue571_breadth_panel.py --phase upload \
        --adapters narrow_s42 --personas assistant --n-questions 2 --tag smoke
    echo "[phase=p1_smoke_pass] === smoke cell narrow_s42 PASS $(date -Iseconds) ==="
fi

if [ "$SMOKE_ONLY" -eq 1 ]; then
    echo "[phase=smoke_only_done] === --smoke-only set; exit after smoke. ==="
    echo "[phase=done]"
    exit 0
fi

# ── Phase 2: remaining trains, parallel, one GPU each ──────────────────────
if [ "$SKIP_SMOKE" -eq 1 ]; then
    TRAIN_CELLS=("narrow 42" "broad 42" "broad 43" "narrow 43")
else
    TRAIN_CELLS=("broad 42" "broad 43" "narrow 43")
fi

echo "[phase=p2_train] === parallel trains: ${TRAIN_CELLS[*]} $(date -Iseconds) ==="
FAILED_FILE="$LOG_DIR/train_failed.txt"
: > "$FAILED_FILE"
pids=()
gpu=0
for cell in "${TRAIN_CELLS[@]}"; do
    read -r panel seed <<<"$cell"
    if [ "$RESUME" -eq 1 ] && adapter_on_hf "$panel" "$seed"; then
        echo "[phase=p2_train_skip_${panel}_s${seed}] ep1 adapter already on HF — resume skip"
        continue
    fi
    log="$LOG_DIR/train_${panel}_s${seed}_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/issue571_train.py \
        --panel "$panel" --seed "$seed" --gpu-id "$gpu" > "$log" 2>&1 &
    pids+=("$!:${panel}_s${seed}")
    gpu=$((gpu + 1))
done
for entry in "${pids[@]}"; do
    pid="${entry%%:*}"
    cell="${entry#*:}"
    if ! wait "$pid"; then
        echo "$cell" >> "$FAILED_FILE"
        echo "TRAIN cell=$cell FAILED (pid=$pid)" >&2
    fi
done
if [ -s "$FAILED_FILE" ]; then
    escalate_and_exit "p2_train" "training cells failed: $(tr '\n' ' ' < "$FAILED_FILE")" 3
fi
echo "[phase=p2_train_done] === all trains complete $(date -Iseconds) ==="

# ── Phase 3: gen, 4-way parallel (one vLLM engine per GPU per adapter) ────
ALL_LABELS=(broad_s42 broad_s43 narrow_s42 narrow_s43)
echo "[phase=p3_gen] === parallel gen: ${ALL_LABELS[*]} $(date -Iseconds) ==="
: > "$FAILED_FILE"
pids=()
gpu=0
for label in "${ALL_LABELS[@]}"; do
    log="$LOG_DIR/gen_${label}_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/issue571_breadth_panel.py \
        --phase gen --adapters "$label" > "$log" 2>&1 &
    pids+=("$!:${label}")
    gpu=$((gpu + 1))
done
for entry in "${pids[@]}"; do
    pid="${entry%%:*}"
    label="${entry#*:}"
    if ! wait "$pid"; then
        echo "gen_$label" >> "$FAILED_FILE"
        echo "GEN label=$label FAILED (pid=$pid)" >&2
    fi
done
if [ -s "$FAILED_FILE" ]; then
    escalate_and_exit "p3_gen" "gen failed: $(tr '\n' ' ' < "$FAILED_FILE")" 3
fi
echo "[phase=p3_gen_done] === gen complete $(date -Iseconds) ==="

# ── Phase 4: scoring, 4-way parallel (trained then base per GPU) ──────────
echo "[phase=p4_score] === parallel scoring $(date -Iseconds) ==="
: > "$FAILED_FILE"
pids=()
gpu=0
for label in "${ALL_LABELS[@]}"; do
    log="$LOG_DIR/score_${label}_gpu${gpu}.log"
    (
        CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/issue571_breadth_panel.py \
            --phase score-trained --adapters "$label" &&
        CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/issue571_breadth_panel.py \
            --phase score-base --adapters "$label"
    ) > "$log" 2>&1 &
    pids+=("$!:${label}")
    gpu=$((gpu + 1))
done
for entry in "${pids[@]}"; do
    pid="${entry%%:*}"
    label="${entry#*:}"
    if ! wait "$pid"; then
        echo "score_$label" >> "$FAILED_FILE"
        echo "SCORE label=$label FAILED (pid=$pid)" >&2
    fi
done
if [ -s "$FAILED_FILE" ]; then
    escalate_and_exit "p4_score" "scoring failed: $(tr '\n' ' ' < "$FAILED_FILE")" 3
fi
echo "[phase=p4_score_done] === scoring complete $(date -Iseconds) ==="

# ── Phase 5: source-check (manipulation check), all 4 adapters, GPU0 ──────
# source-gen (vLLM) and source-score (HF) are separate subprocesses (vLLM
# teardown contract). The canary's files resume-skip automatically.
CUDA_VISIBLE_DEVICES=0 run_phase "source_gen_all" "p5_source_gen" \
    uv run python scripts/issue571_breadth_panel.py --phase source-gen --adapters all
CUDA_VISIBLE_DEVICES=0 run_phase "source_score_all" "p5_source_score" \
    uv run python scripts/issue571_breadth_panel.py --phase source-score --adapters all

# ── Phase 6: upload + sentinel ─────────────────────────────────────────────
run_phase "upload" "p6_upload" \
    uv run python scripts/issue571_breadth_panel.py --phase upload --adapters all

echo "[phase=sweep_done] === i571 full sweep complete $(date -Iseconds) ==="
echo "[phase=done]"
