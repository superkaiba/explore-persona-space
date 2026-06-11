#!/usr/bin/env bash
# Task #571 persona-split-composition pod dispatcher (Stage 2, plan v2 §4.5).
#
# Modeled on scripts/issue571_dispatch.sh. Smoke IS the sweep with one cell:
# --smoke-only runs the full production path for cell split8_s42 (the canary
# exercising the NEW persona-negative builder's 38/37 remainder branch + max
# panel size) — full 600-row ep1 train (this IS the production adapter) ->
# restricted gen (--personas assistant --n-questions 2 --tag smoke, WITH the
# panelneg extra stratum) -> both scoring sides -> source-gen + source-score
# -> tagged upload. The sweep then runs the remaining 5 cells + the full
# eval, resume-skipping the canary's outputs.
#
# Per the cuInit gotcha each parallel cell exports CUDA_VISIBLE_DEVICES in
# the LAUNCHER env AND passes the matching --gpu-id (both required).
#
# Pod-side code NEVER shells out to scripts/task.py — sentinel files under
# /workspace/logs/issue-571-*.json are the only escalation channel; the
# production upload phase writes /workspace/logs/issue-571-psplit-run-
# complete.json.
#
# Usage (pod, repo root, issue-571 branch):
#   nohup bash scripts/issue571_psplit_dispatch.sh > logs/issue_571/psplit_dispatch.log 2>&1 < /dev/null &
#   bash scripts/issue571_psplit_dispatch.sh --smoke-only
#   bash scripts/issue571_psplit_dispatch.sh --skip-smoke --resume

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1

# ── §4.0 HARD ORDERING CONTRACT (top-of-script assert) ─────────────────────
# Stage 1 must be committed to this checkout BEFORE any Stage-2 phase runs.
STAGE1_JSON="eval_results/issue_571/persona-split-composition/stage1_geometry_join.json"
if [ ! -f "$STAGE1_JSON" ]; then
    echo "[phase=failed] FATAL: $STAGE1_JSON missing from the checkout — Stage 1 must run," >&2
    echo "commit, and be pulled onto this pod BEFORE Stage 2 (plan §4.0 ordering contract)." >&2
    exit 7
fi

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

echo "[phase=p0_preflight] === i571 psplit dispatcher $(date -Iseconds) smoke_only=$SMOKE_ONLY skip_smoke=$SKIP_SMOKE resume=$RESUME ==="

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
PREFLIGHT_JSON="$LOG_DIR/psplit_preflight.json"
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
    local sentinel="/workspace/logs/issue-571-psplit-failure-$(date +%s).json"
    mkdir -p "$(dirname "$sentinel")" 2>/dev/null || sentinel="$LOG_DIR/issue-571-psplit-failure-$(date +%s).json"
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
    "by": "issue571_psplit_dispatch.sh",
    "ts": datetime.now(UTC).isoformat(),
    "note": json.dumps(
        {
            "failure_class": "code",
            "followup_label": "persona-split-composition",
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

PANEL_PERSONAS="eval_results/issue_571/persona-split-composition/geometry/panel_personas.json"

# ── Phase 0: driver smoke gates (registry asserts, pinned a9fc5a9 fetch,
# exposure classification, #532 scoring-path gate, #534 vLLM-LoRA gate) ────
CUDA_VISIBLE_DEVICES=0 run_phase "p0_psplit_smoke_gates" "p0_smoke_gates" \
    uv run python scripts/issue571_breadth_panel.py --phase smoke --registry psplit

# ── Phase 0.5: union-bank centroids + panel gate (Stage-1 linkage applied) ─
CUDA_VISIBLE_DEVICES=0 run_phase "p0p5_geometry" "p0p5_geometry" \
    uv run python scripts/issue571_psplit_geometry.py

if [ ! -f "$PANEL_PERSONAS" ]; then
    escalate_and_exit "p0p5_geometry" "$PANEL_PERSONAS missing after geometry phase" 2
fi

# ── Phase 1: frozen R for the realized panel personas (vLLM, #460 recipe) ──
CUDA_VISIBLE_DEVICES=0 run_phase "p1_rgen" "p1_rgen" \
    uv run python scripts/issue571_psplit_rgen.py

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

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -ge 1 ] || NGPU=1
echo "=== visible GPUs: $NGPU ==="

# ── Phase 2: smoke == sweep cell split8_s42, end to end ────────────────────
if [ "$SKIP_SMOKE" -eq 0 ]; then
    if [ "$RESUME" -eq 1 ] && adapter_on_hf split8 42; then
        echo "[phase=p2_smoke_train_skip] split8_s42 ep1 already on HF — resume skip"
    else
        run_phase "smoke_train_split8_s42" "p2_smoke_train" train_cell split8 42 0
    fi
    CUDA_VISIBLE_DEVICES=0 run_phase "smoke_gen" "p2_smoke_gen" \
        uv run python scripts/issue571_breadth_panel.py --phase gen --registry psplit \
        --adapters split8_s42 --personas assistant --n-questions 2 --tag smoke \
        --extra-personas-file "$PANEL_PERSONAS"
    CUDA_VISIBLE_DEVICES=0 run_phase "smoke_score_trained" "p2_smoke_score_trained" \
        uv run python scripts/issue571_breadth_panel.py --phase score-trained --registry psplit \
        --adapters split8_s42 --personas assistant --n-questions 2 --tag smoke \
        --extra-personas-file "$PANEL_PERSONAS"
    CUDA_VISIBLE_DEVICES=0 run_phase "smoke_score_base" "p2_smoke_score_base" \
        uv run python scripts/issue571_breadth_panel.py --phase score-base --registry psplit \
        --adapters split8_s42 --personas assistant --n-questions 2 --tag smoke \
        --extra-personas-file "$PANEL_PERSONAS"
    CUDA_VISIBLE_DEVICES=0 run_phase "smoke_source_gen" "p2_smoke_source_gen" \
        uv run python scripts/issue571_breadth_panel.py --phase source-gen --registry psplit \
        --adapters split8_s42
    CUDA_VISIBLE_DEVICES=0 run_phase "smoke_source_score" "p2_smoke_source_score" \
        uv run python scripts/issue571_breadth_panel.py --phase source-score --registry psplit \
        --adapters split8_s42
    run_phase "smoke_upload" "p2_smoke_upload" \
        uv run python scripts/issue571_breadth_panel.py --phase upload --registry psplit \
        --adapters split8_s42 --personas assistant --n-questions 2 --tag smoke
    echo "[phase=p2_smoke_pass] === smoke cell split8_s42 PASS $(date -Iseconds) ==="
fi

if [ "$SMOKE_ONLY" -eq 1 ]; then
    echo "[phase=smoke_only_exit] === --smoke-only set; exit after smoke. ==="
    echo "[phase=done]"
    exit 0
fi

# ── Phase 3: remaining trains, GPU-waved, one GPU each ─────────────────────
if [ "$SKIP_SMOKE" -eq 1 ]; then
    TRAIN_CELLS=("split8 42" "split2 42" "split2 43" "split4 42" "split4 43" "split8 43")
else
    TRAIN_CELLS=("split2 42" "split2 43" "split4 42" "split4 43" "split8 43")
fi

echo "[phase=p3_train] === waved trains: ${TRAIN_CELLS[*]} $(date -Iseconds) ==="
FAILED_FILE="$LOG_DIR/psplit_train_failed.txt"
: > "$FAILED_FILE"
pids=()
gpu=0
for cell in "${TRAIN_CELLS[@]}"; do
    read -r panel seed <<<"$cell"
    if [ "$RESUME" -eq 1 ] && adapter_on_hf "$panel" "$seed"; then
        echo "[phase=p3_train_skip_${panel}_s${seed}] ep1 adapter already on HF — resume skip"
        continue
    fi
    if [ "$gpu" -ge "$NGPU" ]; then
        # Wave boundary: drain before reusing GPUs.
        for entry in "${pids[@]}"; do
            pid="${entry%%:*}"; cname="${entry#*:}"
            wait "$pid" || { echo "$cname" >> "$FAILED_FILE"; echo "TRAIN cell=$cname FAILED" >&2; }
        done
        pids=()
        gpu=0
        [ -s "$FAILED_FILE" ] && escalate_and_exit "p3_train" "training cells failed: $(tr '\n' ' ' < "$FAILED_FILE")" 3
    fi
    log="$LOG_DIR/train_${panel}_s${seed}_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/issue571_train.py \
        --panel "$panel" --seed "$seed" --gpu-id "$gpu" > "$log" 2>&1 &
    pids+=("$!:${panel}_s${seed}")
    gpu=$((gpu + 1))
done
for entry in "${pids[@]}"; do
    pid="${entry%%:*}"; cname="${entry#*:}"
    wait "$pid" || { echo "$cname" >> "$FAILED_FILE"; echo "TRAIN cell=$cname FAILED" >&2; }
done
if [ -s "$FAILED_FILE" ]; then
    escalate_and_exit "p3_train" "training cells failed: $(tr '\n' ' ' < "$FAILED_FILE")" 3
fi
echo "[phase=p3_train_complete] === all trains complete $(date -Iseconds) ==="

# ── Phase 4: gen, GPU-waved (one vLLM engine per GPU per adapter) ──────────
ALL_LABELS=(split2_s42 split2_s43 split4_s42 split4_s43 split8_s42 split8_s43)
echo "[phase=p4_gen] === waved gen: ${ALL_LABELS[*]} $(date -Iseconds) ==="
: > "$FAILED_FILE"
pids=()
gpu=0
for label in "${ALL_LABELS[@]}"; do
    if [ "$gpu" -ge "$NGPU" ]; then
        for entry in "${pids[@]}"; do
            pid="${entry%%:*}"; lname="${entry#*:}"
            wait "$pid" || { echo "gen_$lname" >> "$FAILED_FILE"; echo "GEN label=$lname FAILED" >&2; }
        done
        pids=()
        gpu=0
        [ -s "$FAILED_FILE" ] && escalate_and_exit "p4_gen" "gen failed: $(tr '\n' ' ' < "$FAILED_FILE")" 3
    fi
    log="$LOG_DIR/psplit_gen_${label}_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/issue571_breadth_panel.py \
        --phase gen --registry psplit --adapters "$label" \
        --extra-personas-file "$PANEL_PERSONAS" > "$log" 2>&1 &
    pids+=("$!:${label}")
    gpu=$((gpu + 1))
done
for entry in "${pids[@]}"; do
    pid="${entry%%:*}"; lname="${entry#*:}"
    wait "$pid" || { echo "gen_$lname" >> "$FAILED_FILE"; echo "GEN label=$lname FAILED" >&2; }
done
if [ -s "$FAILED_FILE" ]; then
    escalate_and_exit "p4_gen" "gen failed: $(tr '\n' ' ' < "$FAILED_FILE")" 3
fi
echo "[phase=p4_gen_complete] === gen complete $(date -Iseconds) ==="

# ── Phase 5: scoring, GPU-waved (trained then base per GPU) ────────────────
echo "[phase=p5_score] === waved scoring $(date -Iseconds) ==="
: > "$FAILED_FILE"
pids=()
gpu=0
for label in "${ALL_LABELS[@]}"; do
    if [ "$gpu" -ge "$NGPU" ]; then
        for entry in "${pids[@]}"; do
            pid="${entry%%:*}"; lname="${entry#*:}"
            wait "$pid" || { echo "score_$lname" >> "$FAILED_FILE"; echo "SCORE label=$lname FAILED" >&2; }
        done
        pids=()
        gpu=0
        [ -s "$FAILED_FILE" ] && escalate_and_exit "p5_score" "scoring failed: $(tr '\n' ' ' < "$FAILED_FILE")" 3
    fi
    log="$LOG_DIR/psplit_score_${label}_gpu${gpu}.log"
    (
        CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/issue571_breadth_panel.py \
            --phase score-trained --registry psplit --adapters "$label" \
            --extra-personas-file "$PANEL_PERSONAS" &&
        CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/issue571_breadth_panel.py \
            --phase score-base --registry psplit --adapters "$label" \
            --extra-personas-file "$PANEL_PERSONAS"
    ) > "$log" 2>&1 &
    pids+=("$!:${label}")
    gpu=$((gpu + 1))
done
for entry in "${pids[@]}"; do
    pid="${entry%%:*}"; lname="${entry#*:}"
    wait "$pid" || { echo "score_$lname" >> "$FAILED_FILE"; echo "SCORE label=$lname FAILED" >&2; }
done
if [ -s "$FAILED_FILE" ]; then
    escalate_and_exit "p5_score" "scoring failed: $(tr '\n' ' ' < "$FAILED_FILE")" 3
fi
echo "[phase=p5_score_complete] === scoring complete $(date -Iseconds) ==="

# ── Phase 6: source-check (manipulation check), all 6 adapters, GPU0 ───────
CUDA_VISIBLE_DEVICES=0 run_phase "psplit_source_gen_all" "p6_source_gen" \
    uv run python scripts/issue571_breadth_panel.py --phase source-gen --registry psplit \
    --adapters all
CUDA_VISIBLE_DEVICES=0 run_phase "psplit_source_score_all" "p6_source_score" \
    uv run python scripts/issue571_breadth_panel.py --phase source-score --registry psplit \
    --adapters all

# ── Phase 7: upload + sentinel ──────────────────────────────────────────────
run_phase "psplit_upload" "p7_upload" \
    uv run python scripts/issue571_breadth_panel.py --phase upload --registry psplit \
    --adapters all

echo "[phase=sweep_complete] === i571 psplit full sweep complete $(date -Iseconds) ==="
echo "[phase=done]"
