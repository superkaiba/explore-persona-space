#!/usr/bin/env bash
# Issue #617 unified dispatcher: WildChat-category corpus + separability.
#
# Plan §4 DAG. Runs steps 1-7 end-to-end. Steps 3 (extraction) and 6
# (completions) are the only GPU phases; steps 1/2/4/5/7 are CPU. The pod
# holds the GPU for steps 3 + 6; CPU steps run on the same box but need no GPU.
#
# ARCHITECTURALLY UNIFIED smoke (PASS_UNIFIED): `--smoke` IS this same
# dispatcher with a one-cell parameterization — it threads a tiny-N subset
# through EVERY phase via the per-step CLI flags (slice target 200, K={5},
# HDBSCAN min_cluster_size derived from target//200=1 -> max(30,1)=30 floor via
# the cluster script's own arithmetic, battery pool-cap 12 + floor 3, scoring
# floor 3, completion cap 4, tiny model). Same subprocess shape, same env
# injection, same [phase=] logging surface, same sentinel writer, same
# [phase=done] terminal. NO separate smoke code path.
#
# Per-phase cell-subset source (PASS_UNIFIED requirement — every phase derives
# its subset from the SAME --smoke switch):
#   step1 slice    : --target / --scan-cap (smoke: 200 / 20000)
#   step2 cluster  : --ks 5 (smoke), HDBSCAN min_cluster_size = max(30, target//200)
#   step3 battery  : --pool-cap / --per-cluster-floor (smoke: 12 / 3)
#   step3 extract  : the capped battery (every extracted instance derives from
#                    step3's --pool-cap subset; cross-eval reads the SAME
#                    membership map produced from that subset)
#   step4 score    : --per-cluster-floor 3 (smoke) over the SAME membership map
#   step5 figures  : reads the SAME separability.json
#   step6 complete : --max-prefixes 4 (smoke) over the winning pair's members
#   step7 upload   : --dry-run under smoke (no real HF write)
#
# Env passthrough: load .env at dispatcher top so subprocesses inherit
# HF_TOKEN / WANDB / ANTHROPIC even under `uv run` (which does NOT auto-load
# .env). CVD pin is N/A here — single GPU, no parallel per-GPU fan-out.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

# Load credentials at entry so every `uv run python` subprocess inherits them.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

SMOKE=0
SKIP_UPLOAD=0
NO_GPU=0
CHAT_DATASET="allenai/WildChat-1M"
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=1 ;;
        --skip-upload) SKIP_UPLOAD=1 ;;
        # --no-gpu: GPU-less host (local VM smoke). The two GPU phases run their
        # CPU-runnable portion only — extraction on a tiny CPU model, completions
        # via --stub-completions (build_prompts + file-write, no vLLM). On a pod
        # (has GPU) the smoke omits this flag and exercises the real GPU path.
        --no-gpu) NO_GPU=1 ;;
        --chat-dataset=*) CHAT_DATASET="${arg#*=}" ;;
        *) ;;
    esac
done

LOG_DIR="logs/issue_617"
mkdir -p "$LOG_DIR"

# Per-phase parameters — the ONE place the smoke vs sweep cell-subset is set.
if [ "$SMOKE" -eq 1 ]; then
    SLICE_TARGET=200
    SLICE_SCAN_CAP=20000
    CLUSTER_KS="5"
    POOL_CAP=12
    PER_CLUSTER_FLOOR=3
    N_PERMS=200
    COMPLETION_MAX_PREFIXES=4
    COMPLETION_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
    if [ "$NO_GPU" -eq 1 ]; then
        # GPU-less host: tiny model on CPU for extraction; stub completions.
        EMBED_DEVICE="cpu"
        EXTRACT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
        EXTRACT_EXTRA="--device cpu --expected-layers 24 --expected-hidden 896 --no-upload --wandb-mode disabled --n-probes 3"
        COMPLETION_EXTRA="--max-model-len 2048 --stub-completions"
    else
        # Pod smoke (has GPU): real vLLM + GPU extraction on the tiny model.
        EMBED_DEVICE="auto"
        EXTRACT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
        EXTRACT_EXTRA="--gpu-id 0 --expected-layers 24 --expected-hidden 896 --no-upload --wandb-mode disabled --n-probes 3"
        COMPLETION_EXTRA="--max-model-len 2048"
    fi
else
    SLICE_TARGET=20000
    SLICE_SCAN_CAP=200000
    CLUSTER_KS="5 10 20"
    POOL_CAP=400
    PER_CLUSTER_FLOOR=30
    N_PERMS=1000
    COMPLETION_MAX_PREFIXES=200
    EMBED_DEVICE="auto"
    EXTRACT_MODEL="Qwen/Qwen2.5-7B-Instruct"
    EXTRACT_EXTRA="--gpu-id 0"
    COMPLETION_MODEL="Qwen/Qwen2.5-7B-Instruct"
    COMPLETION_EXTRA="--max-model-len 4096"
fi

fail_sentinel() {
    local step="$1"
    local reason="$2"
    local sentinel="/workspace/logs/issue-617-epm_failure-$(date +%s).json"
    mkdir -p "$(dirname "$sentinel")" 2>/dev/null || sentinel="$LOG_DIR/issue-617-epm_failure-$(date +%s).json"
    uv run python - "$sentinel" "$step" "$reason" <<'PY'
import json, sys, datetime
path, step, reason = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {
    "sentinel_schema_version": 1, "kind": "epm:failure", "version": 1,
    "issue": 617, "phase": step, "failure_class": "code", "reason": reason,
    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote failure sentinel: {path}")
PY
    echo "[phase=failed] FATAL at $step: $reason" >&2
    exit 2
}

echo "[phase=start] === i617 dispatcher $(date -Iseconds) smoke=$SMOKE chat_dataset=$CHAT_DATASET ==="

# ── Step 1: WildChat slice (CPU pre-provision gate) ─────────────────────────
echo "[phase=slice] === Step 1 WildChat slice (target=$SLICE_TARGET) ==="
uv run python scripts/issue617_build_wildchat_slice.py \
    --chat-dataset "$CHAT_DATASET" --target "$SLICE_TARGET" --scan-cap "$SLICE_SCAN_CAP" \
    > "$LOG_DIR/step1_slice.log" 2>&1 || fail_sentinel slice "step 1 rc=$? (see step1_slice.log)"

# ── Step 2: embed + cluster sweep (CPU / L4) ────────────────────────────────
echo "[phase=cluster] === Step 2 embed + cluster sweep (ks=$CLUSTER_KS) ==="
# shellcheck disable=SC2086
uv run python scripts/issue617_cluster.py \
    --ks $CLUSTER_KS --device "$EMBED_DEVICE" \
    > "$LOG_DIR/step2_cluster.log" 2>&1 || fail_sentinel cluster "step 2 rc=$? (see step2_cluster.log)"

# ── Step 3a: build extraction battery (CPU) ─────────────────────────────────
echo "[phase=battery] === Step 3a build extraction battery (pool-cap=$POOL_CAP floor=$PER_CLUSTER_FLOOR) ==="
uv run python scripts/issue617_build_extraction_battery.py \
    --pool-cap "$POOL_CAP" --per-cluster-floor "$PER_CLUSTER_FLOOR" \
    > "$LOG_DIR/step3a_battery.log" 2>&1 || fail_sentinel battery "step 3a rc=$? (see step3a_battery.log)"

# ── Step 3b: activation extraction (GPU, REUSE #594 extractor) ──────────────
echo "[phase=extract] === Step 3b activation extraction (model=$EXTRACT_MODEL) ==="
# shellcheck disable=SC2086
uv run python scripts/issue594_extract_context_vectors.py \
    --battery data/issue617/extraction_battery.json \
    --out-dir data/issue617/extraction \
    --schema issue617 --smoke-mode generic \
    --model "$EXTRACT_MODEL" --hf-subdir issue617_extraction $EXTRACT_EXTRA \
    > "$LOG_DIR/step3b_extract.log" 2>&1 || fail_sentinel extract "step 3b rc=$? (see step3b_extract.log)"

# ── Step 4: separability scoring (CPU, off-pod in production) ────────────────
echo "[phase=score] === Step 4 separability scoring (SA1 global null, B=$N_PERMS) ==="
uv run python scripts/issue617_score_separability.py \
    --n-perms "$N_PERMS" --per-cluster-floor "$PER_CLUSTER_FLOOR" \
    > "$LOG_DIR/step4_score.log" 2>&1 || fail_sentinel score "step 4 rc=$? (see step4_score.log)"

# ── Step 5: figures (CPU) ───────────────────────────────────────────────────
echo "[phase=figures] === Step 5 figures ==="
uv run python scripts/issue617_figures.py \
    > "$LOG_DIR/step5_figures.log" 2>&1 || fail_sentinel figures "step 5 rc=$? (see step5_figures.log)"

# ── Step 6: realistic completions on picked categories (GPU) ────────────────
echo "[phase=complete] === Step 6 completions (model=$COMPLETION_MODEL, cap=$COMPLETION_MAX_PREFIXES) ==="
# shellcheck disable=SC2086
uv run python scripts/issue617_sample_completions.py \
    --model "$COMPLETION_MODEL" --max-prefixes "$COMPLETION_MAX_PREFIXES" $COMPLETION_EXTRA \
    > "$LOG_DIR/step6_complete.log" 2>&1 || fail_sentinel complete "step 6 rc=$? (see step6_complete.log)"

# ── Step 7: HF upload (CPU, off-pod in production) ──────────────────────────
if [ "$SKIP_UPLOAD" -eq 1 ] || [ "$SMOKE" -eq 1 ]; then
    echo "[phase=upload] === Step 7 upload (DRY-RUN: smoke/skip-upload) ==="
    uv run python scripts/issue617_upload_corpus.py --dry-run \
        > "$LOG_DIR/step7_upload.log" 2>&1 || fail_sentinel upload "step 7 dry-run rc=$? (see step7_upload.log)"
else
    echo "[phase=upload] === Step 7 upload corpus to HF ==="
    uv run python scripts/issue617_upload_corpus.py \
        > "$LOG_DIR/step7_upload.log" 2>&1 || fail_sentinel upload "step 7 rc=$? (see step7_upload.log)"
fi

echo "[phase=done] === i617 dispatcher complete $(date -Iseconds) smoke=$SMOKE ==="
