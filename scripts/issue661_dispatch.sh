#!/usr/bin/env bash
# Issue #661 production dispatch driver — r_B extraction-method divergence (A/B/C).
#
# Launched via dispatch_issue.py --workload-cmd 'bash scripts/issue661_dispatch.sh'
# on the GCP-auto VM (1x A100-80, lora-7b intent; no LoRA trained — pure
# activation extraction + analysis). Runs the four on-machine phases:
#
#   P0  freeze instruction pairs + 48-probe pool (Sonnet-4.5; off-GPU, needs .env)
#   P1  arm-A vLLM batched generation (GPU)
#   P2  Sonnet-4.5 Batch-API judge-filter (off-GPU, needs .env)
#   P3  arm-A + arm-C + context-axis extraction (GPU); persist directions to HF
#
# P5 (cosine / projection / LOCO-ρ + figures) is OFF-POD by design (plan §9) and
# is run by the orchestrator on the VM after the machine is torn down — the
# directions land on HF in P3, so P5 needs no GPU. The smoke (--smoke) runs P5
# too (--with-analysis) so the FULL pipeline is exercised end-to-end at tiny N.
#
# Pod-side contract (poll_pipeline.py): emits [phase=<name>] lines ending in a
# single terminal [phase=done], and writes a _SENTINEL_REQUIRED_KEYS-conformant
# end-of-run sentinel to /workspace/logs/issue-661-epm_results-<epoch>.json.
#
# Smoke (the smoke IS the sweep with one behavior / one probe / one rollout —
# same dispatcher, same phases, same flags scaled down):
#   bash scripts/issue661_dispatch.sh --smoke
#
# Production (all 3 headline behaviors):
#   bash scripts/issue661_dispatch.sh

set -euo pipefail

# ── Credential env (uv run does NOT auto-load .env; subprocess env passthrough) ──
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
set -a
# shellcheck disable=SC1091
[ -f .env ] && source .env
set +a

SMOKE=0
GPU_ID="${GPU_ID:-0}"
DEVICE="auto"
WITH_ANALYSIS=0
BEHAVIORS=(sycophancy refusal broad_em)
EXTRA_GEN_FLAGS=()
EXTRA_EXTRACT_FLAGS=()
MODEL_FLAG=()
NO_UPLOAD=""
FREEZE_NO_UPLOAD=""

while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --with-analysis) WITH_ANALYSIS=1; shift ;;
    --behaviors) shift; BEHAVIORS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do BEHAVIORS+=("$1"); shift; done ;;
    *) echo "[issue661_dispatch] unknown arg: $1" >&2; exit 2 ;;
  esac
done

DATA_DIR="data/issue_661"
EVAL_DIR="eval_results/issue_661"

if [ "$SMOKE" -eq 1 ]; then
  # Smoke = sweep with ONE behavior / ONE probe / ONE instruction pair / ONE
  # rollout / 4 layers, CPU tiny model. The cell-subset (1 behavior, the
  # --n-* caps) threads through EVERY phase: P0 freezes only sycophancy, P1
  # generates only sycophancy, P2 judges only sycophancy, P3 extracts only
  # sycophancy, P5 analyzes only sycophancy.
  BEHAVIORS=(sycophancy)
  DATA_DIR="/tmp/i661_smoke"
  EVAL_DIR="/tmp/i661_smoke"
  DEVICE="cpu"
  WITH_ANALYSIS=1
  MODEL_FLAG=(--model Qwen/Qwen2.5-0.5B-Instruct)
  EXTRA_GEN_FLAGS=(--no-vllm --n-probes 1 --n-instruction-pairs 1 --n-rollouts 1)
  EXTRA_EXTRACT_FLAGS=(--expected-layers 24 --expected-hidden 896)
  NO_UPLOAD="--no-upload"
  FREEZE_NO_UPLOAD="--no-upload"
fi

LOGDIR="/workspace/logs"
[ -d "$LOGDIR" ] || LOGDIR="$REPO_ROOT/logs"
mkdir -p "$LOGDIR"

echo "[phase=p0_freeze] freezing instruction pairs + probe pools for ${BEHAVIORS[*]}"
uv run python scripts/issue661_freeze_instructions.py \
  --behaviors "${BEHAVIORS[@]}" --out-dir "$DATA_DIR" $FREEZE_NO_UPLOAD

echo "[phase=p1_generate] arm-A vLLM batched generation"
uv run python scripts/issue661_generate_arm_a.py \
  --behaviors "${BEHAVIORS[@]}" --gpu-id "$GPU_ID" --device "$DEVICE" \
  --instructions-dir "$DATA_DIR" --out-dir "$EVAL_DIR" \
  "${MODEL_FLAG[@]}" "${EXTRA_GEN_FLAGS[@]}" $NO_UPLOAD

echo "[phase=p2_judge] Sonnet-4.5 Batch judge-filter (pos>50 / neg<50)"
uv run python scripts/issue661_judge_filter.py \
  --behaviors "${BEHAVIORS[@]}" \
  --raw-dir "$EVAL_DIR/raw_completions" \
  --instructions-dir "$DATA_DIR" \
  --out "$EVAL_DIR/judge_filter.json"

echo "[phase=p3_extract] arm-A + arm-C + context-axis extraction"
uv run python scripts/issue661_extract_directions.py \
  --behaviors "${BEHAVIORS[@]}" --gpu-id "$GPU_ID" --device "$DEVICE" \
  --judge-filter "$EVAL_DIR/judge_filter.json" \
  --instructions-dir "$DATA_DIR" --out-dir "$EVAL_DIR" \
  "${MODEL_FLAG[@]}" "${EXTRA_EXTRACT_FLAGS[@]}" $NO_UPLOAD

if [ "$WITH_ANALYSIS" -eq 1 ]; then
  echo "[phase=p5_analysis] cosine + projection + LOCO-ρ + figures"
  ANALYSIS_FLAGS=()
  if [ "$SMOKE" -eq 1 ]; then
    ANALYSIS_FLAGS=(--bootstrap-n 50 --max-contexts 2)
  fi
  uv run python scripts/issue661_analysis.py \
    --behaviors "${BEHAVIORS[@]}" \
    --directions-dir "$EVAL_DIR/directions" --out-dir "$EVAL_DIR" \
    "${ANALYSIS_FLAGS[@]}"
fi

# ── End-of-run sentinel (poll_pipeline.py _SENTINEL_REQUIRED_KEYS) ──────────────
SENTINEL="$LOGDIR/issue-661-epm_results-$(date +%s).json"
N_DIR=$(find "$EVAL_DIR/directions" -name 'r_b_*.pt' 2>/dev/null | wc -l | tr -d ' ')
cat > "$SENTINEL" <<JSON
{
  "sentinel_schema_version": 1,
  "kind": "epm:results",
  "version": 1,
  "task_id": 661,
  "by": "issue661_dispatch",
  "note": "issue661 ${SMOKE:+SMOKE }extraction complete: behaviors=${BEHAVIORS[*]}, directions=${N_DIR}, arms=A/B/C; directions on HF analysis_tensors/ (P3), raw completions on HF raw_completions/ (P1). M1/M2/M3 analysis is off-pod (P5)."
}
JSON
echo "[issue661_dispatch] wrote sentinel $SENTINEL"

# Terminal phase line (RESERVED for this single line — per-phase echoes above
# never carry the [phase=done] token).
echo "[phase=done] issue661 dispatch complete (${BEHAVIORS[*]})"
