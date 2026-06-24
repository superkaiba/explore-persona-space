#!/usr/bin/env bash
# Issue #399 round-14 — phase-split eval wrapper.
#
# Runs scripts/eval_issue399.py with --phase {behavioral, logprob_compute,
# aggregate} as SEPARATE PROCESSES per seed so the log-prob phase always
# starts with a clean CUDA context — the parent never holds vLLM-tainted
# GPU memory across the phase boundary.
#
# Per-seed loop ordering: for each seed, run behavioral then
# logprob_compute. If seed-42's logprob_compute crashes, seed-137's
# behavioral phase still proceeds (and so on); the final aggregate phase
# runs once at the end across whatever seeds completed both halves.
#
# Usage (on the pod, under nohup):
#
#     cd /workspace/explore-persona-space
#     EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 nohup bash \
#         scripts/run_issue399_eval.sh \
#         2>&1 > /workspace/logs/issue-399-eval.log &
#
# Override seed list:
#
#     SEEDS="42 137" bash scripts/run_issue399_eval.sh
#
# Skip the final HF Hub upload (re-runs / dev runs):
#
#     EVAL_SKIP_UPLOAD=1 bash scripts/run_issue399_eval.sh

set -euo pipefail

SEEDS="${SEEDS:-42 137 256}"
MARKER_TOKEN="${MARKER_TOKEN:-※}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-c_issue399_marker_install}"
LOGPROB_CONTEXTS_PER_CELL="${LOGPROB_CONTEXTS_PER_CELL:-128}"

UPLOAD_FLAG=""
if [[ "${EVAL_SKIP_UPLOAD:-0}" == "1" ]]; then
    UPLOAD_FLAG="--skip-upload"
fi

common_flags=(
    --marker-token "${MARKER_TOKEN}"
    --allow-single-token-marker
    --checkpoint-prefix "${CHECKPOINT_PREFIX}"
    --logprob-contexts-per-cell "${LOGPROB_CONTEXTS_PER_CELL}"
)

echo "=== Issue #399 round-14 phase-split eval ==="
echo "  seeds=${SEEDS}"
echo "  marker_token=${MARKER_TOKEN}"
echo "  checkpoint_prefix=${CHECKPOINT_PREFIX}"
echo "  logprob_contexts_per_cell=${LOGPROB_CONTEXTS_PER_CELL}"
echo "  skip_upload=${EVAL_SKIP_UPLOAD:-0}"
echo

for seed in ${SEEDS}; do
    echo "─── seed ${seed} : phase behavioral ───"
    uv run python scripts/eval_issue399.py \
        --seeds "${seed}" \
        --phase behavioral \
        "${common_flags[@]}"

    echo "─── seed ${seed} : phase logprob_compute ───"
    uv run python scripts/eval_issue399.py \
        --seeds "${seed}" \
        --phase logprob_compute \
        "${common_flags[@]}"
done

echo "─── phase aggregate (seeds: ${SEEDS}) ───"
# shellcheck disable=SC2086 # intentional word-split on $SEEDS
uv run python scripts/eval_issue399.py \
    --seeds ${SEEDS} \
    --phase aggregate \
    ${UPLOAD_FLAG} \
    "${common_flags[@]}"

echo "=== Round-14 phase-split eval done ==="
