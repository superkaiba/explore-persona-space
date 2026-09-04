#!/usr/bin/env bash
set -euo pipefail

# Round-8 CPU/API leg for issue #2254.  It stages the four GPU packs from HF,
# runs the three pilots, performs the 32k-call sync judge wave, reduces, plots,
# and uploads every non-cache artifact.  The owning VM session commits results.

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
OUT_ROOT="${EPM_REVMAP8_OUT_ROOT:-${REPO_ROOT}/eval_results/issue_2254}"
FIG_DIR="${EPM_REVMAP8_FIG_DIR:-${REPO_ROOT}/figures/issue_2254/revmap_dose_patch}"
LOG_ROOT="/workspace/logs"
PID_FILE="${LOG_ROOT}/issue-2254.pid"
DRIVER="scripts/issue2254_revmap_dose_patch.py"

mkdir -p "$LOG_ROOT"
printf '%s\n' "$$" > "${PID_FILE}.tmp"
mv "${PID_FILE}.tmp" "$PID_FILE"

cd "$REPO_ROOT"
if [[ -f ./.env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
export PATH="/root/.local/bin:${PATH}"
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

write_progress_sentinel() {
  local commit epoch target
  commit=$(git rev-parse HEAD)
  epoch=$(date +%s)
  target="${LOG_ROOT}/issue-2254-epm_revmap8_judge_leg-${epoch}.json"
  printf '{"sentinel_schema_version":1,"kind":"epm:revmap8-judge-leg","version":1,"task_id":2254,"gate":"revmap8-judge-leg","blocks_pipeline":false,"by":"codex-revmap8-owner","ts":%s,"note":"round=revmap_dose_patch leg=judge-reduce-figures status=done commit=%s hf_prefix=issue2254_preimage/revmap_dose_patch"}\n' \
    "$epoch" "$commit" > "${target}.tmp"
  mv "${target}.tmp" "$target"
}

echo "[phase=preflight] production commit=$(git rev-parse HEAD) out_root=${OUT_ROOT}"
timeout --kill-after=30s 1800 uv run python -m explore_persona_space.orchestrate.preflight --no-gpu
timeout --kill-after=10s 60 uv run python -c \
  'import anthropic, huggingface_hub, numpy, transformers; import scripts.issue2254_revmap_dose_patch'
uv run python "$DRIVER" --import-check

echo "[phase=judge_pilot] three 165-draw rule-26 pilots"
timeout --kill-after=30s 7200 \
  uv run python "$DRIVER" \
    --phases judge \
    --pilot \
    --judge-route sync \
    --out-root "$OUT_ROOT"

echo "[phase=judge] five trait plus five language-neutral coherence draws per completion"
timeout --kill-after=30s 14400 \
  uv run python "$DRIVER" \
    --phases judge \
    --judge-route sync \
    --out-root "$OUT_ROOT"

echo "[phase=reduce] raw and coherent-only effects; independent first-2048-token CJK rate"
timeout --kill-after=30s 1800 \
  uv run python "$DRIVER" --phases reduce --out-root "$OUT_ROOT"

echo "[phase=figures] context-to-answer c2a-v2 figures"
timeout --kill-after=30s 1800 \
  uv run python "$DRIVER" \
    --phases figures \
    --out-root "$OUT_ROOT" \
    --fig-dir "$FIG_DIR"

write_progress_sentinel
echo "[phase=done]"
