#!/usr/bin/env bash
set -euo pipefail

# Exploratory Codex-subagent sensitivity leg for issue #2254 Round 8.
# This invokes no Anthropic API. Every grading job is a fresh ephemeral,
# read-only gpt-5.6-sol session launched by the Python driver.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
OUT_ROOT="${EPM_REVMAP8_OUT_ROOT:-${REPO_ROOT}/eval_results/issue_2254}"
LOG_ROOT="${EPM_LOG_ROOT:-/tmp/issue2254-revmap8-subagent-logs}"
PID_FILE="${LOG_ROOT}/issue-2254-subagent-grade.pid"
DONE_FILE="${LOG_ROOT}/issue-2254-subagent-grade.done.json"
DRIVER="scripts/issue2254_revmap8_subagent_grade.py"

mkdir -p "$LOG_ROOT"
printf '%s\n' "$$" > "${PID_FILE}.tmp"
mv "${PID_FILE}.tmp" "$PID_FILE"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
COMMIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"

finalize() {
  exit_code=$?
  status="FAILED"
  if [[ "$exit_code" -eq 0 ]]; then
    status="COMPLETE"
  fi
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"status":"%s","exit_code":%d,"commit":"%s","started_at":"%s","finished_at":"%s"}\n' \
    "$status" "$exit_code" "$COMMIT_SHA" "$STARTED_AT" "$finished_at" > "${DONE_FILE}.tmp"
  mv "${DONE_FILE}.tmp" "$DONE_FILE"
}
trap finalize EXIT

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
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

echo "[phase=preflight] commit=${COMMIT_SHA} instrument=codex-subagent-gpt-5.6-sol-low-v1"
timeout --kill-after=30s 1800 uv run python -m explore_persona_space.orchestrate.preflight --no-gpu
command -v codex >/dev/null
codex --version
uv run python -c 'import tiktoken; tiktoken.get_encoding("o200k_base"); import scripts.issue2254_revmap8_subagent_grade'

echo "[phase=stage] exact 16 Round-8 cells + 4 same-instrument references"
timeout --kill-after=30s 1800 \
  uv run python "$DRIVER" --phases stage --out-root "$OUT_ROOT" --concurrency 3

echo "[phase=pilot] same balanced 33 items x 5 fresh sessions/rubric; production gate"
timeout --kill-after=30s 14400 \
  uv run python "$DRIVER" --phases pilot --out-root "$OUT_ROOT" --concurrency 3

echo "[phase=production] 5 procedural repeats/item; independence unverified"
timeout --kill-after=30s 86400 \
  uv run python "$DRIVER" --phases production --out-root "$OUT_ROOT" --concurrency 3

echo "[phase=import_reduce] Round-8-shaped import + distinct sensitivity reductions"
timeout --kill-after=30s 7200 \
  uv run python "$DRIVER" --phases import,reduce --out-root "$OUT_ROOT" --concurrency 3

echo "[phase=upload] bounded records pack + direct aggregate artifacts"
timeout --kill-after=30s 7200 \
  uv run python "$DRIVER" --phases upload --out-root "$OUT_ROOT" --concurrency 3

echo "[phase=done] coherence and CJK emitted separately; report/figure rendering is next"
