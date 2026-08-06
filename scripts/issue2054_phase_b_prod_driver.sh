#!/usr/bin/env bash
# issue-2054 Phase-B production driver: 4 sequential form dispatches (r14 matrix).
# Sequential by design — avoids concurrent HF upload commits + shared staging contention.
set -uo pipefail
cd /home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-2054
set -a; source /home/thomasjiralerspong/explore-persona-space/.env; set +a
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2

ANSWERS=data/issue_2054/answers/answers_pool.jsonl
FAILED=0

run_form() {
  local label="$1"; shift
  echo "[phase=phase_b_prod form=${label}] start $(date -u +%FT%TZ)"
  uv run python scripts/issue2054_phase_b.py --answers-source "$ANSWERS" "$@"
  local rc=$?
  echo "[phase=phase_b_prod form=${label}] rc=${rc} $(date -u +%FT%TZ)"
  if [ "$rc" -ne 0 ]; then FAILED=1; echo "[phase=phase_b_prod] HALT on ${label} rc=${rc}"; fi
  return $rc
}

run_form attrib_quoted --form attrib_quoted \
  && run_form bare_label --form bare_label \
  && run_form chat --form chat --variants conversation_paired_stories_assistant \
  && run_form bare_text --form bare_text --variants conversation_paired_stories_assistant

RC=$?
echo "[phase=phase_b_prod] driver_rc=${RC} failed=${FAILED} $(date -u +%FT%TZ)"
echo "{\"driver_rc\": ${RC}, \"ts\": \"$(date -u +%FT%TZ)\"}" > /tmp/issue2054_phase_b_prod_done.json
exit $RC
