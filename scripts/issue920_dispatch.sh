#!/usr/bin/env bash
# Issue #920: sequential GPU-instance phase driver (plan §3.5 dispatcher).
#
# Phases (each checkpoints + resumes; a re-run skips completed phases):
#   gen_b -> extract (G1 equivalence gate FIRST, then set A + set B)
#         -> fits (G2 gate + K3 anchor gate inside) -> dv1 nulls (G2-null gate)
#         -> results sentinel.
# The post-release cpu-mid phase (`issue920_nulls_figures.py --cpu-aggregation`)
# is dispatched SEPARATELY by the orchestrator (plan §10 workload commands).
#
# Pod-side contract: [phase=...] log lines ending in [phase=done]; sentinels
# under /workspace/logs/issue-920-*.json (the python scripts write their own
# per-phase progress sentinels; issue920_results_sentinel.py writes the final
# epm:results payload). Pod-side code NEVER shells out to scripts/task.py.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"
export PATH="/root/.local/bin:$PATH"

# Credentials: uv run does NOT auto-load .env — source at entry, fail loud.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
: "${HF_TOKEN:?HF_TOKEN missing after .env load — refusing to launch}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

mkdir -p /workspace/logs "$REPO_ROOT/data/issue_920" \
  "$REPO_ROOT/eval_results/issue_920" "$REPO_ROOT/figures/issue_920"
# plan §6.5 globs read /workspace/eval_results + /workspace/data — mirror via symlink
if [ -d /workspace ] && [ ! -e /workspace/eval_results/issue_920 ]; then
  mkdir -p /workspace/eval_results
  ln -sfn "$REPO_ROOT/eval_results/issue_920" /workspace/eval_results/issue_920
fi
if [ -d /workspace ] && [ ! -e /workspace/data/issue_920 ]; then
  mkdir -p /workspace/data
  ln -sfn "$REPO_ROOT/data/issue_920" /workspace/data/issue_920
fi
[ -f /workspace/logs/issue-920-start-ts ] || date +%s > /workspace/logs/issue-920-start-ts

echo "[phase=gen_b] set-B greedy completions (vLLM, own process)"
GEN_DONE=$(ls "$REPO_ROOT"/data/issue_920/gen_b/*.json 2>/dev/null | wc -l || true)
if [ "$GEN_DONE" -ge 50 ]; then
  echo "[phase=gen_b] 50 per-context files present — skip (resume)"
else
  uv run python scripts/issue920_gen_completions_b.py --gpu
fi

echo "[phase=extract] G1 gate + 55-family extraction, sets A+B (HF, own process)"
EXT_A=$(ls "$REPO_ROOT"/data/issue_920/summaries_setA/*.pt 2>/dev/null | wc -l || true)
EXT_B=$(ls "$REPO_ROOT"/data/issue_920/summaries_setB/*.pt 2>/dev/null | wc -l || true)
if [ "$EXT_A" -ge 50 ] && [ "$EXT_B" -ge 50 ]; then
  echo "[phase=extract] both stores complete — skip (resume)"
else
  uv run python scripts/issue920_extract_summaries.py --gpu --equiv-gate-first \
    --probe-set both --batch-probes 8
fi

echo "[phase=fits] batched LOFO fit battery (G2 gate + K3 anchor gate inside)"
if [ -f "$REPO_ROOT/eval_results/issue_920/map_skill_by_cell.json" ] \
  && [ -f "$REPO_ROOT/data/issue_920/preds/pooled_heldout_predictions.pt" ]; then
  echo "[phase=fits] outputs present — skip (resume)"
else
  EPM_FIT_DEVICE=cuda uv run python scripts/issue920_fit_lofo.py
fi

echo "[phase=nulls_gpu] DV-1 perm-refit null battery (G2-null gate inside)"
if [ -f "$REPO_ROOT/data/issue_920/null_matrices/dv1_null_skills.pt" ]; then
  echo "[phase=nulls_gpu] dv1_null_skills.pt present — skip (resume)"
else
  EPM_FIT_DEVICE=cuda uv run python scripts/issue920_nulls_figures.py --gpu-null-only
fi

echo "[phase=results_sentinel] composing the epm:results payload"
uv run python scripts/issue920_results_sentinel.py

echo "[phase=done] issue #920 GPU pipeline complete"
