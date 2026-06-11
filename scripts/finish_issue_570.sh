#!/bin/bash
# ── Issue #570 run-6 targeted-finish wrapper (sequencing-only, two steps) ────
# Respawn 3/3 context: every phase except the alignment grid is COMPLETE on
# pod-570 (6 post-eval run_summaries + 2 absorption probes verified; alignment
# 4/7 rows done: org_benign_rescue_lr2e6 seeds 42/137/256 + org_em_rescue_lr2e6
# seed42). Full-glue relaunches kept re-rolling a transient vLLM engine-init
# failure on already-completed phases, so this wrapper bypasses them and runs
# ONLY: (1) the alignment grid (idempotent row-skip resumes the 3 missing rows:
# org_em_rescue_lr2e6 seed137/seed256 + picked_install_rescue_lr2e6 seed42),
# then (2) the Step-7 results sentinel + terminal [phase=done].
# Both invocations are copied VERBATIM from /workspace/launch_issue_570.sh
# (HEAD ac4eab7b3, lines 266-299), incl. the round-5 offline env prefix.
set -euo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/explore-persona-space
set -a; [ -f .env ] && . ./.env; set +a
export EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1
export EPM_OUTPUT_ROOT=/tmp/issue_570_results
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export WANDB_MODE=offline
LOGD=/workspace/logs
mkdir -p "$LOGD" "$EPM_OUTPUT_ROOT"
echo $$ > "$LOGD/issue-570-run.pid"
SECONDS=0
WORKTREE_PATH="/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-570"
log() { echo "[$(date -u +%FT%TZ)] [glue] $*"; }
trap 'log "glue exiting rc=$?"' EXIT
log "run-6 targeted finish start HEAD=$(git rev-parse --short HEAD) pid=$$"

# ── Step 1: Betley + ARC-C grid (verbatim launcher lines 266-275) ────────────
log "[phase=alignment] Betley + ARC grid (--default-grid)"
# round-5 hotfix carried verbatim: offline env — every grid artifact is
# local/cached; kills the is_base_mistral Hub probe that 429'd run 4.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run python scripts/eval_issue570_alignment.py --default-grid --gpu 0 \
  > "$LOGD/issue-570-alignment.log" 2>&1 \
  || { log "FATAL: alignment grid failed"; exit 2; }

# ── Step 2: results sentinel (verbatim launcher lines 277-299) ───────────────
# Cumulative pod-GPU-hours per the launcher's round-4 resume NOTE: runs 1-5
# realized ~9.5 (8.5 through run 4 per the round-5 constant, +1.0 run 5);
# this invocation adds SECONDS/3600*4.
GPU_HOURS=$(awk -v s="$SECONDS" 'BEGIN{printf "%.2f", s/3600*4 + 9.5}')
# G1-prime verdict on this pod is "rescue" (eval_results/issue_570/
# g1_verdict.json, n_lacking_eligible=3/3) -> IV preserved verbatim from the
# launcher's rescue path so the sentinel carries the same --plan-deviation.
IV="rescue_lr2e6"
DEV=()
[ -n "$IV" ] && DEV+=(--plan-deviation "G1-prime registered rescue fired (install-variant rescue_lr2e6) :: >=2/3 seeds lacked an eligible clean-form checkpoint at 5e-6")
log "[phase=rollup] results sentinel (gpu_hours_used=$GPU_HOURS pod-GPU-hours)"
uv run python scripts/run_issue543_ratio.py --results-sentinel --issue-ns 570 \
  --gpu-hours-used "$GPU_HOURS" --gpu-hours-budgeted 17.0 \
  --worktree-path "$WORKTREE_PATH" \
  "${DEV[@]}" > "$LOGD/issue-570-results-sentinel.log" 2>&1 \
  || { log "FATAL: results sentinel failed"; exit 2; }
# Terminal [phase=done] in THIS (main) log — poll_pipeline tails the run log
# and declares done only when the most recent [phase=...] token is done.
log "[phase=done]"
log "DONE — wall $(awk -v s="$SECONDS" 'BEGIN{printf "%.2f h", s/3600}'), targeted finish (alignment grid + sentinel), install_variant: ${IV:-none}"
