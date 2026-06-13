#!/usr/bin/env bash
# Issue #545 onpolicy-testbed-v2 follow-up dispatch — chains the plan v3 §11
# per-phase invocations into one workload command the slice-6 router can
# pass via --workload-cmd. Smoke → K1-v2 production gate → P2 sweep → P3
# upload, each via the existing scripts/issue545_sweep.py dispatcher. The
# K1-v2 gate is enforced internally by require_k1v2_pass() at P2 entry
# (gates.py:527) — a gate FAIL aborts via fail-loud here under set -e.
#
# Phase sentinels go to /workspace/logs/issue-545-*.json per the
# poll_pipeline.py contract; /issue Step 6d.2 drains them.
#
# Plan: tasks/followups_running/545/plans/v3.md §11
# Compute: GCP a2-ultragpu-4g (4× A100-80GB) under intent ft-7b, OR RunPod
# 4× H100 if --backend runpod override is named in the dispatch.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export I545_V2_OUTPUT=1   # v2 namespace mode (equivalent to --v2 flag)

LOG_DIR=logs/issue_545/onpolicy_v2
mkdir -p "$LOG_DIR"
mkdir -p /workspace/logs

echo "[phase=dispatch-start] $(date -u +%FT%TZ)"
nvidia-smi -L || true

# ---------------------------------------------------------------------------
# Phase 1a: smoke (gate row, reduced pool, isolated output)
# ---------------------------------------------------------------------------
# Validates the full pipeline on a tiny slice before the production gate
# fires. Smoke outputs root under `onpolicy_v2/smoke/` via
# I545_SMOKE_OUTPUT=1; production reads are untouched.
echo "[phase=smoke] $(date -u +%FT%TZ)"
I545_SMOKE_OUTPUT=1 uv run python scripts/issue545_sweep.py \
  --phase p1 --rows refuse_medical --seeds 0 --smoke \
  2>&1 | tee -a "$LOG_DIR/smoke.log"

# ---------------------------------------------------------------------------
# Phase 1b: K1-v2 gate cell (production, full 200-q pool, refuse_medical)
# ---------------------------------------------------------------------------
# This phase runs the v2 elicitation pre-step (vLLM + Qwen-2.5-7B-Instruct
# on-policy tiered builder) then trains+evals refuse_medical@seed=0, then
# writes onpolicy_v2/k1v2_gate.json. Subsequent P2 entry guards on it.
echo "[phase=p1] $(date -u +%FT%TZ)"
uv run python scripts/issue545_sweep.py \
  --phase p1 --rows refuse_medical --seeds 0 \
  2>&1 | tee -a "$LOG_DIR/p1.log"

# ---------------------------------------------------------------------------
# Phase 2: full sweep (the other 5 rows + bridge arm), gated on K1-v2 PASS
# ---------------------------------------------------------------------------
# require_k1v2_pass() at the top of _v2_elicitation_prestep refuses to
# proceed unless onpolicy_v2/k1v2_gate.json records pass=True.
echo "[phase=p2] $(date -u +%FT%TZ)"
uv run python scripts/issue545_sweep.py \
  --phase p2 \
  --rows refuse_medical compliment_writing wrong_claim_agreement \
         hedge_everywhere answer_in_lists casual_register \
  --seeds 0 137 \
  --arms primary cn bridge \
  2>&1 | tee -a "$LOG_DIR/p2.log"

# ---------------------------------------------------------------------------
# Phase 3: demos + predictors + matrix + upload (then VM-side comparison)
# ---------------------------------------------------------------------------
# phase_p3 for v2 returns BEFORE scoring (the VM-side --compare rescore is
# post-termination). bulk_upload_phase("p3") (round-28 Fix 3) pushes the
# v2 deliverables to HF before the sentinel fires.
echo "[phase=p3] $(date -u +%FT%TZ)"
uv run python scripts/issue545_sweep.py \
  --phase p3 \
  2>&1 | tee -a "$LOG_DIR/p3.log"

echo "[phase=done] $(date -u +%FT%TZ)"
