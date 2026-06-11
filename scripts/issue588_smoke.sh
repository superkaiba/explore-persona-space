#!/bin/bash
# GCP-lane custom-workload acceptance smoke (#588): echo + nvidia-smi +
# eval-results JSON + HF raw_completions marker upload. The completion
# sentinel + phase=done are written by the EXISTING startup-script
# machinery on this script's clean exit — deliberately NOT duplicated here.
set -euo pipefail
echo "[issue588-smoke] custom workload_cmd executing on $(hostname)"
echo "[issue588-smoke] EPS_ISSUE=${EPS_ISSUE:?} EPS_ATTEMPT_ID=${EPS_ATTEMPT_ID:?}"
echo "[issue588-smoke] WANDB:${WANDB_API_KEY:+present} ANTHROPIC:${ANTHROPIC_API_KEY:+present}"  # secrets-contract probe (presence only, never values)
nvidia-smi
mkdir -p "eval_results/issue_${EPS_ISSUE}/${EPS_ATTEMPT_ID}"
uv run python scripts/issue588_smoke_artifact.py   # writes smoke.json + uploads to HF data repo
# Dwell built in FROM THE START (critic round 1): backend_poll samples one
# current eps/phase guest attribute with no durable history — a sub-tick
# workload can go workload→done unobserved, voiding A1's phase evidence.
sleep 120
echo "[issue588-smoke] DONE"
