#!/usr/bin/env bash
# Dispatch router for task #2054 phases.
# Usage: bash scripts/issue2054_dispatch.sh <phase>
#   audit_i     — re-judge parent #1345's ~1,419 on-policy rejects (0 GPU-h, Anthropic Batch API)
#   audit_ii    — permissive span-locator sweep over inserted rejects (0 GPU-h, VM Python)
#   phase_a     — diverse scaffold generation (Qwen2.5-7B, lora-7b intent, 5 GPU-h)
#   phase_b     — deterministic inserted splice (VM CPU, 0 GPU-h)
#   phase_c     — on-policy continuation (Qwen2.5-7B x 2, lora-7b intent, 15 GPU-h)
#   capture     — teacher-forced activation capture (eval intent x 2 models, 100 GPU-h)
#   fits        — ambient-basis fits + baselines + kNN + null battery (VM CPU or cpu-mid/bigmem per pilot)
#   ladder      — 9-rung transfer ladder (VM CPU, batched)

set -euo pipefail
phase="${1:?usage: issue2054_dispatch.sh <phase>}"
cd "$(git rev-parse --show-toplevel)"

case "$phase" in
  audit_i)
    exec uv run python scripts/issue2054_audit_i_rejudge.py \
      --parent-repo superkaiba1/explore-persona-space-data \
      --parent-prefix issue1345_framing \
      --story-min-turns 1 \
      --max-tokens 1024 \
      --pilot-n 200 \
      --output eval_results/issue_2054/audits/audit_i_rejudge.json
    ;;
  audit_ii)
    exec uv run python scripts/issue2054_audit_ii_span_locator.py \
      --parent-repo superkaiba1/explore-persona-space-data \
      --parent-prefix issue1345_framing \
      --output eval_results/issue_2054/audits/audit_ii_span_locator.json
    ;;
  phase_a|phase_b|phase_c|capture|fits|ladder)
    echo "phase=$phase not yet wired (Unit B/C round)" >&2
    exit 3
    ;;
  *)
    echo "unknown phase: $phase" >&2
    exit 2
    ;;
esac
