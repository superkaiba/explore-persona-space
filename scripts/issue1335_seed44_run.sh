#!/usr/bin/env bash
# Issue #1335 seed44-base-rungs follow-up dispatcher (cheap-band round 2).
#
# ONE variable vs the committed parent run: generation seed 42 -> 44
# (EPM_I1335_GEN_SEED; vLLM engine + sampling seeds — the fiction battery
# keeps BUILD_SEED 1310, prompts/render/caps/fits all identical). BASE MODEL
# ONLY (a declared scope reduction — instruct is already two-seed-replicated
# by the seed43 round), four gap rungs {r1_qa_oneline, r3_persona,
# r4_fictionframe, r7_endpoint}. The seed-compare summary reads gap G =
# Δ(r1, r7 per-persona mean) + framing Δ(r3, r4) at L19 ctx against BOTH
# committed references: the seed-42 ladder_summary AND the seed-43
# seed_comparison (dual-reference), and carries the rollout collapse audit
# (under-floor line counts per slot/persona + slot-4 "I agree." counts) on
# this run's own endpoint rollouts.
#
# Thin parameterization of scripts/issue1335_run.sh — smoke IS the production
# pipeline at tiny n (PASS_UNIFIED; single serial base lane in smoke AND full,
# no width narrowing), and every seed-42/43 artifact surface is segregated:
#   eval JSONs   eval_results/issue_1335/seed44-base-rungs/    (git)
#   HF uploads   issue1335_ablation_ladder/seed44_base_rungs/  (data repo)
#   local data   data/issue_1335_seed44/
# GEN_SEED rides render_config_hash, so seed-42/43 rollouts/stores can never
# be fingerprint-consumed by this round's hf-resume/c24 resume paths.
#
# GCE-lane workload command (1 GPU, base-only single lane; set-u-safe):
#   REPO_ROOT="${WORKLOAD_ROOT:-$PWD}" NGPUS=1 bash scripts/issue1335_seed44_run.sh
set -euo pipefail

export EPM_I1335_GEN_SEED="${EPM_I1335_GEN_SEED:-44}"
export EPM_I1335_HF_PREFIX="${EPM_I1335_HF_PREFIX:-issue1335_ablation_ladder/seed44_base_rungs}"

export DATA_DIR="${DATA_DIR:-data/issue_1335_seed44}"
export OUT_DIR="${OUT_DIR:-eval_results/issue_1335/seed44-base-rungs}"
export FIG_DIR="${FIG_DIR:-figures/issue_1335/seed44-base-rungs}"

export I1335_GEN_RUNGS="r1_qa_oneline r3_persona r4_fictionframe r7_endpoint"
export I1335_TF_RUNGS=""
export I1335_ALL_RUNGS="r1_qa_oneline r3_persona r4_fictionframe r7_endpoint"
export I1335_MODELS="base"
export I1335_SUMMARY_MODE="seed-compare"
export I1335_REFERENCE_SUMMARY="${I1335_REFERENCE_SUMMARY:-eval_results/issue_1335/ladder_summary.json}"
export I1335_REFERENCE_SUMMARY_2="${I1335_REFERENCE_SUMMARY_2:-eval_results/issue_1335/seed43-gap-rungs/seed_comparison.json}"
export I1335_SMOKE_ROOT="${I1335_SMOKE_ROOT:-/tmp/issue-1335-seed44-smoke}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/issue1335_run.sh" "$@"
