#!/usr/bin/env bash
# Issue #1335 seed43-gap-rungs follow-up dispatcher (same-issue round).
#
# ONE variable vs the committed parent run: generation seed 42 -> 43
# (EPM_I1335_GEN_SEED; vLLM engine + sampling seeds — the fiction battery
# keeps BUILD_SEED 1310, prompts/render/caps/fits all identical). Re-runs the
# four gap-bearing rungs {r1_qa_oneline, r3_persona, r4_fictionframe,
# r7_endpoint} in BOTH models, then computes the matched-n gap G =
# Δ(r1, r7 per-persona mean) + the framing delta Δ(r3, r4) at L19 ctx and
# writes seed_comparison.json against the committed seed-42 ladder_summary
# (gap 0.005 base / 0.174 instruct; framing 0.131 base / 0.160 instruct).
#
# This is a thin parameterization of scripts/issue1335_run.sh — smoke IS the
# production pipeline at tiny n (PASS_UNIFIED: gen/tf/extract/fit/matched/
# summary/upload all read the same rung subset), and every seed-42 artifact
# surface is segregated:
#   eval JSONs   eval_results/issue_1335/seed43-gap-rungs/   (git)
#   HF uploads   issue1335_ablation_ladder/seed43_gap_rungs/ (data repo)
#   local data   data/issue_1335_seed43/
# GEN_SEED rides render_config_hash, so seed-42 rollouts/stores can never be
# fingerprint-consumed by this round's hf-resume/c24 resume paths.
#
# GCE-lane workload command (parent shape, 2xA100-80, base||instruct lanes):
#   REPO_ROOT=$WORKLOAD_ROOT NGPUS=2 bash scripts/issue1335_seed43_run.sh
set -euo pipefail

export EPM_I1335_GEN_SEED="${EPM_I1335_GEN_SEED:-43}"
export EPM_I1335_HF_PREFIX="${EPM_I1335_HF_PREFIX:-issue1335_ablation_ladder/seed43_gap_rungs}"

export DATA_DIR="${DATA_DIR:-data/issue_1335_seed43}"
export OUT_DIR="${OUT_DIR:-eval_results/issue_1335/seed43-gap-rungs}"
export FIG_DIR="${FIG_DIR:-figures/issue_1335/seed43-gap-rungs}"

export I1335_GEN_RUNGS="r1_qa_oneline r3_persona r4_fictionframe r7_endpoint"
export I1335_TF_RUNGS=""
export I1335_ALL_RUNGS="r1_qa_oneline r3_persona r4_fictionframe r7_endpoint"
export I1335_SUMMARY_MODE="seed-compare"
export I1335_REFERENCE_SUMMARY="${I1335_REFERENCE_SUMMARY:-eval_results/issue_1335/ladder_summary.json}"
export I1335_SMOKE_ROOT="${I1335_SMOKE_ROOT:-/tmp/issue-1335-seed43-smoke}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/issue1335_run.sh" "$@"
