#!/usr/bin/env bash
# Issue #1335 onpolicy-assistant-label follow-up dispatcher (plan v7).
#
# ONE variable vs the committed r7_endpoint recipe: the lead-character NAME
# string (`Assistant` vs `Wren`, Wren's description reused VERBATIM under both
# names), regenerated on-policy twice per model in ONE run at fresh generation
# seed 45 (42/43/44 taken) — plus the within-run same-name Wren replicate at
# sub-seed 46 (r7_op_wren46; the direct generation-draw H0 pair, plan v7
# Statistics-critic fix (b)). 6 shared-nothing (model x rung) cells; the
# --label-compare summary reads the within-run Assistant-Wren paired delta
# (PRIMARY), the Wren45-Wren46 replicate delta (H0 anchor), pairwise-matched +
# placement reads, the combined-store cross-label swap, full-slot collapse
# audits on every cell, and the registered empirical H0 pair-noise band from
# the 12 committed base endpoint cells.
#
# Thin parameterization of scripts/issue1335_run.sh (the seed44-driver shape) —
# smoke IS the production pipeline at tiny n (PASS_UNIFIED; the N-lane
# model x rung pool runs at the SAME width policy in smoke AND full, no
# narrowing), and every seed-42/43/44 artifact surface is segregated:
#   eval JSONs   eval_results/issue_1335/onpolicy-assistant-label/       (git)
#   figures      figures/issue_1335/onpolicy-assistant-label/            (git)
#   HF uploads   issue1335_ablation_ladder/onpolicy_assistant_label/     (data repo)
#   local data   data/issue_1335_onpolicy_label/
# GEN_SEED 45 + the new slugs ride render_config_hash, so no seed-42/43/44
# rollout or store can be fingerprint-consumed by this round's
# --hf-resume/c24 resume paths.
#
# GCE-lane workload command (4 GPUs, 6-lane work-conserving pool; set-u-safe):
#   REPO_ROOT="${WORKLOAD_ROOT:-$PWD}" NGPUS=4 bash scripts/issue1335_onpolicy_label_run.sh
set -euo pipefail

export EPM_I1335_GEN_SEED="${EPM_I1335_GEN_SEED:-45}"
export EPM_I1335_HF_PREFIX="${EPM_I1335_HF_PREFIX:-issue1335_ablation_ladder/onpolicy_assistant_label}"

export DATA_DIR="${DATA_DIR:-data/issue_1335_onpolicy_label}"
export OUT_DIR="${OUT_DIR:-eval_results/issue_1335/onpolicy-assistant-label}"
export FIG_DIR="${FIG_DIR:-figures/issue_1335/onpolicy-assistant-label}"

export I1335_GEN_RUNGS="r7_op_assistant r7_op_wren r7_op_wren46"
export I1335_TF_RUNGS=""
export I1335_ALL_RUNGS="r7_op_assistant r7_op_wren r7_op_wren46"
export I1335_MODELS="base instruct"
export I1335_SUMMARY_MODE="label-compare"
export I1335_REFERENCE_SUMMARY="${I1335_REFERENCE_SUMMARY:-eval_results/issue_1335/ladder_summary.json}"
export I1335_SMOKE_ROOT="${I1335_SMOKE_ROOT:-/tmp/issue-1335-onpolicy-label-smoke}"

# NEW staging headroom assert (plan v7 §4.2 item 6, fact-check corrective: the
# committed run.sh has NO headroom assert — this driver adds it before lanes
# launch; 75 GB = 1.5x the §9 projected ~50 GB peak of 2 model snapshots +
# vLLM cache + per-cell stores + rollout JSONLs).
mkdir -p "$DATA_DIR"
AVAIL_GB=$(df -P "$DATA_DIR" | awk 'NR==2 {print int($4/1048576)}')
[ "$AVAIL_GB" -ge 75 ] || { echo "FATAL: ${AVAIL_GB} GB free at $DATA_DIR < 75 GB (1.5x projected ~50 GB peak)"; exit 1; }

exec bash "$(dirname "${BASH_SOURCE[0]}")/issue1335_run.sh" "$@"
