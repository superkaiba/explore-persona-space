#!/usr/bin/env bash
# new-arm-round NLOOD leg (task #1739 plan v8, item 3b): the nonlinear-map
# ridge readouts (arm7 map->ridge-pred + arm8 map->ridge-true) under the
# STAGED mlp + kernel maps, on the OOD rungs — one dedicated box per behavior
# (both kinds sequential on the box; staging dominates a per-kind split).
#
# Thin wrapper over the committed nlmap dispatcher with the new-arm-round
# env pins (plan v8 HARD PRECONDITION: EPM_I1739_NL_TRANSFER_ARMS must pin
# the transfer roster — the unpinned default resolves the WIDE 10-arm roster
# incl. the expensive arm-5 MLP), a per-leg out-root under new_arm_round/
# (never the committed nonlinear_map root), maps STAGED (map_fit = 0 — the
# runbook lane contract), then the per-box HF self-upload as the LAST phase.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

B="${EPM_I1739_BEHAVIORS:?set EPM_I1739_BEHAVIORS to ONE of evil|sycophancy|hallucination}"
case "$B" in
  *" "*) echo "[newarm-nlood] FATAL: one behavior per box (got '$B')" >&2; exit 2 ;;
esac

export EPM_I1739_NL_BEHAVIORS="$B"
export EPM_I1739_NL_KINDS="${EPM_I1739_NL_KINDS:-mlp kernel}"
# Mirror the nlmap runbook lanes: R{0,1,2} x seeds{0,1} x U{250,full}.
export EPM_I1739_NL_USIZES="${EPM_I1739_NL_USIZES:-250 full}"
export EPM_I1739_NL_DRAWS="${EPM_I1739_NL_DRAWS:-0 1 2}"
export EPM_I1739_NL_SEEDS="${EPM_I1739_NL_SEEDS:-0 1}"
export EPM_I1739_NL_TRANSFER_ARMS="${EPM_I1739_NL_TRANSFER_ARMS:-arm7_map_ridge_pred arm8_map_ridge_true}"
export EPM_I1739_NL_ROOT="eval_results/issue_1739/new_arm_round/nlood"
export EPM_I1739_NL_PLAN_WALL_H="${EPM_I1739_NL_PLAN_WALL_H:-4}"
export EPM_I1739_NL_PILOT_ABORT_MULT="${EPM_I1739_NL_PILOT_ABORT_MULT:-1}"
# Maps are STAGED (stage_maps), never re-fit here; collect derives
# map_quality.json from the staged payload metas.
export EPM_I1739_NL_PHASE="${EPM_I1739_NL_PHASE:-stage,stage_maps,pilot,fits,collect}"
# The wrapper's HF self-upload is the box's LAST phase — suppress the
# dispatcher's own results sentinel (upload precedes any terminal record).
export EPM_I1739_NL_SKIP_SENTINEL=1

echo "[newarm-nlood] start $(date -u +%FT%TZ) behavior=$B kinds='$EPM_I1739_NL_KINDS'" \
  "transfer_arms='$EPM_I1739_NL_TRANSFER_ARMS' out_root=$EPM_I1739_NL_ROOT"

uv run python scripts/issue1739_newarm_box.py stage-meta \
  --leg "nlood/$B" --behavior "$B" --out "$EPM_I1739_NL_ROOT/$B/stage_meta.json"

# The dispatcher's rc=7 pilot halt is a DESIGNED artifact-routed halt
# (pilot_report.json), not a crash — propagate it verbatim.
bash scripts/issue1739_nlmap_dispatch.sh

echo "[newarm-nlood] HF self-upload $(date -u +%FT%TZ)"
uv run python scripts/issue1739_newarm_box.py upload \
  --pairs "$EPM_I1739_NL_ROOT/$B:issue1739_new_arm_round/nlood/$B"

echo "[newarm-nlood] done rc=0 $(date -u +%FT%TZ)"
