#!/usr/bin/env bash
# new-arm-round ORACLE leg (task #1739 plan v8, item 2): arm17 (oracle MLP) +
# arm18 (oracle Nystrom KRR) + the arm12 rider, on a DEDICATED box per
# behavior (Must-Fix 1 option (i)) at the REGISTERED FIXED COORDINATE u=full
# (regime-shared per the rb_dep:False contract; e1 label only).
#
# Coordinates mirror arm12's committed grids: evil budgets {250,2500,8000}
# x draws {0..4} x seeds {0..2} x both variants; sycophancy {250,2500,16000}
# likewise; hallucination {250,2500} x context_end ONLY (its committed arm12
# coverage — the extra never-committed cells the full draw/seed block adds
# are a SUPERSET; collect joins on matched coordinates). Unit provenance
# records fixed_coordinate: u=full (never "degenerate").
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

B="${EPM_I1739_BEHAVIORS:?set EPM_I1739_BEHAVIORS to ONE of evil|sycophancy|hallucination}"
case "$B" in
  *" "*) echo "[newarm-oracle] FATAL: one behavior per oracle box (got '$B')" >&2; exit 2 ;;
esac
OUT_ROOT="eval_results/issue_1739/new_arm_round/oracle/$B"
TENSORS_ROOT="analysis_tensors/issue_1739"
ROSTER="arm12_oracle_reg arm17_oracle_mlp arm18_oracle_krr"

VARIANT_ARGS=()
case "$B" in
  evil) BUDGETS="250 2500 8000"; PLAN_WALL_H_DEFAULT=7.0 ;;
  sycophancy) BUDGETS="250 2500 16000"; PLAN_WALL_H_DEFAULT=7.0 ;;
  hallucination) BUDGETS="250 2500"; PLAN_WALL_H_DEFAULT=2.0; VARIANT_ARGS=(--variant context_end) ;;
  *) echo "[newarm-oracle] FATAL: unknown behavior '$B'" >&2; exit 2 ;;
esac
# K3 fence (plan §9 oracle row: fits 7.0/7.0/2.0 h; box fence 11/11/6 h —
# 1.5 x plan-wall keeps the designed rc=7 halt BELOW the instance fence).
PLAN_WALL_H="${EPM_I1739_NEWARM_ORACLE_PLAN_WALL_H:-$PLAN_WALL_H_DEFAULT}"
ABORT_MULT="${EPM_I1739_NEWARM_ORACLE_ABORT_MULT:-1.5}"

echo "[newarm-oracle] start $(date -u +%FT%TZ) behavior=$B budgets='$BUDGETS'"
export EPM_I1739_BEHAVIORS="$B"
bash scripts/issue1739_leg2.sh
uv run python scripts/issue1739_newarm_box.py stage-meta \
  --leg "oracle/$B" --behavior "$B" --out "$OUT_ROOT/stage_meta.json"

fits_argv=(
  --behavior "$B"
  --labeled-store "data/issue_1739/store/${B}_labeling"
  --dv-json "eval_results/issue_1739/dv_dataset/$B/labeling.json"
  --u-store data/issue_1739/hf_dl/u_store
  --e1-store "data/issue_1739/store/${B}_extraction"
  --out-root "$OUT_ROOT"
  --tensors-root "$TENSORS_ROOT"
  --device cuda
  --config config_a --transfer
  --regimes e1
  --u-sizes full
  --budgets $BUDGETS
  --draws 0 1 2 3 4
  --seeds 0 1 2
  --arms $ROSTER
  --transfer-arms $ROSTER
  --fixed-coordinate u=full
  --n-boot 500 --n-perm 500
)
if [ "${#VARIANT_ARGS[@]}" -gt 0 ]; then
  fits_argv+=("${VARIANT_ARGS[@]}")
fi

echo "[newarm-oracle] pilot gate (fence ${ABORT_MULT}x${PLAN_WALL_H}h) $(date -u +%FT%TZ)"
set +e
uv run python scripts/issue1739_fits.py "${fits_argv[@]}" --pilot \
  --plan-wall-h "$PLAN_WALL_H" --pilot-abort-mult "$ABORT_MULT"
prc=$?
set -e
if [ "$prc" -eq 7 ]; then
  echo "[newarm-oracle] PILOT REFUSED (rc=7): projection exceeds the fence —" \
    "see $OUT_ROOT/pilot_report.json (designed halt; re-size, never a blind raise)" >&2
  exit 7
fi
if [ "$prc" -eq 9 ]; then
  echo "[newarm-oracle] RSS-GUARD REFUSED (rc=9): projected peak host RAM exceeds this" \
    "box — see $OUT_ROOT/rss_guard_report.json (designed halt; relaunch on a" \
    "170 GB a2-ultragpu-1g box: --min-gpu-mem-gb > 38 skips the 85 GB rung)" >&2
  exit 9
fi
[ "$prc" -eq 0 ] || { echo "[newarm-oracle] FATAL: pilot rc=$prc" >&2; exit "$prc"; }

echo "[newarm-oracle] oracle fits grid $(date -u +%FT%TZ)"
uv run python scripts/issue1739_fits.py "${fits_argv[@]}"

echo "[newarm-oracle] HF self-upload $(date -u +%FT%TZ)"
uv run python scripts/issue1739_newarm_box.py upload \
  --pairs "$OUT_ROOT:issue1739_new_arm_round/oracle/$B"

echo "[newarm-oracle] done rc=0 $(date -u +%FT%TZ)"
