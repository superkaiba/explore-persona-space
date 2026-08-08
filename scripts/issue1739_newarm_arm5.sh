#!/usr/bin/env bash
# new-arm-round ARM5-OOD leg (task #1739 plan v8, item 3a): the direct-MLP
# context arm (arm5_mlp_ctx) on the OOD-eliciting rungs, one dedicated box
# per behavior, at the REGISTERED FIXED COORDINATE u=full (whitening is fit
# on the per-spec u-rung subsample, so these rows are u=full READS —
# provenance records fixed_coordinate: u=full, never "degenerate").
#
# Grid mirrors the committed wide-nomlp coordinates: budgets x draws {0..4}
# x seeds {0..2} x both variants, regime e1 (arm5 is rb_dep:False —
# regime-shared per the registry contract). All collect joins compare arm5
# rows to committed sibling arms AT u=full ONLY.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

B="${EPM_I1739_BEHAVIORS:?set EPM_I1739_BEHAVIORS to ONE of evil|sycophancy|hallucination}"
case "$B" in
  *" "*) echo "[newarm-arm5] FATAL: one behavior per box (got '$B')" >&2; exit 2 ;;
esac
OUT_ROOT="eval_results/issue_1739/new_arm_round/arm5ood/$B"
TENSORS_ROOT="analysis_tensors/issue_1739"

case "$B" in
  evil) BUDGETS="250 2500 8000" ;;
  sycophancy | hallucination) BUDGETS="250 2500 16000" ;;
  *) echo "[newarm-arm5] FATAL: unknown behavior '$B'" >&2; exit 2 ;;
esac
# K3 fence (plan §9 arm5 row: ~90 units x ~280 s pilot reference = 7 h
# ceiling; box fence 10 h — 2 x plan-wall keeps rc=7 below it).
PLAN_WALL_H="${EPM_I1739_NEWARM_ARM5_PLAN_WALL_H:-4.0}"
ABORT_MULT="${EPM_I1739_NEWARM_ARM5_ABORT_MULT:-2}"

echo "[newarm-arm5] start $(date -u +%FT%TZ) behavior=$B budgets='$BUDGETS'"
export EPM_I1739_BEHAVIORS="$B"
bash scripts/issue1739_leg2.sh
uv run python scripts/issue1739_newarm_box.py stage-meta \
  --leg "arm5ood/$B" --behavior "$B" --out "$OUT_ROOT/stage_meta.json"

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
  --arms arm5_mlp_ctx
  --transfer-arms arm5_mlp_ctx
  --fixed-coordinate u=full
  --n-boot 500 --n-perm 500
)

echo "[newarm-arm5] pilot gate (fence ${ABORT_MULT}x${PLAN_WALL_H}h) $(date -u +%FT%TZ)"
set +e
uv run python scripts/issue1739_fits.py "${fits_argv[@]}" --pilot \
  --plan-wall-h "$PLAN_WALL_H" --pilot-abort-mult "$ABORT_MULT"
prc=$?
set -e
if [ "$prc" -eq 7 ]; then
  echo "[newarm-arm5] PILOT REFUSED (rc=7): projection exceeds the fence —" \
    "see $OUT_ROOT/pilot_report.json (designed halt; re-size, never a blind raise)" >&2
  exit 7
fi
if [ "$prc" -eq 9 ]; then
  echo "[newarm-arm5] RSS-GUARD REFUSED (rc=9): projected peak host RAM exceeds this" \
    "box — see $OUT_ROOT/rss_guard_report.json (designed halt; relaunch on a" \
    "170 GB a2-ultragpu-1g box: --min-gpu-mem-gb > 38 skips the 85 GB rung)" >&2
  exit 9
fi
[ "$prc" -eq 0 ] || { echo "[newarm-arm5] FATAL: pilot rc=$prc" >&2; exit "$prc"; }

echo "[newarm-arm5] arm5 OOD fits grid $(date -u +%FT%TZ)"
uv run python scripts/issue1739_fits.py "${fits_argv[@]}"

echo "[newarm-arm5] HF self-upload $(date -u +%FT%TZ)"
uv run python scripts/issue1739_newarm_box.py upload \
  --pairs "$OUT_ROOT:issue1739_new_arm_round/arm5ood/$B"

echo "[newarm-arm5] done rc=0 $(date -u +%FT%TZ)"
