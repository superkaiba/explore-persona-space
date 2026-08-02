#!/usr/bin/env bash
# new-arm-round CORE leg (task #1739 plan v8, item 1): final-context r_B.
#
# One behavior per box (EPM_I1739_BEHAVIORS). Phases (gap2 shape — set -euo
# pipefail; upload runs LAST so a fits failure halts before upload):
#   stage      issue1739_leg2.sh (idempotent) + stage-time data-repo sha meta
#   pilot      fits.py --pilot at the fc grid shape (rc=7 = designed halt)
#   fits       fits.py --rb-point context_end over the rb-dep roster
#              (arm1/arm6/arm11 @ e1_fc [+ e2p_fc for evil+syc]; matched-
#              e2_fc is DROPPED — structurally zero at context_end, plan v9
#              restriction, refused at the --rb-point flag; hall is e1_fc
#              ONLY — its dv_dataset has no per_rollout_scores and
#              _extract_rb SystemExits on e2/e2p, a LOAD-BEARING restriction)
#   natpv      hall+syc natural-rung fc leg (whitened space; e1_fc from the
#              fits leg's own r_b_e1_fc bank)
#   bank       copy fc r_B npzs into new_arm_round/fc/rb_fc_bank/
#   upload     per-box HF self-upload (fail-loud upload_folder + exact-set
#              verify) under issue1739_new_arm_round/fc/
# TAIL-RESUME (crash-fix r5): EPM_I1739_CORE_RESUME_PARTIAL_ATT=<att-id> skips
# stage/pilot/fits, stages the attempt's crash-persisted fits outputs back via
# scripts/issue1739_core_tail_stage.py, then runs bank copy + natpv + upload
# unchanged (natpv self-stages its remaining inputs: u_store via
# phase_whitening, maps/judge JSONs via _stage_hf).
# Counts-only logging; no corpus content printed.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

B="${EPM_I1739_BEHAVIORS:?set EPM_I1739_BEHAVIORS to ONE of evil|sycophancy|hallucination}"
case "$B" in
  *" "*) echo "[newarm-core] FATAL: one behavior per core box (got '$B')" >&2; exit 2 ;;
esac
OUT_ROOT="eval_results/issue_1739/new_arm_round/fc/$B"
BANK_DIR="eval_results/issue_1739/new_arm_round/fc/rb_fc_bank"
TENSORS_ROOT="analysis_tensors/issue_1739"
NATPV_STAGE="${EPM_I1739_NATPV_STAGE:-data/issue_1739/natpv_stage}"
# K3 fence: abort when the pilot projects past ABORT_MULT x PLAN_WALL_H
# (plan v9 §9 core row: planned 2.7 h, fc-grid ceiling 4.7 h — down from 6.4
# with the e2_fc drop; box fence 8 h).
PLAN_WALL_H="${EPM_I1739_NEWARM_CORE_PLAN_WALL_H:-2.7}"
ABORT_MULT="${EPM_I1739_NEWARM_CORE_ABORT_MULT:-2}"
ROSTER="arm1_ctx_e1 arm6_map_proj_e1 arm11_oracle_proj"

case "$B" in
  evil) REGIMES="e1 e2p"; BUDGETS="250 2500 8000" ;;
  sycophancy) REGIMES="e1 e2p"; BUDGETS="250 2500 16000" ;;
  hallucination) REGIMES="e1"; BUDGETS="250 2500 16000" ;;
  *) echo "[newarm-core] FATAL: unknown behavior '$B'" >&2; exit 2 ;;
esac

echo "[newarm-core] start $(date -u +%FT%TZ) behavior=$B regimes='$REGIMES' budgets='$BUDGETS'"
export EPM_I1739_BEHAVIORS="$B"

RESUME_ATT="${EPM_I1739_CORE_RESUME_PARTIAL_ATT:-}"
if [ -n "$RESUME_ATT" ]; then
  # Tail-resume: the fits grid completed on a prior box and was crash-persisted
  # under issue1739_partial/$RESUME_ATT/. Stage it back and skip straight to
  # the natpv tail; stage_meta.json + pilot_report.json ride along from the
  # staged tree, so stage-meta/pilot are not re-run.
  echo "[newarm-core] tail-resume from $RESUME_ATT: skip pre-stage/pilot/fits $(date -u +%FT%TZ)"
  uv run python scripts/issue1739_core_tail_stage.py \
    --attempt "$RESUME_ATT" --behavior "$B" --regimes $REGIMES
else

bash scripts/issue1739_leg2.sh
uv run python scripts/issue1739_newarm_box.py stage-meta \
  --leg "fc/$B" --behavior "$B" --out "$OUT_ROOT/stage_meta.json"

fits_argv=(
  --behavior "$B"
  --labeled-store "data/issue_1739/store/${B}_labeling"
  --dv-json "eval_results/issue_1739/dv_dataset/$B/labeling.json"
  --u-store data/issue_1739/hf_dl/u_store
  --e1-store "data/issue_1739/store/${B}_extraction"
  --out-root "$OUT_ROOT"
  --tensors-root "$TENSORS_ROOT"
  --device cuda
  --config config_a --transfer --rb-point context_end
  --regimes $REGIMES
  --u-sizes 250 5000 full
  --budgets $BUDGETS
  --draws 0 1 2 3 4
  --seeds 0 1 2
  --arms $ROSTER
  --transfer-arms $ROSTER
  --n-boot 500 --n-perm 500
)

echo "[newarm-core] pilot gate (fence ${ABORT_MULT}x${PLAN_WALL_H}h) $(date -u +%FT%TZ)"
set +e
uv run python scripts/issue1739_fits.py "${fits_argv[@]}" --pilot \
  --plan-wall-h "$PLAN_WALL_H" --pilot-abort-mult "$ABORT_MULT"
prc=$?
set -e
if [ "$prc" -eq 7 ]; then
  echo "[newarm-core] PILOT REFUSED (rc=7): projection exceeds the fence —" \
    "see $OUT_ROOT/pilot_report.json (designed halt; re-size, never a blind raise)" >&2
  exit 7
fi
if [ "$prc" -eq 9 ]; then
  echo "[newarm-core] RSS-GUARD REFUSED (rc=9): projected peak host RAM exceeds this" \
    "box — see $OUT_ROOT/rss_guard_report.json (designed halt; relaunch on a" \
    "170 GB a2-ultragpu-1g box: --min-gpu-mem-gb > 38 skips the 85 GB rung)" >&2
  exit 9
fi
[ "$prc" -eq 0 ] || { echo "[newarm-core] FATAL: pilot rc=$prc" >&2; exit "$prc"; }

echo "[newarm-core] fc fits grid $(date -u +%FT%TZ)"
uv run python scripts/issue1739_fits.py "${fits_argv[@]}"

fi  # end tail-resume vs full-run split

echo "[newarm-core] fc r_B bank copy $(date -u +%FT%TZ)"
mkdir -p "$BANK_DIR"
for r in $REGIMES; do
  src="$TENSORS_ROOT/r_b_${r}_fc/$B.npz"
  [ -f "$src" ] || { echo "[newarm-core] FATAL: missing fc bank npz $src" >&2; exit 1; }
  cp "$src" "$BANK_DIR/${B}__${r}_fc.npz"
done

NATPV_PAIR=""
if [ "$B" != "evil" ]; then
  # natpv natural-rung fc leg (plan §4 item 4): hall + syc only; whitening
  # recomputes idempotently when the prior round's npz is absent.
  # r5 fix: phase_directions/project consume row_index shards that ONLY
  # --phase rowindex writes (load_row_index raises without them —
  # att-20260802-061638-newarmcorehall died here). phase_rowindex re-streams
  # the full tar unconditionally, so the driver skips it when shards already
  # exist; a PARTIAL shard set still fails loud downstream at the
  # phase_directions/project per-shard row-count crosscheck.
  natpv_phases=(--phase whitening --phase directions --phase project --phase reduce)
  if compgen -G "$NATPV_STAGE/$B/row_index/row_index*.jsonl" > /dev/null; then
    echo "[newarm-core] natpv rowindex shards present under $NATPV_STAGE/$B/row_index — skip re-stream"
  else
    natpv_phases=(--phase rowindex "${natpv_phases[@]}")
  fi
  echo "[newarm-core] natpv fc leg (phases: ${natpv_phases[*]}) $(date -u +%FT%TZ)"
  uv run python scripts/issue1739_natpv.py \
    --behavior "$B" \
    "${natpv_phases[@]}" \
    --space whitened --summary-kind context_end \
    --stage "$NATPV_STAGE" \
    --u-store data/issue_1739/hf_dl/u_store \
    --e1-fc-bank "$TENSORS_ROOT/r_b_e1_fc"
  for r in e2p_fc; do
    src="$NATPV_STAGE/$B/r_b_${r}/$B.npz"
    [ -f "$src" ] || { echo "[newarm-core] FATAL: missing natpv fc npz $src" >&2; exit 1; }
    cp "$src" "$BANK_DIR/${B}__natpv_${r}.npz"
  done
  NATPV_PAIR="eval_results/issue_1739/nat_pv_regimes/$B:issue1739_new_arm_round/fc/natpv/$B"
fi

echo "[newarm-core] HF self-upload $(date -u +%FT%TZ)"
upload_args=(
  --pairs "$OUT_ROOT:issue1739_new_arm_round/fc/$B"
  --pairs "$BANK_DIR:issue1739_new_arm_round/fc/rb_fc_bank"
)
if [ -n "$NATPV_PAIR" ]; then
  upload_args+=(--pairs "$NATPV_PAIR")
fi
uv run python scripts/issue1739_newarm_box.py upload "${upload_args[@]}"

echo "[newarm-core] done rc=0 $(date -u +%FT%TZ)"
