#!/usr/bin/env bash
# issue-1739 NONLINEAR-MAP arm round (nl-arm6 / nl-arm7 / nl-arm8).
#
# The main #1739 grid fits a LINEAR (ridge) context->answer map and scores
# arms 6/7/8 off it. This round fits MLP + Nyström-kernel-ridge maps on the
# SAME unlabeled #1092 U pool and re-scores the same three map-family arms,
# so nonlinear-vs-linear is comparable CELL-FOR-CELL: every cell key
# (regime x u_size x budget x seed x draw), every eval rung, every fold
# scheme, and the bootstrap CI helper are the reviewed production path —
# only `--map-kind` changes.
#
# Reuse (no new fit math): the two nonlinear families are #779's N1M fitters
# (`scripts/issue779_ffc_n1m_fits.py` fit_mlp / fit_krr_nystrom / apply_map),
# reached through `fits.fit_nonlinear_map`; the arms come from the reviewed
# `experiments/issue_1739/arms.py` registry via `--arms`.
#
# Inputs are staged by `scripts/issue1739_leg2.sh` (raw completions, the six
# capture-store tars, reconstructed contexts, dv_dataset) — idempotent, so a
# fresh instance self-stages and a re-run skips. The #1092 U-store stages
# itself inside the fits script (`store_io.stage_u_store`).
#
# Per-MAP-KIND out-roots: `--map-kind` is a regime key for the resume/output
# state, so each kind owns its own out-root (crash-fix-rounds.md § Per-leg
# out-roots for regime-keyed drivers) — a shared root would let one kind's
# `all_arms_spearman.json` overwrite the other's.
#
# Counts-only logging; no corpus content printed.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

STORE_ROOT="data/issue_1739/store"
RESULTS_ROOT="eval_results/issue_1739"
NL_ROOT="$RESULTS_ROOT/nonlinear_map"
TENSORS_ROOT="analysis_tensors/issue_1739"
U_STORE_DIR="data/issue_1739/hf_dl/u_store"
LOG_DIR="/workspace/logs"

# The three map-family arms. nl-arm6 = map -> E1 persona-vector projection;
# nl-arm7 = ridge readout trained on PREDICTED answer vectors; nl-arm8 =
# ridge readout trained on REAL answer vectors, applied to map predictions.
NL_ARMS="arm6_map_proj_e1 arm7_map_ridge_pred arm8_map_ridge_true"

BEHAVIORS="${EPM_I1739_NL_BEHAVIORS:-evil sycophancy hallucination}"
KINDS="${EPM_I1739_NL_KINDS:-mlp kernel}"
# Grid defaults hold FULL parity with the main grid's fits phase so every
# nonlinear cell keys to a linear sibling. A budget-driven descope narrows
# DRAWS / REGIMES via these env knobs and is recorded as a deviation — never
# silently, and never by changing a cell's VALUES (that would unmatch it).
USIZES="${EPM_I1739_NL_USIZES:-250 5000 full}"
SEEDS="${EPM_I1739_NL_SEEDS:-0 1 2}"
DRAWS="${EPM_I1739_NL_DRAWS:-0 1 2 3 4}"
# Pilot gate budget. The ROUND ceiling is ~5 GPU-h across all
# (behavior x kind) invocations; with 6 invocations that is ~0.83 h each, and
# the pilot runs on the HEAVIEST behavior (3 regimes), so a per-invocation pass
# bounds the lighter ones too. abort-mult 1 makes the gate enforce that share
# directly rather than the fits default 3x plan-§9 re-size fence.
PLAN_WALL_H="${EPM_I1739_NL_PLAN_WALL_H:-0.83}"
PILOT_ABORT_MULT="${EPM_I1739_NL_PILOT_ABORT_MULT:-1}"
# Comma/space list of phases, or "all". Two-leg dispatch (stage,pilot then
# fits,collect,upload) keeps the round-level STOP decision on measured numbers.
PHASE="${EPM_I1739_NL_PHASE:-all}"

want_phase() {
  # want_phase <name> — true when PHASE is "all" or lists <name>.
  case "$PHASE" in
    all) return 0 ;;
  esac
  local p
  for p in ${PHASE//,/ }; do
    [ "$p" = "$1" ] && return 0
  done
  return 1
}

FITS_DEVICE="${EPM_I1739_FITS_DEVICE:-}"
if [ -z "$FITS_DEVICE" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    FITS_DEVICE=cuda
  else
    FITS_DEVICE=cpu
  fi
fi

mkdir -p "$LOG_DIR"
echo "[nlmap] start $(date -u +%FT%TZ) repo_root=$REPO_ROOT device=$FITS_DEVICE phase=$PHASE"
echo "[nlmap] behaviors='$BEHAVIORS' kinds='$KINDS' usizes='$USIZES' seeds='$SEEDS' draws='$DRAWS'"

# Per-behavior grid axes — copied from issue1739_dispatch.sh so the nonlinear
# cells key to the SAME (regime, budget) values the linear grid used.
behavior_budgets() {
  case "$1" in
    evil) echo "250 2500 8000" ;;
    *) echo "250 2500 16000" ;;
  esac
}
behavior_regimes() {
  if [ -n "${EPM_I1739_NL_REGIMES:-}" ]; then echo "$EPM_I1739_NL_REGIMES"; return; fi
  case "$1" in
    hallucination) echo "e1" ;;
    *) echo "e1 e2 e2p" ;;
  esac
}

fits_args() {
  # fits_args <behavior> <kind> — the production fits invocation, subset to
  # the map-family arms with a per-kind out-root.
  local b="$1" kind="$2"
  printf '%s\n' \
    --behavior "$b" \
    --labeled-store "$STORE_ROOT/${b}_labeling" \
    --dv-json "$RESULTS_ROOT/dv_dataset/$b/labeling.json" \
    --u-store "$U_STORE_DIR" \
    --e1-store "$STORE_ROOT/${b}_extraction" \
    --out-root "$NL_ROOT/$b/$kind" \
    --tensors-root "$TENSORS_ROOT" \
    --device "$FITS_DEVICE" \
    --map-kind "$kind" \
    --arms $NL_ARMS \
    --config config_a \
    --transfer \
    --regimes $(behavior_regimes "$b") \
    --u-sizes $USIZES \
    --budgets $(behavior_budgets "$b") \
    --draws $DRAWS \
    --seeds $SEEDS \
    --n-boot 500 \
    --n-perm 500
}

# ---- stage -----------------------------------------------------------------
if want_phase stage; then
  echo "[nlmap] phase=stage: pre-staging inputs via issue1739_leg2.sh"
  bash scripts/issue1739_leg2.sh
  for b in $BEHAVIORS; do
    for s in "${b}_labeling" "${b}_extraction"; do
      [ -d "$STORE_ROOT/$s" ] || { echo "[nlmap] FATAL: store $s missing after stage" >&2; exit 1; }
    done
    [ -f "$RESULTS_ROOT/dv_dataset/$b/labeling.json" ] \
      || { echo "[nlmap] FATAL: dv_dataset/$b missing after stage" >&2; exit 1; }
  done
  echo "[nlmap] phase=stage: complete ($(date -u +%FT%TZ))"
fi

# ---- pilot -----------------------------------------------------------------
# MEASURED 1-cell pilot through the PRODUCTION entrypoint at production shape
# (plan-compute-sizing.md § Per-cell fit phases — an asserted per-call cost is
# never a sizing basis). The fits script's own `--pilot` times one cell and
# projects the grid wall; rc=7 means the projection exceeds PLAN_WALL_H, which
# this round treats as STOP-and-report, not a silent descope.
if want_phase pilot; then
  pb="$(echo "$BEHAVIORS" | awk '{print $1}')"
  pk="$(echo "$KINDS" | awk '{print $1}')"
  echo "[nlmap] phase=pilot: $pb/$pk vs plan_wall_h=$PLAN_WALL_H x mult=$PILOT_ABORT_MULT"
  set +e
  mapfile -t _pa < <(fits_args "$pb" "$pk")
  uv run python scripts/issue1739_fits.py "${_pa[@]}" --pilot \
    --plan-wall-h "$PLAN_WALL_H" --pilot-abort-mult "$PILOT_ABORT_MULT"
  prc=$?
  set -e
  if [ "$prc" -eq 7 ]; then
    echo "[nlmap] PILOT REFUSED (rc=7): projected wall exceeds" \
      "${PILOT_ABORT_MULT}x${PLAN_WALL_H}h." >&2
    echo "[nlmap] STOP — reporting instead of launching (see pilot_report.json)." >&2
    exit 7
  fi
  [ "$prc" -eq 0 ] || { echo "[nlmap] FATAL: pilot exited rc=$prc" >&2; exit "$prc"; }
  echo "[nlmap] phase=pilot: PASS ($(date -u +%FT%TZ))"
fi

# ---- fits ------------------------------------------------------------------
if want_phase fits; then
  for b in $BEHAVIORS; do
    for kind in $KINDS; do
      echo "[nlmap] phase=fits behavior=$b kind=$kind start $(date -u +%FT%TZ)"
      mapfile -t _fa < <(fits_args "$b" "$kind")
      uv run python scripts/issue1739_fits.py "${_fa[@]}"
      echo "[nlmap] phase=fits behavior=$b kind=$kind done $(date -u +%FT%TZ)"
    done
  done
fi

# ---- collect map_quality.json ---------------------------------------------
# Derived from the frozen `maps/*.pt` metas (which carry the held-out
# r2_map + identity+bias baseline + kNN retrieval per layer), so the standing
# mapping-companion reads survive without re-running any fit.
if want_phase collect; then
  echo "[nlmap] phase=collect: map_quality.json"
  uv run python scripts/issue1739_nlmap_collect.py \
    --tensors-root "$TENSORS_ROOT" \
    --out "$NL_ROOT/map_quality.json" \
    --kinds $KINDS
fi

# ---- upload ----------------------------------------------------------------
if want_phase upload; then
  echo "[nlmap] phase=upload: nonlinear map payloads -> HF analysis_tensors"
  uv run python scripts/issue1739_upload.py --stage tensors
  echo "[nlmap] phase=upload: results -> git (fetch+rebase first; #1880 push race)"
  uv run python scripts/issue1739_upload.py --stage results-git
fi

# ---- sentinel + terminal line ---------------------------------------------
uv run python -c "
import json, pathlib, time
p = pathlib.Path('$LOG_DIR/issue-1739-nlmap-results.json')
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({
    'issue': 1739,
    'round': 'nonlinear_map',
    'status': 'ok',
    'behaviors': '$BEHAVIORS'.split(),
    'kinds': '$KINDS'.split(),
    'arms': '$NL_ARMS'.split(),
    'out_root': '$NL_ROOT',
    'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}, indent=2))
print('[nlmap] sentinel written ->', p)
"
echo "[phase=done]"
