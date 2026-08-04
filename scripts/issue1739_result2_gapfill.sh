#!/usr/bin/env bash
# Result-2 gap-fill (#1739 Job C): hallucination arms 7/8/12/17/18 at the
# PLOTTED operating slice -- u_rung = full (18,793), budget_l = 16000,
# regime e1, LINEAR map -- on the train + nqopen + simpleqa rungs, both
# variants where the arm supports them.
#
# Why this corner is missing. Result 2's hallucination OOD panels render 11
# arms against 16 elsewhere: the committed coverage for these five arms at
# (u=full, L=16000) on nqopen/simpleqa is either absent (arm12/17/18: only
# L in {250,2500}) or exists ONLY under the NONLINEAR map legs
# (arm7/arm8 -- new_arm_round/nlood/hallucination/{kernel,mlp}). Reusing a
# kernel/MLP-map row to fill a LINEAR-map bar group is a silent methodology
# error, so those rows are NOT usable here and this leg refits the corner
# under `--map-kind linear` (the fits.py default). Arms 17/18 additionally
# have the same hole on their own TRAIN rung at L=16000.
#
# Lane. One 1x H200 box. The prior #1739 fit round measured 3.1% mean GPU
# utilisation and was HOST-RAM bound, so a wide box is the wrong shape; the
# H200 is chosen for its host-RAM tier, not its HBM -- the per-group
# whitening + LINEAR map fit projects ~116 GB of additional host RAM at
# (28 layers x 18,793 U rows x 3,584 dims), which is what killed the 85 GB
# boxes (fits.py rc=9 RSS guard).
#
# Disk. The hallucination labeling capture store is a single 69.9 GB tar;
# downloading it *and* untarring it under /workspace would peak ~140 GB and
# trip the RunPod MooseFS per-pod quota (~130 GB, EDQUOT). Stage phase points
# leg2's tar directory at the container overlay disk instead, so the tar and
# the extracted store never co-reside on the same surface.
#
# Phases (EPM_I1739_GAPFILL_PHASES, comma-separated): stage,pilot,fits.
# Uploading is deliberately NOT a phase here -- every committed #1739 driver
# writes a FIXED HF prefix that ignores --out-root, which would co-mingle this
# round with the wide-roster results. The round uploads separately, to
# issue1739_result2_gapfill/.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

B=hallucination
OUT_ROOT="${EPM_I1739_GAPFILL_OUT_ROOT:-eval_results/issue_1739/result2_gapfill/$B}"
TENSORS_ROOT="${EPM_I1739_GAPFILL_TENSORS_ROOT:-analysis_tensors/issue_1739}"
VARIANT="${EPM_I1739_GAPFILL_VARIANT:-both}"
DRAWS="${EPM_I1739_GAPFILL_DRAWS:-0 1 2 3 4}"
SEEDS="${EPM_I1739_GAPFILL_SEEDS:-0 1 2}"
PHASES="${EPM_I1739_GAPFILL_PHASES:-stage,pilot,fits}"
# Container-disk staging dir for leg2's store tars (see "Disk" above). Set
# empty to keep leg2's default in-repo location.
TAR_STAGE="${EPM_I1739_GAPFILL_TAR_STAGE:-/root/i1739_store_tars}"
# Pilot fence: ONE production-shape unit, projected against this wall.
PLAN_WALL_H="${EPM_I1739_GAPFILL_PLAN_WALL_H:-4.0}"
ABORT_MULT="${EPM_I1739_GAPFILL_ABORT_MULT:-2.0}"

# The five arms whose (full, 16000) cells Result 2 is missing. Passed as BOTH
# --arms (the train-rung roster) and --transfer-arms (the OOD-rung roster):
# arms 17/18 need their own train rung filled at this budget too.
ROSTER="arm7_map_ridge_pred arm8_map_ridge_true arm12_oracle_reg arm17_oracle_mlp arm18_oracle_krr"

has_phase() { case ",$PHASES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

echo "[gapfill] start $(date -u +%FT%TZ) repo_root=$REPO_ROOT phases=$PHASES"
echo "[gapfill] out_root=$OUT_ROOT variant=$VARIANT draws='$DRAWS' seeds='$SEEDS'"
echo "[gapfill] MemAvailable: $(awk '/MemAvailable/{printf "%.1f GB", $2/1048576}' /proc/meminfo)"

if has_phase stage; then
  echo "[gapfill] phase=stage $(date -u +%FT%TZ)"
  if [ -n "$TAR_STAGE" ]; then
    # leg2 downloads each capture-store tar into data/issue_1739/hf_dl/store_tars
    # and deletes it right after untarring. Redirect that directory to the
    # container overlay so the 69.9 GB tar never shares a quota with the
    # 69.9 GB extracted store.
    mkdir -p "$TAR_STAGE"
    mkdir -p data/issue_1739/hf_dl
    if [ ! -e data/issue_1739/hf_dl/store_tars ]; then
      ln -s "$TAR_STAGE" data/issue_1739/hf_dl/store_tars
    fi
    echo "[gapfill] tar staging -> $(readlink -f data/issue_1739/hf_dl/store_tars)"
  fi

  EPM_I1739_BEHAVIORS="$B" bash scripts/issue1739_leg2.sh

  # leg2 stages the capture stores / raw completions / dv_dataset but NOT the
  # #1092 U-pool (the dispatcher's fits phase does that, and this leg does not
  # go through the dispatcher). Idempotent + short-circuits when loadable.
  echo "[gapfill] stage #1092 U-store $(date -u +%FT%TZ)"
  uv run python -c "
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from pathlib import Path
from explore_persona_space.experiments.issue_1739 import store_io
from explore_persona_space.experiments.issue_1739.constants import N_LAYERS
store_io.stage_u_store(Path('data/issue_1739/hf_dl/u_store'), layers=tuple(range(N_LAYERS)))
print('[gapfill] u_store staged/verified', flush=True)
"
  df -h . | tail -1
fi

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
  --map-kind linear
  --regimes e1
  --u-sizes full
  --budgets 16000
  --draws $DRAWS
  --seeds $SEEDS
  --arms $ROSTER
  --transfer-arms $ROSTER
  --fixed-coordinate u=full
  --variant "$VARIANT"
  --n-boot 500 --n-perm 500
)

if has_phase pilot; then
  echo "[gapfill] phase=pilot (fence ${ABORT_MULT}x${PLAN_WALL_H}h) $(date -u +%FT%TZ)"
  set +e
  uv run python scripts/issue1739_fits.py "${fits_argv[@]}" --pilot \
    --plan-wall-h "$PLAN_WALL_H" --pilot-abort-mult "$ABORT_MULT"
  prc=$?
  set -e
  case "$prc" in
    0) : ;;
    7) echo "[gapfill] PILOT REFUSED (rc=7): projection exceeds the fence — see" \
         "$OUT_ROOT/pilot_report.json (designed halt; re-size, never a blind raise)" >&2
       exit 7 ;;
    9) echo "[gapfill] RSS-GUARD REFUSED (rc=9): projected peak host RAM exceeds this" \
         "box — see $OUT_ROOT/rss_guard_report.json (designed halt; relaunch on a" \
         "bigger-RAM box)" >&2
       exit 9 ;;
    *) echo "[gapfill] FATAL: pilot rc=$prc" >&2; exit "$prc" ;;
  esac
fi

if has_phase fits; then
  echo "[gapfill] phase=fits $(date -u +%FT%TZ)"
  uv run python scripts/issue1739_fits.py "${fits_argv[@]}"
fi

echo "[gapfill] done rc=0 $(date -u +%FT%TZ)"
