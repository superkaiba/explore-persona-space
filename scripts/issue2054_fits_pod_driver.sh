#!/usr/bin/env bash
# issue-2054 fits pod driver (cpu-bigmem, one (model x condition) shard).
#
# Usage: FITS_MODEL=<slug> FITS_COND=<inserted|on_policy> bash scripts/issue2054_fits_pod_driver.sh
#
# Stages the capture npz store from HF, then runs TWO concurrent fits
# processes split by form pairs ({attrib_quoted,chat} / {bare_label,bare_text}
# — 6 variant-cells each, balanced), OMP=8 each (16 vCPU box; peak RSS
# ~21 GiB/proc vs 128 GB). Per-cell fit JSONs are disjoint by cell key; the
# HF mirror is fail-loud per process. Routing per the M5 pilot measurement
# (VM pilot: 328.8 s/unit-fold, 20.31 GiB peak — plan §9 off-VM rule).
set -uo pipefail
: "${FITS_MODEL:?set FITS_MODEL (qwen2.5-7b | qwen2.5-7b-instruct)}"
: "${FITS_COND:?set FITS_COND (inserted | on_policy)}"
cd "$(git rev-parse --show-toplevel)"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

LOG_DIR="${ISSUE2054_FITS_LOG_DIR:-/workspace/logs}"
STAGE_ROOT="${ISSUE2054_FITS_STAGE_ROOT:-/workspace/issue2054_fits_stage}"
ACT_DIR="${STAGE_ROOT}/issue2054_lattice/activations"
mkdir -p "$LOG_DIR"

echo "[phase=fits_pod] driver start model=${FITS_MODEL} cond=${FITS_COND} $(date -u +%FT%TZ)"

echo "[phase=fits_pod stage=stage_activations] start $(date -u +%FT%TZ)"
STAGE_ROOT="$STAGE_ROOT" uv run python - > "$LOG_DIR/issue-2054-fits-stage.log" 2>&1 <<'PYEOF'
import os
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.hub import stage_hub_prefix

REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue2054_lattice/activations"
dest = Path(os.environ["STAGE_ROOT"])
dest.mkdir(parents=True, exist_ok=True)
stage_hub_prefix(REPO, PREFIX, dest, repo_type="dataset")
mirror = dest / PREFIX
npz = list(mirror.rglob("*.npz"))
if len(npz) != 48:
    raise RuntimeError(f"expected 48 npz under {mirror}, found {len(npz)}")
print("[stage] verified 48/48 npz present", flush=True)
PYEOF
rc=$?
echo "[phase=fits_pod stage=stage_activations] rc=${rc} $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  echo "[phase=fits_pod] HALT stage_activations rc=${rc} (tail follows)"
  tail -20 "$LOG_DIR/issue-2054-fits-stage.log" || true
  exit "$rc"
fi

run_split() {
  local label="$1" forms="$2" log="$3"
  env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    uv run python scripts/issue2054_fits.py \
    --activations-dir "$ACT_DIR" \
    --fold-map eval_results/issue_2054/shared_fold_map.json \
    --output-dir data/issue_2054/fits/ \
    --seed 137 --layer 19 --n-null-draws 100 \
    --models "$FITS_MODEL" --conditions "$FITS_COND" --forms "$forms" \
    --skip-pilot-gate \
    > "$log" 2>&1
}

echo "[phase=fits_pod stage=fits] concurrent start $(date -u +%FT%TZ)"
run_split a "attrib_quoted,chat" "$LOG_DIR/issue-2054-fits-a.log" &
P1=$!
run_split b "bare_label,bare_text" "$LOG_DIR/issue-2054-fits-b.log" &
P2=$!
wait "$P1"; R1=$?
wait "$P2"; R2=$?
echo "[phase=fits_pod stage=fits] split-a rc=${R1}; split-b rc=${R2} $(date -u +%FT%TZ)"
if [ "$R1" -ne 0 ] || [ "$R2" -ne 0 ]; then
  echo "[phase=fits_pod] HALT split-a rc=${R1} split-b rc=${R2} (tails follow)"
  tail -25 "$LOG_DIR/issue-2054-fits-a.log" || true
  tail -25 "$LOG_DIR/issue-2054-fits-b.log" || true
  exit 1
fi
echo "[phase=fits_pod] driver_rc=0 model=${FITS_MODEL} cond=${FITS_COND} $(date -u +%FT%TZ)"
