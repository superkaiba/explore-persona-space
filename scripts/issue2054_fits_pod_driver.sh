#!/usr/bin/env bash
# issue-2054 fits pod driver (cpu-bigmem, one (model x condition) shard).
#
# Usage: FITS_MODEL=<slug> FITS_COND=<inserted|on_policy|cell_c> bash scripts/issue2054_fits_pod_driver.sh
#
# Stages the capture npz store from HF, then runs TWO concurrent fits
# processes split by form pairs ({attrib_quoted,chat} / {bare_label,bare_text}
# — 6 variant-cells each, balanced), OMP=8 each (16 vCPU box; peak RSS
# ~21 GiB/proc vs 128 GB). Per-cell fit JSONs are disjoint by cell key; the
# HF mirror is fail-loud per process. Routing per the M5 pilot measurement
# (VM pilot: 328.8 s/unit-fold, 20.31 GiB peak — plan §9 off-VM rule).
#
# cell_c dispatch (the (c) cells exist ONLY in the chat form, 4 cells per
# model — fits.py consumes them as-is; the knobs below adapt THIS driver's
# two-form fan-out + staging floor):
#   FITS_MODEL=<slug> FITS_COND=cell_c \
#   ISSUE2054_FITS_FORMS_A=chat ISSUE2054_FITS_FORMS_B= \
#   ISSUE2054_FITS_EXPECTED_NPZ=56 \
#   bash scripts/issue2054_fits_pod_driver.sh
# (56 = 48 a/b/d npz + 8 cell_c npz; an EMPTY ISSUE2054_FITS_FORMS_B skips
# split-b — the default two-split fan-out would resolve zero cell_c cells on
# split-b and rc-fail. Defaults leave a/b/d dispatches byte-equivalent.)
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

# Staging floor: >= the expected store size (FLOOR, not equality — the prefix
# is a superset once the cell_c capture npz land beside the 48 a/b/d ones;
# extra cells are inert to this shard's cell-key enumeration). Default 48
# (the a/b/d store); a cell_c dispatch sets 56 to assert its own cells staged.
EXPECTED_NPZ="${ISSUE2054_FITS_EXPECTED_NPZ:-48}"

echo "[phase=fits_pod stage=stage_activations] start $(date -u +%FT%TZ)"
STAGE_ROOT="$STAGE_ROOT" EXPECTED_NPZ="$EXPECTED_NPZ" \
  uv run python - > "$LOG_DIR/issue-2054-fits-stage.log" 2>&1 <<'PYEOF'
import os
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.hub import stage_hub_prefix

REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue2054_lattice/activations"
dest = Path(os.environ["STAGE_ROOT"])
expected = int(os.environ["EXPECTED_NPZ"])
dest.mkdir(parents=True, exist_ok=True)
stage_hub_prefix(REPO, PREFIX, dest, repo_type="dataset")
mirror = dest / PREFIX
npz = list(mirror.rglob("*.npz"))
if len(npz) < expected:
    raise RuntimeError(f"expected >= {expected} npz under {mirror}, found {len(npz)}")
print(f"[stage] verified {len(npz)} npz present (floor {expected})", flush=True)
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

# Form splits are env-overridable for conditions that realize fewer forms
# (cell_c exists only in chat): an EMPTY ISSUE2054_FITS_FORMS_B skips split-b
# (a split resolving zero cells would rc-fail fits.py by design). Defaults
# keep the a/b/d two-split fan-out byte-equivalent.
FORMS_A="${ISSUE2054_FITS_FORMS_A:-attrib_quoted,chat}"
# colon-less default (unset -> default; set-but-EMPTY -> empty) so the
# documented empty-B skip is actually reachable — ${VAR:-} substitutes the
# default on set-but-empty too and made the skip branch dead code (caught
# live on the first cell_c dispatch: forms_b=bare_label,bare_text despite
# an explicit empty export).
FORMS_B="${ISSUE2054_FITS_FORMS_B-bare_label,bare_text}"

echo "[phase=fits_pod stage=fits] concurrent start forms_a=${FORMS_A} forms_b=${FORMS_B:-<skipped>} $(date -u +%FT%TZ)"
run_split a "$FORMS_A" "$LOG_DIR/issue-2054-fits-a.log" &
P1=$!
P2=""
if [ -n "$FORMS_B" ]; then
  run_split b "$FORMS_B" "$LOG_DIR/issue-2054-fits-b.log" &
  P2=$!
fi
wait "$P1"; R1=$?
R2=0
if [ -n "$P2" ]; then
  wait "$P2"; R2=$?
  echo "[phase=fits_pod stage=fits] split-a rc=${R1}; split-b rc=${R2} $(date -u +%FT%TZ)"
else
  echo "[phase=fits_pod stage=fits] split-a rc=${R1}; split-b skipped (FORMS_B empty) $(date -u +%FT%TZ)"
fi
if [ "$R1" -ne 0 ] || [ "$R2" -ne 0 ]; then
  echo "[phase=fits_pod] HALT split-a rc=${R1} split-b rc=${R2} (tails follow)"
  tail -25 "$LOG_DIR/issue-2054-fits-a.log" || true
  if [ -n "$P2" ]; then tail -25 "$LOG_DIR/issue-2054-fits-b.log" || true; fi
  exit 1
fi
echo "[phase=fits_pod] driver_rc=0 model=${FITS_MODEL} cond=${FITS_COND} $(date -u +%FT%TZ)"
