#!/usr/bin/env bash
# issue-2054 ladder pod driver (cpu-bigmem, one pair-class shard).
#
# Usage: LADDER_CLASSES=<comma list> [LADDER_MODELS=<comma list>] \
#          bash scripts/issue2054_ladder_pod_driver.sh
#
# Stages the capture npz store + the 48 fit JSONs from HF, then runs TWO
# concurrent ladder processes split by ARM (context / prefix) over the given
# pair classes, OMP=8 each (16 vCPU; peak RSS ~21.6 GiB/proc vs 128 GB).
# Sharding basis: the committed pilot report (49.0 s/unit-fold measured) +
# the exact unit census — twobytwo 288 / cross_character 192 / cross_model 96
# / cross_framing 80 units. --skip-pilot-gate is legitimate: the standalone
# pilot ran on f5 (report committed at
# eval_results/issue_2054/ladder_pilot_gate_report.json).
set -uo pipefail
: "${LADDER_CLASSES:?set LADDER_CLASSES (comma list of pair classes)}"
cd "$(git rev-parse --show-toplevel)"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

LOG_DIR="${ISSUE2054_LADDER_LOG_DIR:-/workspace/logs}"
STAGE_ROOT="${ISSUE2054_LADDER_STAGE_ROOT:-/workspace/issue2054_ladder_stage}"
ACT_DIR="${STAGE_ROOT}/issue2054_lattice/activations"
FITS_DIR="${STAGE_ROOT}/issue2054_lattice/fits"
mkdir -p "$LOG_DIR"

echo "[phase=ladder_pod] driver start classes=${LADDER_CLASSES} models=${LADDER_MODELS:-all} $(date -u +%FT%TZ)"

echo "[phase=ladder_pod stage=stage_inputs] start $(date -u +%FT%TZ)"
STAGE_ROOT="$STAGE_ROOT" uv run python - > "$LOG_DIR/issue-2054-ladder-stage.log" 2>&1 <<'PYEOF'
import os
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.hub import stage_hub_prefix

REPO = "superkaiba1/explore-persona-space-data"
dest = Path(os.environ["STAGE_ROOT"])
dest.mkdir(parents=True, exist_ok=True)
for prefix, glob_pat, want in (
    ("issue2054_lattice/activations", "*.npz", 48),
    ("issue2054_lattice/fits", "*.json", 48),
):
    stage_hub_prefix(REPO, prefix, dest, repo_type="dataset")
    found = list((dest / prefix).rglob(glob_pat))
    if len(found) < want:
        raise RuntimeError(f"expected >={want} {glob_pat} under {dest / prefix}, found {len(found)}")
    print(f"[stage] {prefix}: {len(found)} {glob_pat} present", flush=True)
print("[stage] inputs staged", flush=True)
PYEOF
rc=$?
echo "[phase=ladder_pod stage=stage_inputs] rc=${rc} $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  echo "[phase=ladder_pod] HALT stage_inputs rc=${rc} (tail follows)"
  tail -20 "$LOG_DIR/issue-2054-ladder-stage.log" || true
  exit "$rc"
fi

run_arm() {
  local arm="$1" log="$2"
  local extra=()
  if [ -n "${LADDER_MODELS:-}" ]; then extra=(--models "$LADDER_MODELS"); fi
  env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    uv run python scripts/issue2054_ladder.py \
    --activations-dir "$ACT_DIR" \
    --fits-dir "$FITS_DIR" \
    --fold-map eval_results/issue_2054/shared_fold_map.json \
    --output-dir data/issue_2054/ladder/ \
    --seed 137 \
    --pair-classes "$LADDER_CLASSES" \
    --arms "$arm" \
    --skip-pilot-gate \
    "${extra[@]}" \
    > "$log" 2>&1
}

echo "[phase=ladder_pod stage=ladder] concurrent start $(date -u +%FT%TZ)"
run_arm context "$LOG_DIR/issue-2054-ladder-ctx.log" &
P1=$!
run_arm prefix "$LOG_DIR/issue-2054-ladder-pfx.log" &
P2=$!
wait "$P1"; R1=$?
wait "$P2"; R2=$?
echo "[phase=ladder_pod stage=ladder] context rc=${R1}; prefix rc=${R2} $(date -u +%FT%TZ)"
if [ "$R1" -ne 0 ] || [ "$R2" -ne 0 ]; then
  echo "[phase=ladder_pod] HALT context rc=${R1} prefix rc=${R2} (tails follow)"
  tail -25 "$LOG_DIR/issue-2054-ladder-ctx.log" || true
  tail -25 "$LOG_DIR/issue-2054-ladder-pfx.log" || true
  exit 1
fi
echo "[phase=ladder_pod] driver_rc=0 classes=${LADDER_CLASSES} $(date -u +%FT%TZ)"
