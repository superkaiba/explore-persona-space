#!/usr/bin/env bash
# issue-2054 ladder pod driver (cpu-bigmem).
#
# Stages the capture npz store + the 48 fit JSONs from HF (cross-machine
# seam — both were produced on other machines), then runs the 9-rung
# transfer ladder through its built-in auto pilot gate (measured 1-unit wall
# extrapolated to the pending fleet; projection > --max-fleet-wall-hours
# exits 7 with pilot_gate_report.json — route on the artifact, plan M-R2-1).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

LOG_DIR="${ISSUE2054_LADDER_LOG_DIR:-/workspace/logs}"
STAGE_ROOT="${ISSUE2054_LADDER_STAGE_ROOT:-/workspace/issue2054_ladder_stage}"
ACT_DIR="${STAGE_ROOT}/issue2054_lattice/activations"
FITS_DIR="${STAGE_ROOT}/issue2054_lattice/fits"
mkdir -p "$LOG_DIR"

echo "[phase=ladder_pod] driver start $(date -u +%FT%TZ)"

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

echo "[phase=ladder_pod stage=ladder] start $(date -u +%FT%TZ)"
env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
  uv run python scripts/issue2054_ladder.py \
  --activations-dir "$ACT_DIR" \
  --fits-dir "$FITS_DIR" \
  --fold-map eval_results/issue_2054/shared_fold_map.json \
  --output-dir data/issue_2054/ladder/ \
  --seed 137
rc=$?
echo "[phase=ladder_pod stage=ladder] rc=${rc} $(date -u +%FT%TZ)"
echo "[phase=ladder_pod] driver_rc=${rc} $(date -u +%FT%TZ)"
exit "$rc"
