#!/usr/bin/env bash
# issue-2054 fits production driver (VM-side, CPU).
#
# Stages the capture npz store from HF onto the DATA DISK (the capture ran
# pod-side — cross-machine seam; ~9.7 GB routes off `/` per the staging rule),
# then runs the fits battery through its own built-in 1-cell measured pilot
# gate (M5/plan §9) before the fleet. Fit JSONs land in the worktree
# data/issue_2054/fits/ (small) and mirror to HF in-script (fail-loud).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

STAGE_ROOT="${ISSUE2054_FITS_STAGE_ROOT:-/mnt/eps-data/${USER}/issue2054_fits}"
ACT_DIR="${STAGE_ROOT}/issue2054_lattice/activations"
LOG_DIR="${ISSUE2054_FITS_LOG_DIR:-/tmp}"

echo "[phase=fits_prod] driver start $(date -u +%FT%TZ)"

echo "[phase=fits_prod stage=stage_activations] start $(date -u +%FT%TZ)"
STAGE_ROOT="$STAGE_ROOT" uv run python - > "$LOG_DIR/issue2054_fits_stage.log" 2>&1 <<'PYEOF'
import os
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.hub import stage_hub_prefix

REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue2054_lattice/activations"
dest = Path(os.environ["STAGE_ROOT"])
dest.mkdir(parents=True, exist_ok=True)
staged = stage_hub_prefix(REPO, PREFIX, dest, repo_type="dataset")
n = len(staged) if staged is not None else -1
print(f"[stage] activations staged under {dest}/{PREFIX}: {n} entries", flush=True)
mirror = dest / PREFIX
npz = list(mirror.rglob("*.npz"))
if len(npz) != 48:
    raise RuntimeError(f"expected 48 npz under {mirror}, found {len(npz)}")
print(f"[stage] verified 48/48 npz present", flush=True)
PYEOF
rc=$?
echo "[phase=fits_prod stage=stage_activations] rc=${rc} $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  echo "[phase=fits_prod] HALT stage_activations rc=${rc} (tail follows)"
  tail -20 "$LOG_DIR/issue2054_fits_stage.log" || true
  exit "$rc"
fi

echo "[phase=fits_prod stage=fits] start $(date -u +%FT%TZ)"
uv run python scripts/issue2054_fits.py \
  --activations-dir "$ACT_DIR" \
  --fold-map eval_results/issue_2054/shared_fold_map.json \
  --output-dir data/issue_2054/fits/ \
  --seed 137 \
  --layer 19 \
  --n-null-draws 100
rc=$?
echo "[phase=fits_prod stage=fits] rc=${rc} $(date -u +%FT%TZ)"
echo "[phase=fits_prod] driver_rc=${rc} $(date -u +%FT%TZ)"
exit "$rc"
