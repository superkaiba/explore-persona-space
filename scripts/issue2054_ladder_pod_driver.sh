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
# HF prefix knob (#2054 regen plan §4 item 4): the coordinated-regen round
# passes ISSUE2054_LADDER_HF_PREFIX=issue2054_lattice/common_regen so staging
# reads the ROUND's stores/fits and uploads never clobber the parent's.
HF_PREFIX="${ISSUE2054_LADDER_HF_PREFIX:-issue2054_lattice}"
FOLD_MAP="${ISSUE2054_LADDER_FOLD_MAP:-eval_results/issue_2054/shared_fold_map.json}"
OUT_DIR="${ISSUE2054_LADDER_OUT_DIR:-data/issue_2054/ladder/}"
# Plan §9 R5: the driver threads --max-fleet-wall-hours 14 (the 12.0 default
# leaves 0.6 h headroom over the 11.4 h/pod projection).
MAX_FLEET_WALL_HOURS="${ISSUE2054_LADDER_MAX_FLEET_WALL_HOURS:-14}"
EXPECTED_NPZ="${ISSUE2054_LADDER_EXPECTED_NPZ:-48}"
EXPECTED_FITS="${ISSUE2054_LADDER_EXPECTED_FITS:-48}"
ACT_DIR="${STAGE_ROOT}/${HF_PREFIX}/activations"
FITS_DIR="${STAGE_ROOT}/${HF_PREFIX}/fits"
mkdir -p "$LOG_DIR"

echo "[phase=ladder_pod] driver start classes=${LADDER_CLASSES} models=${LADDER_MODELS:-all} prefix=${HF_PREFIX} $(date -u +%FT%TZ)"

# Per-leg out-root headroom preamble (plan §4 item 7; resume-aware: a fully
# staged store re-asserts on the residual floor only).
HEADROOM_GB="${ISSUE2054_LADDER_HEADROOM_GB:-25}"
N_STAGED=$(find "$ACT_DIR" -name '*.npz' 2>/dev/null | wc -l || true)
N_STAGED=${N_STAGED:-0}
if [ "$N_STAGED" -ge "$EXPECTED_NPZ" ]; then
  HEADROOM_GB="${ISSUE2054_LADDER_RESIDUAL_HEADROOM_GB:-5}"
fi
echo "[phase=ladder_pod stage=headroom] floor=${HEADROOM_GB}GB staged_npz=${N_STAGED} $(date -u +%FT%TZ)"
STAGE_ROOT="$STAGE_ROOT" HEADROOM_GB="$HEADROOM_GB" uv run python - <<'PYEOF'
import os

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

assert_out_root_headroom(
    os.environ["STAGE_ROOT"], float(os.environ["HEADROOM_GB"]), phase="ladder_pod"
)
PYEOF
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "[phase=ladder_pod] HALT headroom rc=${rc}"
  exit "$rc"
fi

echo "[phase=ladder_pod stage=stage_inputs] start $(date -u +%FT%TZ)"
STAGE_ROOT="$STAGE_ROOT" HF_PREFIX="$HF_PREFIX" EXPECTED_NPZ="$EXPECTED_NPZ" \
  EXPECTED_FITS="$EXPECTED_FITS" \
  uv run python - > "$LOG_DIR/issue-2054-ladder-stage.log" 2>&1 <<'PYEOF'
import os
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.hub import stage_hub_prefix

REPO = "superkaiba1/explore-persona-space-data"
root = os.environ["HF_PREFIX"]
dest = Path(os.environ["STAGE_ROOT"])
dest.mkdir(parents=True, exist_ok=True)
for prefix, glob_pat, want in (
    (f"{root}/activations", "*.npz", int(os.environ["EXPECTED_NPZ"])),
    (f"{root}/fits", "*.json", int(os.environ["EXPECTED_FITS"])),
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

# Built-in pilot gate default ON for the regen round (the committed parent
# pilot report does not cover the regen store's n; the fence needs the
# projection). ISSUE2054_LADDER_SKIP_PILOT_GATE=1 restores parent behavior.
SKIP_PILOT_ARGS=()
if [ "${ISSUE2054_LADDER_SKIP_PILOT_GATE:-0}" = "1" ]; then
  SKIP_PILOT_ARGS=(--skip-pilot-gate)
fi

run_arm() {
  local arm="$1" log="$2"
  local extra=()
  if [ -n "${LADDER_MODELS:-}" ]; then extra=(--models "$LADDER_MODELS"); fi
  env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    uv run python scripts/issue2054_ladder.py \
    --activations-dir "$ACT_DIR" \
    --fits-dir "$FITS_DIR" \
    --fold-map "$FOLD_MAP" \
    --output-dir "$OUT_DIR" \
    --seed 137 \
    --pair-classes "$LADDER_CLASSES" \
    --arms "$arm" \
    --hf-prefix "$HF_PREFIX" \
    --max-fleet-wall-hours "$MAX_FLEET_WALL_HOURS" \
    "${SKIP_PILOT_ARGS[@]}" \
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
