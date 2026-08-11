#!/usr/bin/env bash
# issue-2054 rb789 pod driver (cpu-bigmem, ONE shard per pod) — plan v16 §4 P1.
#
# Usage (class shards):
#   RB789_CLASS=<prose|twobytwo|boundary|model> RB789_ARM=<context|prefix> \
#     bash scripts/issue2054_rb789_pod_driver.sh
# Usage (matched-n shards):
#   RB789_MATCHEDN=<boundary|twobytwo> bash scripts/issue2054_rb789_pod_driver.sh
#
# One linear leg under `set -euo pipefail` (plan §4 exit-path trace): every
# failure arm is that command's own non-zero exit propagating to the driver's
# exit; the pilot fence's exit 7 is a DESIGNED halt (projection persisted in
# pilot_gate_report__<shard>.json) the dispatcher/poller reads with its report
# JSON — never an anonymous crash. Steps: (1) deterministic same-prefix jitter
# (sha256(slug) mod 120 s — v223 blocker 1; launches are ALSO dispatched
# sequentially at ~4-min offsets); (2) out-root headroom assert (branch
# preflight.py:620); (3) stage the READ-ONLY parent prefixes (activations +
# ladder) via stage_hub_prefix (branch hub.py:2272); (4) run the shard — the
# fold-map floor assert, pilot (identity gates 1+2 + n^2-scaled fence), unit
# loop, M-C2 upload cadence, and the final scoped verify (branch hub.py:1343)
# all live INSIDE scripts/issue2054_rb789.py. No sentinel writes (plan
# assumption 12): pod legs signal via HF artifact presence + the poller.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

RB789_CLASS="${RB789_CLASS:-}"
RB789_ARM="${RB789_ARM:-}"
RB789_MATCHEDN="${RB789_MATCHEDN:-}"
if [ -n "$RB789_MATCHEDN" ] && [ -n "$RB789_CLASS" ]; then
  echo "[phase=rb789_pod] FATAL: set exactly ONE of RB789_CLASS / RB789_MATCHEDN" >&2
  exit 2
fi
if [ -n "$RB789_MATCHEDN" ]; then
  SHARD_SLUG="matchedn-${RB789_MATCHEDN}"
  MODE_ARGS=(--matchedn "$RB789_MATCHEDN")
  OUT_DIR="${RB789_OUT_DIR:-data/issue_2054/rb789/ladder_matchedn}"
elif [ -n "$RB789_CLASS" ]; then
  : "${RB789_ARM:?set RB789_ARM (context|prefix) with RB789_CLASS}"
  SHARD_SLUG="${RB789_CLASS}_${RB789_ARM}"
  MODE_ARGS=(--shard-class "$RB789_CLASS" --arm "$RB789_ARM")
  OUT_DIR="${RB789_OUT_DIR:-data/issue_2054/rb789/ladder}"
else
  echo "[phase=rb789_pod] FATAL: set RB789_CLASS+RB789_ARM or RB789_MATCHEDN" >&2
  exit 2
fi

LOG_DIR="${RB789_LOG_DIR:-/workspace/logs}"
STAGE_ROOT="${RB789_STAGE_ROOT:-/workspace/issue2054_rb789_stage}"
# READ-ONLY parent prefixes (staging source); the round's own upload prefix is
# threaded separately below and NEVER points at the parent's (plan §10).
PARENT_PREFIX="${RB789_PARENT_PREFIX:-issue2054_lattice}"
TASK_PREFIX="${RB789_HF_PREFIX:-issue2054_lattice/reduced_basis_refit_rungs789}"
FOLD_MAP="${RB789_FOLD_MAP:-eval_results/issue_2054/shared_fold_map.json}"
MAX_FLEET_WALL_HOURS="${RB789_MAX_FLEET_WALL_HOURS:-14}"
EXPECTED_NPZ="${RB789_EXPECTED_NPZ:-48}"
EXPECTED_LADDER_JSON="${RB789_EXPECTED_LADDER_JSON:-816}"
OMP="${RB789_OMP:-16}"
ACT_DIR="${STAGE_ROOT}/${PARENT_PREFIX}/activations"
LADDER_DIR="${STAGE_ROOT}/${PARENT_PREFIX}/ladder"
mkdir -p "$LOG_DIR"

echo "[phase=rb789_pod] driver start shard=${SHARD_SLUG} task_prefix=${TASK_PREFIX} $(date -u +%FT%TZ)"

# (1) Deterministic driver-side jitter BEFORE the shared-prefix pull (v223
# blocker 1 / plan §9 staging shape: sha256(shard slug) mod 120 s).
JITTER_HEX=$(printf '%s' "$SHARD_SLUG" | sha256sum | cut -c1-8)
JITTER=$((16#$JITTER_HEX % 120))
echo "[phase=rb789_pod stage=jitter] sleeping ${JITTER}s (sha256(${SHARD_SLUG}) mod 120)"
sleep "$JITTER"

# (2) Out-root headroom assert (resume-aware residual floor — ladder-driver
# convention; branch preflight.py:620).
HEADROOM_GB="${RB789_HEADROOM_GB:-25}"
N_STAGED=$(find "$ACT_DIR" -name '*.npz' 2>/dev/null | wc -l || true)
N_STAGED=${N_STAGED:-0}
if [ "$N_STAGED" -ge "$EXPECTED_NPZ" ]; then
  HEADROOM_GB="${RB789_RESIDUAL_HEADROOM_GB:-5}"
fi
echo "[phase=rb789_pod stage=headroom] floor=${HEADROOM_GB}GB staged_npz=${N_STAGED} $(date -u +%FT%TZ)"
rc=0
STAGE_ROOT="$STAGE_ROOT" HEADROOM_GB="$HEADROOM_GB" uv run python - <<'PYEOF' || rc=$?
import os

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

assert_out_root_headroom(
    os.environ["STAGE_ROOT"], float(os.environ["HEADROOM_GB"]), phase="rb789_pod"
)
PYEOF
if [ "$rc" -ne 0 ]; then
  echo "[phase=rb789_pod] HALT headroom rc=${rc}"
  exit "$rc"
fi

# (3) Stage the READ-ONLY parent inputs: the shared activations prefix (~12 GB)
# + the parent per-pair ladder JSONs (gate 2 needs one control pair's JSON on
# EVERY shard — I1 — so the full small prefix is staged everywhere).
echo "[phase=rb789_pod stage=stage_inputs] start $(date -u +%FT%TZ)"
rc=0
STAGE_ROOT="$STAGE_ROOT" PARENT_PREFIX="$PARENT_PREFIX" EXPECTED_NPZ="$EXPECTED_NPZ" \
  EXPECTED_LADDER_JSON="$EXPECTED_LADDER_JSON" \
  uv run python - > "$LOG_DIR/issue-2054-rb789-stage-${SHARD_SLUG}.log" 2>&1 <<'PYEOF' || rc=$?
import os
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
from explore_persona_space.orchestrate.hub import stage_hub_prefix

REPO = "superkaiba1/explore-persona-space-data"
root = os.environ["PARENT_PREFIX"]
dest = Path(os.environ["STAGE_ROOT"])
dest.mkdir(parents=True, exist_ok=True)
for prefix, glob_pat, want in (
    (f"{root}/activations", "*.npz", int(os.environ["EXPECTED_NPZ"])),
    (f"{root}/ladder", "*.json", int(os.environ["EXPECTED_LADDER_JSON"])),
):
    stage_hub_prefix(REPO, prefix, dest, repo_type="dataset")
    found = list((dest / prefix).rglob(glob_pat))
    if len(found) < want:
        raise RuntimeError(f"expected >={want} {glob_pat} under {dest / prefix}, found {len(found)}")
    print(f"[stage] {prefix}: {len(found)} {glob_pat} present", flush=True)
print("[stage] inputs staged", flush=True)
PYEOF
echo "[phase=rb789_pod stage=stage_inputs] rc=${rc} $(date -u +%FT%TZ)"
if [ "$rc" -ne 0 ]; then
  echo "[phase=rb789_pod] HALT stage_inputs rc=${rc} (tail follows)"
  tail -20 "$LOG_DIR/issue-2054-rb789-stage-${SHARD_SLUG}.log" || true
  exit "$rc"
fi

# (4) The shard: fold-map floor assert -> pilot (gates 1+2 + n^2-scaled fence,
# exit 7 designed halt) -> unit loop (per-unit checkpoints + M-C2 cadence) ->
# final scoped verify. Single process, OMP=${OMP} (16 vCPU cpu5m-16-128).
echo "[phase=rb789_pod stage=run] start shard=${SHARD_SLUG} $(date -u +%FT%TZ)"
rc=0
env OMP_NUM_THREADS="$OMP" MKL_NUM_THREADS="$OMP" OPENBLAS_NUM_THREADS="$OMP" \
  NUMEXPR_NUM_THREADS="$OMP" \
  uv run python scripts/issue2054_rb789.py \
  --mode run \
  "${MODE_ARGS[@]}" \
  --activations-dir "$ACT_DIR" \
  --parent-ladder-dir "$LADDER_DIR" \
  --fold-map "$FOLD_MAP" \
  --out-dir "$OUT_DIR" \
  --hf-prefix "$TASK_PREFIX" \
  --max-fleet-wall-hours "$MAX_FLEET_WALL_HOURS" \
  > "$LOG_DIR/issue-2054-rb789-${SHARD_SLUG}.log" 2>&1 || rc=$?
echo "[phase=rb789_pod stage=run] rc=${rc} $(date -u +%FT%TZ)"
if [ "$rc" -eq 7 ]; then
  echo "[phase=rb789_pod] DESIGNED HALT rc=7 (fleet-wall fence; see pilot_gate_report__${SHARD_SLUG}.json — M-C4 raise mechanism or re-shard wider)"
  tail -15 "$LOG_DIR/issue-2054-rb789-${SHARD_SLUG}.log" || true
  exit 7
fi
if [ "$rc" -ne 0 ]; then
  echo "[phase=rb789_pod] HALT run rc=${rc} (tail follows)"
  tail -30 "$LOG_DIR/issue-2054-rb789-${SHARD_SLUG}.log" || true
  exit "$rc"
fi
echo "[phase=rb789_pod] driver_rc=0 shard=${SHARD_SLUG} $(date -u +%FT%TZ)"
