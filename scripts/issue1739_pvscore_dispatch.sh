#!/usr/bin/env bash
# issue-1739 PV-SYNTH ARM SCORING leg (the PV-eval rung's final phase).
#
# Runs pvsynth-runner's `scripts/issue1739_pvsynth_arms.py` (6-arm transfer
# roster x both variants x 28 layers) against the SAME train capture stores the
# nonlinear-map round stages. This dispatcher owns staging + the scorer +
# durability only; it NEVER edits the scorer (another lane's file set) and
# NEVER re-runs the judge — the DV/judge outputs are already committed and the
# scorer reads them.
#
# Co-location note: this leg was originally scoped to ride the nonlinear-map
# round's live instance (shared staging). That instance self-terminated on its
# pilot-gate rc=7 halt, so this is the documented fallback — a fresh instance
# that stages the stores itself.
#
# Staging reuse: `scripts/issue1739_leg2.sh` already stages
# `data/issue_1739/store/{behavior}_{labeling,extraction}` AND
# `eval_results/issue_1739/dv_dataset/{behavior}/labeling.json` for all three
# behaviors (idempotent, so a re-run skips). Only the pvsynth capture store is
# extra, and it is a 1.81 GB / 519-file prefix on the data repo.
#
# Counts-only logging; no corpus content printed.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$PWD}}"
cd "$REPO_ROOT"

STORE_ROOT="data/issue_1739/store"
RESULTS_ROOT="eval_results/issue_1739"
PV_ROOT="$RESULTS_ROOT/pvsynth"
U_STORE_DIR="data/issue_1739/hf_dl/u_store"
PVCAP_MIRROR="data/issue_1739/hf_dl/pvcap_mirror"
DATA_REPO="superkaiba1/explore-persona-space-data"
PVCAP_PREFIX="issue1739_ctxmap/pvsynth/capture_store"
BRANCH="issue-1739"
LOG_DIR="/workspace/logs"
SENTINEL="$LOG_DIR/issue-1739-pvscore-results.json"

BEHAVIORS="${EPM_I1739_PV_BEHAVIORS:-evil hallucination sycophancy}"
# LINEAR map — matches the map family the committed real-rung baselines used,
# so the pvsynth-rung rho sits next to them cell-for-cell. The scorer also
# accepts mlp/kernel; a nonlinear pvsynth read is a separate question.
MAP_KIND="${EPM_I1739_PV_MAP_KIND:-linear}"
DEVICE="${EPM_I1739_PV_DEVICE:-cuda}"

PHASE="${PHASE:-all}"
want_phase() {
  case "$PHASE" in
    all) return 0 ;;
  esac
  local p
  for p in ${PHASE//,/ }; do
    [ "$p" = "$1" ] && return 0
  done
  return 1
}

mkdir -p "$LOG_DIR"

# ---- stage -----------------------------------------------------------------
if want_phase stage; then
  echo "[pvscore] phase=stage: pre-staging train/E1/DV inputs via issue1739_leg2.sh"
  bash scripts/issue1739_leg2.sh
  for b in $BEHAVIORS; do
    for s in "${b}_labeling" "${b}_extraction"; do
      [ -d "$STORE_ROOT/$s" ] || { echo "[pvscore] FATAL: store $s missing after stage" >&2; exit 1; }
    done
    [ -f "$RESULTS_ROOT/dv_dataset/$b/labeling.json" ] \
      || { echo "[pvscore] FATAL: dv_dataset/$b/labeling.json missing after stage" >&2; exit 1; }
  done
  [ -d "$U_STORE_DIR" ] || { echo "[pvscore] FATAL: u_store missing after stage" >&2; exit 1; }

  echo "[pvscore] phase=stage: staging pvsynth capture store ($PVCAP_PREFIX)"
  mkdir -p "$PVCAP_MIRROR"
  uv run python -c "
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
from pathlib import Path
from explore_persona_space.orchestrate.hub import stage_hub_prefix
paths = stage_hub_prefix('$DATA_REPO', '$PVCAP_PREFIX', Path('$PVCAP_MIRROR'))
print(f'[pvscore] staged {len(paths)} pvsynth capture files')
"
  echo "[pvscore] phase=stage: complete ($(date -u +%FT%TZ))"
fi

# stage_hub_prefix lands files at <dest>/<repo-relative path> (a VERBATIM
# prefix mirror, #928) — NOT flat in <dest>. Resolve the real store dir by
# locating the capture manifest rather than hard-coding the mirrored depth,
# so a producer-side prefix change cannot silently mis-point the consumer.
resolve_pvcap() {
  local m
  m="$(find "$PVCAP_MIRROR" -name '_capture_manifest.json' -print -quit 2>/dev/null || true)"
  [ -n "$m" ] || { echo "[pvscore] FATAL: _capture_manifest.json not found under $PVCAP_MIRROR" >&2; exit 1; }
  dirname "$m"
}

# ---- score -----------------------------------------------------------------
if want_phase score; then
  PVCAP_DIR="$(resolve_pvcap)"
  echo "[pvscore] phase=score: pvsynth_store=$PVCAP_DIR map_kind=$MAP_KIND behaviors='$BEHAVIORS'"
  # Per-behavior checkpoint/resume lives in the scorer
  # (<out-root>/<behavior>/percell/pvsynth_transfer.jsonl), so an interrupted
  # leg resumes rather than recomputing.
  uv run python scripts/issue1739_pvsynth_arms.py \
    --behaviors $BEHAVIORS \
    --store-root "$STORE_ROOT" \
    --u-store "$U_STORE_DIR" \
    --pvsynth-store "$PVCAP_DIR" \
    --map-kind "$MAP_KIND" \
    --device "$DEVICE" \
    --out-root "$PV_ROOT"
  for b in $BEHAVIORS; do
    [ -f "$PV_ROOT/$b/all_arms_spearman.json" ] \
      || { echo "[pvscore] FATAL: $b/all_arms_spearman.json missing after score" >&2; exit 1; }
  done
  echo "[pvscore] phase=score: complete ($(date -u +%FT%TZ))"
fi

# ---- upload ----------------------------------------------------------------
if want_phase upload; then
  # #1880 push race: three rounds write this branch concurrently, and the
  # upload helper only fetch-then-retries (it never rebases), so a stale base
  # would fail both attempts. Advance the clone to the tip FIRST — the results
  # are still uncommitted working-tree files here, so a ff-merge is clean and
  # the helper's commit then lands directly on the current tip.
  echo "[pvscore] phase=upload: syncing clone to origin/$BRANCH before commit"
  git fetch origin "$BRANCH"
  git merge --ff-only "origin/$BRANCH"
  uv run python scripts/issue1739_upload.py --stage results-git --branch "$BRANCH"
  echo "[pvscore] phase=upload: complete ($(date -u +%FT%TZ))"
fi

# ---- sentinel + terminal marker -------------------------------------------
# FAIL LOUD rather than declaring done on absent artifacts: `[phase=done]` and
# the sentinel are the poller's completion contract, so a run that skipped or
# lost the score phase must NOT emit them (a `present: false` row plus
# [phase=done] would read as a successful empty leg).
for b in $BEHAVIORS; do
  [ -f "$PV_ROOT/$b/all_arms_spearman.json" ] || {
    echo "[pvscore] FATAL: refusing to write sentinel — $b/all_arms_spearman.json absent" >&2
    exit 1
  }
done
uv run python -c "
import json, subprocess, time
from pathlib import Path
root = Path('$PV_ROOT')
per = []
for b in '$BEHAVIORS'.split():
    p = root / b / 'all_arms_spearman.json'
    d = json.loads(p.read_text())
    per.append({
        'behavior': b,
        'present': True,
        'n_transfer_rows': len(d.get('transfer_rows') or []),
        'n_per_layer_rows': len(d.get('per_layer_rows') or []),
    })
sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
Path('$SENTINEL').write_text(json.dumps({
    'issue': 1739, 'leg': 'pvsynth_arm_scoring', 'rung': 'pvsynth',
    'map_kind': '$MAP_KIND', 'behaviors': '$BEHAVIORS'.split(),
    'per_behavior': per, 'git_commit': sha,
    'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}, indent=2))
print('[pvscore] sentinel written -> $SENTINEL')
"
echo "[phase=done] pvsynth arm scoring complete ($(date -u +%FT%TZ))"
