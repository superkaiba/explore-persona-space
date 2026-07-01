#!/usr/bin/env bash
# Issue #811 — turn_nl-vs-mean map-change measurement driver.
#
# ONE entrypoint for BOTH the smoke and the production sweep (smoke = the sweep
# with --smoke scaling each phase to a tiny cell subset). Every phase derives its
# cell subset from the SAME --behaviors / --sources / --layers filters, so the
# smoke exercises the IDENTICAL dispatcher the sweep runs (PASS_UNIFIED — see the
# epm:smoke-architecture-check marker). The base-leg (M0) validity read the KILL-1
# gate needs is produced INSIDE the extract pass (the base leg is half the paired
# store), so there is no separate Phase-0 forward-pass phase.
#
# Phases:
#   1. extract  (GPU) — per-source-adapter paired re-extraction via
#      scripts/issue667_extract.py --turn-nl, writing v0/v_plus + v0_turn_nl/
#      v_plus_turn_nl per cell under eval_results/issue_811/analysis_tensors/
#      {behavior}/{source}_seed42/. CVD-pinned per source (#545). KILL-2 lives in
#      the extractor's _locate_turn_close_newline assert (non-zero exit -> HALT).
#   2. upload   (CPU) — bulk-commit + fresh-listing-verify the store to HF
#      (issue811_turn_nl_mapchange/analysis_tensors/) BEFORE releasing the GPU
#      (Upload Policy #521; #664/#778 GPU-release-before-CPU-phase). Skipped with
#      EPM_SKIP_UPLOAD=1 (a local smoke reads the store from disk).
#   3. fit      (CPU) — scripts/issue811_fit.py: ridge headline (byte-identical
#      #722) + vectorized MLP-vs-shuffle validity gate per (behavior, layer,
#      summary); writes the KILL-1 base-leg decision (kill1_base_leg_validity.json).
#      On the GCP lane the GPU pod is RELEASED before this phase (plan §9).
#   4. analyze  (CPU) — scripts/issue811_analyze.py: side-by-side deliverables.
#      GATED on KILL-1: if the base-leg validity gate fired, HALT here with a loud
#      KILL1_TRIGGERED line (the orchestrator persists epm:failure data).
#   5. figures  (CPU) — scripts/issue811_figures.py: the hero + candidate figures.
#
# Pod-side contract (CLAUDE.md / poll_pipeline.py): emits [phase=<name>] lines, a
# terminal [phase=done] (RESERVED), and an end-of-run sentinel JSON at
# /workspace/logs/issue-811-<kind_slug>-<epoch>.json with _SENTINEL_REQUIRED_KEYS.
#
# Usage:
#   bash scripts/issue811_dispatch.sh                 # full production sweep
#   bash scripts/issue811_dispatch.sh --smoke         # tiny end-to-end smoke
#   bash scripts/issue811_dispatch.sh --skip-extract  # store already on disk/HF
#   EPM_SKIP_UPLOAD=1 bash scripts/issue811_dispatch.sh --smoke   # local CPU smoke
set -euo pipefail

REPO_ROOT="${WORKLOAD_ROOT:-/workspace/explore-persona-space}"
[ -d "$REPO_ROOT" ] || REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_PROJECT="${WANDB_PROJECT:-issue811}"
# uv run python does NOT auto-load .env; put creds in the env for every child.
set -a
# shellcheck disable=SC1091
[ -f .env ] && source .env
set +a

# ---- flags (order-independent) ----
SMOKE=""
SKIP_EXTRACT=""
BEHAVIORS="em,sycophancy,fact"
LAYERS="7 14 21"
PRIMARY_LAYER="14"
SOURCES=""          # empty = the 16 #537 train cids per behavior
GPU_ID="${EPM_GPU_ID:-0}"
LOCAL_ROOT=""       # fit reads the store from a LOCAL mirror (dispatcher smoke)
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE="1"; shift ;;
    --skip-extract) SKIP_EXTRACT="1"; shift ;;
    --behaviors) BEHAVIORS="${2:?}"; shift 2 ;;
    --layers) LAYERS="${2:?}"; shift 2 ;;
    --primary-layer) PRIMARY_LAYER="${2:?}"; shift 2 ;;
    --sources) SOURCES="${2:?}"; shift 2 ;;
    --gpu-id) GPU_ID="${2:?}"; shift 2 ;;
    --local-root) LOCAL_ROOT="${2:?}"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

STORE_DIR="eval_results/issue_811/analysis_tensors"

# The smoke scales the SAME phases down: 1 behavior, layer 14 only, >=2 sources,
# a handful of targets + probes. Every phase reads these same filters.
if [ -n "$SMOKE" ]; then
  BEHAVIORS="$(echo "$BEHAVIORS" | cut -d, -f1)"
  LAYERS="14"
  PRIMARY_LAYER="14"
  [ -n "$SOURCES" ] || SOURCES="default,sp_swe"
  EXTRACT_SMOKE_ARGS="--max-probes 2 --targets default,sp_swe,sp_doctor"
  FIT_SMOKE_ARGS="--smoke"
else
  EXTRACT_SMOKE_ARGS=""
  FIT_SMOKE_ARGS=""
fi

# ---- Phase 1: paired re-extraction (GPU) ----
if [ -n "$SKIP_EXTRACT" ]; then
  echo "[phase=extract] skipped (--skip-extract; store already on disk/HF)"
else
  echo "[phase=extract] starting behaviors=$BEHAVIORS layers=$LAYERS sources=${SOURCES:-<16 train cids>}"
  IFS=',' read -r -a BEH_ARR <<< "$BEHAVIORS"
  for beh in "${BEH_ARR[@]}"; do
    # Resolve the source cids for this behavior (the 16 #537 train cids, or the
    # --sources subset). select_sources filters + validates against the train grid.
    if [ -n "$SOURCES" ]; then
      SRC_LIST="$SOURCES"
    else
      SRC_LIST="$(uv run python -c "
import sys; sys.path.insert(0, 'scripts')
from issue667_dispatch import select_sources
print(','.join(select_sources('$beh', None)))
")"
    fi
    IFS=',' read -r -a SRC_ARR <<< "$SRC_LIST"
    for src in "${SRC_ARR[@]}"; do
      # CVD pinned in the LAUNCHER env per cell (#545) + matching --gpu-id.
      echo "[phase=extract] cell behavior=$beh source=$src"
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES="$GPU_ID" VLLM_WORKER_MULTIPROC_METHOD=spawn \
        uv run python scripts/issue667_extract.py \
          --behavior "$beh" --source-cid "$src" \
          --layers $LAYERS --primary-layer "$PRIMARY_LAYER" \
          --turn-nl --out "$STORE_DIR" --gpu-id "$GPU_ID" $EXTRACT_SMOKE_ARGS
    done
  done
  echo "[phase=extract] all cells extracted -> $STORE_DIR"
fi

# ---- Phase 2: upload the store to HF, then RELEASE the GPU pod (Upload Policy) ----
echo "[phase=upload] starting"
uv run python scripts/issue811_upload_store.py
echo "[phase=upload] store uploaded + verified (GPU pod may be released here on the GCP lane)"

# ---- Phase 3: vectorized fits (CPU) ----
echo "[phase=fit] starting"
FIT_LOCAL_ARGS=""
[ -n "$LOCAL_ROOT" ] && FIT_LOCAL_ARGS="--local-root $LOCAL_ROOT"
# shellcheck disable=SC2086
uv run python scripts/issue811_fit.py --behaviors ${BEHAVIORS//,/ } --layers $LAYERS \
  $FIT_SMOKE_ARGS $FIT_LOCAL_ARGS

# ---- KILL-1 gate: HALT before analyze if the base-leg validity gate fired ----
KILL1_JSON="eval_results/issue_811/kill1_base_leg_validity.json"
if [ -f "$KILL1_JSON" ]; then
  FIRED="$(uv run python -c "import json;print(json.load(open('$KILL1_JSON')).get('fired'))")"
  if [ "$FIRED" = "True" ]; then
    echo "[phase=fit] KILL1_TRIGGERED: turn_nl base-leg validity gate collapsed vs mean at L$PRIMARY_LAYER (>=2/3 behaviors). See $KILL1_JSON. HALT before analyze/figures (plan §7, failure_class: data)." >&2
    # Non-zero exit -> the orchestrator persists epm:failure failure_class: data.
    exit 3
  fi
  echo "[phase=fit] KILL-1 PASS (turn_nl base-leg validity gate holds)"
fi

# ---- Phase 4: assemble side-by-side deliverables (CPU) ----
echo "[phase=analyze] starting"
uv run python scripts/issue811_analyze.py

# ---- Phase 5: figures (CPU) ----
echo "[phase=figures] starting"
uv run python scripts/issue811_figures.py

# ---- End-of-run sentinel + terminal phase line (poll_pipeline contract) ----
uv run python - <<'PY'
import datetime, json, os, time
from pathlib import Path
log_dir = Path(os.environ.get("EPM_LOG_DIR", "/workspace/logs"))
if not log_dir.exists():
    log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 811,
    "by": "issue811_dispatch",
    "ts": datetime.datetime.now(datetime.UTC).isoformat(),
    "note": "issue811 turn_nl-vs-mean map-change: extract + upload + fit + analyze + figures complete. "
            "Store: issue811_turn_nl_mapchange/analysis_tensors/. Deliverables under eval_results/issue_811/.",
}
out = log_dir / f"issue-811-epm_results-{time.time_ns()}.json"
out.write_text(json.dumps(payload, indent=2))
print(f"sentinel written: {out}")
PY

echo "[phase=done] issue811 dispatch complete"
