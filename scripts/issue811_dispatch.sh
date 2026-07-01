#!/usr/bin/env bash
# Issue #811 — turn_nl-vs-mean map-change measurement driver.
#
# ONE entrypoint for BOTH the smoke and the production sweep (smoke = the sweep
# with --smoke scaling each phase to a tiny cell subset). Every phase derives its
# cell subset from the SAME --behaviors / --sources / --layers filters, so the
# smoke exercises the IDENTICAL dispatcher the sweep runs (PASS_UNIFIED — see the
# epm:smoke-architecture-check marker).
#
# KILL-1 is a GENUINE PRE-SPEND gate (round-2 BLOCKER kill1-not-pre-spend-gate,
# plan §4.0/§7). Phase 0 does a BASE-LEG-ONLY re-extraction (no #537 adapter, base
# model θ0 forward pass, ~1 GPU-h) at the PRIMARY layer, fits the MLP-vs-shuffle
# base-leg validity gate, and CHECKS KILL-1 INLINE — HALTing before the ~7 GPU-h
# Phase-1 paired re-extraction if turn_nl's gate collapses vs mean. The base leg
# is genuinely cheaper: one model, no adapter apply, layer 14 only.
#
# Phases:
#   0. phase0    (GPU) — BASE-LEG-ONLY re-extraction (scripts/issue811_phase0_extract.py):
#      base θ0 mean + turn_nl v0 at the primary layer over 16 sources × 30 targets,
#      per source (CVD-pinned, #545). Then:
#        0a. parity  (CPU) — scripts/issue811_mean_parity_check.py: assert the
#            re-extracted base mean v0 reproduces #667's stored v0 on 2-3 cells
#            (plan §13 check 1; fail-loud on drift, failure_class: code).
#        0b. phase0-gate (CPU) — scripts/issue811_fit.py --phase0-gate: the
#            MLP-vs-shuffle base-leg validity gate at the primary layer, writes
#            eval_results/issue_811/kill1_base_leg_validity.json. EXIT 3 on
#            collapse -> the dispatcher HALTs here (failure_class: data). PASS ->
#            proceed to Phase 1.
#   1. extract  (GPU) — per-source-adapter PAIRED re-extraction via
#      scripts/issue667_extract.py --turn-nl, writing v0/v_plus + v0_turn_nl/
#      v_plus_turn_nl per cell under eval_results/issue_811/analysis_tensors/
#      {behavior}/{source}_seed42/. CVD-pinned per source (#545). KILL-2 lives in
#      the extractor's _locate_turn_close_newline assert (non-zero exit -> HALT).
#   2. upload   (CPU) — bulk-commit + fresh-listing-verify the store to HF
#      (issue811_turn_nl_mapchange/analysis_tensors/ + .../phase0_base_leg/) BEFORE
#      releasing the GPU (Upload Policy #521; #664/#778 GPU-release-before-CPU-phase).
#      Skipped with EPM_SKIP_UPLOAD=1 (a local smoke reads the store from disk).
#   3. fit      (CPU) — scripts/issue811_fit.py: ridge headline (byte-identical
#      #722) + vectorized MLP-vs-shuffle validity gate per (behavior, layer,
#      summary). KILL-1 is NOT re-decided here (Phase 0 already gated it, pre-spend);
#      the Phase-2 gate margins are logged as an informational re-check only.
#      On the GCP lane the GPU pod is RELEASED before this phase (plan §9).
#   4. analyze  (CPU) — scripts/issue811_analyze.py: side-by-side deliverables.
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
PHASE0_DIR="eval_results/issue_811/phase0_base_leg"
KILL1_JSON="eval_results/issue_811/kill1_base_leg_validity.json"

# The smoke scales the SAME phases down: 1 behavior, layer 14 only, >=2 sources,
# a handful of targets + probes. Every phase reads these same filters.
if [ -n "$SMOKE" ]; then
  BEHAVIORS="$(echo "$BEHAVIORS" | cut -d, -f1)"
  LAYERS="14"
  PRIMARY_LAYER="14"
  [ -n "$SOURCES" ] || SOURCES="default,sp_swe"
  EXTRACT_SMOKE_ARGS="--max-probes 2 --targets default,sp_swe,sp_doctor"
  PHASE0_SMOKE_ARGS="--max-probes 2 --targets default,sp_swe,sp_doctor"
  FIT_SMOKE_ARGS="--smoke"
  PARITY_CELLS="2"
else
  EXTRACT_SMOKE_ARGS=""
  PHASE0_SMOKE_ARGS=""
  FIT_SMOKE_ARGS=""
  PARITY_CELLS="3"
fi

# Resolve the per-behavior source list (the 16 #537 train cids, or the --sources
# subset). Shared by Phase 0 (base leg) + Phase 1 (paired) so both extract the
# SAME cell grid.
resolve_sources() {  # $1 = behavior
  if [ -n "$SOURCES" ]; then
    echo "$SOURCES"
  else
    uv run python -c "
import sys; sys.path.insert(0, 'scripts')
from issue667_dispatch import select_sources
print(','.join(select_sources('$1', None)))
"
  fi
}

# ---- Phase 0: BASE-LEG-ONLY re-extraction + KILL-1 PRE-SPEND gate (GPU->CPU) ----
if [ -n "$SKIP_EXTRACT" ]; then
  echo "[phase=phase0] skipped (--skip-extract; phase0 store already on disk/HF)"
else
  echo "[phase=phase0] starting base-leg-only extraction behaviors=$BEHAVIORS primary_layer=$PRIMARY_LAYER sources=${SOURCES:-<16 train cids>}"
  IFS=',' read -r -a BEH_ARR <<< "$BEHAVIORS"
  for beh in "${BEH_ARR[@]}"; do
    SRC_LIST="$(resolve_sources "$beh")"
    IFS=',' read -r -a SRC_ARR <<< "$SRC_LIST"
    for src in "${SRC_ARR[@]}"; do
      # CVD pinned in the LAUNCHER env per cell (#545) + matching --gpu-id.
      echo "[phase=phase0] base-leg cell behavior=$beh source=$src"
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES="$GPU_ID" VLLM_WORKER_MULTIPROC_METHOD=spawn \
        uv run python scripts/issue811_phase0_extract.py \
          --behavior "$beh" --source-cid "$src" \
          --layers "$PRIMARY_LAYER" --primary-layer "$PRIMARY_LAYER" \
          --out "$PHASE0_DIR" --gpu-id "$GPU_ID" $PHASE0_SMOKE_ARGS
    done
  done
  echo "[phase=phase0] base-leg cells extracted -> $PHASE0_DIR"
fi

# ---- Phase 0a: mean-parity check vs #667 (fail-loud BEFORE any spend) ----
# Local smoke (EPM_SKIP_UPLOAD=1) has no HF-store reference wired for #811-owned
# cells, but #667's reference IS on HF; the parity check reads the local phase0
# store vs #667's HF store, so it runs in both smoke + production (it needs the
# #667 reference cells to exist for the chosen behavior/cells).
if [ -z "$SKIP_EXTRACT" ]; then
  echo "[phase=parity] starting mean-parity check vs #667 (plan §13)"
  PARITY_BEH="$(echo "$BEHAVIORS" | cut -d, -f1)"
  # shellcheck disable=SC2086
  uv run python scripts/issue811_mean_parity_check.py \
    --phase0-root "$PHASE0_DIR" --behavior "$PARITY_BEH" \
    --layer "$PRIMARY_LAYER" --n-cells "$PARITY_CELLS"
  echo "[phase=parity] mean-parity check PASS (re-extracted mean reproduces #667)"
fi

# ---- Phase 0b: KILL-1 pre-spend gate on the base leg ----
echo "[phase=phase0_gate] starting KILL-1 base-leg validity gate"
FIT_LOCAL_ARGS=""
[ -n "$LOCAL_ROOT" ] && FIT_LOCAL_ARGS="--local-root $LOCAL_ROOT"
# The KILL-1 gate is a PRE-SPEND gate: on a fresh run it runs in the SAME run that
# JUST wrote the phase0 base-leg store to $PHASE0_DIR (Phase 0's extractor), and
# BEFORE the store is uploaded to HF (Phase 2, after the gate PASSes). On the fresh
# path the store therefore ALWAYS exists on disk at gate time by construction but is
# NOT yet on HF, so the gate MUST read the local mirror — reading the HF prefix would
# 404 or, worse, read an empty tree and PASS vacuously. Default to the local phase0
# store whenever it exists on disk (every fresh run); an explicit --local-root
# override always wins. The HF-prefix fallback (PHASE0_LOCAL_ARGS empty) is RESTRICTED
# to a --skip-extract resume (that store IS on HF from the prior run, and the fit-side
# fail-loud empty-store guard catches an empty/missing HF tree instead of PASSing
# vacuously — round-3 BLOCKER phase0-gate-reads-unuploaded-hf-store). On a FRESH
# NON-skip run the local phase0 store MUST exist by construction — Phase 0's extractor
# just wrote it; a missing $PHASE0_DIR means Phase 0 silently produced nothing, and
# falling back to HF (a stale/other-run store, or empty) would let the gate read the
# WRONG store, so HARD-FAIL instead (round-4 BLOCKER phase0-hf-fallback-not-skip-gated).
if [ -n "$LOCAL_ROOT" ]; then
  PHASE0_LOCAL_ARGS="--local-root $LOCAL_ROOT"
elif [ -d "$PHASE0_DIR" ]; then
  PHASE0_LOCAL_ARGS="--local-root $PHASE0_DIR"
elif [ -n "$SKIP_EXTRACT" ]; then
  echo "[phase=phase0_gate] no local phase0 store at $PHASE0_DIR (--skip-extract resume) — reading the HF prefix (the fit-side empty-store guard catches a vacuous HF tree)" >&2
  PHASE0_LOCAL_ARGS=""  # allowed ONLY on --skip-extract resume; HF has the store
else
  echo "[phase=phase0_gate] local phase0 store $PHASE0_DIR missing after Phase 0 extraction on a NON-skip run; refusing to fall back to HF on a non-skip run (Phase 0 produced nothing — would read a stale/empty store) — HALT (round-4 BLOCKER phase0-hf-fallback-not-skip-gated)." >&2
  exit 5
fi
# The gate EXITs 3 on KILL-1 fire; do not let set -e mask the intent — capture rc.
set +e
# shellcheck disable=SC2086
uv run python scripts/issue811_fit.py --phase0-gate \
  --behaviors ${BEHAVIORS//,/ } --primary-layer "$PRIMARY_LAYER" \
  $FIT_SMOKE_ARGS $PHASE0_LOCAL_ARGS
PHASE0_RC=$?
set -e
if [ "$PHASE0_RC" -eq 3 ]; then
  echo "[phase=phase0_gate] KILL1_TRIGGERED: turn_nl base-leg validity gate collapsed vs mean at L$PRIMARY_LAYER (>=2/3 behaviors). See $KILL1_JSON. HALT BEFORE the ~7 GPU-h Phase-1 paired re-extraction (plan §7, failure_class: data)." >&2
  exit 3
elif [ "$PHASE0_RC" -ne 0 ]; then
  echo "[phase=phase0_gate] phase0 gate failed with rc=$PHASE0_RC (not a KILL-1 collapse) — HALT." >&2
  exit "$PHASE0_RC"
fi
echo "[phase=phase0_gate] KILL-1 PASS (turn_nl base-leg validity gate holds) — proceed to Phase 1"

# ---- Phase 1: PAIRED re-extraction (GPU) — only reached after KILL-1 PASS ----
if [ -n "$SKIP_EXTRACT" ]; then
  echo "[phase=extract] skipped (--skip-extract; paired store already on disk/HF)"
else
  echo "[phase=extract] starting paired re-extraction behaviors=$BEHAVIORS layers=$LAYERS sources=${SOURCES:-<16 train cids>}"
  IFS=',' read -r -a BEH_ARR <<< "$BEHAVIORS"
  for beh in "${BEH_ARR[@]}"; do
    SRC_LIST="$(resolve_sources "$beh")"
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
# shellcheck disable=SC2086
uv run python scripts/issue811_fit.py --behaviors ${BEHAVIORS//,/ } --layers $LAYERS \
  --primary-layer "$PRIMARY_LAYER" $FIT_SMOKE_ARGS $FIT_LOCAL_ARGS

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
    "note": "issue811 turn_nl-vs-mean map-change: phase0 base-leg + KILL-1 pre-spend gate + "
            "paired extract + upload + fit + analyze + figures complete. Store: "
            "issue811_turn_nl_mapchange/analysis_tensors/ (+ phase0_base_leg/). "
            "Deliverables under eval_results/issue_811/.",
}
out = log_dir / f"issue-811-epm_results-{time.time_ns()}.json"
out.write_text(json.dumps(payload, indent=2))
print(f"sentinel written: {out}")
PY

echo "[phase=done] issue811 dispatch complete"
