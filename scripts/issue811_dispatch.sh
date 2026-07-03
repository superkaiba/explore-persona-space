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
#   bash scripts/issue811_dispatch.sh                 # full production sweep (v1 turn_nl round)
#   bash scripts/issue811_dispatch.sh --smoke         # tiny end-to-end smoke
#   bash scripts/issue811_dispatch.sh --skip-extract  # store already on disk/HF
#   EPM_SKIP_UPLOAD=1 bash scripts/issue811_dispatch.sh --smoke   # local CPU smoke
#   # maxp-winner-mapchange round (plan §10 dispatch card; adds --maxp extraction,
#   # --test-summary maxp KILL-1, 3-summary fits, committed-mean parity + F1
#   # offset decomposition, round-owned dirs + HF prefix + artifact revision pins):
#   SUMMARY_VARIANT=maxp BEHAVIORS=em,fact,sycophancy LAYERS="7 14 21" \
#     PRIMARY_LAYER=14 bash scripts/issue811_dispatch.sh
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

# ---- flags (order-independent; BEHAVIORS/LAYERS/PRIMARY_LAYER/SOURCES also
# accept env-var defaults — the plan §10 dispatch line passes them as env) ----
SMOKE=""
SKIP_EXTRACT=""
BEHAVIORS="${BEHAVIORS:-em,sycophancy,fact}"
LAYERS="${LAYERS:-7 14 21}"
PRIMARY_LAYER="${PRIMARY_LAYER:-14}"
SOURCES="${SOURCES:-}"          # empty = the 16 #537 train cids per behavior
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

# ---- SUMMARY_VARIANT arm (constants only — the phase code paths are IDENTICAL) ----
# Default (turn_nl) preserves the completed v1 round VERBATIM. SUMMARY_VARIANT=maxp
# is the #811 maxp-winner-mapchange round (plan §4 item 5): the round's OWN dirs +
# HF prefix, --maxp added to the extract flags, the KILL-1 gate re-pointed at
# --test-summary maxp, three-summary fits, and the critic-advisory artifact
# revision pins. Everything else (phase order, filters, smoke scaling) is shared.
SUMMARY_VARIANT="${SUMMARY_VARIANT:-turn_nl}"
case "$SUMMARY_VARIANT" in
  turn_nl)
    ROUND_DIR="eval_results/issue_811"
    HF_ROUND_PREFIX="issue811_turn_nl_mapchange"
    CELLS_DIR="eval_results/issue_811/cells"
    FIG_DIR="figures/issue_811"
    EXTRACT_SUMMARY_FLAGS="--turn-nl"
    FIT_SUMMARIES="mean turn_nl"
    TEST_SUMMARY="turn_nl"
    SUMMARY_DOC="mean_vs_turn_nl_summary.json"
    ;;
  maxp)
    ROUND_DIR="eval_results/issue_811/maxp-winner-mapchange"
    HF_ROUND_PREFIX="issue811_maxp_mapchange"
    CELLS_DIR="$ROUND_DIR/cells"
    FIG_DIR="figures/issue_811/maxp-winner-mapchange"
    EXTRACT_SUMMARY_FLAGS="--turn-nl --maxp"
    FIT_SUMMARIES="mean turn_nl maxp"
    TEST_SUMMARY="maxp"
    SUMMARY_DOC="summary_three_summaries.json"
    # Critic advisory: PIN the reused-artifact revisions (full shas resolved live
    # from the short pins e663b7cc / f6b7b0d0 via HfApi.repo_info on 2026-07-02).
    export EPM_I537_ADAPTER_REVISION="e663b7cc6f9bb133b4df6d8508afa8c091b388dc"
    export EPM_RB_REVISION="f6b7b0d0a94bf896e5816af2fe6421a900d0ae6e"
    # Phase-0 staging is INVALID for this round: no prior store carries v0_maxp
    # (plan §4 item 5 — the r7 stage-completeness verifier would refuse anyway,
    # but fail fast + loud here).
    if [ -n "${EPM_PHASE0_STAGE_PREFIX:-}" ]; then
      echo "ERROR: EPM_PHASE0_STAGE_PREFIX must be UNSET for SUMMARY_VARIANT=maxp — no prior phase-0 store carries v0_maxp (plan §4 item 5); this round extracts phase 0 fresh." >&2
      exit 2
    fi
    ;;
  *) echo "unknown SUMMARY_VARIANT: $SUMMARY_VARIANT (turn_nl|maxp)" >&2; exit 2 ;;
esac

STORE_DIR="$ROUND_DIR/analysis_tensors"
PHASE0_DIR="$ROUND_DIR/phase0_base_leg"
KILL1_JSON="$ROUND_DIR/kill1_base_leg_validity.json"
# Round args shared by BOTH issue811_fit invocations (phase0 gate + Phase-3 fits).
FIT_ROUND_ARGS="--test-summary $TEST_SUMMARY --out-dir $CELLS_DIR --store-prefix $HF_ROUND_PREFIX/analysis_tensors --phase0-prefix $HF_ROUND_PREFIX/phase0_base_leg"

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
  # Plan §13 smoke check 1: print the mean/maxp span bounds + last-3 token ids
  # per teacher-forced read (the extractor's [maxp-span] debug line).
  export EPM_I811_SPAN_DEBUG=1
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

# ---- Phase 0-stage: reuse a COMPLETED phase-0 base-leg store from HF (opt-in) ----
# The att-20260701-233116 production run completed the phase-0 base leg cleanly (3
# behaviors x 16 sources x 30 targets @ L14) and the crash's EXIT trap uploaded the
# FULL store (incl. every per-source .done sentinel) to HF under
# issue811_partial/att-.../eval_results_issue_811/phase0_base_leg. Re-extracting costs
# ~5.6h wall on the L4 lane — do not repay it. When EPM_PHASE0_STAGE_PREFIX is set
# (value = that HF prefix), snapshot_download it into $PHASE0_DIR, then VERIFY the
# store is COMPLETE for THIS run's resolved (behaviors x sources) grid: each cell dir
# has its atomic .done sentinel AND the expected per-source target npz at the primary
# layer. On a COMPLETE stage, skip the phase-0 extraction loop (parity 0a + gate 0b
# still run on the staged store). On a SHORTFALL, FAIL LOUD listing the missing cells
# — the phase-0 extractor has NO per-cell resume-skip (it re-runs every cell it is
# handed), so a partial stage cannot be safely "topped up" by falling through to
# extraction (that would re-extract ALL cells, clobbering the staged good ones), and
# stamping a trusted-complete flag over a partial dir is banned. Fail-fast: the
# operator re-stages a complete prefix or unsets the env var to extract from scratch.
PHASE0_STAGED=""
if [ -n "${EPM_PHASE0_STAGE_PREFIX:-}" ] && [ -z "$SKIP_EXTRACT" ]; then
  echo "[phase=phase0_stage] staging completed phase-0 store from HF prefix=$EPM_PHASE0_STAGE_PREFIX -> $PHASE0_DIR"
  # Resolve the per-behavior source lists + this run's target set for the verify.
  # (Production: 30 eval targets/source; smoke: the --targets subset.)
  P0_TARGETS_ARG=""
  case "$PHASE0_SMOKE_ARGS" in
    *"--targets "*) P0_TARGETS_ARG="$(echo "$PHASE0_SMOKE_ARGS" | sed -n 's/.*--targets \([^ ]*\).*/\1/p')" ;;
  esac
  P0_SOURCES_JSON=""
  IFS=',' read -r -a P0_BEH_ARR <<< "$BEHAVIORS"
  for beh in "${P0_BEH_ARR[@]}"; do
    P0_SOURCES_JSON="${P0_SOURCES_JSON}${beh}=$(resolve_sources "$beh");"
  done
  # shellcheck disable=SC2086
  uv run python scripts/issue811_stage_phase0.py \
    --hf-prefix "$EPM_PHASE0_STAGE_PREFIX" \
    --out "$PHASE0_DIR" \
    --primary-layer "$PRIMARY_LAYER" \
    --sources-spec "$P0_SOURCES_JSON" \
    ${P0_TARGETS_ARG:+--targets "$P0_TARGETS_ARG"}
  PHASE0_STAGED="1"
  echo "[phase=phase0_stage] staged + verified complete for behaviors=$BEHAVIORS at L$PRIMARY_LAYER — skipping phase-0 re-extraction"
fi

# ---- Phase 0: BASE-LEG-ONLY re-extraction + KILL-1 PRE-SPEND gate (GPU->CPU) ----
if [ -n "$PHASE0_STAGED" ]; then
  echo "[phase=phase0] skipped (staged a completed phase-0 store from HF; parity + gate run on it)"
elif [ -n "$SKIP_EXTRACT" ]; then
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
  $FIT_ROUND_ARGS $FIT_SMOKE_ARGS $PHASE0_LOCAL_ARGS
PHASE0_RC=$?
set -e
if [ "$PHASE0_RC" -eq 3 ]; then
  echo "[phase=phase0_gate] KILL1_TRIGGERED: $TEST_SUMMARY base-leg validity gate collapsed vs mean at L$PRIMARY_LAYER (>=2/3 behaviors). See $KILL1_JSON. HALT BEFORE the ~7 GPU-h Phase-1 paired re-extraction (plan §7, failure_class: data)." >&2
  exit 3
elif [ "$PHASE0_RC" -ne 0 ]; then
  echo "[phase=phase0_gate] phase0 gate failed with rc=$PHASE0_RC (not a KILL-1 collapse) — HALT." >&2
  exit "$PHASE0_RC"
fi
echo "[phase=phase0_gate] KILL-1 PASS ($TEST_SUMMARY base-leg validity gate holds) — proceed to Phase 1"

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
          $EXTRACT_SUMMARY_FLAGS --out "$STORE_DIR" --gpu-id "$GPU_ID" $EXTRACT_SMOKE_ARGS
    done
  done
  echo "[phase=extract] all cells extracted -> $STORE_DIR"
fi

# ---- Phase 2: upload the store to HF, then RELEASE the GPU pod (Upload Policy) ----
echo "[phase=upload] starting"
if [ "$SUMMARY_VARIANT" = "maxp" ]; then
  EPM_I811_ROUND_DIR="$ROUND_DIR" EPM_I811_HF_PREFIX="$HF_ROUND_PREFIX" EPM_I811_REQUIRE_RAW=1 \
    uv run python scripts/issue811_upload_store.py
else
  uv run python scripts/issue811_upload_store.py
fi
echo "[phase=upload] store uploaded + verified (GPU pod may be released here on the GCP lane)"

# ---- Phase 3: vectorized fits (CPU) ----
echo "[phase=fit] starting"
# maxp arm: when no explicit --local-root was given and the just-extracted store
# is on disk, read it locally (skips the HF tree walk, which 504s on this repo;
# an explicit --local-root always wins; a --skip-extract resume with no local
# store falls back to the round's HF prefix via $FIT_ROUND_ARGS --store-prefix).
# The default turn_nl arm keeps the v1 behavior verbatim (HF read unless
# --local-root is passed).
FIT_STORE_LOCAL_ARGS="$FIT_LOCAL_ARGS"
if [ "$SUMMARY_VARIANT" = "maxp" ] && [ -z "$FIT_STORE_LOCAL_ARGS" ] && [ -d "$STORE_DIR" ]; then
  FIT_STORE_LOCAL_ARGS="--local-root $STORE_DIR"
fi
# shellcheck disable=SC2086
uv run python scripts/issue811_fit.py --behaviors ${BEHAVIORS//,/ } --layers $LAYERS \
  --primary-layer "$PRIMARY_LAYER" --summaries $FIT_SUMMARIES \
  $FIT_ROUND_ARGS $FIT_SMOKE_ARGS $FIT_STORE_LOCAL_ARGS

# ---- Phase 3b (maxp round only): committed-mean replication read (plan §6b) ----
# Report-only: compares THIS round's re-extracted mean cells vs the completed v1
# run's committed cells; a flipped Δ/floor CALL is a replication-stability
# finding, never fatal. Same phase in smoke + production (report scale differs).
if [ "$SUMMARY_VARIANT" = "maxp" ]; then
  echo "[phase=parity_committed] starting committed-mean replication read"
  uv run python scripts/issue811_mean_parity_check.py --compare-committed \
    --run-cells-dir "$CELLS_DIR" \
    --committed-cells-dir eval_results/issue_811/cells \
    --committed-out "$ROUND_DIR/mean_call_replication_vs_v1.json"
  echo "[phase=parity_committed] report written -> $ROUND_DIR/mean_call_replication_vs_v1.json"
fi

# ---- Phase 4: assemble side-by-side deliverables (CPU) ----
echo "[phase=analyze] starting"
uv run python scripts/issue811_analyze.py --cells-dir "$CELLS_DIR" --out-dir "$ROUND_DIR" \
  --summary-filename "$SUMMARY_DOC"

# ---- Phase 4b (maxp round only): F1 offset decomposition (CPU, plan §5/§9) ----
# Standard read of this line; runs against the round's OWN local store + cells
# (no HF re-download). Smoke passes the same caps the smoke store was built with.
if [ "$SUMMARY_VARIANT" = "maxp" ]; then
  echo "[phase=offset_decomposition] starting F1 offset decomposition"
  OD_SMOKE_ARGS=""
  if [ -n "$SMOKE" ]; then
    OD_SMOKE_ARGS="--max-sources 2 --max-targets-per-source 3 --target-dim 4"
  fi
  # shellcheck disable=SC2086
  uv run python scripts/issue811_offset_decomposition.py \
    --behaviors ${BEHAVIORS//,/ } --layers $LAYERS --summaries $FIT_SUMMARIES \
    --local-store-root "$STORE_DIR" --cells-dir "$CELLS_DIR" \
    --store-prefix "$HF_ROUND_PREFIX/analysis_tensors" \
    --out "$ROUND_DIR/offset_decomposition.json" $OD_SMOKE_ARGS
  echo "[phase=offset_decomposition] wrote $ROUND_DIR/offset_decomposition.json"
fi

# ---- Phase 5: figures (CPU) ----
echo "[phase=figures] starting"
# shellcheck disable=SC2086
uv run python scripts/issue811_figures.py --summary-json "$ROUND_DIR/$SUMMARY_DOC" \
  --out-dir "$FIG_DIR" --summaries $FIT_SUMMARIES

# ---- End-of-run sentinel + terminal phase line (poll_pipeline contract) ----
EPM_I811_VARIANT="$SUMMARY_VARIANT" EPM_I811_HF_PREFIX="$HF_ROUND_PREFIX" \
  EPM_I811_ROUND_DIR="$ROUND_DIR" uv run python - <<'PY'
import datetime, json, os, time
from pathlib import Path
log_dir = Path(os.environ.get("EPM_LOG_DIR", "/workspace/logs"))
if not log_dir.exists():
    log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
variant = os.environ.get("EPM_I811_VARIANT", "turn_nl")
hf_prefix = os.environ.get("EPM_I811_HF_PREFIX", "issue811_turn_nl_mapchange")
round_dir = os.environ.get("EPM_I811_ROUND_DIR", "eval_results/issue_811")
payload = {
    "sentinel_schema_version": 1,
    "kind": "epm:results",
    "version": 1,
    "task_id": 811,
    "by": "issue811_dispatch",
    "ts": datetime.datetime.now(datetime.UTC).isoformat(),
    "note": f"issue811 {variant}-vs-mean map-change: phase0 base-leg + KILL-1 pre-spend gate + "
            f"paired extract + upload + fit + analyze + figures complete. Store: "
            f"{hf_prefix}/analysis_tensors/ (+ phase0_base_leg/ + raw_completions/). "
            f"Deliverables under {round_dir}/.",
}
out = log_dir / f"issue-811-epm_results-{time.time_ns()}.json"
out.write_text(json.dumps(payload, indent=2))
print(f"sentinel written: {out}")
PY

echo "[phase=done] issue811 dispatch complete"
