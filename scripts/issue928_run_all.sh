#!/usr/bin/env bash
# Issue #928 linear pipeline driver (plan §4.1): Phase 0 gate → G generation →
# P parse → B capture → U upload (issue928_extract_thinking_store.py, all in
# one process), then Phase F fits + nulls + bootstrap (issue928_fit_
# decomposition.py, EPM_FIT_DEVICE=cuda), then figures (+ HF upload), then the
# finalize step that writes the ONE epm:results sentinel (round 2 — the
# extract phase's sentinel is epm:progress). Smoke = the SAME driver with
# CONTEXTS=5 (unification default — the subset threads through every phase:
# extract reads --contexts, fits/figures enumerate the produced store
# manifest, upload commits what exists). Restartability: rollouts persist per
# generation group, store blobs skip-if-valid, fit layers checkpoint under
# eval_results/issue_928/partial/ — a crash-restart with EXTRA_EXTRACT=
# "--skip-gen" reuses everything already completed.
#
#   full run  : REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue928_run_all.sh
#   pod smoke : CONTEXTS=5 REPO_ROOT="$WORKLOAD_ROOT" bash scripts/issue928_run_all.sh
#
# Terminal [phase=done] is printed HERE (the component scripts print
# extract_done / fits_done / figures_done so the poller never sees a
# premature done). Non-zero rc from any phase aborts (set -e) — the gate-fail
# path exits 3 with a failure sentinel written by the extract script.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/explore-persona-space}"
cd "$REPO_ROOT"
# GCE lane exports tokens via its startup script and has NO .env — source
# conditionally, never unconditionally inside a classified chain (gotchas.md #923).
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi

CONTEXTS="${CONTEXTS:-}"            # empty = all 50 battery contexts
N_PERMS="${N_PERMS:-1000}"          # plan §6: 1,000 selection-symmetric draws
N_BOOT="${N_BOOT:-2000}"            # plan §6: 2,000 paired bootstrap draws
EXTRA_EXTRACT="${EXTRA_EXTRACT:-}"  # e.g. "--rung greedy" / "--skip-gen"
EXTRA_FIT="${EXTRA_FIT:-}"          # e.g. "--layers 0 1 --no-mlp" (smoke)
UPLOAD_PREFIX="${UPLOAD_PREFIX:-issue928_cot_decomposition/fit_results}"
FIGURES_UPLOAD_PREFIX="${FIGURES_UPLOAD_PREFIX:-issue928_cot_decomposition/figures}"

CTX_FLAG=""
if [ -n "$CONTEXTS" ]; then CTX_FLAG="--contexts $CONTEXTS"; fi

echo "[phase=extract]"
# shellcheck disable=SC2086
uv run python scripts/issue928_extract_thinking_store.py --gpu $CTX_FLAG $EXTRA_EXTRACT

echo "[phase=fits]"
# shellcheck disable=SC2086
EPM_FIT_DEVICE="${EPM_FIT_DEVICE:-cuda}" uv run python scripts/issue928_fit_decomposition.py \
  --store "$REPO_ROOT/data/issue_928/store" \
  --out "$REPO_ROOT/eval_results/issue_928" \
  --n-perms "$N_PERMS" --n-boot "$N_BOOT" \
  --upload-prefix "$UPLOAD_PREFIX" $EXTRA_FIT

echo "[phase=figures]"
# Pod-side figures upload to the HF issue prefix (round-2 fix: git is the
# canonical figure home but the pod/GCE instance cannot commit — the VM-side
# analyzer pulls + commits after teardown).
uv run python scripts/issue928_figures.py \
  --results "$REPO_ROOT/eval_results/issue_928" \
  --store "$REPO_ROOT/data/issue_928/store" \
  --rollouts "$REPO_ROOT/data/issue_928/raw_completions/thinking_rollouts" \
  --out "$REPO_ROOT/figures/issue_928" \
  --upload-prefix "$FIGURES_UPLOAD_PREFIX"

echo "[phase=finalize]"
# The ONE epm:results sentinel of the pipeline — fires only after fits +
# figures + uploads all completed (round-2 sentinel-timing fix; the extract
# phase's sentinel is epm:progress).
uv run python scripts/issue928_finalize.py \
  --store "$REPO_ROOT/data/issue_928/store" \
  --results "$REPO_ROOT/eval_results/issue_928" \
  --figures "$REPO_ROOT/figures/issue_928" \
  --out-dir "$REPO_ROOT/data/issue_928"

echo "[phase=done]"
